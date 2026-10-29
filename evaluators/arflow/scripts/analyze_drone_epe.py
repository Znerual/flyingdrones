"""Drone-vs-background EPE analysis for ARFlow on the SyntheticDrones dataset.

Computes the paper's optical-flow metrics split by drone pixels vs background
(fd_eval_common.compute_metrics):
  - epe:       mean endpoint error (pixels)
  - epe_norm:  mean ||(du/W, dv/H)|| x 100
  - px1/3/5:   % of pixels with epe > 1/3/5 px  (error-rate, higher = worse)
  - fl:        % of pixels where (epe > 3 px) AND (epe / gt_magnitude > 5%)
One summary row goes to --output_csv, one row per pair to --pairs_csv.

Ground truth: Nature3 ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so the
model is asked for t -> t-1 with I_t as reference (frame 0000 never a reference,
edge frames clamped):
  - 2-frame checkpoints:  input (I_t, I_{t-1})              -> flows_fw[0]
    (pwclite.py L292: forward_2_frames(x[0], x[1]) = x0 -> x1)
  - 3-frame (_mv) ckpts:  input (I_{t+1}, I_t, I_{t-1})      -> flows_fw[0]
    (pwclite.py forward_3_frames(x0, x1, x2): x2 warped with flow[:, 2:] (L241),
     flows_12 = flo[:, 2:] (L281), res_dict["flows_fw"] = flows_12 (L297), i.e.
     middle -> last = t -> t-1; index 0 = finest level, upsampled to full res).

Usage:
    python scripts/analyze_drone_epe.py \
        --ckpt checkpoints/KITTI15/pwclite_ar.tar \
        --root $FD_DATA/Nature3 \
        --image_size 432 960 \
        --job_id arflow_kitti15_std \
        --output_csv results_csv/arflow_kitti15_std.csv
"""
import argparse
import csv
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
# Script dir goes to sys.path[0] last so the drone_split.py / fd_eval_common.py next
# to this file (bind-mounted in the container) win over any other copy.
for _p in (_repo_root, _script_dir):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from drone_split import load_split_frames, filter_samples
from fd_eval_common import (ARFLOW_WINDOW, TWO_FRAME, PairDumper, aggregate,
                            compute_metrics, list_samples, load_state_dict_checked,
                            ref_base, summary_row)

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from easydict import EasyDict

from models.pwclite import PWCLite
from utils.torch_utils import load_checkpoint


def detect_model_cfg(ckpt_path: str) -> EasyDict:
    """Detect n_frames and reduce_dense from checkpoint weight shapes."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    ctx_in = sd['context_networks.convs.0.0.weight'].shape[1]
    # ctx_in = (feat_dim + 2) * (n_frames - 1)
    # reduce_dense=True → feat_dim=32, so for n_frames=2: 34, n_frames=3: 68
    if ctx_in == 34:
        n_frames, reduce_dense = 2, True
    elif ctx_in == 68:
        n_frames, reduce_dense = 3, True
    elif ctx_in == 565:
        n_frames, reduce_dense = 2, False
    elif ctx_in == 1296:
        n_frames, reduce_dense = 3, False
    else:
        raise ValueError(f"Cannot detect model config from context_networks input dim {ctx_in}")
    return EasyDict({'upsample': True, 'n_frames': n_frames, 'reduce_dense': reduce_dense})


def window_offsets(n_frames: int):
    return TWO_FRAME if n_frames == 2 else ARFLOW_WINDOW


class DroneFlowDataset(data.Dataset):
    """SyntheticDrones dataset returning the model's input window, flow GT and drone mask.

    Samples are ``(*frame_paths, flow_path, label_path)`` from
    ``fd_eval_common.list_samples(root, offsets)``: (I_t, I_{t-1}) for 2-frame and
    (I_{t+1}, I_t, I_{t-1}) for 3-frame checkpoints. __getitem__ returns
    (frames (N,3,H,W) in [0,255], flow (2,H,W), mask (H,W)).
    """

    def __init__(self, root_path, resize_shape=None, offsets=TWO_FRAME):
        self.resize_shape = resize_shape
        self.n_frames = len(offsets)
        self.samples = list_samples(root_path, offsets)

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path, resize_hw=None):
        img = np.array(Image.open(path).convert('RGB')).astype(np.float32)
        if resize_hw is not None:
            H, W = resize_hw
            img = np.array(Image.fromarray(img.astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame_paths, pf, pl = s[:self.n_frames], s[self.n_frames], s[self.n_frames + 1]
        rs = self.resize_shape
        frames = torch.stack([self._load_img(p, rs) for p in frame_paths], dim=0)

        flow = np.load(pf).astype(np.float32)
        mask = (np.array(Image.open(pl)) > 0).astype(np.float32)

        if rs is not None:
            H, W = rs
            oh, ow = flow.shape[:2]
            ft = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            ft = F.interpolate(ft, size=(H, W), mode='bilinear', align_corners=False)
            ft[:, 0] *= W / ow
            ft[:, 1] *= H / oh
            flow = ft.squeeze(0).permute(1, 2, 0).numpy()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mt, size=(H, W), mode='nearest').squeeze().numpy()

        return (frames,
                torch.from_numpy(flow).permute(2, 0, 1),
                torch.from_numpy(mask))


def run_model(model, frames: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Run ARFlow on a window; return flow t -> t-1 as [B,2,H,W].

    frames: (B, N, 3, H, W) in [0,255]; N=2: [I_t, I_{t-1}], N=3: [I_{t+1}, I_t, I_{t-1}].
    Both cases read out['flows_fw'][0] (2-frame: x0 -> x1; 3-frame: x1 -> x2).
    """
    if frames.shape[1] != n_frames:
        raise ValueError(f"model expects {n_frames} frames, window has {frames.shape[1]}")
    inp = torch.cat([frames[:, k] / 255.0 for k in range(n_frames)], dim=1)  # [B, 3N, H, W]
    with torch.no_grad():
        out = model(inp)
    return out['flows_fw'][0]  # [B, 2, H, W] (upsample=True puts it at full res)


def _default_pairs_csv(output_csv, ckpt_label, H, W):
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    return f"{stem}__{ckpt_label}_{W}x{H}_pairs.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        required=True, help='Path to .tar checkpoint')
    parser.add_argument('--root',        required=True, help='Path to dataset (e.g. /path/to/Nature3)')
    parser.add_argument('--image_size',  nargs=2, type=int, default=None, metavar=('H', 'W'),
                        help='Resize images before eval, e.g. --image_size 432 960')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore --max_samples')
    parser.add_argument('--job_id',      default='unknown')
    parser.add_argument('--output_csv',  required=True)
    parser.add_argument('--pairs_csv',   default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<W>x<H>_pairs.csv)')
    cli = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"Checkpoint: {cli.ckpt}")
    print(f"Dataset:    {cli.root}")
    if cli.image_size:
        print(f"Image size: {cli.image_size[0]}×{cli.image_size[1]}")

    cfg = detect_model_cfg(cli.ckpt)
    offsets = window_offsets(cfg.n_frames)
    print(f"Detected:   n_frames={cfg.n_frames}, reduce_dense={cfg.reduce_dense}; "
          f"window offsets {offsets} -> flows_fw[0] = t -> t-1")

    model = PWCLite(cfg)
    _, weights = load_checkpoint(cli.ckpt)
    load_state_dict_checked(model, weights, strip_prefixes=('module.',), name='ARFlow-PWCLite')
    model = model.to(device).eval()

    resize_shape = tuple(cli.image_size) if cli.image_size else None
    dataset = DroneFlowDataset(cli.root, resize_shape=resize_shape, offsets=offsets)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(cli.split_file))
    n = len(dataset) if cli.split_file else min(len(dataset), cli.max_samples)
    print(f"Samples:    {n} / {len(dataset)}")

    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=0, drop_last=False)

    if resize_shape:
        H, W = resize_shape
    else:
        s = dataset[0]   # index 1 = flow tensor [2,H,W]
        H, W = s[1].shape[-2], s[1].shape[-1]

    dataset_name = os.path.basename(cli.root.rstrip('/'))
    ckpt_stem = os.path.splitext(os.path.basename(cli.ckpt))[0]
    pairs_csv = cli.pairs_csv or _default_pairs_csv(cli.output_csv, ckpt_stem, H, W)
    dumper = PairDumper(pairs_csv)
    oom_count = 0

    print(f"Evaluating {n} samples...")
    for i, (frames, flow_gt, drone_mask) in enumerate(loader):
        if i >= n:
            break
        try:
            pred = run_model(model, frames.to(device), cfg.n_frames)
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            if oom_count == 1:
                print(f"  WARNING: CUDA OOM at sample {i} "
                      f"(image_size={resize_shape}). Skipping (no per-pair row).")
            if oom_count >= 5:
                print(f"  OOM on {oom_count} samples — stopping early.")
                break
            continue

        if pred.shape[-2:] != flow_gt.shape[-2:]:
            pred = F.interpolate(pred, size=flow_gt.shape[-2:],
                                 mode='bilinear', align_corners=False)
        m = compute_metrics(pred[0], flow_gt[0], drone_mask[0])
        dumper.add(ref_base(dataset.samples[i]), m)

        done = len(dumper.rows)
        if done % 100 == 0:
            s = summary_row(dumper.rows, ckpt_stem, dataset_name, f"{H}x{W}")
            print(f"  {done}/{n}  epe={s['epe']:.3f}  epe_drone={s['epe_drone']:.3f}")
    dumper.close()

    rows = dumper.rows
    skipped = n - len(rows)
    if skipped:
        print(f"\n  {skipped} sample(s) not evaluated ({oom_count} CUDA OOM)")

    summ = summary_row(rows, cli.ckpt, dataset_name, f"{H}x{W}", skipped=skipped)
    agg = aggregate(rows)
    print("\n" + "=" * 65)
    print(f"Checkpoint: {os.path.basename(cli.ckpt)}")
    print(f"Samples: {len(rows)} (skipped {skipped})  |  Dataset: {dataset_name}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print("-" * 65)
    for metric in ('epe', 'epe_norm', 'px1_err', 'px3_err', 'px5_err', 'fl'):
        print(f"  {metric:<26} {agg[metric + '_all']:>8.4f} {agg[metric + '_drone']:>8.4f} "
              f"{agg[metric + '_bg']:>8.4f}")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(cli.output_csv)), exist_ok=True)
    fieldnames = [
        'job_id', 'checkpoint', 'dataset', 'image_size',
        'epe',      'epe_drone',
        'epe_norm', 'epe_norm_drone',
        'px1_err',  'px1_err_drone',
        'px3_err',  'px3_err_drone',
        'px5_err',  'px5_err_drone',
        'fl',       'fl_drone',
        'samples_processed', 'samples_skipped',
    ]
    row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in summ.items()}
    row['job_id'] = cli.job_id
    with open(cli.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(row)
    print(f"\nCSV written: {cli.output_csv}")
    print(f"Pairs CSV:   {pairs_csv}")


if __name__ == '__main__':
    main()
