"""Drone-vs-background EPE analysis for AllTracker.

AllTracker is a dense point tracker; on a 2-frame clip its `Net.forward()`
degenerates to a plain optical-flow prediction (B,2,H,W), which is what this
script feeds into the same drone/background EPE pipeline used by the other
repos (memflow, waft, sea-raft, ...).

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is BACKWARD flow (t -> t-1), i.e.
I_t(x) ~= I_{t-1}(x + F(x)). The model is therefore run on (I_t, I_{t-1}) with
I_t as the reference frame; frame 0000 (no t-1) is never a reference. Samples
come from fd_eval_common.list_samples(root, TWO_FRAME). One summary row goes to
--output_csv, one row per evaluated pair to --pairs_csv (bootstrap CIs).

Metric conventions: the paper definitions in fd_eval_common.compute_metrics
  - epe:       mean endpoint error (pixels)
  - epe_norm:  mean ||(du/W, dv/H)|| x 100
  - px1/3/5:   % of pixels with epe > 1/3/5 px  (error-rate, higher = worse)
  - fl:        % of pixels where (epe > 3 px) AND (epe / gt_magnitude > 5%)

Weights: loaded via fd_eval_common.load_state_dict_checked (raises on any
missing/unexpected key, prints the loaded-key count).

AllTracker has no CPU inference path: upstream Net.get_fmaps() hardcodes
`.cuda()` on the non-training forward path. --device cpu is rejected up front.

Usage:
    python scripts/analyze_drone_epe.py \\
        --ckpt ckpts/alltracker.pth \\
        --datasets_path $FD_DATA \\
        --root Nature3 \\
        --image_size 432 960 \\
        --output_csv results.csv
"""
import argparse
import csv
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
# Script dir first: fd_eval_common.py / drone_split.py next to this file must win
# over any same-named module elsewhere on sys.path (e.g. a container's /app copy).
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
import fd_eval_common
from fd_eval_common import (TWO_FRAME, list_samples, compute_metrics, ref_base,
                            PairDumper, aggregate, summary_row, SUMMARY_FIELDS,
                            load_state_dict_checked)

from drone_split import load_split_frames, filter_samples

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

from nets.alltracker import Net


class DroneFlowDataset(data.Dataset):
    """FlyingDrones (Nature3) dataset returning drone label mask.

    Handles two layouts:
      - root_path/images/, root_path/flow/, root_path/labels/  (single scene)
      - root_path/scene_001/images/, ...                       (multi-scene)
    """

    def __init__(self, root_path, resize_shape=None):
        self.resize_shape = resize_shape
        # (I_t, I_{t-1}, <t>_flow.npy, <t>_label.png): GT is backward flow t -> t-1,
        # so I_t is the reference (first) frame. Frame 0000 is never a reference.
        self.samples = list_samples(root_path, TWO_FRAME)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, pf, pl = self.samples[idx]  # p1 = I_t (reference), p2 = I_{t-1}
        img1 = np.array(Image.open(p1).convert('RGB')).astype(np.float32)
        img2 = np.array(Image.open(p2).convert('RGB')).astype(np.float32)
        flow = np.load(pf).astype(np.float32)
        mask = (np.array(Image.open(pl)) > 0).astype(np.float32)

        if self.resize_shape is not None:
            H, W = self.resize_shape
            img1 = np.array(Image.fromarray(img1.astype(np.uint8)).resize((W, H))).astype(np.float32)
            img2 = np.array(Image.fromarray(img2.astype(np.uint8)).resize((W, H))).astype(np.float32)
            oh, ow = flow.shape[:2]
            ft = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            ft = F.interpolate(ft, size=(H, W), mode='bilinear', align_corners=False)
            ft[:, 0] *= W / ow
            ft[:, 1] *= H / oh
            flow = ft.squeeze(0).permute(1, 2, 0).numpy()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mt, size=(H, W), mode='nearest').squeeze().numpy()

        return (torch.from_numpy(img1).permute(2, 0, 1),   # C,H,W [0,255]
                torch.from_numpy(img2).permute(2, 0, 1),
                torch.from_numpy(flow).permute(2, 0, 1),   # 2,H,W
                torch.from_numpy(mask))                     # H,W


def _default_pairs_csv(output_csv, ckpt, H, W):
    """<output_csv minus .csv>__<ckpt-stem>_<W>x<H>_pairs.csv (unique per ckpt/res,
    so several checkpoints appending to one results.csv never share a pairs file)."""
    base = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    stem = os.path.basename(str(ckpt).rstrip('/'))
    for ext in ('.tar', '.pth', '.pt', '.ckpt', '.safetensors'):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
    return f"{base}__{stem}_{W}x{H}_pairs.csv"


def _print_summary(rows, ckpt, dataset_name, skipped):
    avgs = aggregate(rows) if rows else {}
    print("\n" + "=" * 65)
    print(f"Checkpoint: {os.path.basename(str(ckpt))}")
    print(f"Pairs: {len(rows)} evaluated, {skipped} skipped  |  Dataset: {dataset_name}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print("-" * 65)
    for metric in ('epe', 'epe_norm', 'px1_err', 'px3_err', 'px5_err', 'fl'):
        a = avgs.get(f'{metric}_all',   float('nan'))
        d = avgs.get(f'{metric}_drone', float('nan'))
        b = avgs.get(f'{metric}_bg',    float('nan'))
        print(f"  {metric:<26} {a:>8.4f} {d:>8.4f} {b:>8.4f}")
    print("=" * 65)


def _write_summary_csv(path, row, job_id, append=False):
    """Legacy results.csv layout: job_id + fd_eval_common.SUMMARY_FIELDS."""
    fields = ['job_id'] + list(SUMMARY_FIELDS)
    out = {'job_id': job_id}
    for k in SUMMARY_FIELDS:
        v = row[k]
        out[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not (append and os.path.exists(path))
    with open(path, 'w' if new else 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(out)
    print(f"\nCSV written: {path}")


def build_model(ckpt_path, device):
    is_tiny = 'tiny' in os.path.splitext(os.path.basename(ckpt_path))[0]
    model = Net(seqlen=16, use_basicencoder=True, no_split=True) if is_tiny else Net(seqlen=16)
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt_data['model'] if 'model' in ckpt_data else ckpt_data
    load_state_dict_checked(model, state_dict, name=f"AllTracker{'-tiny' if is_tiny else ''}")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',  required=True, help="Path to checkpoint .pth file "
                        "(alltracker.pth or alltracker_tiny.pth)")
    parser.add_argument('--datasets_path', default='',
                        help="Parent directory that contains the dataset root")
    parser.add_argument('--root',  default='',
                        help="Dataset subdirectory within datasets_path")
    parser.add_argument('--image_size', nargs=2, type=int, default=None,
                        metavar=('H', 'W'), help="Resize input to H x W (default: native resolution)")
    parser.add_argument('--max_samples',  type=int, default=500)
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore --max_samples')
    parser.add_argument('--num_workers',  type=int, default=1)
    parser.add_argument('--iters',        type=int, default=4, help="Inference iterations")
    parser.add_argument('--job_id',       default='unknown')
    parser.add_argument('--pairs_csv',     default=None,
                        help='Per-pair metrics CSV (default: <output_csv>__<ckpt-stem>_<W>x<H>_pairs.csv)')
    parser.add_argument('--output_csv',   required=True)
    parser.add_argument('--device',       default='cuda',
                        help="Compute device: cuda only (AllTracker has no CPU forward path)")
    cli = parser.parse_args()

    if cli.device != 'cuda':
        print(f"ERROR: AllTracker has no CPU inference path (upstream Net.get_fmaps() "
              f"hardcodes .cuda()); --device {cli.device} is not supported.")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available, and AllTracker has no CPU fallback.")
        sys.exit(1)

    device = torch.device('cuda')

    root_path = os.path.join(cli.datasets_path, cli.root).rstrip('/') if cli.root else cli.datasets_path
    dataset_name = os.path.basename(root_path)

    resize_shape = tuple(cli.image_size) if cli.image_size else None

    print(f"Dataset:    {root_path}")
    print(f"Checkpoint: {cli.ckpt}")
    print(f"Resolution: {resize_shape if resize_shape else 'native'}")
    print(f"Device:     {device}")

    print(f"fd_eval_common: {fd_eval_common.__file__}")
    model = build_model(cli.ckpt, device)

    dataset = DroneFlowDataset(root_path, resize_shape=resize_shape)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(cli.split_file))
    if len(dataset) == 0:
        print(f"ERROR: no samples found under {root_path}")
        sys.exit(1)

    if resize_shape is not None:
        H, W = resize_shape
    else:
        _, _, flow0, _ = dataset[0]
        H, W = flow0.shape[-2:]

    n = len(dataset) if cli.split_file else min(len(dataset), cli.max_samples)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=cli.num_workers, drop_last=False)

    pairs_csv = cli.pairs_csv or _default_pairs_csv(cli.output_csv, cli.ckpt, H, W)
    dumper = PairDumper(pairs_csv)

    print(f"Evaluating {n} pairs (I_t, I_t-1)...")
    with torch.no_grad():
        for i, (img1, img2, flow_gt, drone_mask) in enumerate(loader):
            if i >= n:
                break
            img1, img2 = img1.to(device), img2.to(device)
            flow_gt, drone_mask = flow_gt.to(device), drone_mask.to(device)

            images = torch.stack([img1, img2], dim=1)  # 1,2,3,H,W in [0,255]: (I_t, I_t-1)

            try:
                full_flows, _visconfs, _, _ = model.forward(images, iters=cli.iters, is_training=False)
                pred = full_flows  # 1,2,H,W
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at {H}x{W} — aborting")
                sys.exit(1)

            if pred.shape[-2:] != flow_gt.shape[-2:]:
                pred = F.interpolate(pred, size=flow_gt.shape[-2:],
                                     mode='bilinear', align_corners=False)

            m = compute_metrics(pred[0], flow_gt[0], drone_mask[0])
            dumper.add(ref_base(dataset.samples[i]), m)

            done = i + 1
            if done % 100 == 0 and dumper.rows:
                ag = aggregate(dumper.rows)
                print(f"  {done}/{n}  epe={ag['epe_all']:.3f}  epe_drone={ag['epe_drone']:.3f}")

    torch.cuda.empty_cache()
    dumper.close()
    skipped = n - len(dumper.rows)   # OOM pairs + pairs not reached after an early stop
    print(f"Per-pair CSV: {pairs_csv}")

    _print_summary(dumper.rows, cli.ckpt, dataset_name, skipped)
    row = summary_row(dumper.rows, os.path.basename(cli.ckpt), dataset_name, f"{H}x{W}", skipped=skipped)
    _write_summary_csv(cli.output_csv, row, cli.job_id, append=True)


if __name__ == '__main__':
    main()
