"""Drone-vs-background EPE analysis for MemFlow.

Computes the paper's optical-flow metrics split by drone pixels vs background
(fd_eval_common.compute_metrics):
  - epe:       mean endpoint error (pixels)
  - epe_norm:  mean ||(du/W, dv/H)|| x 100
  - px1/3/5:   % of pixels with epe > 1/3/5 px  (error-rate, higher = worse)
  - fl:        % of pixels where (epe > 3 px) AND (epe / gt_magnitude > 5%)
One summary row is appended to --output_csv, one row per pair goes to --pairs_csv.

Ground truth: Nature3 ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1. MemFlow
is run on the time-reversed clip ``[t+3, t+2, t+1, t, t-1]`` (fd_eval_common.
MEMFLOW_CLIP, edge frames clamped, frame 0000 never a reference) with ONE
InferenceCore per sample, stepping through consecutive pairs exactly like upstream
evaluate_MemFlowNet.py (``step(images[:, ti:ti+2], end=(ti == T-2))``, memory
written on every non-final step since mem_every=1). The LAST flow,
(I_t, I_{t-1}) -> flow t -> t-1, is scored. The tiled path keeps one core per tile
across the clip.

--zero_flow skips model building and inference entirely (pred = 0) and labels the
summary row checkpoint as ``zero_flow``.

Usage:
    python scripts/analyze_drone_epe.py \\
        --name MemFlowNet --stage sintel \\
        --ckpt ckpts/MemFlowNet_sintel.pth \\
        --datasets_path $FD_DATA \\
        --root Nature3 \\
        --image_size 432 960 \\
        --output_csv results.csv
"""
import argparse
import csv
import importlib
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
# Script dir goes to sys.path[0] last so the drone_split.py / fd_eval_common.py next
# to this file (bind-mounted in the container) win over any other copy.
for _p in (os.path.join(_repo_root, 'core'), _repo_root, _script_dir):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from drone_split import load_split_frames, filter_samples
from fd_eval_common import (MEMFLOW_CLIP, TWO_FRAME, PairDumper, aggregate,
                            compute_metrics, list_samples, load_state_dict_checked,
                            ref_base, summary_row)

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

from core.Networks import build_network
from core.utils.utils import InputPadder, forward_interpolate
from inference import inference_core_skflow as inference_core


def load_config(name, stage):
    mapping = {
        'MemFlowNet': {
            'things': 'configs.things_memflownet',
            'sintel': 'configs.sintel_memflownet',
            'kitti':  'configs.kitti_memflownet',
            'spring': 'configs.spring_memflownet',
        },
        'MemFlowNet_T': {
            'things':       'configs.things_memflownet_t',
            'things_kitti': 'configs.things_memflownet_t_kitti',
            'sintel':       'configs.sintel_memflownet_t',
            'kitti':        'configs.kitti_memflownet_t',
        },
    }
    if name not in mapping:
        raise ValueError(f"Unknown model name: {name}")
    stages = mapping[name]
    if stage not in stages:
        raise ValueError(f"Unknown stage '{stage}' for {name}. Valid: {list(stages.keys())}")
    module = importlib.import_module(stages[stage])
    return module.get_cfg()


class DroneFlowDataset(data.Dataset):
    """FlyingDrones (Nature3) dataset returning a reversed clip + GT + drone mask.

    Samples are ``(*frame_paths, flow_path, label_path)`` from
    ``fd_eval_common.list_samples(root, offsets)`` (single scene or parent of scenes).
    __getitem__ returns (frames (N,3,H,W) in [0,255], flow (2,H,W), mask (H,W));
    with load_images=False frames is an empty tensor (zero-flow baseline).
    """

    def __init__(self, root_path, resize_shape=None, offsets=MEMFLOW_CLIP, load_images=True):
        self.resize_shape = resize_shape
        self.n_frames = len(offsets)
        self.load_images = load_images
        self.samples = list_samples(root_path, offsets)

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        # uint8-range float — MemFlow normalises to [-1,1] at inference time
        img = np.array(Image.open(path).convert('RGB')).astype(np.float32)
        if self.resize_shape is not None:
            H, W = self.resize_shape
            img = np.array(Image.fromarray(img.astype(np.uint8)).resize((W, H))).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame_paths, pf, pl = s[:self.n_frames], s[self.n_frames], s[self.n_frames + 1]
        if self.load_images:
            frames = torch.stack([self._load_img(p) for p in frame_paths], dim=0)
        else:
            frames = torch.zeros(0)
        flow = np.load(pf).astype(np.float32)
        mask = (np.array(Image.open(pl)) > 0).astype(np.float32)

        if self.resize_shape is not None:
            H, W = self.resize_shape
            oh, ow = flow.shape[:2]
            ft = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            ft = F.interpolate(ft, size=(H, W), mode='bilinear', align_corners=False)
            ft[:, 0] *= W / ow
            ft[:, 1] *= H / oh
            flow = ft.squeeze(0).permute(1, 2, 0).numpy()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mt, size=(H, W), mode='nearest').squeeze().numpy()

        return (frames,                                     # N,3,H,W [0,255]
                torch.from_numpy(flow).permute(2, 0, 1),    # 2,H,W
                torch.from_numpy(mask))                     # H,W


@torch.no_grad()
def run_clip(model, cfg, images):
    """Step one fresh InferenceCore through ``images`` (1,T,3,H,W, normalised, padded).

    Mirrors evaluate_MemFlowNet.validate_sintel: step(images[:, ti:ti+2],
    end=(ti == T-2)), optional warm start. Returns the LAST flow_up (1,2,H,W), i.e.
    the flow of the final pair (images[:, T-2] -> images[:, T-1]).
    """
    T = images.shape[1]
    proc = inference_core.InferenceCore(model, config=cfg)
    add_pe = ('rope' in cfg and cfg.rope)
    warm = ('warm_start' in cfg and cfg.warm_start)
    flow_prev = None
    flow_pre = None
    for ti in range(T - 1):
        flow_low, flow_pre = proc.step(images[:, ti:ti + 2], end=(ti == T - 2),
                                       add_pe=add_pe, flow_init=flow_prev)
        if warm:
            flow_prev = forward_interpolate(flow_low[0])[None].to(images.device)
    return flow_pre


@torch.no_grad()
def predict(model, cfg, frames, tile=None, overlap=20):
    """frames: (1,T,3,H,W) in [0,255] (reversed clip). Returns (1,2,H,W) last-pair flow."""
    padder = InputPadder(frames.shape)
    images = padder.pad(frames)
    images = 2.0 * (images / 255.0) - 1.0
    if tile is None:
        flow_pre = run_clip(model, cfg, images)
        return padder.unpad(flow_pre[0]).unsqueeze(0)

    th, tw = tile
    H_, W_ = images.shape[-2:]
    th = min(th, H_); tw = min(tw, W_)

    def _axis(d, t):
        if t >= d:
            return [0]
        step = max(1, t - overlap)
        pos = list(range(0, d - t + 1, step))
        if pos[-1] != d - t:
            pos.append(d - t)
        return pos

    grid = [(h, w) for h in _axis(H_, th) for w in _axis(W_, tw)]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, th, device=images.device),
        torch.linspace(-1, 1, tw, device=images.device), indexing='ij')
    w_tile = torch.exp(-(yy * yy + xx * xx) / (2 * 0.5 ** 2))[None, None]
    flow_acc = torch.zeros((1, 2, H_, W_), device=images.device)
    weight_acc = torch.zeros((1, 1, H_, W_), device=images.device)
    for h_i, w_i in grid:
        fp = run_clip(model, cfg, images[..., h_i:h_i + th, w_i:w_i + tw])  # one core per tile
        flow_acc[:, :, h_i:h_i + th, w_i:w_i + tw] += fp[0:1] * w_tile
        weight_acc[:, :, h_i:h_i + th, w_i:w_i + tw] += w_tile
    pred_padded = flow_acc / weight_acc.clamp(min=1e-6)
    return padder.unpad(pred_padded[0]).unsqueeze(0)


def _default_pairs_csv(output_csv, ckpt_label, H, W):
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    return f"{stem}__{ckpt_label}_{W}x{H}_pairs.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',  required=True, choices=['MemFlowNet', 'MemFlowNet_T'],
                        help="Model variant")
    parser.add_argument('--stage', required=True,
                        help="Config stage: things/sintel/kitti/spring (or things_kitti for _T)")
    parser.add_argument('--ckpt',  default=None,
                        help="Path to checkpoint .pth file (not needed with --zero_flow)")
    parser.add_argument('--datasets_path', default='',
                        help="Parent directory that contains the dataset root")
    parser.add_argument('--root',  default='',
                        help="Dataset subdirectory within datasets_path")
    parser.add_argument('--image_size', nargs=2, type=int, default=None,
                        metavar=('H', 'W'), help="Resize input to H x W")
    parser.add_argument('--max_samples',  type=int, default=500)
    parser.add_argument('--zero_flow', action='store_true',
                        help="Predict zero flow instead of running the model, to "
                             "measure the do-nothing baseline this metric scores. "
                             "Drone motion is sub-pixel at low resolution, so EPE "
                             "over drone pixels can reward predicting nothing; "
                             "without this number the leaderboard has no floor. "
                             "No model is built, no checkpoint loaded, no image read; "
                             "the summary row's checkpoint is 'zero_flow'.")
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore --max_samples')
    parser.add_argument('--num_workers',  type=int, default=1)
    parser.add_argument('--job_id',       default='unknown')
    parser.add_argument('--output_csv',   required=True)
    parser.add_argument('--pairs_csv',    default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<W>x<H>_pairs.csv)')
    parser.add_argument('--device',       default='cuda',
                        help="Compute device: cuda or cpu")
    parser.add_argument('--tile',         type=int, nargs=2, default=None, metavar=('TH', 'TW'),
                        help='If set, use Gaussian-blended tiled inference at this tile size.')
    cli = parser.parse_args()
    if not cli.zero_flow and not cli.ckpt:
        parser.error("--ckpt is required unless --zero_flow is given")

    if cli.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to cpu")
        cli.device = 'cpu'

    cfg = load_config(cli.name, cli.stage)

    resize_shape = tuple(cli.image_size) if cli.image_size else tuple(cfg.image_size)
    H, W = resize_shape

    root_path = os.path.join(cli.datasets_path, cli.root).rstrip('/') if cli.root else cli.datasets_path
    dataset_name = os.path.basename(root_path)

    device = torch.device(cli.device)
    ckpt_label = 'zero_flow' if cli.zero_flow else os.path.basename(cli.ckpt)
    ckpt_stem = 'zero_flow' if cli.zero_flow else os.path.splitext(ckpt_label)[0]

    print(f"Dataset:    {root_path}")
    print(f"Checkpoint: {ckpt_label if cli.zero_flow else cli.ckpt}")
    print(f"Resolution: {H}x{W}")
    print(f"Device:     {device}")

    model = None
    if cli.zero_flow:
        offsets = TWO_FRAME
        print("Zero-flow baseline: no model, pred = 0")
    else:
        offsets = MEMFLOW_CLIP
        print(f"Clip offsets {offsets}: step through, score last pair (t -> t-1)")
        model = build_network(cfg).to(device).eval()
        ckpt_data = torch.load(cli.ckpt, map_location='cpu')
        ckpt_model = ckpt_data['model'] if 'model' in ckpt_data else ckpt_data
        load_state_dict_checked(model, ckpt_model, strip_prefixes=('module.',),
                                name=f'{cli.name}-{cli.stage}')

    dataset = DroneFlowDataset(root_path, resize_shape=resize_shape, offsets=offsets,
                               load_images=not cli.zero_flow)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(cli.split_file))
    n = len(dataset) if cli.split_file else min(len(dataset), cli.max_samples)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=cli.num_workers, drop_last=False)

    pairs_csv = cli.pairs_csv or _default_pairs_csv(cli.output_csv, ckpt_stem, H, W)
    dumper = PairDumper(pairs_csv)
    oom_count = 0

    print(f"Evaluating {n} samples...")
    with torch.no_grad():
        for i, (frames, flow_gt, drone_mask) in enumerate(loader):
            if i >= n:
                break
            if cli.zero_flow:
                pred = torch.zeros_like(flow_gt)
            else:
                try:
                    pred = predict(model, cfg, frames.to(device), tile=cli.tile)
                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    torch.cuda.empty_cache()
                    if oom_count == 1:
                        print(f"  WARNING: CUDA OOM at sample {i} ({H}x{W}). "
                              "Skipping sample (no per-pair row).")
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
                s = summary_row(dumper.rows, ckpt_label, dataset_name, f"{H}x{W}")
                print(f"  {done}/{n}  epe={s['epe']:.3f}  epe_drone={s['epe_drone']:.3f}")
    dumper.close()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    rows = dumper.rows
    skipped = n - len(rows)
    if skipped:
        print(f"\n  {skipped} sample(s) not evaluated ({oom_count} CUDA OOM)")

    summ = summary_row(rows, ckpt_label, dataset_name, f"{H}x{W}", skipped=skipped)
    agg = aggregate(rows)
    print("\n" + "=" * 65)
    print(f"Checkpoint: {ckpt_label}")
    print(f"Samples: {len(rows)} (skipped {skipped})  |  Dataset: {dataset_name}  |  Res: {H}x{W}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print("-" * 65)
    for metric in ('epe', 'epe_norm', 'px1_err', 'px3_err', 'px5_err', 'fl'):
        print(f"  {metric:<26} {agg[metric + '_all']:>8.4f} {agg[metric + '_drone']:>8.4f} "
              f"{agg[metric + '_bg']:>8.4f}")
    print("=" * 65)

    fieldnames = [
        'job_id', 'checkpoint', 'dataset', 'image_size',
        'epe', 'epe_drone',
        'epe_norm', 'epe_norm_drone',
        'px1_err', 'px1_err_drone',
        'px3_err', 'px3_err_drone',
        'px5_err', 'px5_err_drone',
        'fl', 'fl_drone',
        'samples_processed', 'samples_skipped',
    ]
    row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in summ.items()}
    row['job_id'] = cli.job_id

    out_dir = os.path.dirname(os.path.abspath(cli.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.exists(cli.output_csv)
    with open(cli.output_csv, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nCSV written: {cli.output_csv}")
    print(f"Pairs CSV:   {pairs_csv}")


if __name__ == '__main__':
    main()
