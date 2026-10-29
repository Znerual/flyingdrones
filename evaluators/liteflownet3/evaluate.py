"""LiteFlowNet3 evaluation on the synthetic drones dataset.

Computes all standard optical flow metrics split by drone pixels vs background,
using the label PNGs in the dataset. Appends a single CSV row to --output_csv.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so the
model is run on (I_t, I_{t-1}) with I_t as reference; frame 0000 is never a
reference (fd_eval_common.list_samples).

Metric conventions = paper definitions (fd_eval_common.compute_metrics):
  - epe:       mean endpoint error (pixels)
  - epe_norm:  sqrt((du/W)^2 + (dv/H)^2) * 100
  - px1/3/5:   % of pixels with epe > 1/3/5 px  (error-rate, higher = worse)
  - fl:        % of pixels where (epe > 3 px) AND (epe / gt_magnitude > 5%)
Summary = frame-weighted mean over pairs; one row per pair -> --pairs_csv.
CUDA-OOM pairs get no row and are counted in samples_skipped; after 5 OOMs the
resolution is aborted and an all-NaN summary row is written (as before).

Available model names (ptlflow):
  liteflownet3           — full model, sintel pretrained
  liteflownet3_pseudoreg — full model + pseudo-reg, kitti pretrained
  liteflownet3s          — small model, sintel pretrained
  liteflownet3s_pseudoreg— small model + pseudo-reg, kitti pretrained

Usage:
    python evaluate.py \\
        --cfg         config/drones-M.json \\
        --model_name  liteflownet3 \\
        --ckpt        models/liteflownet3-sintel.ckpt \\
        --datasets_path $FD_DATA \\
        --root        Nature3 \\
        --output_csv  results.csv
"""
import argparse
import csv
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from glob import glob
from PIL import Image

import ptlflow

# Script dir first so the local drone_split / fd_eval_common win over any other copy.
_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from drone_split import load_split_frames, filter_samples  # noqa: E402

# Shared Nature3 helpers (pairing, paper metrics, per-pair dump). Re-insert our own
# dir right before the import so a stale copy elsewhere on sys.path (e.g. baked into
# a container under /app) can never shadow the bind-mounted / local one.
sys.path.insert(0, _SCRIPT_DIR)
import fd_eval_common as fdc  # noqa: E402


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def json_to_args(path: str):
    with open(path) as f:
        cfg = json.load(f)
    args = argparse.Namespace(**cfg)
    if not hasattr(args, 'image_size'):
        args.image_size = [432, 960]
    return args


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DroneFlowDataset(data.Dataset):
    """FlyingDrones (Nature3) dataset — returns (img_t, img_{t-1}, flow_t, drone_mask_t)."""

    def __init__(self, root_path, resize_shape=None):
        self.resize_shape = resize_shape
        self.samples = []
        # Pairing (fixed 2026-10-07): ``flow/<t>_flow.npy`` is the BACKWARD flow
        # t -> t-1, so each sample is (I_t, I_{t-1}, <t>_flow, <t>_label) with I_t as
        # the reference frame; frame 0000 is never a reference. Labels are required
        # (as before). Flat scene dir or parent of scene dirs.
        self.samples = fdc.list_samples(root_path, fdc.TWO_FRAME, require_labels=True)
        print(f"DroneFlowDataset: {len(self.samples)} pairs (I_t, I_t-1) from {root_path}")

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
            oh, ow = img1.shape[:2]
            img1 = np.array(Image.fromarray(img1.astype(np.uint8)).resize((W, H))).astype(np.float32)
            img2 = np.array(Image.fromarray(img2.astype(np.uint8)).resize((W, H))).astype(np.float32)
            ft = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            ft = F.interpolate(ft, size=(H, W), mode='bilinear', align_corners=False)
            ft[:, 0] *= W / ow
            ft[:, 1] *= H / oh
            flow = ft.squeeze(0).permute(1, 2, 0).numpy()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mt, size=(H, W), mode='nearest').squeeze().numpy()

        return (
            torch.from_numpy(img1).permute(2, 0, 1),   # 3 x H x W  [0, 255]
            torch.from_numpy(img2).permute(2, 0, 1),
            torch.from_numpy(flow).permute(2, 0, 1),   # 2 x H x W
            torch.from_numpy(mask),                    # H x W
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(pred, flow_gt, drone_mask):
    """Paper metrics for one (2, H, W) pair (delegates to fd_eval_common.compute_metrics)."""
    return fdc.compute_metrics(pred, flow_gt, drone_mask)


def _verify_weights(model, ckpt_path):
    """Fail loudly unless the model holds exactly the checkpoint's tensors.

    ptlflow.restore_model already uses a strict load_state_dict; this re-checks it
    and prints the loaded-key count (RAFT-loader-bug guard).
    """
    if not osp.exists(ckpt_path):
        print(f"[liteflownet3] '{ckpt_path}' is a ptlflow pretrained name; "
              f"relying on ptlflow's strict load_state_dict")
        return
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = sd.get('state_dict', sd)
    msd = model.state_dict()
    missing = sorted(set(msd) - set(sd))
    unexpected = sorted(set(sd) - set(msd))
    differ = [k for k in msd if k in sd and not torch.equal(msd[k].detach().cpu(), sd[k].to(msd[k].dtype))]
    print(f"[liteflownet3] loaded {len(msd) - len(missing)}/{len(msd)} keys; "
          f"missing={len(missing)} unexpected={len(unexpected)} value_mismatch={len(differ)}")
    if missing or unexpected or differ:
        raise RuntimeError(f"[liteflownet3] weight/key mismatch: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]} differ={differ[:5]}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, img1, img2):
    """Run model on a batch; img1/img2 are [0, 255] float tensors B×3×H×W."""
    i1 = img1 / 255.0
    i2 = img2 / 255.0
    inputs = {'images': torch.stack([i1, i2], dim=1)}
    with torch.no_grad():
        preds = model(inputs)
    return preds['flows'][:, 0]   # B × 2 × H × W, pixel units


def _tile_grid(H, W, th, tw, min_overlap=20):
    """FlowFormer++-style: positions covering image, last tile flush to edge."""
    if min_overlap >= th or min_overlap >= tw:
        raise ValueError("min_overlap must be < tile size")
    def axis(dim, t):
        if t >= dim:
            return [0]
        step = max(1, t - min_overlap)
        pos = list(range(0, dim - t + 1, step))
        if pos[-1] != dim - t:
            pos.append(dim - t)
        return pos
    return [(h, w) for h in axis(H, th) for w in axis(W, tw)]


def _gaussian_weight(th, tw, sigma, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, th, device=device),
        torch.linspace(-1, 1, tw, device=device),
        indexing='ij',
    )
    return torch.exp(-(yy * yy + xx * xx) / (2 * sigma * sigma))[None, None]


def run_inference_tiled(model, img1, img2, tile_size, min_overlap=20, sigma=0.5):
    """Gaussian-blended tiled inference."""
    B, _, H, W = img1.shape
    th, tw = tile_size
    th = min(th, H); tw = min(tw, W)
    grid = _tile_grid(H, W, th, tw, min_overlap)
    flow_acc = torch.zeros((B, 2, H, W), device=img1.device)
    weight_acc = torch.zeros((B, 1, H, W), device=img1.device)
    w_tile = _gaussian_weight(th, tw, sigma, img1.device)
    for h, w in grid:
        t1 = img1[:, :, h:h + th, w:w + tw]
        t2 = img2[:, :, h:h + th, w:w + tw]
        f = run_inference(model, t1, t2)
        if f.shape[-2:] != (th, tw):
            f = F.interpolate(f, size=(th, tw), mode='bilinear', align_corners=False)
        flow_acc[:, :, h:h + th, w:w + tw] += f * w_tile
        weight_acc[:, :, h:h + th, w:w + tw] += w_tile
    return flow_acc / weight_acc.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Maps checkpoint filename stems to their ptlflow model class name
CKPT_TO_MODEL = {
    'liteflownet3-sintel':        'liteflownet3',
    'liteflownet3-kitti':         'liteflownet3_pseudoreg',
    'liteflownet3s-sintel':       'liteflownet3s',
    'liteflownet3s-kitti':        'liteflownet3s_pseudoreg',
}


def infer_model_name(ckpt_path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    stem = osp.splitext(osp.basename(ckpt_path))[0]
    for key, name in CKPT_TO_MODEL.items():
        if stem.startswith(key):
            return name
    raise ValueError(
        f"Cannot infer model name from '{stem}'. "
        f"Pass --model_name explicitly. Choices: {list(CKPT_TO_MODEL.values())}"
    )


def default_pairs_csv(output_csv, ckpt_path, res_str):
    """<output_csv stem>__<ckpt stem>_<HxW>_pairs.csv"""
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    ckpt_stem = osp.splitext(osp.basename(str(ckpt_path).rstrip('/')))[0]
    return f'{stem}__{ckpt_stem}_{res_str}_pairs.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',           required=True,  help='path to JSON config')
    parser.add_argument('--ckpt',          required=True,  help='path to .ckpt file or pretrained name (sintel/kitti)')
    parser.add_argument('--model_name',    default=None,   help='ptlflow model name; inferred from ckpt filename if omitted')
    parser.add_argument('--datasets_path', default=None)
    parser.add_argument('--root',          default=None)
    parser.add_argument('--max_samples',   type=int, default=500)
    parser.add_argument('--output_csv',    default='results.csv')
    parser.add_argument('--cpu',           action='store_true')
    parser.add_argument('--job_id',        default='unknown',
                        help='Job identifier written to CSV (default: unknown)')
    parser.add_argument('--tile',          type=int, nargs=2, default=None, metavar=('TH', 'TW'),
                        help='If set, use Gaussian-blended tiled inference at this tile size.')
    parser.add_argument('--pairs_csv', default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<HxW>_pairs.csv)')
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore any sample cap')
    cli = parser.parse_args()

    cfg = json_to_args(cli.cfg)
    if cli.datasets_path:
        cfg.datasets_path = cli.datasets_path
    if cli.root is not None:
        cfg.root = cli.root

    root_path = osp.join(cfg.datasets_path, cfg.root) if cfg.root else cfg.datasets_path
    H, W = cfg.image_size
    _base = osp.basename(root_path.rstrip('/')) or osp.basename(cfg.datasets_path.rstrip('/'))
    dataset_name = f"{_base}_{W}x{H}"
    ckpt_name    = osp.basename(cli.ckpt)

    print(f"Dataset:    {root_path}")
    print(f"Checkpoint: {cli.ckpt}")

    use_cuda = torch.cuda.is_available() and not cli.cpu
    device   = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Device:     {device}")

    model_name = infer_model_name(cli.ckpt, cli.model_name)
    print(f"Model:      {model_name}")
    model = ptlflow.get_model(model_name, cli.ckpt)
    _verify_weights(model, cli.ckpt)
    model = model.to(device).eval()

    dataset = DroneFlowDataset(root_path, resize_shape=cfg.image_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(cli.split_file))
    if len(dataset) == 0:
        print(f"ERROR: no samples found in {root_path}")
        sys.exit(1)

    n      = len(dataset) if cli.split_file or not cli.max_samples else min(len(dataset), cli.max_samples)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=1, drop_last=False)

    res_str   = f"{H}x{W}"
    pairs_csv = cli.pairs_csv or default_pairs_csv(cli.output_csv, cli.ckpt, res_str)
    dumper = fdc.PairDumper(pairs_csv)

    print(f"Evaluating {n} samples (image size {H}x{W})...")
    oom_count = 0
    oom_abort = False
    with torch.no_grad():
        # shuffle=False, batch_size=1 -> loader index i == dataset.samples[i]
        for i, (img1, img2, flow_gt, drone_mask) in enumerate(loader):
            if i >= n:
                break
            img1       = img1.to(device)   # I_t   (reference)
            img2       = img2.to(device)   # I_t-1
            flow_gt    = flow_gt.to(device)
            drone_mask = drone_mask.to(device)

            try:
                if cli.tile is not None:
                    pred = run_inference_tiled(model, img1, img2, tuple(cli.tile))
                else:
                    pred = run_inference(model, img1, img2)
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                print(f"WARNING: CUDA OOM at sample {i} ({H}x{W}); skipped (no row).")
                if oom_count >= 5:
                    print("WARNING: 5 OOMs — aborting this resolution, writing NaN summary.")
                    oom_abort = True
                    break
                continue

            if pred.shape[-2:] != flow_gt.shape[-2:]:
                pred = F.interpolate(pred, size=flow_gt.shape[-2:],
                                     mode='bilinear', align_corners=False)

            dumper.add(fdc.ref_base(dataset.samples[i]), _metrics(pred[0], flow_gt[0], drone_mask[0]))
            if len(dumper.rows) % 100 == 0:
                a = fdc.aggregate(dumper.rows)
                print(f"  {len(dumper.rows)}/{n}  epe={a['epe_all']:.3f}  epe_drone={a['epe_drone']:.3f}")
    dumper.close()

    if not dumper.rows and not oom_abort:
        print("ERROR: no samples processed")
        sys.exit(1)
    row = fdc.summary_row(dumper.rows, ckpt_name, _base, res_str, skipped=oom_count)
    if oom_abort:   # incomplete resolution: keep the legacy all-NaN summary
        for k in fdc.SUMMARY_FIELDS[3:-2]:
            row[k] = float('nan')
    avgs = fdc.aggregate(dumper.rows) if not oom_abort else {}

    # Summary
    print('\n' + '=' * 70)
    print(f"Checkpoint: {ckpt_name}  |  Model: {model_name}")
    print(f"Samples: {len(dumper.rows)}  skipped: {oom_count}  |  Dataset: {dataset_name}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print('-' * 70)
    for metric in fdc.METRIC_NAMES:
        a = avgs.get(f'{metric}_all',   float('nan'))
        d = avgs.get(f'{metric}_drone', float('nan'))
        b = avgs.get(f'{metric}_bg',    float('nan'))
        print(f"  {metric:<26} {a:>8.4f} {d:>8.4f} {b:>8.4f}")
    print('=' * 70)

    # CSV (append; legacy layout: job_id + the shared summary columns)
    csv_path    = cli.output_csv
    write_header = not osp.exists(csv_path) or osp.getsize(csv_path) == 0
    os.makedirs(osp.dirname(osp.abspath(csv_path)), exist_ok=True)
    row['job_id'] = cli.job_id
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['job_id'] + fdc.SUMMARY_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results appended to {csv_path}")
    print(f"Per-pair rows written to {pairs_csv}")


if __name__ == '__main__':
    main()
