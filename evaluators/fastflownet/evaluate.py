"""
FastFlowNet evaluation on SyntheticDrones dataset.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so the
model is run on (I_t, I_{t-1}) with I_t as reference; frame 0000 is never a
reference (fd_eval_common.list_samples).

Metrics = paper definitions (fd_eval_common.compute_metrics), drone/bg split:
  - epe:       mean endpoint error (pixels)
  - epe_norm:  sqrt((du/W)^2+(dv/H)^2)*100
  - px1/3/5:   % of pixels with EPE > 1/3/5 px  (error rate, higher = worse)
  - fl:        % of pixels where (EPE > 3px) AND (EPE/gt_magnitude > 5%)
Summary = frame-weighted mean over pairs; one row per pair -> --pairs_csv.
OOM-skipped pairs get no row and are counted in samples_skipped.

Usage:
    python evaluate.py \\
        --checkpoint checkpoints/fastflownet_ft_sintel.pth \\
        --datasets_path $FD_DATA \\
        --root Nature3 \\
        --image_size 432 960 \\
        --max_samples 500 \\
        --output_csv results.csv \\
        --job_id fastflownet_ft_sintel_std
"""

import sys
import os
import os.path as osp
import csv
import math
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

# Ensure repo root on path so models/ is importable
_repo_root = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _repo_root)

from drone_split import load_split_frames, filter_samples

# Shared Nature3 helpers (pairing, paper metrics, per-pair dump). Re-insert our own
# dir right before the import so a stale copy elsewhere on sys.path (e.g. baked into
# a container under /app) can never shadow the bind-mounted / local one.
sys.path.insert(0, _repo_root)
import fd_eval_common as fdc  # noqa: E402

from models.FastFlowNet_pt import FastFlowNet

DIV_FLOW = 20.0
DIV_SIZE = 64


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def centralize(img1: torch.Tensor, img2: torch.Tensor):
    """Subtract joint mean (as in original demo.py)."""
    B, C = img1.shape[:2]
    rgb_mean = torch.cat([img1, img2], dim=2).view(B, C, -1).mean(2).view(B, C, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean


def pad_to_multiple(img: torch.Tensor, div: int = DIV_SIZE):
    """Pad H and W up to nearest multiple of div. Returns (padded, (orig_H, orig_W))."""
    H, W = img.shape[-2:]
    pH = math.ceil(H / div) * div
    pW = math.ceil(W / div) * div
    if pH == H and pW == W:
        return img, (H, W)
    pad = [0, pW - W, 0, pH - H]
    return F.pad(img, pad, mode='replicate'), (H, W)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_flow(model: FastFlowNet, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Run FastFlowNet on an image pair. Images should be [B,3,H,W] float in [0,255].
    Returns flow [B,2,H,W] in pixels at the original resolution.
    """
    orig_H, orig_W = img1.shape[-2:]

    # Normalize to [0,1] then centralize
    i1 = img1.float() / 255.0
    i2 = img2.float() / 255.0
    i1, i2 = centralize(i1, i2)

    # Pad to multiple of DIV_SIZE
    i1, (pH, pW) = pad_to_multiple(i1)
    i2, _ = pad_to_multiple(i2)

    inp = torch.cat([i1, i2], dim=1)
    raw = model(inp)  # [B, 2, pH/4, pW/4] (coarsest output, upsampled by model to pH/2..?)
    # Model outputs at 1/4 resolution; upsample to padded size
    raw = F.interpolate(raw, size=(pH, pW), mode='bilinear', align_corners=False)

    # Scale flow by DIV_FLOW and adjust for resolution
    # raw output is flow in padded-image space at 1/4 scale, so we only
    # need the div_flow correction (the upsample above does spatial scaling)
    flow = raw * DIV_FLOW

    # Crop back to original size
    flow = flow[:, :, :orig_H, :orig_W]

    return flow


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DroneFlowDataset(data.Dataset):
    """SyntheticDrones loader returning (img_t, img_{t-1}, flow_t, drone_mask_t)."""

    def __init__(self, root_path: str, resize_shape=None):
        self.resize_shape = resize_shape
        self.samples = []

        # Pairing (fixed 2026-10-07): ``flow/<t>_flow.npy`` is the BACKWARD flow
        # t -> t-1, so each sample is (I_t, I_{t-1}, <t>_flow, <t>_label) with I_t as
        # the reference frame; frame 0000 is never a reference. Built by the shared
        # fd_eval_common.list_samples (flat scene dir or parent of scene dirs).
        # Labels stay optional per scene: if a scene has a labels/ dir, samples
        # without a label file are dropped; otherwise the mask is all-zero.
        _all = fdc.list_samples(root_path, fdc.TWO_FRAME, require_labels=False)
        self.samples = [s for s in _all
                        if s[-1] is not None
                        or not osp.isdir(osp.join(osp.dirname(osp.dirname(s[-2])), 'labels'))]

        print(f"DroneFlowDataset: {len(self.samples)} pairs (I_t, I_t-1) from {root_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        p1, p2, pf, pl = self.samples[idx]  # p1 = I_t (reference), p2 = I_{t-1}

        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB).astype(np.float32)
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB).astype(np.float32)
        flow = np.load(pf).astype(np.float32)

        if pl is not None:
            mask = (cv2.imread(pl, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
        else:
            mask = np.zeros(img1.shape[:2], dtype=np.float32)

        if self.resize_shape is not None:
            H, W = self.resize_shape
            oh, ow = img1.shape[:2]
            img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_AREA)
            ft = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            ft = F.interpolate(ft, size=(H, W), mode='bilinear', align_corners=False)
            ft[:, 0] *= W / ow
            ft[:, 1] *= H / oh
            flow = ft.squeeze(0).permute(1, 2, 0).numpy()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mt, size=(H, W), mode='nearest').squeeze().numpy()

        return (
            torch.from_numpy(img1).permute(2, 0, 1).float(),
            torch.from_numpy(img2).permute(2, 0, 1).float(),
            torch.from_numpy(flow).permute(2, 0, 1).float(),
            torch.from_numpy(mask).float(),
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_np, gt_np, mask_np):
    """Paper metrics for one pair (delegates to fd_eval_common.compute_metrics).

    pred_np:  [H, W, 2], gt_np: [H, W, 2], mask_np: [H, W] float (drone=1, bg=0)
    Returns dict with keys like epe_all, epe_drone, epe_norm_all, etc.
    epe_norm = sqrt((u_err/W)^2 + (v_err/H)^2) * 100
    """
    return fdc.compute_metrics(pred_np, gt_np, mask_np)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def default_pairs_csv(output_csv, ckpt_path, res_str):
    """<output_csv stem>__<ckpt stem>_<HxW>_pairs.csv"""
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    ckpt_stem = osp.splitext(osp.basename(str(ckpt_path).rstrip('/')))[0]
    return f'{stem}__{ckpt_stem}_{res_str}_pairs.csv'


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = FastFlowNet().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    # Raises on any missing / unexpected key and prints the loaded-key count.
    fdc.load_state_dict_checked(model, ckpt, name='fastflownet')
    print(f"Loaded: {args.checkpoint}")

    root_path    = osp.join(args.datasets_path, args.root) if args.datasets_path else args.root
    dataset_name = osp.basename(root_path.rstrip('/'))
    eval_size    = tuple(args.image_size) if args.image_size else None

    dataset = DroneFlowDataset(root_path, resize_shape=eval_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(args.split_file))
    if len(dataset) == 0:
        print(f"ERROR: no samples found in {root_path}")
        return

    if args.split_file:
        n = len(dataset)
    else:
        n = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=0, drop_last=False)

    res_str = f"{eval_size[0]}x{eval_size[1]}" if eval_size else "native"
    print(f"Evaluating {n}/{len(dataset)} pairs @ {res_str}")

    ckpt_name = osp.basename(args.checkpoint)
    pairs_csv = args.pairs_csv or default_pairs_csv(args.output_csv, args.checkpoint, res_str)
    dumper = fdc.PairDumper(pairs_csv)

    oom_count  = 0
    # shuffle=False, batch_size=1 -> loader index i == dataset.samples[i]
    for i, (img1, img2, flow_gt, drone_mask) in enumerate(tqdm(loader, total=n)):
        if i >= n:
            break
        try:
            img1 = img1.to(device)   # I_t   (reference)
            img2 = img2.to(device)   # I_t-1

            pred = infer_flow(model, img1, img2)

            if pred.shape[-2:] != flow_gt.shape[-2:]:
                _, _, H, W = flow_gt.shape
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
            gt_np   = flow_gt[0].permute(1, 2, 0).cpu().numpy()
            mask_np = drone_mask[0].cpu().numpy()

        except torch.cuda.OutOfMemoryError:
            # skipped pair: no per-pair row, counted in samples_skipped
            oom_count += 1
            torch.cuda.empty_cache()
            if oom_count >= 5:
                print(f"OOM on {oom_count} samples — stopping early.")
                break
            continue

        dumper.add(fdc.ref_base(dataset.samples[i]), compute_metrics(pred_np, gt_np, mask_np))
    dumper.close()

    samples_skipped_total = oom_count
    avgs = fdc.aggregate(dumper.rows)
    row = fdc.summary_row(dumper.rows, ckpt_name, dataset_name, res_str, skipped=samples_skipped_total)

    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_name}  |  Eval size: {res_str}  |  pairs: {len(dumper.rows)}  skipped: {oom_count}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print('-' * 70)
    for m in fdc.METRIC_NAMES:
        a = avgs.get(f'{m}_all',   float('nan'))
        d = avgs.get(f'{m}_drone', float('nan'))
        b = avgs.get(f'{m}_bg',    float('nan'))
        print(f"  {m:<26} {a:>8.4f} {d:>8.4f} {b:>8.4f}")
    print('=' * 70)

    # --- write CSV (legacy layout: job_id + the shared summary columns) ---
    csv_path = args.output_csv
    write_header = not osp.exists(csv_path) or osp.getsize(csv_path) == 0
    os.makedirs(osp.dirname(osp.abspath(csv_path)), exist_ok=True)
    row['job_id'] = args.job_id
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['job_id'] + fdc.SUMMARY_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results appended to {csv_path}")
    print(f"Per-pair rows written to {pairs_csv}")

    return avgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--datasets_path', default='',
                        help='Base path containing dataset directory')
    parser.add_argument('--root',          default='Nature3',
                        help='Dataset root subdirectory relative to datasets_path')
    parser.add_argument('--image_size',    nargs=2, type=int, default=None,
                        metavar=('H', 'W'),
                        help='Resize images to H x W before evaluation')
    parser.add_argument('--output_csv',    default='results.csv')
    parser.add_argument('--max_samples',   type=int, default=None)
    parser.add_argument('--job_id',        default='unknown')
    parser.add_argument('--pairs_csv', default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<HxW>_pairs.csv)')
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore any sample cap')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
