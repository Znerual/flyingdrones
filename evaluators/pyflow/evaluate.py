"""
Evaluation script for pyflow (Ce Liu's Coarse2Fine Optical Flow) on SyntheticDrones dataset.

pyflow is a classical variational method — there are no learned weights.
"Configurations" play the role of checkpoints: different hyperparameter sets.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so pyflow
is run on (I_t, I_{t-1}) with I_t as reference; frame 0000 is never a reference
(fd_eval_common.list_samples).

Metrics = paper definitions (fd_eval_common.compute_metrics), drone/bg split:
  - epe:       mean endpoint error (pixels)
  - epe_norm:  sqrt((du/W)^2+(dv/H)^2)*100
  - px1/3/5:   % of pixels with EPE > 1/3/5 px  (error rate, higher = worse)
  - fl:        % of pixels where (EPE > 3px) AND (EPE / gt_magnitude > 5%)
Summary = frame-weighted mean over pairs; one row per pair -> --pairs_csv.

Usage:
    python evaluate.py \\
        --datasets_path $FD_DATA \\
        --root Nature3 \\
        --image_size 270 480 \\
        --config default \\
        --max_samples 500 \\
        --output_csv results.csv
"""

import sys
import os
import os.path as osp
import argparse
import csv
import math
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob

# pyflow .so must be on the path
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import pyflow  # noqa: E402  (Cython extension)

from drone_split import load_split_frames, filter_samples

# Shared Nature3 helpers (pairing, paper metrics, per-pair dump). Re-insert our own
# dir right before the import so a stale copy elsewhere on sys.path (e.g. baked into
# a container under /app) can never shadow the bind-mounted / local one.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import fd_eval_common as fdc  # noqa: E402


# ---------------------------------------------------------------------------
# Parameter configurations  (no learned weights — configs act as "checkpoints")
# ---------------------------------------------------------------------------

CONFIGS = {
    "default": dict(
        alpha=0.012, ratio=0.75, minWidth=20,
        nOuterFPIterations=7, nInnerFPIterations=1, nSORIterations=30,
        colType=0,
    ),
    "fast": dict(
        alpha=0.012, ratio=0.75, minWidth=40,
        nOuterFPIterations=5, nInnerFPIterations=1, nSORIterations=20,
        colType=0,
    ),
    "accurate": dict(
        alpha=0.008, ratio=0.75, minWidth=10,
        nOuterFPIterations=10, nInnerFPIterations=2, nSORIterations=40,
        colType=0,
    ),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DroneFlowDataset:
    """SyntheticDrones loader returning numpy HWC (img_t, img_{t-1}, flow_t, mask_t).

    Handles two layouts:
      A) root_path/{scene}/images|flow|labels   (container of scenes)
      B) root_path/images|flow|labels            (root_path IS the scene)
    """

    def __init__(self, root_path, resize_shape=None):
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
            flow_t = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            flow_t = F.interpolate(flow_t, size=(H, W), mode='bilinear', align_corners=False)
            flow_t[:, 0] *= W / ow
            flow_t[:, 1] *= H / oh
            flow = flow_t.squeeze(0).permute(1, 2, 0).numpy()
            mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask_t, size=(H, W), mode='nearest').squeeze().numpy()

        return img1, img2, flow, mask


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_flow(img1: np.ndarray, img2: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Run pyflow on a pair of RGB uint8 images.
    Returns flow [H, W, 2]  where flow[...,0]=u (x) and flow[...,1]=v (y).
    """
    im1 = img1.astype(float) / 255.0
    im2 = img2.astype(float) / 255.0
    # Redirect stdout to suppress verbose C++ pyramid output
    import ctypes
    import io
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        u, v, _ = pyflow.coarse2fine_flow(
            im1, im2,
            cfg['alpha'], cfg['ratio'], cfg['minWidth'],
            cfg['nOuterFPIterations'], cfg['nInnerFPIterations'],
            cfg['nSORIterations'], cfg['colType'],
        )
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)
    return np.stack([u, v], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, flow_gt: np.ndarray, drone_mask: np.ndarray) -> dict:
    """Paper metrics for one pair (delegates to fd_eval_common.compute_metrics).

    epe_norm = sqrt((u_err/W)^2 + (v_err/H)^2) * 100
    """
    return fdc.compute_metrics(pred, flow_gt, drone_mask)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    'job_id', 'checkpoint', 'dataset', 'image_size',
    'epe', 'epe_drone',
    'epe_norm', 'epe_norm_drone',
    'px1_err', 'px1_err_drone',
    'px3_err', 'px3_err_drone',
    'px5_err', 'px5_err_drone',
    'fl', 'fl_drone',
    'samples_processed', 'samples_skipped',
]


def default_pairs_csv(output_csv, ckpt_or_config, res_str):
    """<output_csv stem>__<ckpt stem>_<HxW>_pairs.csv"""
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    ckpt_stem = osp.splitext(osp.basename(str(ckpt_or_config).rstrip('/')))[0]
    return f'{stem}__{ckpt_stem}_{res_str}_pairs.csv'


def evaluate(args):
    cfg = CONFIGS[args.config]
    root_path    = osp.join(args.datasets_path, args.root) if args.datasets_path else args.root
    dataset_name = osp.basename(root_path.rstrip('/'))
    eval_size    = tuple(args.image_size) if args.image_size else None  # (H, W)

    dataset = DroneFlowDataset(root_path, resize_shape=eval_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(args.split_file))
    if len(dataset) == 0:
        print(f"Error: no samples found in {root_path}")
        return

    if args.split_file:
        n = len(dataset)
    else:
        n = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    size_str = f"{eval_size[0]}x{eval_size[1]}" if eval_size else "native"

    print(f"\nConfig: {args.config} | Eval size: {size_str} | Samples: {n}/{len(dataset)}")

    pairs_csv = args.pairs_csv or default_pairs_csv(args.output_csv, args.config, size_str)
    dumper = fdc.PairDumper(pairs_csv)
    t_total = 0.0

    for idx in tqdm(range(n), desc=f"{args.config} @ {size_str}"):
        img1, img2, flow_gt, drone_mask = dataset[idx]   # img1 = I_t, img2 = I_t-1

        t0 = time.perf_counter()
        pred = infer_flow(img1, img2, cfg)
        t_total += time.perf_counter() - t0

        dumper.add(fdc.ref_base(dataset.samples[idx]), compute_metrics(pred, flow_gt, drone_mask))
    dumper.close()

    avgs = fdc.aggregate(dumper.rows)
    row = fdc.summary_row(dumper.rows, args.config, dataset_name, size_str, skipped=0)

    # print summary
    print(f"\n{'='*70}")
    print(f"Config: {args.config}  |  Eval size: {size_str}  |  pairs: {len(dumper.rows)}  |  Avg time/pair: {t_total/max(n, 1):.2f}s")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print('-' * 70)
    for metric in fdc.METRIC_NAMES:
        a = avgs.get(f'{metric}_all',   float('nan'))
        d = avgs.get(f'{metric}_drone', float('nan'))
        b = avgs.get(f'{metric}_bg',    float('nan'))
        print(f"  {metric:<26} {a:>8.4f} {d:>8.4f} {b:>8.4f}")
    print('=' * 70)

    # write CSV (legacy layout: job_id + the shared summary columns)
    csv_path   = args.output_csv
    write_hdr  = not osp.exists(csv_path) or osp.getsize(csv_path) == 0
    os.makedirs(osp.dirname(osp.abspath(csv_path)), exist_ok=True)
    row['job_id'] = args.job_id
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if write_hdr:
            writer.writeheader()
        writer.writerow(row)
    print(f"Results appended to {csv_path}")
    print(f"Per-pair rows written to {pairs_csv}")

    return avgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path', default='',
                        help='Base path to datasets directory')
    parser.add_argument('--root',          default='Nature3',
                        help='Dataset root subdirectory relative to datasets_path')
    parser.add_argument('--image_size',    nargs=2, type=int, default=None,
                        metavar=('H', 'W'),
                        help='Resize images to H x W before evaluation')
    parser.add_argument('--config',        default='default',
                        choices=list(CONFIGS.keys()),
                        help='Parameter configuration to use')
    parser.add_argument('--max_samples',   type=int, default=None,
                        help='Max pairs to evaluate (default: all)')
    parser.add_argument('--output_csv',    default='results.csv',
                        help='CSV file to append results to')
    parser.add_argument('--job_id',        default='pyflow',
                        help='Job identifier written to CSV')
    parser.add_argument('--pairs_csv', default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<config>_<HxW>_pairs.csv)')
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore any sample cap')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
