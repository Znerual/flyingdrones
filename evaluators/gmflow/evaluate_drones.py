"""
Evaluate GMFlow on the SyntheticDrones (Nature3) dataset.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so the
model is run on (I_t, I_{t-1}) with I_t as reference; frame 0000 is never a
reference (see fd_eval_common.py / datasets/drones.py).

Metrics are the paper definitions from fd_eval_common.compute_metrics:
  - epe:       mean endpoint error (pixels)
  - epe_norm:  mean ||(du/W, dv/H)|| x 100
  - px1/3/5:   % pixels with EPE > 1 / 3 / 5 px
  - fl:        % pixels where EPE > 3px AND EPE/gt_mag > 5%
Results are split into all / drone / background pixels; the summary is the
frame-weighted mean over pairs (pairs without drone pixels are skipped for the
drone columns). One row per pair is written to --pairs_csv (bootstrap CIs).

Usage:
    python evaluate_drones.py \\
        --resume pretrained/gmflow_things-e9887eda.pth \\
        --datasets_path /path/to/datasets \\
        --root SyntheticDrones \\
        --image_size 432 960 \\
        --output_csv results.csv
"""

import sys
import os
import os.path as osp

_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _dir)

import csv
import math
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

from gmflow.gmflow import GMFlow
from datasets.drones import DroneFlowDataset
from drone_split import load_split_frames, filter_samples

# Re-insert our own dir right before importing the shared module so that a stale
# copy elsewhere on sys.path (e.g. baked into a container) can never shadow it.
sys.path.insert(0, _dir)
import fd_eval_common as fdc  # noqa: E402


# ---------------------------------------------------------------------------
# Padding helper (mirrors GMFlow's own InputPadder)
# ---------------------------------------------------------------------------

class InputPadder:
    """Pads images so H and W are divisible by padding_factor."""
    def __init__(self, dims, padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


# ---------------------------------------------------------------------------
# Tiled inference for large images
# ---------------------------------------------------------------------------

def _tiled_inference(model, image1, image2, train_size, model_kwargs):
    """Uniform-weighted tiled inference (avoids Gaussian elliptical artefacts)."""
    H, W = image1.shape[-2:]
    th, tw = train_size
    min_overlap = 20

    hs = list(range(0, H, th - min_overlap))
    ws = list(range(0, W, tw - min_overlap))
    if hs[-1] + th > H:
        hs[-1] = max(0, H - th)
    if ws[-1] + tw > W:
        ws[-1] = max(0, W - tw)

    hws = [(h, w) for h in hs for w in ws]

    flow_acc   = torch.zeros(1, 2, H, W, device=image1.device)
    weight_acc = torch.zeros(1, 1, H, W, device=image1.device)

    for h, w in hws:
        h_end = min(h + th, H)
        w_end = min(w + tw, W)
        tile1 = image1[:, :, h:h_end, w:w_end]
        tile2 = image2[:, :, h:h_end, w:w_end]

        padder = InputPadder(tile1.shape, padding_factor=model_kwargs.get('padding_factor', 8))
        t1p, t2p = padder.pad(tile1, tile2)

        results = model(t1p, t2p, **{k: v for k, v in model_kwargs.items() if k != 'padding_factor'})
        tile_flow = results['flow_preds'][-1]
        tile_flow = padder.unpad(tile_flow)

        # Crop to actual tile region (tile might be smaller than train_size at edges)
        tile_flow = tile_flow[:, :, :h_end - h, :w_end - w]
        w_tile = torch.ones(1, 1, h_end - h, w_end - w, device=image1.device)

        flow_acc[:, :, h:h_end, w:w_end]   += tile_flow * w_tile
        weight_acc[:, :, h:h_end, w:w_end] += w_tile

    return flow_acc / weight_acc.clamp(min=1e-6)


@torch.no_grad()
def infer_flow(model, image1, image2, train_size, model_kwargs, padding_factor=8):
    """Run model; use tiling if image larger than train_size."""
    H, W = image1.shape[-2:]
    th, tw = train_size
    fwd_kwargs = {k: v for k, v in model_kwargs.items() if k != 'padding_factor'}

    if H > th or W > tw:
        return _tiled_inference(model, image1, image2, train_size,
                                {**fwd_kwargs, 'padding_factor': padding_factor})

    padder = InputPadder(image1.shape, padding_factor=padding_factor)
    image1p, image2p = padder.pad(image1, image2)
    results = model(image1p, image2p, **fwd_kwargs)
    flow = results['flow_preds'][-1]
    return padder.unpad(flow)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_np, gt_np, drone_mask_np):
    """Paper metrics for one pair (delegates to fd_eval_common.compute_metrics).

    Args: pred_np / gt_np [H, W, 2], drone_mask_np [H, W] (>0.5 = drone).
    Returns dict keyed like 'epe_all', 'epe_drone', 'fl_bg', ...
    """
    return fdc.compute_metrics(pred_np, gt_np, drone_mask_np)


# ---------------------------------------------------------------------------
# Checkpoint config lookup
# ---------------------------------------------------------------------------

# Maps checkpoint filename substrings → GMFlow model kwargs + padding_factor
_CKPT_CONFIGS = {
    'with_refine': dict(
        num_scales=2,
        upsample_factor=4,
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],
        padding_factor=32,
    ),
    'default': dict(
        num_scales=1,
        upsample_factor=8,
        attn_splits_list=[2],
        corr_radius_list=[-1],
        prop_radius_list=[-1],
        padding_factor=16,  # feature map H/8 must be divisible by attn_splits=2 → pad to 16
    ),
}


def get_ckpt_config(ckpt_path):
    name = osp.basename(ckpt_path)
    if 'with_refine' in name:
        return dict(_CKPT_CONFIGS['with_refine'])
    return dict(_CKPT_CONFIGS['default'])


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(ckpt_path, device, cfg_override=None):
    cfg = get_ckpt_config(ckpt_path)
    if cfg_override:
        cfg.update(cfg_override)

    model = GMFlow(
        num_scales=cfg['num_scales'],
        upsample_factor=cfg['upsample_factor'],
        feature_channels=128,
        attention_type='swin',
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    weights = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    # Raises on any missing / unexpected key (prints the loaded-key count).
    fdc.load_state_dict_checked(model, weights, name='gmflow')
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def default_pairs_csv(output_csv, ckpt_path, res_str):
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    ckpt_stem = osp.splitext(osp.basename(ckpt_path.rstrip('/')))[0]
    return f'{stem}__{ckpt_stem}_{res_str}_pairs.csv'


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    root_path    = osp.join(args.datasets_path, args.root) if args.datasets_path else args.root
    dataset_name = osp.basename(root_path.rstrip('/'))

    eval_size = list(args.image_size) if args.image_size else None

    dataset = DroneFlowDataset(root_path, resize_shape=eval_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(args.split_file))
    if len(dataset) == 0:
        print(f'ERROR: no samples found in {root_path}')
        return

    n = len(dataset) if args.split_file else min(len(dataset), args.max_samples)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=2, drop_last=False)

    ckpt_name = osp.basename(args.resume)
    model, cfg = build_model(args.resume, device)
    padding_factor = cfg['padding_factor']
    fwd_kwargs = dict(
        attn_splits_list=cfg['attn_splits_list'],
        corr_radius_list=cfg['corr_radius_list'],
        prop_radius_list=cfg['prop_radius_list'],
    )

    # Tile size for inference: images larger than this are processed in overlapping tiles.
    # Keep at 432x960 so 1080x1920 inputs are tiled and don't OOM.
    train_size = [432, 960]

    res_str = f"{eval_size[0]}x{eval_size[1]}" if eval_size else "native"
    pairs_csv = args.pairs_csv or default_pairs_csv(args.output_csv, args.resume, res_str)
    dumper = fdc.PairDumper(pairs_csv)

    print(f'\nEvaluating {ckpt_name}  |  eval_size={eval_size}  |  samples={n}/{len(dataset)}  |  device={device}')

    # shuffle=False, batch_size=1 -> loader index i == dataset.samples[i]
    for i, (img1, img2, flow_gt, drone_mask) in enumerate(tqdm(loader, total=n)):
        if i >= n:
            break
        img1 = img1.to(device)   # I_t   (reference)
        img2 = img2.to(device)   # I_t-1

        pred = infer_flow(model, img1, img2, train_size, fwd_kwargs, padding_factor)

        # Align spatial dims if tiling introduced rounding
        if pred.shape[-2:] != flow_gt.shape[-2:]:
            _, _, H, W = flow_gt.shape
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

        pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
        gt_np   = flow_gt[0].permute(1, 2, 0).cpu().numpy()
        mask_np = drone_mask[0].cpu().numpy()

        dumper.add(fdc.ref_base(dataset.samples[i]), compute_metrics(pred_np, gt_np, mask_np))
    dumper.close()

    samples_skipped_total = 0
    row = fdc.summary_row(dumper.rows, ckpt_name, dataset_name, res_str, skipped=samples_skipped_total)
    avgs = fdc.aggregate(dumper.rows)

    # Print table
    print(f'\n{"=" * 70}')
    print(f'Checkpoint: {ckpt_name}  |  Eval size: {res_str}  |  pairs: {len(dumper.rows)}')
    print(f'{"Metric":<28} {"All":>8} {"Drone":>8} {"BG":>8}')
    print('-' * 70)
    for m in fdc.METRIC_NAMES:
        a = avgs.get(f'{m}_all',   float('nan'))
        d = avgs.get(f'{m}_drone', float('nan'))
        b = avgs.get(f'{m}_bg',    float('nan'))
        print(f'  {m:<26} {a:>8.4f} {d:>8.4f} {b:>8.4f}')
    print('=' * 70)

    # Write CSV (legacy gmflow layout: job_id + the shared summary columns)
    csv_path = args.output_csv
    write_header = not osp.exists(csv_path) or osp.getsize(csv_path) == 0
    os.makedirs(osp.dirname(osp.abspath(csv_path)), exist_ok=True)
    row['job_id'] = args.job_id
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['job_id'] + fdc.SUMMARY_FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f'Results appended to {csv_path}')
    print(f'Per-pair rows written to {pairs_csv}')

    return avgs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',        required=True,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--datasets_path', default='',
                        help='Base datasets directory')
    parser.add_argument('--root',          default='SyntheticDrones',
                        help='Dataset subdirectory under datasets_path')
    parser.add_argument('--image_size',    nargs=2, type=int, default=None,
                        metavar=('H', 'W'),
                        help='Resize images to H x W before evaluation')
    parser.add_argument('--output_csv',    default='results.csv',
                        help='CSV file to append results to')
    parser.add_argument('--max_samples',   type=int, default=500,
                        help='Max number of pairs to evaluate (default: 500)')
    parser.add_argument('--split_file', default=None,
                        help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore --max_samples')
    parser.add_argument('--pairs_csv',     default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<HxW>_pairs.csv)')
    parser.add_argument('--cpu',           action='store_true',
                        help='Force CPU inference')
    parser.add_argument('--job_id',        default='unknown',
                        help='Job identifier written to CSV (default: unknown)')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
