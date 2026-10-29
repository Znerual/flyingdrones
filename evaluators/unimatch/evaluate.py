"""
evaluate.py — UniMatch (GMFlow) optical flow evaluation on SyntheticDrones.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so the
model is run on (I_t, I_{t-1}) with I_t as reference; frame 0000 is never a
reference (fd_eval_common.list_samples).

Metrics = paper definitions (fd_eval_common.compute_metrics):
  epe        — mean endpoint error (pixels)
  epe_norm   — sqrt((du/W)²+(dv/H)²)*100
  px1/3/5    — % pixels with EPE > 1/3/5 px
  fl         — % pixels where EPE>3px AND EPE/|GT|>5%
All metrics split: all / drone / bg. Summary = frame-weighted mean over pairs;
one row per pair goes to --pairs_csv (bootstrap CIs).

Usage:
    python evaluate.py \\
        --model checkpoints/gmflow-scale1-things-e9887eda.pth \\
        --datasets_path $FD_DATA \\
        --root Nature3 \\
        --image_size 432 960 \\
        --output_csv results.csv
"""

import sys
import os
import os.path as osp

_script_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _script_dir)

from drone_split import load_split_frames, filter_samples

# Shared Nature3 helpers (pairing, paper metrics, per-pair dump). Re-insert our own
# dir right before the import so a stale copy elsewhere on sys.path (e.g. baked into
# a container under /app) can never shadow the bind-mounted / local one.
sys.path.insert(0, _script_dir)
import fd_eval_common as fdc  # noqa: E402

import math
import csv
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

from unimatch.unimatch import UniMatch
from utils.utils import InputPadder


# ---------------------------------------------------------------------------
# Checkpoint → model config
# ---------------------------------------------------------------------------

def _cfg_from_name(filename):
    """Infer UniMatch model hyperparameters from checkpoint filename."""
    n = filename.lower()
    if 'scale2' in n and 'regrefine' in n:
        return dict(num_scales=2, upsample_factor=4, padding_factor=32,
                    attn_splits_list=[2, 8], corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1], reg_refine=True, num_reg_refine=6)
    elif 'scale2' in n:
        return dict(num_scales=2, upsample_factor=4, padding_factor=32,
                    attn_splits_list=[2, 8], corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1], reg_refine=False, num_reg_refine=1)
    else:  # scale1: feature at 1/8 must be divisible by attn_splits=2 → pad to multiple of 16
        return dict(num_scales=1, upsample_factor=8, padding_factor=16,
                    attn_splits_list=[2], corr_radius_list=[-1],
                    prop_radius_list=[-1], reg_refine=False, num_reg_refine=1)


def load_model(ckpt_path, device):
    cfg = _cfg_from_name(osp.basename(ckpt_path))
    model = UniMatch(
        feature_channels=128,
        num_scales=cfg['num_scales'],
        upsample_factor=cfg['upsample_factor'],
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=cfg['reg_refine'],
        task='flow',
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    new_state = OrderedDict(
        (k[7:] if k.startswith('module.') else k, v) for k, v in state.items()
    )
    # Raises on any missing / unexpected key and prints the loaded-key count.
    fdc.load_state_dict_checked(model, new_state, name='unimatch')
    model = model.to(device).eval()
    print(f"Loaded {osp.basename(ckpt_path)}  cfg={cfg}")
    return model, cfg


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

TILE_SIZE = (432, 960)  # native eval resolution used as tile size for large images


def _forward(model, i1, i2, cfg):
    """Padded single forward pass, returns flow [B,2,H,W]."""
    padder = InputPadder(i1.shape, padding_factor=cfg['padding_factor'])
    i1p, i2p = padder.pad(i1, i2)
    out = model(
        i1p, i2p,
        attn_type='swin',
        attn_splits_list=cfg['attn_splits_list'],
        corr_radius_list=cfg['corr_radius_list'],
        prop_radius_list=cfg['prop_radius_list'],
        num_reg_refine=cfg['num_reg_refine'],
        task='flow',
    )
    return padder.unpad(out['flow_preds'][-1])


@torch.no_grad()
def _tiled_inference(model, image1, image2, cfg, tile_size=TILE_SIZE):
    """Uniform-weighted tiled inference (avoids Gaussian elliptical artefacts)."""
    H, W   = image1.shape[-2:]
    th, tw = tile_size
    overlap = 20

    hs = list(range(0, H, th - overlap))
    ws = list(range(0, W, tw - overlap))
    if hs[-1] + th > H:
        hs[-1] = max(0, H - th)
    if ws[-1] + tw > W:
        ws[-1] = max(0, W - tw)

    flow_acc   = torch.zeros(1, 2, H, W, device=image1.device)
    weight_acc = torch.zeros(1, 1, H, W, device=image1.device)

    for h in hs:
        for w in ws:
            he = min(h + th, H)
            we = min(w + tw, W)
            t1 = image1[:, :, h:he, w:we]
            t2 = image2[:, :, h:he, w:we]
            tile_flow = _forward(model, t1, t2, cfg)
            tile_flow = tile_flow[:, :, : he - h, : we - w]
            wt = torch.ones(1, 1, he - h, we - w, device=image1.device)
            flow_acc[:, :, h:he, w:we]   += tile_flow * wt
            weight_acc[:, :, h:he, w:we] += wt

    return flow_acc / weight_acc.clamp(min=1e-6)


@torch.no_grad()
def run_model(model, image1, image2, cfg, tile_size=TILE_SIZE):
    """Run model, falling back to tiling when image exceeds tile_size."""
    H, W   = image1.shape[-2:]
    th, tw = tile_size
    if H > th or W > tw:
        return _tiled_inference(model, image1, image2, cfg, tile_size)
    return _forward(model, image1, image2, cfg)


# ---------------------------------------------------------------------------
# Metrics — identical to flowformerplusplus/evaluate.py
# ---------------------------------------------------------------------------

def _compute_metrics(pred, flow_gt, drone_mask):
    """Paper metrics for one pair (delegates to fd_eval_common.compute_metrics).

    pred, flow_gt: [H, W, 2] numpy; drone_mask: [H, W] (>0.5 = drone pixel).
    epe_norm = sqrt((u_err/W)² + (v_err/H)²) * 100
    """
    return fdc.compute_metrics(pred, flow_gt, drone_mask)


# ---------------------------------------------------------------------------
# Dataset — identical to flowformerplusplus/evaluate.py
# ---------------------------------------------------------------------------

class DroneFlowDataset(data.Dataset):
    """SyntheticDrones loader returning (img_t, img_{t-1}, flow_t, drone_mask_t)."""

    def __init__(self, root_path, resize_shape=None):
        import cv2
        from glob import glob
        self.resize_shape = resize_shape
        self.samples = []
        self._cv2 = cv2

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
        cv2 = self._cv2
        p1, p2, pf, pl = self.samples[idx]  # p1 = I_t (reference), p2 = I_{t-1}
        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB).astype(np.float32)
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB).astype(np.float32)
        flow = np.load(pf).astype(np.float32)
        mask = ((cv2.imread(pl, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
                if pl is not None else np.zeros(img1.shape[:2], dtype=np.float32))

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

        return (torch.from_numpy(img1).permute(2, 0, 1).float(),
                torch.from_numpy(img2).permute(2, 0, 1).float(),
                torch.from_numpy(flow).permute(2, 0, 1).float(),
                torch.from_numpy(mask).float())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_pairs_csv(output_csv, ckpt_path, res_str):
    """<output_csv stem>__<ckpt stem>_<HxW>_pairs.csv"""
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    ckpt_stem = osp.splitext(osp.basename(str(ckpt_path).rstrip('/')))[0]
    return f'{stem}__{ckpt_stem}_{res_str}_pairs.csv'


def evaluate(args):
    device = torch.device('cpu')
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("WARNING: CUDA not available, falling back to CPU")

    model, cfg = load_model(args.model, device)
    eval_size  = list(args.image_size) if args.image_size else None
    size_str   = f"{eval_size[0]}x{eval_size[1]}" if eval_size else "native"

    root_path    = osp.join(args.datasets_path, args.root) if args.datasets_path else args.root
    dataset_name = osp.basename(root_path.rstrip('/'))
    dataset      = DroneFlowDataset(root_path, resize_shape=eval_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(args.split_file))
    if len(dataset) == 0:
        print(f"No samples found in {root_path}")
        return

    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    max_samples = getattr(args, 'max_samples', None)
    if args.split_file:
        n = len(dataset)
    else:
        n = min(len(dataset), max_samples) if max_samples else len(dataset)
    ckpt_name = osp.basename(args.model)
    pairs_csv = args.pairs_csv or default_pairs_csv(args.output_csv, args.model, size_str)
    dumper = fdc.PairDumper(pairs_csv)
    print(f"Evaluating {n}/{len(dataset)} pairs at {size_str}")
    # shuffle=False, batch_size=1 -> loader index i == dataset.samples[i]
    for i, (img1, img2, flow_gt, drone_mask) in enumerate(tqdm(loader, total=n)):
        if i >= n:
            break
        img1 = img1.to(device)   # I_t   (reference)
        img2 = img2.to(device)   # I_t-1
        pred = run_model(model, img1, img2, cfg)
        if pred.shape[-2:] != flow_gt.shape[-2:]:
            _, _, H, W = flow_gt.shape
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
        m = _compute_metrics(
            pred[0].permute(1, 2, 0).cpu().numpy(),
            flow_gt[0].permute(1, 2, 0).cpu().numpy(),
            drone_mask[0].cpu().numpy(),
        )
        dumper.add(fdc.ref_base(dataset.samples[i]), m)
    dumper.close()

    samples_skipped_total = 0
    avgs = fdc.aggregate(dumper.rows)
    row = fdc.summary_row(dumper.rows, ckpt_name, dataset_name, size_str, skipped=samples_skipped_total)

    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_name}  |  Eval size: {size_str}  |  pairs: {len(dumper.rows)}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print('-' * 70)
    for m in fdc.METRIC_NAMES:
        print(f"  {m:<26} "
              f"{avgs.get(m+'_all',float('nan')):>8.4f} "
              f"{avgs.get(m+'_drone',float('nan')):>8.4f} "
              f"{avgs.get(m+'_bg',float('nan')):>8.4f}")
    print('=' * 70)

    # Legacy unimatch layout: job_id + the shared summary columns.
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
    parser = argparse.ArgumentParser(description='Evaluate UniMatch on SyntheticDrones')
    parser.add_argument('--model',        required=True,  help='checkpoint .pth path')
    parser.add_argument('--datasets_path', default='',   help='base datasets directory')
    parser.add_argument('--root',          default='SyntheticDrones')
    parser.add_argument('--image_size',    nargs=2, type=int, default=None, metavar=('H', 'W'),
                        help='resize images to H×W before evaluation')
    parser.add_argument('--output_csv',    default='results.csv')
    parser.add_argument('--device',        default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--job_id',        default='unknown',
                        help='Job identifier written to CSV (default: unknown)')
    parser.add_argument('--max_samples',   type=int, default=None,
                        help='Limit number of samples (default: all)')
    parser.add_argument('--pairs_csv', default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<HxW>_pairs.csv)')
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore any sample cap')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
