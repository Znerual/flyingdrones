"""
Evaluation script for FlowFormerPlusPlus on the SyntheticDrones dataset.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so the
model is run on (I_t, I_{t-1}) with I_t as reference; frame 0000 is never a
reference (fd_eval_common.list_samples).

Metrics = paper definitions (fd_eval_common.compute_metrics), drone/bg split:
  - epe:       mean endpoint error in pixels
  - epe_norm:  sqrt((du/W)²+(dv/H)²) * 100
  - px1/3/5:   % of pixels with EPE > 1/3/5 px  (error rate, higher = worse)
  - fl:        % of pixels where (EPE > 3px) AND (EPE/gt_magnitude > 5%)
Summary = frame-weighted mean over pairs; one row per pair -> --pairs_csv.

Usage:
    python evaluate.py \\
        --cfg configs/sintel \\
        --model checkpoints/sintel.pth \\
        --datasets_path /path/to/datasets \\
        --root SyntheticDrones \\
        --image_size 432 960 \\
        --output_csv results.csv
"""

import sys
import os
import os.path as osp

# sys.path setup: script_dir first so our datasets/ package is found before core/datasets.py
_script_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _script_dir)                        # datasets/ package (our code)
sys.path.insert(1, osp.join(_script_dir, 'core'))      # FlowFormerPlusPlus internal imports
# Optional container fallback (no-op unless /app/FlowFormerPlusPlus exists). It is
# APPENDED, so an older copy baked into a container image can never shadow this
# script's own drone_split / fd_eval_common / configs / core; it only fills in
# modules that are missing locally.
if osp.exists('/app/FlowFormerPlusPlus') and osp.realpath('/app/FlowFormerPlusPlus') != osp.realpath(_script_dir):
    sys.path.append('/app/FlowFormerPlusPlus')
    sys.path.append('/app/FlowFormerPlusPlus/core')

from drone_split import load_split_frames, filter_samples

# Shared Nature3 helpers (pairing, paper metrics, per-pair dump). Re-insert our own
# dir right before the import so a stale copy elsewhere on sys.path (e.g. baked into
# a container under /app) can never shadow the bind-mounted / local one.
sys.path.insert(0, _script_dir)
import fd_eval_common as fdc  # noqa: E402

import math
import csv
import argparse
import importlib
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

from core.FlowFormer import build_flowformer
from datasets import SyntheticDrones


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

TRAIN_SIZE_DEFAULT = [432, 960]


class InputPadder:
    """Pads images so H and W are divisible by pad_to."""
    def __init__(self, dims, pad_to=8):
        h, w = dims[-2:]
        pad_h = (-h) % pad_to
        pad_w = (-w) % pad_to
        self._pad = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        h, w = x.shape[-2:]
        p = self._pad
        return x[..., p[2]: h - p[3] if p[3] else h, p[0]: w - p[1] if p[1] else w]


def _run_model(model, image1, image2):
    """Single forward pass. Returns flow tensor [B, 2, H, W]."""
    padder = InputPadder(image1.shape)
    image1p, image2p = padder.pad(image1, image2)
    with torch.no_grad():
        result = model(image1p, image2p)
    # eval mode returns (flow_up, coords_low); training mode returns list
    if isinstance(result, (tuple, list)) and not isinstance(result[0], torch.Tensor):
        # list of flow tensors (training mode accidentally triggered)
        flow = result[-1]
    elif isinstance(result, (tuple, list)):
        flow = result[0]  # (flow_up, coords_low) → take flow_up
    else:
        flow = result
    return padder.unpad(flow)


def _tiled_inference(model, image1, image2, train_size):
    """Uniform-weighted tiled inference for images larger than train_size."""
    H, W = image1.shape[-2:]
    th, tw = train_size
    min_overlap = 20

    # compute tile origins, last tile flush with image boundary
    hs = list(range(0, H, th - min_overlap))
    ws = list(range(0, W, tw - min_overlap))
    if hs[-1] + th > H:
        hs[-1] = max(0, H - th)
    if ws[-1] + tw > W:
        ws[-1] = max(0, W - tw)

    hws = [(h, w) for h in hs for w in ws]

    flow_acc = torch.zeros(1, 2, H, W, device=image1.device)
    weight_acc = torch.zeros(1, 1, H, W, device=image1.device)

    for h, w in hws:
        h_end = min(h + th, H)
        w_end = min(w + tw, W)
        tile1 = image1[:, :, h:h_end, w:w_end]
        tile2 = image2[:, :, h:h_end, w:w_end]

        # pad tile up to multiple-of-8 (model requirement); do NOT resize to train_size
        # — resizing distorts flow magnitudes and breaks tile alignment.
        padder = InputPadder((1, 3, h_end - h, w_end - w), pad_to=8)
        t1p, t2p = padder.pad(tile1, tile2)

        with torch.no_grad():
            result = model(t1p, t2p)

        if isinstance(result, (tuple, list)):
            tile_flow = result[0]
        else:
            tile_flow = result

        # unpad back to (h_end - h, w_end - w)
        tile_flow = padder.unpad(tile_flow)
        w_tile = torch.ones(1, 1, h_end - h, w_end - w, device=image1.device)

        flow_acc[:, :, h:h_end, w:w_end] += tile_flow * w_tile
        weight_acc[:, :, h:h_end, w:w_end] += w_tile

    return flow_acc / weight_acc.clamp(min=1e-6)


def infer_flow(model, image1, image2, train_size):
    """Infer flow, using tiling when image is larger than train_size."""
    H, W = image1.shape[-2:]
    th, tw = train_size
    if H > th or W > tw:
        return _tiled_inference(model, image1, image2, train_size)
    return _run_model(model, image1, image2)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val += val
        self.count += n

    @property
    def avg(self):
        return self.val / self.count if self.count > 0 else float('nan')


def _compute_metrics(pred, flow_gt, drone_mask):
    """Paper metrics for one pair (delegates to fd_eval_common.compute_metrics).

    Args:
        pred:       [H, W, 2] numpy, predicted flow  (u, v)
        flow_gt:    [H, W, 2] numpy, ground-truth flow
        drone_mask: [H, W] numpy, >0.5 / True on drone pixels

    Returns dict with keys like 'epe_all', 'epe_drone', 'epe_norm_all', etc.
    epe_norm = sqrt((u_err/W)^2 + (v_err/H)^2) * 100.
    (Signature kept: train_nature3.py imports this name.)
    """
    return fdc.compute_metrics(pred, flow_gt, drone_mask)


# ---------------------------------------------------------------------------
# Dataset with labels
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

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()

        return img1, img2, flow, mask


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def default_pairs_csv(output_csv, ckpt_path, res_str):
    """<output_csv stem>__<ckpt stem>_<HxW>_pairs.csv"""
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    ckpt_stem = osp.splitext(osp.basename(str(ckpt_path).rstrip('/')))[0]
    return f'{stem}__{ckpt_stem}_{res_str}_pairs.csv'


def evaluate(args):
    # --- build model ---
    cfg_module = importlib.import_module(args.cfg.replace('/', '.').replace('.py', '').lstrip('.'))
    cfg = cfg_module.get_cfg()

    if args.image_size is not None:
        # Override image_size from CLI (used for multi-resolution evaluation)
        # We keep model train_size fixed at config image_size; images are resized to args.image_size
        pass

    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    if device.type == 'cuda':
        model = torch.nn.DataParallel(build_flowformer(cfg))
    else:
        model = build_flowformer(cfg)

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    # support both bare state-dicts and wrapped checkpoints
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    # Checked load (2026-10-07; was strict=False, the RAFT-loader-bug failure class):
    # load into the unwrapped module, strip a 'module.' prefix, raise on any missing
    # or unexpected key and print the loaded-key count.
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    fdc.load_state_dict_checked(target, ckpt, name=f'ffpp:{osp.basename(args.model)}')
    model = model.to(device).eval()
    print(f"Loaded model from {args.model} (device={device})")

    train_size = list(cfg.image_size)  # [H, W] the model was trained on
    eval_size  = list(args.image_size) if args.image_size else train_size

    # --- dataset ---
    root_path    = osp.join(args.datasets_path, args.root) if args.datasets_path else args.root
    dataset_name = osp.basename(root_path.rstrip('/'))
    dataset      = DroneFlowDataset(root_path, resize_shape=eval_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(args.split_file))
    if len(dataset) == 0:
        print(f"Error: no samples found in {root_path}")
        return

    max_samples = getattr(args, 'max_samples', None)
    if args.split_file:
        n = len(dataset)
    else:
        n = min(len(dataset), max_samples) if max_samples else len(dataset)

    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=1, drop_last=False)

    ckpt_name = osp.basename(args.model)
    res_str   = f"{eval_size[0]}x{eval_size[1]}"
    pairs_csv = args.pairs_csv or default_pairs_csv(args.output_csv, args.model, res_str)
    dumper = fdc.PairDumper(pairs_csv)

    print(f"Evaluating {n}/{len(dataset)} pairs at eval_size={eval_size}, train_size={train_size}")
    # shuffle=False, batch_size=1 -> loader index i == dataset.samples[i]
    for i, (img1, img2, flow_gt, drone_mask) in enumerate(tqdm(loader, total=n)):
        if i >= n:
            break
        img1 = img1.to(device)   # I_t   (reference)
        img2 = img2.to(device)   # I_t-1

        pred = infer_flow(model, img1, img2, train_size)

        # Resize prediction to eval_size if they differ (e.g. tiling with rounding)
        if pred.shape[-2:] != flow_gt.shape[-2:]:
            _, _, H, W = flow_gt.shape
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

        pred_np  = pred[0].permute(1, 2, 0).cpu().numpy()
        gt_np    = flow_gt[0].permute(1, 2, 0).cpu().numpy()
        mask_np  = drone_mask[0].cpu().numpy()

        dumper.add(fdc.ref_base(dataset.samples[i]), _compute_metrics(pred_np, gt_np, mask_np))
    dumper.close()

    samples_skipped_total = 0
    avgs = fdc.aggregate(dumper.rows)
    row = fdc.summary_row(dumper.rows, ckpt_name, dataset_name, res_str, skipped=samples_skipped_total)

    # --- print summary ---
    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_name}  |  Eval size: {res_str}  |  pairs: {len(dumper.rows)}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print('-' * 70)
    for m in fdc.METRIC_NAMES:
        a = avgs.get(f'{m}_all',   float('nan'))
        d = avgs.get(f'{m}_drone', float('nan'))
        b = avgs.get(f'{m}_bg',    float('nan'))
        print(f"  {m:<26} {a:>8.4f} {d:>8.4f} {b:>8.4f}")
    print('=' * 70)

    # --- write CSV row (legacy layout: job_id + the shared summary columns) ---
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
    parser.add_argument('--cfg',           required=True,
                        help='Config module, e.g. configs.sintel or configs/sintel.py')
    parser.add_argument('--model',         required=True,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--datasets_path', default='',
                        help='Base path to datasets directory')
    parser.add_argument('--root',          default='SyntheticDrones',
                        help='Dataset root subdirectory relative to datasets_path')
    parser.add_argument('--image_size',    nargs=2, type=int, default=None,
                        metavar=('H', 'W'),
                        help='Resize dataset images to H x W before evaluation. '
                             'Defaults to the model training size from the config.')
    parser.add_argument('--output_csv',    default='results.csv',
                        help='CSV file to append results to')
    parser.add_argument('--max_samples',   type=int, default=None,
                        help='Limit number of samples evaluated (default: all)')
    parser.add_argument('--device',        default='cuda', choices=['cuda', 'cpu'],
                        help='Device for inference (default: cuda)')
    parser.add_argument('--job_id',        default='unknown',
                        help='Job identifier written to CSV (default: unknown)')
    parser.add_argument('--pairs_csv', default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<HxW>_pairs.csv)')
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore any sample cap')
    args = parser.parse_args()

    # normalise cfg to dotted module path
    args.cfg = args.cfg.replace('/', '.').replace('.py', '').lstrip('.')
    # make sure we can find the configs package
    sys.path.insert(0, _script_dir)

    evaluate(args)


if __name__ == '__main__':
    main()
