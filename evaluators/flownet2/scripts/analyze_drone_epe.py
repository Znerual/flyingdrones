"""Drone-vs-background EPE analysis with full metric suite.

Computes all standard optical flow metrics split by drone pixels vs background,
using the label PNGs of the FlyingDrones (Nature3) dataset. Writes a single CSV row to
--output_csv (one file per run, merge with merge_results.py).

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

Weights: loaded via fd_eval_common.load_state_dict_checked (``module.`` stripped,
raises on any missing/unexpected key, prints the loaded-key count).

Usage:
    python scripts/analyze_drone_epe.py \\
        --cfg config/drones-flownet2.json \\
        --ckpt /path/to/FlowNet2_checkpoint.pth.tar \\
        --datasets_path $FD_DATA \\
        --root Nature3 \\
        --job_id 1234 \\
        --output_csv /path/to/results/1234_flownet2.csv
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from config.parser import json_to_args
import models


class InputPadder:
    def __init__(self, shape, divisor=64):
        h, w = shape[-2:]
        self.pad_h = (divisor - h % divisor) % divisor
        self.pad_w = (divisor - w % divisor) % divisor

    def pad(self, *imgs):
        if self.pad_h == 0 and self.pad_w == 0:
            return imgs
        return tuple(F.pad(x, (0, self.pad_w, 0, self.pad_h), mode='replicate') for x in imgs)

    def unpad(self, x):
        h, w = x.shape[-2:]
        return x[..., :h - self.pad_h if self.pad_h else h, :w - self.pad_w if self.pad_w else w]


class DroneFlowDataset(data.Dataset):
    """SyntheticDrones dataset that also returns the drone label mask."""

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
            oh, ow = flow.shape[:2]
            img1 = np.array(Image.fromarray(img1.astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32)
            img2 = np.array(Image.fromarray(img2.astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32)
            ft = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            ft = F.interpolate(ft, size=(H, W), mode='bilinear', align_corners=False)
            ft[:, 0] *= W / ow
            ft[:, 1] *= H / oh
            flow = ft.squeeze(0).permute(1, 2, 0).numpy()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mt, size=(H, W), mode='nearest').squeeze().numpy()

        return (torch.from_numpy(img1).permute(2, 0, 1),
                torch.from_numpy(img2).permute(2, 0, 1),
                torch.from_numpy(flow).permute(2, 0, 1),
                torch.from_numpy(mask))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',           required=True)
    parser.add_argument('--ckpt',          required=True)
    parser.add_argument('--datasets_path', default='')
    parser.add_argument('--root',          default='datasets/AITSyntheticParsed')
    parser.add_argument('--image_size',    type=int, nargs=2, default=None, metavar=('H', 'W'))
    parser.add_argument('--max_samples',   type=int, default=500)
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore --max_samples')
    parser.add_argument('--job_id',        default='unknown')
    parser.add_argument('--pairs_csv',     default=None,
                        help='Per-pair metrics CSV (default: <output_csv>__<ckpt-stem>_<W>x<H>_pairs.csv)')
    parser.add_argument('--output_csv',    required=True)
    parser.add_argument('--device',        default='cuda')
    cli = parser.parse_args()

    args = json_to_args(cli.cfg)
    if cli.datasets_path:
        args.datasets_path = cli.datasets_path
    if cli.root:
        args.root = cli.root
    if cli.image_size is not None:
        args.image_size = cli.image_size

    root_path = os.path.join(args.datasets_path, args.root)
    dataset_name = os.path.basename(root_path.rstrip('/'))
    print(f"Dataset:    {root_path}")
    print(f"Checkpoint: {cli.ckpt}")

    device = torch.device(cli.device)
    print(f"Device:     {device}")

    model_class = getattr(models, args.model)
    model = model_class(args)
    print(f"fd_eval_common: {fd_eval_common.__file__}")
    state = torch.load(cli.ckpt, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    load_state_dict_checked(model, state, name=f'FlowNet2/{args.model}')
    model = model.to(device).eval()

    dataset = DroneFlowDataset(root_path, resize_shape=args.image_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(cli.split_file))
    n = len(dataset) if cli.split_file else min(len(dataset), cli.max_samples)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=1, drop_last=False)

    H, W = args.image_size
    pairs_csv = cli.pairs_csv or _default_pairs_csv(cli.output_csv, cli.ckpt, H, W)
    dumper = PairDumper(pairs_csv)

    print(f"Evaluating {n} pairs (I_t, I_t-1)...")
    with torch.no_grad():
        for i, (img1, img2, flow_gt, drone_mask) in enumerate(loader):
            if i >= n:
                break
            img1, img2 = img1.to(device), img2.to(device)
            flow_gt, drone_mask = flow_gt.to(device), drone_mask.to(device)

            padder = InputPadder(img1.shape)
            img1p, img2p = padder.pad(img1, img2)
            imgs = torch.stack([img1p, img2p], dim=2)   # (I_t, I_t-1) -> flow t -> t-1
            pred = model(imgs)
            pred = padder.unpad(pred)

            if pred.shape[-2:] != flow_gt.shape[-2:]:
                pred = F.interpolate(pred, size=flow_gt.shape[-2:],
                                     mode='bilinear', align_corners=False)

            m = compute_metrics(pred[0], flow_gt[0], drone_mask[0])
            dumper.add(ref_base(dataset.samples[i]), m)

            done = i + 1
            if done % 100 == 0 and dumper.rows:
                ag = aggregate(dumper.rows)
                print(f"  {done}/{n}  epe={ag['epe_all']:.3f}  epe_drone={ag['epe_drone']:.3f}")

    dumper.close()
    skipped = n - len(dumper.rows)   # OOM pairs + pairs not reached after an early stop
    print(f"Per-pair CSV: {pairs_csv}")

    _print_summary(dumper.rows, cli.ckpt, dataset_name, skipped)
    row = summary_row(dumper.rows, cli.ckpt, dataset_name, f"{H}x{W}", skipped=skipped)
    _write_summary_csv(cli.output_csv, row, cli.job_id, append=False)


if __name__ == '__main__':
    main()
