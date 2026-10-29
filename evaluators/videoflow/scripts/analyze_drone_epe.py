"""Drone-vs-background EPE analysis for VideoFlow (BOFNet / MOFNet).

Computes the paper's optical-flow metrics split by drone pixels vs background,
using the label PNGs of FlyingDrones (Nature3). Writes one summary row to
--output_csv (one file per run, merge with merge_results.py) and one row per
evaluated pair to --pairs_csv (for bootstrap CIs).

Metrics: fd_eval_common.compute_metrics (paper "Metrics" section):
  - epe:       mean endpoint error (pixels)
  - epe_norm:  mean ||(du/W, dv/H)|| x 100
  - px1/3/5:   % of pixels with epe > 1/3/5 px  (error-rate, higher = worse)
  - fl:        % of pixels where (epe > 3 px) AND (epe / gt_magnitude > 5%)

Ground truth: Nature3 ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1
(I_t(x) ~= I_{t-1}(x + F)). The model is asked for exactly that flow, with I_t as
the reference frame, using a genuine time-reversed window from the full sequence
(edge frames clamped; frame 0000 is never a reference):

  BOF mode (3-frame BOFNet):
    window [t+1, t, t-1]  -> flow_pre[:, 0]
    BOFNet/network.py: fmap2 = centre frame (L95), corr_fn_23 = CorrBlock(fmap2,
    fmap3) (L103), output stack([flow_23, flow_21]) (L140/L145), so slot 0 =
    centre -> third frame = t -> t-1.
  MOF mode (N-frame MOFNetStack, N odd, default 5):
    window [t+k, ..., t, ..., t-k], k=(N-1)//2 -> flow_pre[:, (N-3)//2]
    MOFNetStack/network.py: forward_corr_fn = corr(fmaps[1:N-1], fmaps[2:N])
    (L157), output cat([forward (N-2), backward (N-2)]) (L210), so forward slot j
    = frame j+1 -> frame j+2; j=(N-3)//2 is centre -> next = t -> t-1 (slot 1 for
    N=5, same index as upstream evaluate_MOFNet.py gt_index).

Usage:
    python scripts/analyze_drone_epe.py \\
        --mode MOF \\
        --ckpt /path/to/MOF_sintel.pth \\
        --image_size 432 960 \\
        --datasets_path $FD_DATA \\
        --root Nature3 \\
        --split_file nature3_split_frames.txt \\
        --job_id 1234 \\
        --output_csv /path/to/results/1234_MOF-sintel.csv
"""
import argparse
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Path resolution: works both locally and inside a container image.
# The VideoFlow source bundled next to this script (../VideoFlow) wins; the copy
# baked into the image at /app/VideoFlow is only a fallback. The script dir is put
# at sys.path[0] LAST so the drone_split.py / fd_eval_common.py next to this file
# (bind-mounted in the container) shadow any stale copy elsewhere on the path.
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.abspath(os.path.join(_script_dir, '..', 'VideoFlow')),
    '/app/VideoFlow',
]
_videoflow_root = next((p for p in _candidates
                        if os.path.isdir(os.path.join(p, 'core', 'Networks'))), None)
if _videoflow_root is None:
    raise RuntimeError(
        f"Cannot find VideoFlow source. Searched: {_candidates}\n"
        "Expected VideoFlow/ to be bundled inside this repo."
    )
for _p in (os.path.join(_videoflow_root, 'core'), _videoflow_root, _script_dir):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from drone_split import load_split_frames, filter_samples
from fd_eval_common import (BOF_WINDOW, MOF_WINDOW, PairDumper, aggregate,
                            compute_metrics, list_samples, load_state_dict_checked, ref_base,
                            summary_row)

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

from core.Networks import build_network
from core.utils.utils import InputPadder


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _bof_cfg(ckpt: str, image_size):
    from configs.sintel_submission import get_cfg
    cfg = get_cfg()
    cfg.model = ckpt
    cfg.image_size = list(image_size)
    cfg.input_frames = 3
    return cfg


def _mof_cfg(ckpt: str, image_size, input_frames: int):
    from configs.multiframes_sintel_submission import get_cfg
    cfg = get_cfg()
    cfg.model = ckpt
    cfg.image_size = list(image_size)
    cfg.input_frames = input_frames
    return cfg


def window_offsets(mode: str, input_frames: int = 5):
    """Time-reversed frame offsets (relative to the reference frame t)."""
    if mode == 'BOF':
        return BOF_WINDOW
    if input_frames < 3 or input_frames % 2 == 0:
        raise ValueError(f"--input_frames must be odd and >= 3, got {input_frames}")
    k = (input_frames - 1) // 2
    offs = tuple(range(k, -k - 1, -1))
    assert input_frames != 5 or offs == MOF_WINDOW
    return offs


def read_slot(mode: str, n_frames: int) -> int:
    """Forward-flow slot holding centre -> next frame (= t -> t-1 for reversed windows)."""
    return 0 if mode == 'BOF' else (n_frames - 3) // 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DroneFlowDataset(data.Dataset):
    """FlyingDrones (Nature3) dataset returning a time-reversed frame window + GT + mask.

    Samples are ``(*frame_paths, flow_path, label_path)`` from
    ``fd_eval_common.list_samples(root, offsets)``; accepts a direct scene dir
    (images/, flow/, labels/) or a parent of scene dirs.
    __getitem__ returns (frames (N,3,H,W) in [0,255], flow (2,H,W), mask (H,W)).
    """

    def __init__(self, root_path, resize_shape=None, offsets=BOF_WINDOW):
        self.resize_shape = resize_shape
        self.n_frames = len(offsets)
        self.samples = list_samples(root_path, offsets)

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        img = np.array(Image.open(path).convert('RGB')).astype(np.float32)
        if self.resize_shape is not None:
            H, W = self.resize_shape
            img = np.array(Image.fromarray(img.astype(np.uint8)).resize((W, H))).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame_paths, pf, pl = s[:self.n_frames], s[self.n_frames], s[self.n_frames + 1]
        frames = torch.stack([self._load_img(p) for p in frame_paths], dim=0)
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

        return (frames,
                torch.from_numpy(flow).permute(2, 0, 1),
                torch.from_numpy(mask))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _tile_grid(image_shape, tile_size, min_overlap=20):
    """FlowFormer++-style grid: origins covering image, last tile flush to edge."""
    H, W = image_shape
    th, tw = tile_size
    if min_overlap >= th or min_overlap >= tw:
        raise ValueError("min_overlap must be smaller than tile_size")
    hs = list(range(0, H, th - min_overlap))
    ws = list(range(0, W, tw - min_overlap))
    hs[-1] = H - th
    ws[-1] = W - tw
    hs = sorted(set(hs))
    ws = sorted(set(ws))
    return [(h, w) for h in hs for w in ws]


def _gaussian_tile_weight(tile_size, sigma, device):
    """Single 2D Gaussian mask centred on a tile (unnormalised); shape (1,1,th,tw)."""
    th, tw = tile_size
    y = torch.arange(th, device=device, dtype=torch.float32) / th - 0.5
    x = torch.arange(tw, device=device, dtype=torch.float32) / tw - 0.5
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(yy ** 2 + xx ** 2) / sigma
    w = torch.exp(-0.5 * r ** 2)
    return w[None, None]


def _as_window(frames) -> torch.Tensor:
    if isinstance(frames, (list, tuple)):
        frames = torch.stack(list(frames), dim=1)
    if not torch.is_tensor(frames) or frames.dim() != 5:
        raise TypeError("frames must be a (B, N, 3, H, W) tensor or a list of N (B, 3, H, W) "
                        "tensors in time-reversed window order")
    return frames


@torch.no_grad()
def _predict_tiled(model, frames, mode: str, tile_size, sigma=0.05, min_overlap=20) -> torch.Tensor:
    """Gaussian-blended tiled version of ``_predict`` (same window / slot semantics)."""
    frames = _as_window(frames)
    B, N, _, H, W = frames.shape
    th, tw = tile_size
    if H <= th and W <= tw:
        return _predict(model, frames, mode)
    th, tw = min(th, H), min(tw, W)
    hws = _tile_grid((H, W), (th, tw), min_overlap)
    wmask = _gaussian_tile_weight((th, tw), sigma, frames.device)

    flows = torch.zeros(B, 2, H, W, device=frames.device)
    wsum  = torch.zeros(B, 1, H, W, device=frames.device)
    for (h, w) in hws:
        tile_flow = _predict(model, frames[..., h:h + th, w:w + tw], mode)
        flows[:, :, h:h + th, w:w + tw] += tile_flow * wmask
        wsum[:,  :, h:h + th, w:w + tw] += wmask
    return flows / wsum


@torch.no_grad()
def _predict(model, frames, mode: str, input_frames=None) -> torch.Tensor:
    """Run VideoFlow on one time-reversed window and return the reference-frame flow.

    frames: (B, N, 3, H, W) tensor (or list of N (B, 3, H, W) tensors), values in
            [0, 255], already on the model's device, ordered as
            ``window_offsets(mode, N)``:
              BOF (N=3): [I_{t+1}, I_t, I_{t-1}]
              MOF (N=5): [I_{t+2}, I_{t+1}, I_t, I_{t-1}, I_{t-2}]
    mode:   'BOF' or 'MOF'.
    input_frames: optional, only checked against N.
    Returns (B, 2, H, W) forward flow of the centre frame to the next window frame,
    i.e. flow t -> t-1 = the Nature3 GT ``<t>_flow.npy``.

    NOTE: the old signature ``_predict(model, img1, img2, mode, input_frames)`` (pair
    duplicated into a window) is gone; a call in that form raises TypeError.
    """
    if torch.is_tensor(mode):
        raise TypeError("_predict(model, img1, img2, mode, N) was replaced by "
                        "_predict(model, frames, mode): pass the full time-reversed window")
    frames = _as_window(frames)
    N = frames.shape[1]
    if input_frames is not None and input_frames != N:
        raise ValueError(f"window has {N} frames, input_frames={input_frames}")
    if mode == 'BOF' and N != 3:
        raise ValueError(f"BOF needs a 3-frame window, got {N}")
    if mode == 'MOF' and (N < 3 or N % 2 == 0):
        raise ValueError(f"MOF needs an odd window >= 3, got {N}")
    padder = InputPadder(frames.shape)
    images = padder.pad(frames)
    flow_pre, _ = model(images, {})
    flow_pre = padder.unpad(flow_pre)          # BOF (B,2,2,H,W) / MOF (B,2(N-2),2,H,W)
    return flow_pre[:, read_slot(mode, N)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_pairs_csv(output_csv, ckpt_label, H, W):
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    return f"{stem}__{ckpt_label}_{W}x{H}_pairs.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',          default='BOF', choices=['BOF', 'MOF'],
                        help='BOF = 3-frame BOFNet, MOF = multi-frame MOFNetStack')
    parser.add_argument('--ckpt',          required=True)
    parser.add_argument('--image_size',    type=int, nargs=2, default=[432, 960],
                        metavar=('H', 'W'))
    parser.add_argument('--input_frames',  type=int, default=5,
                        help='Number of input frames for MOF mode (must be odd >= 3)')
    parser.add_argument('--datasets_path', default='')
    parser.add_argument('--root',          default='datasets/AITSyntheticParsed')
    parser.add_argument('--max_samples',   type=int, default=500)
    parser.add_argument('--split_file', default=None, help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore --max_samples')
    parser.add_argument('--job_id',        default='unknown')
    parser.add_argument('--output_csv',    required=True)
    parser.add_argument('--pairs_csv',     default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<W>x<H>_pairs.csv)')
    parser.add_argument('--device',        default='auto',
                        help='cuda / cpu / auto (auto uses cuda if available)')
    parser.add_argument('--tile_size',     type=int, nargs=2, default=None,
                        metavar=('TH', 'TW'),
                        help='If set, use Gaussian-blended tiled inference at this tile size.')
    parser.add_argument('--tile_overlap',  type=int, default=20,
                        help='Min overlap (px) between tiles when --tile_size is set.')
    parser.add_argument('--tile_sigma',    type=float, default=0.05,
                        help='Gaussian blending sigma (in normalised tile coords) when tiling.')
    cli = parser.parse_args()

    H, W = cli.image_size
    n_frames = 3 if cli.mode == 'BOF' else cli.input_frames
    offsets = window_offsets(cli.mode, n_frames)

    # Resolve device
    if cli.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cli.device)
    print(f"Device:     {device}")
    print(f"VideoFlow:  {_videoflow_root}")

    # Build config & model
    if cli.mode == 'BOF':
        cfg = _bof_cfg(cli.ckpt, cli.image_size)
    else:
        cfg = _mof_cfg(cli.ckpt, cli.image_size, cli.input_frames)

    # CPU doesn't support mixed precision autocast; disable it
    if device.type == 'cpu':
        if cli.mode == 'BOF':
            cfg.BOFNet.mixed_precision = False
        else:
            cfg.MOFNetStack.mixed_precision = False

    model = build_network(cfg)
    sd = torch.load(cli.ckpt, map_location='cpu')
    load_state_dict_checked(model, sd, strip_prefixes=('module.',), name=f'VideoFlow-{cli.mode}')
    model = model.to(device).eval()

    # Dataset
    root_path    = os.path.join(cli.datasets_path, cli.root) if cli.datasets_path else cli.root
    dataset_name = os.path.basename(root_path.rstrip('/'))
    print(f"Mode:       {cli.mode}  window offsets {offsets}  read forward slot "
          f"{read_slot(cli.mode, n_frames)} (t -> t-1)")
    print(f"Checkpoint: {cli.ckpt}")
    print(f"Dataset:    {root_path}")

    dataset = DroneFlowDataset(root_path, resize_shape=cli.image_size, offsets=offsets)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(cli.split_file))
    n = len(dataset) if cli.split_file else min(len(dataset), cli.max_samples)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=1, drop_last=False)

    ckpt_label = os.path.splitext(os.path.basename(cli.ckpt))[0]
    pairs_csv = cli.pairs_csv or _default_pairs_csv(cli.output_csv, ckpt_label, H, W)
    dumper = PairDumper(pairs_csv)
    oom_count = 0

    print(f"Evaluating {n} samples...")
    with torch.no_grad():
        for i, (frames, flow_gt, drone_mask) in enumerate(loader):
            if i >= n:
                break
            try:
                frames = frames.to(device)
                if cli.tile_size is not None:
                    pred = _predict_tiled(model, frames, cli.mode, cli.tile_size,
                                          sigma=cli.tile_sigma, min_overlap=cli.tile_overlap)
                else:
                    pred = _predict(model, frames, cli.mode)
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                if oom_count == 1:
                    print(f"  WARNING: CUDA OOM at sample {i} (image_size={cli.image_size}). "
                          "Skipping sample (no per-pair row).")
                if oom_count >= 5:
                    print(f"  OOM on {oom_count} samples — resolution likely too large "
                          f"for available GPU memory. Stopping early.")
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

    rows = dumper.rows
    skipped = n - len(rows)
    if oom_count > 0:
        print(f"\n  {oom_count} sample(s) skipped due to CUDA OOM; {skipped} not evaluated in total")

    summ = summary_row(rows, cli.ckpt, dataset_name, f"{H}x{W}", skipped=skipped)
    agg = aggregate(rows)
    print("\n" + "=" * 65)
    print(f"Mode: {cli.mode}  |  Checkpoint: {os.path.basename(cli.ckpt)}")
    print(f"Samples: {len(rows)} (skipped {skipped})  |  Dataset: {dataset_name}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print("-" * 65)
    for metric in ('epe', 'epe_norm', 'px1_err', 'px3_err', 'px5_err', 'fl'):
        print(f"  {metric:<26} {agg[metric + '_all']:>8.4f} {agg[metric + '_drone']:>8.4f} "
              f"{agg[metric + '_bg']:>8.4f}")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(cli.output_csv)), exist_ok=True)
    fieldnames = [
        'job_id', 'checkpoint', 'dataset', 'image_size', 'mode',
        'epe', 'epe_drone',
        'epe_norm', 'epe_norm_drone',
        'px1_err', 'px1_err_drone',
        'px3_err', 'px3_err_drone',
        'px5_err', 'px5_err_drone',
        'fl', 'fl_drone',
        'samples_processed', 'samples_skipped',
    ]
    row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in summ.items()}
    row.update({'job_id': cli.job_id, 'mode': cli.mode})
    with open(cli.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(row)
    print(f"\nCSV written: {cli.output_csv}")
    print(f"Pairs CSV:   {pairs_csv}")


if __name__ == '__main__':
    main()
