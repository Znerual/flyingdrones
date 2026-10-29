"""Evaluate FlowDiffusion (DDVM) on the Synthetic Drones dataset.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow t -> t-1, so the
model is run on (I_t, I_{t-1}) with I_t as reference; frame 0000 is never a
reference (fd_eval_common.list_samples). --dataset_path may be a scene dir
(images/ flow/ labels/, e.g. .../Nature3) or a parent of scene dirs.

Metrics = paper definitions (fd_eval_common.compute_metrics):
  epe       — mean endpoint error (pixels)
  epe_norm  — sqrt((u_err/W)²+(v_err/H)²) × 100
  px1/3/5   — % pixels with EPE > 1/3/5 px  (error rate)
  fl        — % pixels where (EPE>3px) AND (EPE/gt_mag>5%)
Drone vs background split uses label PNGs from the dataset. Summary =
frame-weighted mean over pairs; one row per pair -> --pairs_csv. Pairs skipped
after a repeated GPU OOM get no row and are counted in samples_skipped.

Usage:
    python evaluate_drones.py \\
        --pipeline_path checkpoints/pipeline-305000 \\
        --dataset_path $FD_DATA/Nature3 \\
        --image_size 1080 1920 \\
        --ddpm_num_steps 64 \\
        --normalize_range \\
        --output_csv results.csv
"""

import sys
import os
import os.path as osp
import math
import csv
import argparse

# Make local_diffusers importable from anywhere
_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
# Container path fallback. APPENDED (2026-10-07; previously insert(0)), so a stale
# /app/FlowDiffusion baked into the image can no longer shadow this script's own
# drone_split / fd_eval_common / local_diffusers when run from a bind-mounted
# checkout; /app only fills in modules missing locally.
if osp.exists('/app/FlowDiffusion') and osp.realpath('/app/FlowDiffusion') != osp.realpath(_SCRIPT_DIR):
    sys.path.append('/app/FlowDiffusion')

import numpy as np
import cv2
from glob import glob

import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm

from local_diffusers.pipelines.DDPM import DDPMPipeline
from drone_split import load_split_frames, filter_samples

# Shared Nature3 helpers (pairing, paper metrics, per-pair dump). Re-insert our own
# dir right before the import so a stale copy elsewhere on sys.path (e.g. baked into
# a container under /app) can never shadow the bind-mounted / local one.
sys.path.insert(0, _SCRIPT_DIR)
import fd_eval_common as fdc  # noqa: E402

# Train size the model was trained at
TRAIN_SIZE = [320, 448]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DroneFlowDataset(data.Dataset):
    """Returns (img_t, img_{t-1}, flow_t, drone_mask_t).

    dataset_path is either a scene dir (images/, flow/, optionally labels/) or a
    parent of such scene dirs.
    """

    def __init__(self, dataset_path, resize_shape=None):
        self.resize_shape = resize_shape
        self.samples = []

        # Pairing (fixed 2026-10-07): ``flow/<t>_flow.npy`` is the BACKWARD flow
        # t -> t-1, so each sample is (I_t, I_{t-1}, <t>_flow, <t>_label) with I_t as
        # the reference frame; frame 0000 is never a reference. Built by the shared
        # fd_eval_common.list_samples (flat scene dir or parent of scene dirs).
        # Labels stay optional per scene: if a scene has a labels/ dir, samples
        # without a label file are dropped; otherwise the mask is all-zero.
        _all = fdc.list_samples(dataset_path, fdc.TWO_FRAME, require_labels=False)
        self.samples = [s for s in _all
                        if s[-1] is not None
                        or not osp.isdir(osp.join(osp.dirname(osp.dirname(s[-2])), 'labels'))]

        print(f"DroneFlowDataset: {len(self.samples)} pairs (I_t, I_t-1) from {dataset_path}")

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
            ft = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0)
            ft = F.interpolate(ft, size=(H, W), mode='bilinear', align_corners=False)
            ft[:, 0] *= W / ow
            ft[:, 1] *= H / oh
            flow = ft.squeeze(0).permute(1, 2, 0).numpy()
            mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mt, size=(H, W), mode='nearest').squeeze().numpy()

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        mask = torch.from_numpy(np.asarray(mask)).float()
        return img1, img2, flow, mask


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_flow(pipeline, img1, img2, eval_size, ddpm_num_steps, normalize_range):
    """Run pipeline and return pixel-space flow at eval_size.

    img1, img2: (B, 3, H, W) float tensors in [0, 255]
    Returns flow (B, 2, H, W) in pixel coords at eval_size.
    """
    H_eval, W_eval = eval_size
    H_train, W_train = TRAIN_SIZE

    # Normalise to [-1, 1] and resize to train size
    img1_n = 2.0 * (img1 / 255.0) - 1.0
    img2_n = 2.0 * (img2 / 255.0) - 1.0
    img1_r = F.interpolate(img1_n, size=TRAIN_SIZE, mode='bicubic', align_corners=True)
    img2_r = F.interpolate(img2_n, size=TRAIN_SIZE, mode='bicubic', align_corners=True)

    inputs = torch.cat([img1_r, img2_r], dim=1).to(torch.bfloat16).to(pipeline.device)

    flow_train = pipeline(
        inputs=inputs,
        batch_size=inputs.shape[0],
        num_inference_steps=ddpm_num_steps,
        output_type="tensor",
        normalize=normalize_range,
    ).images.to(torch.float32)

    # Scale flow from train size to eval size
    flow_eval = F.interpolate(flow_train, size=(H_eval, W_eval),
                               mode='bicubic', align_corners=True)
    flow_eval = flow_eval * torch.tensor(
        [W_eval / W_train, H_eval / H_train],
        device=flow_eval.device
    ).view(1, 2, 1, 1)

    return flow_eval


# ---------------------------------------------------------------------------
# Metrics — paper definitions (fd_eval_common)
# ---------------------------------------------------------------------------

def compute_metrics(pred, flow_gt, drone_mask):
    """Paper metrics for one pair (delegates to fd_eval_common.compute_metrics).

    pred, flow_gt: (2, H, W) torch tensors in pixel coords
    drone_mask:    (H, W) float tensor, >0.5 = drone pixel
    epe_norm = 100 * sqrt((u_err/W)^2 + (v_err/H)^2)
    """
    return fdc.compute_metrics(pred, flow_gt, drone_mask)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _load_pipeline(pipeline_path, device):
    """Load pipeline onto device. On CPU uses float32 (bfloat16 unsupported)."""
    if device == 'cpu':
        print(f"Loading pipeline (float32, CPU)...")
        pipeline = DDPMPipeline.from_pretrained(pipeline_path).to('cpu')
    else:
        print(f"Loading pipeline (bfloat16, GPU)...")
        pipeline = DDPMPipeline.from_pretrained(pipeline_path,
                                                torch_dtype=torch.bfloat16).to(device)
    pipeline.unet.eval()
    _check_unet_weights(pipeline.unet, pipeline_path)
    return pipeline


def _check_unet_weights(unet, pipeline_path):
    """Fail loudly unless every UNet tensor equals the one in the safetensors file.

    diffusers' from_pretrained can leave weights at init on a key mismatch (warning
    only), the same failure class as the RAFT loader bug.
    """
    from safetensors.torch import load_file
    wf = osp.join(pipeline_path, 'unet', 'diffusion_pytorch_model.safetensors')
    if not osp.exists(wf):
        raise FileNotFoundError(f"[flowdiffusion] weight file not found: {wf}")
    file_sd = load_file(wf, device='cpu')
    model_sd = unet.state_dict()
    missing = sorted(set(model_sd) - set(file_sd))
    unexpected = sorted(set(file_sd) - set(model_sd))
    differ = [k for k in model_sd if k in file_sd and not torch.equal(
        model_sd[k].detach().cpu(), file_sd[k].to(model_sd[k].dtype))]
    print(f"[flowdiffusion] loaded {len(model_sd) - len(missing)}/{len(model_sd)} keys; "
          f"missing={len(missing)} unexpected={len(unexpected)} value_mismatch={len(differ)}")
    if missing or unexpected or differ:
        raise RuntimeError(f"[flowdiffusion] weight/key mismatch: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]} differ={differ[:5]}")


def default_pairs_csv(output_csv, ckpt_path, res_str):
    """<output_csv stem>__<ckpt stem>_<HxW>_pairs.csv"""
    stem = output_csv[:-4] if output_csv.endswith('.csv') else output_csv
    ckpt_stem = osp.splitext(osp.basename(str(ckpt_path).rstrip('/')))[0]
    return f'{stem}__{ckpt_stem}_{res_str}_pairs.csv'


def evaluate(args):
    eval_size  = list(args.image_size)     # [H, W]
    device = getattr(args, 'device', 'cuda')

    print(f"\nLoading pipeline from {args.pipeline_path}")
    try:
        pipeline = _load_pipeline(args.pipeline_path, device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if device != 'cpu' and 'CUDA' in str(e):
            print(f"GPU OOM loading pipeline, retrying on CPU...")
            torch.cuda.empty_cache()
            pipeline = _load_pipeline(args.pipeline_path, 'cpu')
            device = 'cpu'
        else:
            raise

    dataset_name = osp.basename(args.dataset_path.rstrip('/'))
    dataset = DroneFlowDataset(args.dataset_path, resize_shape=eval_size)
    dataset.samples = filter_samples(dataset.samples, load_split_frames(args.split_file))
    if len(dataset) == 0:
        print(f"ERROR: no samples found in {args.dataset_path}")
        return

    if args.split_file or not args.max_samples:
        n_samples = len(dataset)
    else:
        n_samples = min(len(dataset), args.max_samples)
    subset = torch.utils.data.Subset(dataset, list(range(n_samples)))
    loader = data.DataLoader(subset, batch_size=1, shuffle=False,
                             num_workers=1, drop_last=False)

    res_str = f"{eval_size[0]}x{eval_size[1]}"
    ckpt_name = osp.basename(args.pipeline_path.rstrip('/'))
    pairs_csv = args.pairs_csv or default_pairs_csv(args.output_csv, args.pipeline_path, res_str)
    dumper = fdc.PairDumper(pairs_csv)
    print(f"Evaluating {n_samples}/{len(dataset)} pairs | ckpt={ckpt_name} | size={res_str}")

    oom_count = 0
    skipped = 0
    # Subset(range(n)), shuffle=False, batch 1 -> i indexes subset.indices / dataset.samples
    for i, (img1, img2, flow_gt, drone_mask) in enumerate(tqdm(loader, total=n_samples)):
        sample = dataset.samples[subset.indices[i]]
        img1 = img1.to(device)   # I_t   (reference)
        img2 = img2.to(device)   # I_t-1
        try:
            flow_pred = predict_flow(pipeline, img1, img2, eval_size,
                                      args.ddpm_num_steps, args.normalize_range)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if device != 'cpu' and 'CUDA' in str(e):
                oom_count += 1
                if oom_count == 1:
                    print(f"\nGPU OOM during inference — switching to CPU for remainder")
                    torch.cuda.empty_cache()
                    device = 'cpu'
                    pipeline = _load_pipeline(args.pipeline_path, 'cpu')
                    img1 = img1.cpu()
                    img2 = img2.cpu()
                    flow_pred = predict_flow(pipeline, img1, img2, eval_size,
                                              args.ddpm_num_steps, args.normalize_range)
                else:
                    print(f"\nRepeated OOM — skipping sample {fdc.ref_base(sample)}")
                    skipped += 1   # no per-pair row
                    continue
            else:
                raise

        if flow_pred.shape[-2:] != flow_gt.shape[-2:]:
            flow_pred = F.interpolate(flow_pred, size=flow_gt.shape[-2:],
                                       mode='bilinear', align_corners=False)

        m = compute_metrics(flow_pred[0], flow_gt[0].to(flow_pred.device),
                             drone_mask[0].to(flow_pred.device))
        dumper.add(fdc.ref_base(sample), m)
    dumper.close()

    avgs = fdc.aggregate(dumper.rows)
    row = fdc.summary_row(dumper.rows, ckpt_name, dataset_name, res_str, skipped=skipped)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Checkpoint : {ckpt_name}")
    print(f"Eval size  : {res_str}  |  pairs: {len(dumper.rows)}  skipped: {skipped}")
    print(f"{'Metric':<28} {'All':>8} {'Drone':>8} {'BG':>8}")
    print('-' * 70)
    for m in fdc.METRIC_NAMES:
        a = avgs.get(f'{m}_all',   float('nan'))
        d = avgs.get(f'{m}_drone', float('nan'))
        b = avgs.get(f'{m}_bg',    float('nan'))
        print(f"  {m:<26} {a:>8.4f} {d:>8.4f} {b:>8.4f}")
    print('=' * 70)

    # Write CSV row (legacy layout: job_id + the shared summary columns)
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
    parser = argparse.ArgumentParser(
        description='Evaluate FlowDiffusion on Synthetic Drones dataset')
    parser.add_argument('--pipeline_path', required=True,
                        help='Path to HuggingFace-format pipeline directory')
    parser.add_argument('--dataset_path', required=True,
                        help='Scene dir with images/ flow/ labels/ (e.g. .../Nature3) or a parent of scene dirs')
    parser.add_argument('--image_size', nargs=2, type=int, required=True,
                        metavar=('H', 'W'),
                        help='Evaluation resolution, e.g. 1080 1920')
    parser.add_argument('--ddpm_num_steps', type=int, default=64,
                        help='Number of DDPM denoising steps (default: 64)')
    parser.add_argument('--normalize_range', action='store_true',
                        help='Use normalised flow range (required for released checkpoints)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit evaluation to first N samples (default: all)')
    parser.add_argument('--pairs_csv', default=None,
                        help='Per-pair metrics CSV (default: <output_csv stem>__<ckpt stem>_<HxW>_pairs.csv)')
    parser.add_argument('--split_file', default=None,
                        help='Path to nature3_split_frames.txt; if set, restrict eval to that stratified subset and ignore --max_samples')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on. Falls back to CPU on GPU OOM.')
    parser.add_argument('--job_id',     default='unknown',
                        help='Job identifier written to CSV (default: unknown)')
    parser.add_argument('--output_csv', default='results.csv',
                        help='CSV file to append results to')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
