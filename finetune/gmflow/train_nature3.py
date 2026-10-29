"""Finetune GMFlow (gmflow_chairs, non-refine 'default' config) on FlyingDrones/Nature3.

Chosen as a weak-but-nonzero real-footage model: gmflow_chairs shows a drone-motion
signal on real footage that is present but far from saturated.
GMFlow's global matching needs resolution, so we finetune AND evaluate at 720x1280.

Standalone trainer; sources Nature3 via a split file, uses the
gmflow *default* (single-scale) config that matches gmflow_chairs, bf16 autocast (the
-1e9 correlation mask overflows fp16), and best-by-val checkpointing.

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow
t -> t-1 (I_t(x) ~= I_{t-1}(x + F)). Every sample is therefore (I_t, I_{t-1}, <t>_flow)
with I_t as the reference (first) frame, built by fd_eval_common.list_samples(root,
TWO_FRAME); frame 0000 has no t-1 and is never a reference (dropped automatically).

Validation runs the patched canonical evaluator's own path: evaluate_drones.py's
DroneFlowDataset (datasets/drones.py), infer_flow (fp32, InputPadder, tiled at the
evaluator's 432x960 tile size when eval_size is larger) and compute_metrics, aggregated
with fd_eval_common.aggregate -> internal val drone-EPE == canonical eval on the split.

Example:
    python -u train_nature3.py --name gmflow-nat3 \
        --restore_ckpt pretrained/gmflow_chairs-1d776046.pth \
        --datasets_path $FD_DATA --root Nature3 \
        --train_split_file $FD_REPO/splits/nature3_ft_train_frames.txt \
        --val_split_file   $FD_REPO/splits/nature3_ft_val_frames.txt \
        --image_size 512 896 --eval_size 720 1280 --num_steps 1500 \
        --batch_size 1 --lr 5e-5 --mixed_precision
"""
from __future__ import print_function, division
import argparse, os, sys, glob as _glob

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_REPO, "core"))
# Script dir first: fd_eval_common.py / drone_split.py next to this file must win.
while _REPO in sys.path:
    sys.path.remove(_REPO)
sys.path.insert(0, _REPO)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from gmflow.gmflow import GMFlow
from data.transforms import FlowAugmentor
from drone_split import load_split_frames, filter_samples
import fd_eval_common as fdc
from fd_eval_common import TWO_FRAME, list_samples
# canonical (patched) evaluator: dataset + inference + metric -> internal val == canonical
import evaluate_drones as _canon

# gmflow 'default' (single-scale) config, matches gmflow_chairs-1d776046
MODEL_KWARGS = dict(feature_channels=128, num_scales=1, upsample_factor=8,
                    num_head=1, attention_type="swin", ffn_dim_expansion=4,
                    num_transformer_layers=6)
FWD_KWARGS = dict(attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1])
PADDING_FACTOR = 16
# evaluate_drones.evaluate(): ``train_size = [432, 960]`` = tile size for larger inputs
CANON_TILE = [432, 960]
_cc = _canon._CKPT_CONFIGS["default"]           # evaluator's config for gmflow_chairs
assert _cc["padding_factor"] == PADDING_FACTOR and all(_cc[k] == v for k, v in FWD_KWARGS.items()), \
    "train_nature3 FWD_KWARGS drifted from evaluate_drones default config"
MAX_FLOW = 400
SUM_FREQ = 100


def _list_samples(root, split_file):
    """(I_t, I_{t-1}, <t>_flow.npy, <t>_label.png) for every reference frame t >= 1 in
    the split. GT <t>_flow.npy is the backward flow t -> t-1, so I_t is frame 1."""
    return filter_samples(list_samples(root, TWO_FRAME), load_split_frames(split_file))


def _resize_to(img1, img2, flow, H, W):
    oh, ow = flow.shape[:2]
    sx, sy = W / float(ow), H / float(oh)
    r1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(img2, (W, H), interpolation=cv2.INTER_AREA)
    fr = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) * np.array([sx, sy], np.float32)
    return r1, r2, fr.astype(np.float32)


class Nature3PairDataset(Dataset):
    def __init__(self, args, split_file):
        self.samples = _list_samples(os.path.join(args.datasets_path, args.root), split_file)
        self.eH, self.eW = args.eval_size
        self.aug = FlowAugmentor(crop_size=list(args.image_size),
                                 min_scale=args.min_scale, max_scale=args.max_scale, do_flip=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, pf, _ = self.samples[idx]     # p1 = I_t (reference), p2 = I_{t-1}
        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        flow = np.load(pf).astype(np.float32)
        img1, img2, flow = _resize_to(img1, img2, flow, self.eH, self.eW)
        img1, img2, flow = self.aug(img1, img2, flow)
        valid = ((np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000) &
                 np.isfinite(flow).all(axis=-1))
        flow = np.nan_to_num(flow)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()
        return t(img1), t(img2), t(flow), torch.from_numpy(valid).float()


def flow_loss(flow_preds, flow_gt, valid, gamma, max_flow=MAX_FLOW):
    n = len(flow_preds)
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    v = valid[:, None].float()
    loss = 0.0
    for i in range(n):
        loss += gamma ** (n - i - 1) * (v * (flow_preds[i] - flow_gt).abs()).sum() / v.sum().clamp(min=1.0)
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt().view(-1)[valid.view(-1)]
    return loss, {"epe": epe.mean().item(), "1px": (epe < 1).float().mean().item()}


_VAL_DS = {}


def _val_dataset(args):
    """Canonical evaluator dataset (evaluate_drones -> datasets.drones), built once."""
    key = (args.datasets_path, args.root, args.val_split_file, tuple(args.eval_size))
    if key not in _VAL_DS:
        ds = _canon.DroneFlowDataset(os.path.join(args.datasets_path, args.root),
                                     resize_shape=list(args.eval_size))
        ds.samples = filter_samples(ds.samples, load_split_frames(args.val_split_file))
        _VAL_DS[key] = ds
    return _VAL_DS[key]


@torch.no_grad()
def validate_nature3(model, args):
    """evaluate_drones.evaluate()'s per-pair path: fp32 infer_flow (tiled at CANON_TILE
    when eval_size is larger, as the evaluator does) + compute_metrics."""
    was = model.training
    model.eval()
    ds = _val_dataset(args)
    rows = []
    for i in range(len(ds)):
        img1, img2, flow_gt, mask = ds[i]          # img1 = I_t, img2 = I_{t-1}
        img1, img2 = img1[None].cuda(), img2[None].cuda()
        pred = _canon.infer_flow(model, img1, img2, CANON_TILE, FWD_KWARGS, PADDING_FACTOR)
        if pred.shape[-2:] != flow_gt.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=flow_gt.shape[-2:],
                                                   mode="bilinear", align_corners=False)
        rows.append(_canon.compute_metrics(pred[0].permute(1, 2, 0).float().cpu().numpy(),
                                           flow_gt.permute(1, 2, 0).numpy(), mask.numpy()))
    if was:
        model.train()
    agg = fdc.aggregate(rows)
    d = agg["epe_drone"]
    return {"nat3_val_epe": agg["epe_all"],
            "nat3_val_epe_drone": d if d == d else -1.0}


def save_ckpt(path, model, opt, sch, scaler, step):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sch.state_dict(), "scaler": scaler.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)


def train(args):
    model = GMFlow(**MODEL_KWARGS).cuda()
    latest = os.path.join(args.ckpt_dir, f"{args.name}_latest.pth")
    start_step, resume_state = 0, None
    if os.path.exists(latest):
        resume_state = torch.load(latest, map_location="cpu")
        model.load_state_dict(resume_state["model"]); start_step = resume_state["step"]
        print(f"resumed at {start_step}")
    elif args.restore_ckpt is not None:
        ck = torch.load(args.restore_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=True)
        print(f"initialized from {args.restore_ckpt}")
    model.train()

    train_ds = Nature3PairDataset(args, args.train_split_file)
    print(f"train samples: {len(train_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False,
                                              anneal_strategy="linear")
    scaler = torch.cuda.amp.GradScaler(enabled=False)   # bf16 -> no scaling needed
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])

    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "runs", args.name))
    running, total_steps, best_val = {}, start_step, float("inf")
    done = False
    while not done:
        for image1, image2, flow, valid in loader:
            image1, image2 = image1.cuda(non_blocking=True), image2.cuda(non_blocking=True)
            flow, valid = flow.cuda(), valid.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.bfloat16):
                preds = model(image1, image2, **FWD_KWARGS)["flow_preds"]
            loss, metrics = flow_loss(preds, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer); scheduler.step(); scaler.update()
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            total_steps += 1

            if total_steps % SUM_FREQ == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[{total_steps:6d}, lr={lr:.8f}] " +
                      " ".join(f"{k}={running[k]/SUM_FREQ:.4f}" for k in sorted(running)), flush=True)
                for k in running:
                    writer.add_scalar(k, running[k] / SUM_FREQ, total_steps)
                running = {}
            if total_steps % args.save_every == 0:
                save_ckpt(latest, model, optimizer, scheduler, scaler, total_steps)
            if total_steps % args.val_every == 0:
                res = validate_nature3(model, args)
                print(f"[{total_steps:6d}] " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
                for k, v in res.items():
                    writer.add_scalar(k, v, total_steps)
                if 0 <= res["nat3_val_epe_drone"] < best_val:
                    best_val = res["nat3_val_epe_drone"]
                    torch.save({"model": model.state_dict()},
                               os.path.join(args.ckpt_dir, f"{args.name}_best.pth"))
                    print(f"  new best val_epe_drone={best_val:.4f}", flush=True)
            if total_steps >= args.num_steps:
                done = True; break

    torch.save({"model": model.state_dict()}, os.path.join(args.ckpt_dir, f"{args.name}.pth"))
    save_ckpt(latest, model, optimizer, scheduler, scaler, total_steps)
    writer.close()
    print(f"done at {total_steps}, best val_epe_drone={best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="gmflow-nat3")
    p.add_argument("--restore_ckpt", default="pretrained/gmflow_chairs-1d776046.pth")
    p.add_argument("--datasets_path", default=os.environ.get("HOME", ""))
    p.add_argument("--root", default="Nature3")
    p.add_argument("--train_split_file", required=True)
    p.add_argument("--val_split_file", required=True)
    p.add_argument("--ckpt_dir", default="checkpoints_nat3")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 896])
    p.add_argument("--eval_size", type=int, nargs=2, default=[720, 1280])
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--wdecay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--min_scale", type=float, default=-0.2)
    p.add_argument("--max_scale", type=float, default=0.4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--val_every", type=int, default=150)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
