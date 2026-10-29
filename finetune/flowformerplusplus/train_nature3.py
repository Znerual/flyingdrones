"""Finetune FlowFormer++ (2-frame) on the FlyingDrones / Nature3 stratified split.

Standalone trainer (train_FlowFormer.py untouched); sources
data from the Nature3 eval layout (images/ flow/ labels/) restricted to a stratified
train/val split file. FlowFormer++ is a native 2-frame model, so unlike the VideoFlow
MOF case this is *genuine* supervision (no frame duplication).

Pairing: Nature3 GT ``flow/<t>_flow.npy`` is the BACKWARD flow
t -> t-1 (I_t(x) ~= I_{t-1}(x + F)). Every sample is therefore (I_t, I_{t-1}, <t>_flow)
with I_t as the reference (first) frame, built by fd_eval_common.list_samples(root,
TWO_FRAME); frame 0000 has no t-1 and is never a reference (dropped automatically).

Frames are resized to the eval scale (432x960, cv2 INTER_AREA + flow scaling, as
evaluate.py::DroneFlowDataset) then random-cropped to the training size, so the drone
scale matches test. Validation runs the patched evaluator's own path: evaluate.py's
DroneFlowDataset, infer_flow with train_size = the --cfg file's image_size (tiled when
eval_size is larger, e.g. things_288960 -> 288x960 tiles at 432x960) and
_compute_metrics, aggregated with fd_eval_common.aggregate -> internal val drone-EPE ==
canonical eval (evaluate.py --cfg <same cfg> --image_size <eval_size>) on the split.

Paper init: --cfg configs.things_288960 --restore_ckpt checkpoints/things_288960.pth
(the --restore_ckpt default is None = random init; a warning is printed).

Example:
    python -u train_nature3.py --name ffpp-nat3 --cfg configs.things_288960 \
        --restore_ckpt checkpoints/things_288960.pth \
        --datasets_path $FD_DATA --root Nature3 \
        --train_split_file $FD_REPO/splits/nature3_ft_train_frames.txt \
        --val_split_file   $FD_REPO/splits/nature3_ft_val_frames.txt \
        --image_size 384 768 --eval_size 432 960 --num_steps 1500 \
        --batch_size 1 --accum 6 --lr 1.25e-4 --mixed_precision
"""
from __future__ import print_function, division
import argparse, importlib, os, sys, glob as _glob

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_REPO, "core"))
sys.path.append(os.path.join(_REPO, "scripts"))
# Script dir first: fd_eval_common.py / drone_split.py / evaluate.py next to this file win.
while _REPO in sys.path:
    sys.path.remove(_REPO)
sys.path.insert(0, _REPO)

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from core.FlowFormer import build_flowformer
from core.optimizer import fetch_optimizer
from core.utils.augmentor import FlowAugmentor
from drone_split import load_split_frames, filter_samples
import fd_eval_common as fdc
from fd_eval_common import TWO_FRAME, list_samples
# canonical (patched) evaluator: dataset + inference + metric -> internal val == canonical
import evaluate as _canon

SUM_FREQ = 100
MAX_FLOW = 400


def canon_train_size(args):
    """evaluate.py: ``train_size = list(cfg.image_size)`` of the UNMODIFIED --cfg file
    (tile size for inputs larger than it). build_cfg overrides image_size for training."""
    return list(importlib.import_module(args.cfg).get_cfg().image_size)


def build_cfg(args):
    cfg = importlib.import_module(args.cfg).get_cfg()
    cfg.image_size = list(args.image_size)
    cfg.batch_size = args.batch_size
    cfg.gamma = args.gamma
    cfg[cfg.transformer].pretrain = False           # weights from restore_ckpt; skip timm dl
    cfg.trainer.canonical_lr = args.lr
    cfg.trainer.adamw_decay = args.wdecay
    cfg.trainer.num_steps = args.num_steps
    return cfg


def _list_samples(root, split_file):
    """(I_t, I_{t-1}, <t>_flow.npy, <t>_label.png) for every reference frame t >= 1 in
    the split. GT <t>_flow.npy is the backward flow t -> t-1, so I_t is frame 1."""
    return filter_samples(list_samples(root, TWO_FRAME), load_split_frames(split_file))


def _resize_to(img1, img2, flow, H, W):
    """cv2 INTER_AREA resize to (H,W); scale flow vectors per-axis (eval-consistent)."""
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
        img1, img2, flow = self.aug(img1, img2, flow)               # uint8 imgs, cv2 aug
        valid = ((np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000) &
                 np.isfinite(flow).all(axis=-1))
        flow = np.nan_to_num(flow)
        img1 = torch.from_numpy(np.ascontiguousarray(img1)).permute(2, 0, 1).float()
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).permute(2, 0, 1).float()
        flow = torch.from_numpy(np.ascontiguousarray(flow)).permute(2, 0, 1).float()
        return img1, img2, flow, torch.from_numpy(valid).float()


def sequence_loss(flow_preds, flow_gt, valid, gamma, max_flow=MAX_FLOW):
    n = len(flow_preds)
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    v = valid[:, None].float()
    loss = 0.0
    for i in range(n):
        w = gamma ** (n - i - 1)
        loss += w * (v * (flow_preds[i] - flow_gt).abs()).sum() / v.sum().clamp(min=1.0)
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt().view(-1)[valid.view(-1)]
    return loss, {"epe": epe.mean().item(), "1px": (epe < 1).float().mean().item()}


_VAL_DS = {}


def _val_dataset(args):
    """Canonical evaluator dataset (evaluate.DroneFlowDataset), built once."""
    key = (args.datasets_path, args.root, args.val_split_file, tuple(args.eval_size))
    if key not in _VAL_DS:
        ds = _canon.DroneFlowDataset(os.path.join(args.datasets_path, args.root),
                                     resize_shape=list(args.eval_size))
        ds.samples = filter_samples(ds.samples, load_split_frames(args.val_split_file))
        _VAL_DS[key] = ds
    return _VAL_DS[key]


@torch.no_grad()
def validate_nature3(model, args):
    """evaluate.py evaluate()'s per-pair path: infer_flow (tiled at the --cfg file's
    image_size when eval_size is larger) + _compute_metrics."""
    was = model.training
    model.eval()
    ds = _val_dataset(args)
    train_size = canon_train_size(args)
    rows = []
    for i in range(len(ds)):
        img1, img2, flow_gt, mask = ds[i]          # img1 = I_t, img2 = I_{t-1}
        img1, img2 = img1[None].cuda(), img2[None].cuda()
        pred = _canon.infer_flow(model, img1, img2, train_size)
        if pred.shape[-2:] != flow_gt.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=flow_gt.shape[-2:],
                                                   mode="bilinear", align_corners=False)
        rows.append(_canon._compute_metrics(pred[0].permute(1, 2, 0).cpu().numpy(),
                                            flow_gt.permute(1, 2, 0).numpy(), mask.numpy()))
    if was:
        model.train()
    agg = fdc.aggregate(rows)
    d = agg["epe_drone"]
    return {"nat3_val_epe": agg["epe_all"],
            "nat3_val_epe_drone": d if d == d else -1.0}


def save_ckpt(path, model, optimizer, scheduler, scaler, step):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "step": step}, tmp)
    os.replace(tmp, path)


def train(args):
    cfg = build_cfg(args)
    model = nn.DataParallel(build_flowformer(cfg))
    latest = os.path.join(args.ckpt_dir, f"{args.name}_latest.pth")
    start_step, resume_state = 0, None
    if os.path.exists(latest):
        resume_state = torch.load(latest, map_location="cpu")
        model.load_state_dict(resume_state["model"]); start_step = resume_state["step"]
        print(f"resumed from {latest} at step {start_step}")
    elif args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt, map_location="cpu"), strict=True)
        print(f"initialized from {args.restore_ckpt}")
    else:
        print("WARNING: no --restore_ckpt and no resume state -> training from RANDOM init "
              "(paper recipe: --restore_ckpt checkpoints/things_288960.pth)", flush=True)
    model.cuda(); model.train()

    train_ds = Nature3PairDataset(args, args.train_split_file)
    print(f"train samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)

    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)
    scaler = GradScaler(enabled=args.mixed_precision)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])

    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "runs", args.name))
    running, total_steps, micro = {}, start_step, 0
    best_val = float("inf")
    optimizer.zero_grad()
    done = False
    while not done:
        for img1, img2, flow, valid in train_loader:
            img1, img2 = img1.cuda(), img2.cuda()
            flow, valid = flow.cuda(), valid.cuda()
            flow_predictions = model(img1, img2)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg.gamma)
            scaler.scale(loss / args.accum).backward()
            micro += 1
            if micro % args.accum != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            scaler.step(optimizer); scheduler.step(); scaler.update()
            optimizer.zero_grad()
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            total_steps += 1

            if total_steps % SUM_FREQ == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[{total_steps:6d}, lr={lr:.7f}] " +
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
                    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.name}_best.pth"))
                    print(f"  new best val_epe_drone={best_val:.4f}", flush=True)
            if total_steps >= args.num_steps:
                done = True; break

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.name}.pth"))
    save_ckpt(latest, model, optimizer, scheduler, scaler, total_steps)
    writer.close()
    print(f"done at step {total_steps}, best val_epe_drone={best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ffpp-nat3")
    p.add_argument("--cfg", default="configs.things_288960")
    p.add_argument("--restore_ckpt", default=None)
    p.add_argument("--datasets_path", default=os.environ.get("HOME", ""))
    p.add_argument("--root", default="Nature3")
    p.add_argument("--train_split_file", required=True)
    p.add_argument("--val_split_file", required=True)
    p.add_argument("--ckpt_dir", default="checkpoints_nat3")
    p.add_argument("--lr", type=float, default=1.25e-4)
    p.add_argument("--num_steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--accum", type=int, default=6)
    p.add_argument("--image_size", type=int, nargs=2, default=[384, 768])
    p.add_argument("--eval_size", type=int, nargs=2, default=[432, 960])
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--wdecay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--min_scale", type=float, default=0.0)
    p.add_argument("--max_scale", type=float, default=0.4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--val_every", type=int, default=150)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
