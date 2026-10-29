"""Finetune VideoFlow MOF (5-frame) on the FlyingDrones / Nature3 stratified split.

Standalone trainer (train_MOFNet.py untouched); sources
data from the Nature3 eval layout (images/ flow/ labels/) restricted to a
stratified train/val split file. Reuses VideoFlow's own multi-frame augmentor,
sequence loss, optimizer builder and the sintel multi-frame finetune config.

MOF takes 5 frames [W0..W4] and predicts 3 forward flows and 3 backward flows,
see VideoFlow/core/Networks/MOFNetStack/network.py: forward_corr_fn = corr(fmaps[1:N-1],
fmaps[2:N]) (L157) and the output is cat([forward (N-2), backward (N-2)]) (L210), so
forward slot j = W(j+1) -> W(j+2) and backward slot j = W(j+1) -> W(j).

Ground truth: Nature3 ``flow/<t>_flow.npy`` is the BACKWARD flow of
frame t, t -> t-1 (I_t(x) ~= I_{t-1}(x + F)). The window is therefore the
time-reversed MOF_WINDOW of fd_eval_common, exactly as the patched evaluator
(scripts/analyze_drone_epe.py) feeds it:

    W = [t+2, t+1, t, t-1, t-2]          (fd_eval_common.list_samples(root, MOF_WINDOW))
    forward slot 0 = W1->W2 = (t+1)->t     GT <t+1>_flow.npy
    forward slot 1 = W2->W3 =  t->(t-1)    GT <t>_flow.npy   (evaluator's read slot)
    forward slot 2 = W3->W4 = (t-1)->(t-2) GT <t-1>_flow.npy
    backward slots 0..2 = W(j+1)->W(j) = forward-in-time flows: not stored -> masked
                          invalid (INVALID_SENTINEL), as before.

Every slot thus holds the backward-flow file of its source frame, read from the full
contiguous 9001-frame sequence. list_samples clamps out-of-range offsets to the nearest
frame (edge replication); a slot whose two frames coincide after clamping (only
t=1: slot 2 = 0->0, and t=last: slot 0) is masked invalid instead of carrying GT.
Frame 0000 is never a reference.

Validation runs the patched evaluator's own path: analyze_drone_epe.DroneFlowDataset
(reversed window, PIL resize, per-axis flow scaling), analyze_drone_epe._predict(model,
frames, 'MOF') (InputPadder, forward slot 1 = t -> t-1) and fd_eval_common.compute_metrics
+ aggregate, with the evaluator's inference config (multiframes_sintel_submission:
decoder_depth 32, AMP on) applied to the network for the duration of validation. So the
internal val drone-EPE equals `analyze_drone_epe.py --mode MOF --image_size <eval_size>`
on the same split.

Example:
    python -u train_nature3.py --name videoflow-mof-nat3 \
        --restore_ckpt VideoFlow_ckpt/MOF_sintel.pth \
        --datasets_path $FD_DATA --root Nature3 \
        --train_split_file $FD_REPO/splits/nature3_ft_train_frames.txt \
        --val_split_file   $FD_REPO/splits/nature3_ft_val_frames.txt \
        --image_size 384 768 --eval_size 432 960 --mixed_precision \
        --num_steps 1500 --val_every 150 --batch_size 1 --accum 4 --lr 6.25e-5
"""
from __future__ import print_function, division
import argparse, os, random, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_DIR, "VideoFlow"),
              os.path.abspath(os.path.join(_DIR, "..", "VideoFlow"))):
    if os.path.isdir(os.path.join(_cand, "core")):
        sys.path.insert(0, _cand); break
else:
    raise ImportError("VideoFlow repo (core/, configs/) not found next to train_nature3.py")
sys.path.insert(0, os.path.join(_DIR, "scripts"))   # drone_split.py, analyze_drone_epe.py
# Script dir first: fd_eval_common.py next to this file must win over any stale copy.
while _DIR in sys.path:
    sys.path.remove(_DIR)
sys.path.insert(0, _DIR)
import fd_eval_common as fdc
from fd_eval_common import MOF_WINDOW, list_samples

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.Networks import build_network
from core.utils.augmentor_multiframes import FlowAugmentor
from configs.sintel_multiframes import get_cfg
from drone_split import load_split_frames, filter_samples
# patched canonical evaluator: reversed-window dataset, inference (slot read) and config
import analyze_drone_epe as _canon

SUM_FREQ = 100
INVALID_SENTINEL = 1e4
N_FRAMES = 5
READ_SLOT = _canon.read_slot("MOF", N_FRAMES)   # forward slot t -> t-1 = 1
assert tuple(_canon.window_offsets("MOF", N_FRAMES)) == MOF_WINDOW


def make_cfg(args):
    cfg = get_cfg()
    cfg.image_size = list(args.image_size)
    cfg.batch_size = args.batch_size
    cfg.input_frames = N_FRAMES
    cfg.restore_ckpt = args.restore_ckpt
    cfg.mixed_precision = args.mixed_precision
    cfg.MOFNetStack.mixed_precision = args.mixed_precision
    cfg.MOFNetStack.pretrain = False       # full ckpt restored; skip timm download
    cfg.trainer.canonical_lr = args.lr
    cfg.trainer.adamw_decay = args.wdecay
    cfg.trainer.num_steps = args.num_steps
    cfg.gamma = args.gamma
    return cfg


def _base(path):
    return os.path.splitext(os.path.basename(path))[0]


def _list_temporal_samples(root, split_file):
    """Reversed MOF windows for every reference frame t >= 1 in the split.

    Returns tuples (window, slot_flows, flow_t, label_t): ``window`` = 5 frame paths
    [t+2, t+1, t, t-1, t-2] (edge-clamped by fd_eval_common.list_samples);
    ``slot_flows[j]`` = GT for forward slot j = W(j+1) -> W(j+2), i.e. the backward-flow
    file ``<W(j+1)>_flow.npy`` of the slot's source frame, or None when clamping made
    the two frames identical (slot masked invalid). slot_flows[READ_SLOT] == flow_t.
    """
    samples = filter_samples(list_samples(root, MOF_WINDOW), load_split_frames(split_file))
    out = []
    for s in samples:
        win, ff, lf = list(s[:N_FRAMES]), s[N_FRAMES], s[N_FRAMES + 1]
        slots = []
        for j in range(N_FRAMES - 2):
            src, dst = win[j + 1], win[j + 2]
            if src == dst:                       # clamped at a sequence edge -> no GT
                slots.append(None)
                continue
            if int(_base(src)) - int(_base(dst)) != 1:
                raise RuntimeError(f"non-contiguous frames {src} -> {dst}: the MOF trainer "
                                   "needs the full contiguous Nature3 sequence")
            fp = os.path.join(os.path.dirname(os.path.dirname(src)), "flow",
                              f"{_base(src)}_flow.npy")
            if not os.path.exists(fp):
                raise FileNotFoundError(fp)
            slots.append(fp)
        assert slots[READ_SLOT] == ff, (slots, ff)
        out.append((win, slots, ff, lf))
    return out


def _resize_window(imgs, flows, H, W):
    """Resize a list of images and a list of flows to (H,W); images returned as
    uint8, flow vectors scaled by the per-axis ratio (eval-consistent). All
    flows are assumed to share the images' native resolution."""
    oh, ow = flows[0].shape[:2]
    sx, sy = W / float(ow), H / float(oh)
    r = lambda im: np.asarray(Image.fromarray(im.astype(np.uint8)).resize((W, H),
                              Image.BILINEAR))                       # uint8 HxWx3
    def rf(flow):
        f = np.asarray(Image.fromarray(flow[..., 0]).resize((W, H), Image.BILINEAR))
        g = np.asarray(Image.fromarray(flow[..., 1]).resize((W, H), Image.BILINEAR))
        return np.stack([f * sx, g * sy], axis=-1).astype(np.float32)
    return [r(im) for im in imgs], [rf(f) for f in flows]


class Nature3MOFDataset(Dataset):
    """(images 5x3xHxW, flows 6x2xHxW, valids 6xHxW) from reversed Nature3 windows
    [t+2, t+1, t, t-1, t-2]; forward slots = [<t+1>_flow, <t>_flow, <t-1>_flow]
    (all backward-flow files = W(j+1) -> W(j+2)), edge-clamped slots and all backward
    slots masked invalid (see _list_temporal_samples)."""

    def __init__(self, args, split_file):
        self.samples = _list_temporal_samples(os.path.join(args.datasets_path, args.root), split_file)
        self.H, self.W = args.eval_size          # resize frames to eval scale first
        self.aug = FlowAugmentor(crop_size=list(args.image_size),   # then random-crop
                                 min_scale=args.min_scale, max_scale=args.max_scale,
                                 do_flip=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        win, slots, ff, _ = self.samples[idx]
        raw_imgs = [np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for p in win]
        ref = np.load(ff).astype(np.float32)                      # slot READ_SLOT, always real
        fwd = [ref if j == READ_SLOT else
               (np.load(p).astype(np.float32) if p is not None else np.full_like(ref, INVALID_SENTINEL))
               for j, p in enumerate(slots)]
        imgs, fwd = _resize_window(raw_imgs, fwd, self.H, self.W)
        # invalid slots: resizing an INVALID_SENTINEL-filled array by bilinear
        # interpolation preserves the (huge, constant) value, so it still masks correctly.

        bwd = [np.full((self.H, self.W, 2), INVALID_SENTINEL, np.float32) for _ in range(N_FRAMES - 2)]
        flows = fwd + bwd                                          # forward 0..2, backward 0..2
        imgs, flows = self.aug(imgs, flows)

        imgs = [torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() for im in imgs]
        flows = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() for f in flows]
        valids = [((f[0].abs() < 1000) & (f[1].abs() < 1000)).float() for f in flows]
        return torch.stack(imgs), torch.stack(flows), torch.stack(valids)


# Inference config of the canonical evaluator (analyze_drone_epe._mof_cfg ->
# configs/multiframes_sintel_submission, AMP on for CUDA). Only these keys differ from
# the training cfg in a way that matters at inference; applied during validation only.
_EVAL_NET_CFG = _canon._mof_cfg("", [432, 960], N_FRAMES).MOFNetStack
_EVAL_KEYS = ("decoder_depth", "mixed_precision")
_IGNORED_KEYS = _EVAL_KEYS + ("pretrain", "critical_params", "corr_radius")


def _check_net_cfg(net_cfg):
    diff = [k for k in _EVAL_NET_CFG if k not in _IGNORED_KEYS and k in net_cfg
            and net_cfg[k] != _EVAL_NET_CFG[k]]
    if diff:
        raise RuntimeError(f"training MOFNetStack cfg differs from evaluator cfg in {diff}")


_VAL_DS = {}


def _val_dataset(args):
    """Canonical evaluator dataset (reversed MOF window), built once."""
    key = (args.datasets_path, args.root, args.val_split_file, tuple(args.eval_size))
    if key not in _VAL_DS:
        ds = _canon.DroneFlowDataset(os.path.join(args.datasets_path, args.root),
                                     resize_shape=list(args.eval_size),
                                     offsets=_canon.window_offsets("MOF", N_FRAMES))
        ds.samples = filter_samples(ds.samples, load_split_frames(args.val_split_file))
        _VAL_DS[key] = ds
    return _VAL_DS[key]


@torch.no_grad()
def validate_nature3(model, args):
    """analyze_drone_epe.py --mode MOF per-pair path on the val split: reversed window,
    _predict (forward slot 1 = t -> t-1), compute_metrics; evaluator's decoder_depth/AMP."""
    was = model.training
    net = model.module if isinstance(model, nn.DataParallel) else model
    saved = {k: net.cfg[k] for k in _EVAL_KEYS}
    for k in _EVAL_KEYS:
        net.cfg[k] = _EVAL_NET_CFG[k]
    model.eval()
    try:
        ds = _val_dataset(args)
        rows = []
        for i in range(len(ds)):
            frames, flow_gt, mask = ds[i]
            pred = _canon._predict(model, frames[None].cuda(), "MOF")     # (1,2,H,W) t -> t-1
            if pred.shape[-2:] != flow_gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred, size=flow_gt.shape[-2:],
                                                       mode="bilinear", align_corners=False)
            rows.append(fdc.compute_metrics(pred[0], flow_gt, mask))
    finally:
        for k, v in saved.items():
            net.cfg[k] = v
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


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + worker_id); random.seed(seed + worker_id)


def train(args):
    cfg = make_cfg(args)
    model = nn.DataParallel(build_network(cfg))
    print("Parameter Count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))

    latest = os.path.join(args.ckpt_dir, f"{args.name}_latest.pth")
    start_step, resume_state = 0, None
    if os.path.exists(latest):
        resume_state = torch.load(latest, map_location="cpu")
        model.load_state_dict(resume_state["model"]); start_step = resume_state["step"]
        print(f"resumed from {latest} at step {start_step}")
    elif args.restore_ckpt is not None:
        fdc.load_state_dict_checked(model.module, torch.load(args.restore_ckpt, map_location="cpu"),
                                    name=f"VideoFlow-MOF:{os.path.basename(args.restore_ckpt)}")
        print(f"initialized from {args.restore_ckpt}")
    else:
        print("WARNING: no --restore_ckpt and no resume state -> training from RANDOM init "
              "(paper recipe: --restore_ckpt .../MOF_sintel.pth)", flush=True)
    _check_net_cfg(model.module.cfg)
    model.cuda(); model.train()

    train_ds = Nature3MOFDataset(args, args.train_split_file)
    print(f"train samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=_worker_init,
                              persistent_workers=args.num_workers > 0)

    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])

    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, "runs", args.name))
    running, total_steps, micro = {}, start_step, 0
    best_val = float("inf")
    optimizer.zero_grad()

    done = False
    while not done:                     # cycle the finite loader until num_steps
        for images, flows, valids in train_loader:
            images = images.cuda(non_blocking=True)
            flows, valids = flows.cuda(), valids.cuda()
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                images = (images + stdv * torch.randn_like(images)).clamp(0.0, 255.0)

            flow_predictions = model(images, {})
            loss, metrics, nan_flag = sequence_loss(flow_predictions, flows, valids, cfg)
            if nan_flag:
                print(f"[step {total_steps}] NaN in loss terms", flush=True)
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
                msg = " ".join(f"{k}={running[k] / SUM_FREQ:.4f}" for k in sorted(running))
                print(f"[{total_steps:6d}, lr={lr:.7f}] {msg}", flush=True)
                for k in running:
                    writer.add_scalar(k, running[k] / SUM_FREQ, total_steps)
                running = {}

            if total_steps % args.save_every == 0:
                save_ckpt(latest, model, optimizer, scheduler, scaler, total_steps)

            if total_steps % args.val_every == 0:
                results = validate_nature3(model, args)
                print(f"[{total_steps:6d}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in results.items()), flush=True)
                for k, v in results.items():
                    writer.add_scalar(k, v, total_steps)
                if results["nat3_val_epe_drone"] >= 0 and results["nat3_val_epe_drone"] < best_val:
                    best_val = results["nat3_val_epe_drone"]
                    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{args.name}_best.pth"))
                    print(f"  new best val_epe_drone={best_val:.4f} -> {args.name}_best.pth", flush=True)

            if total_steps >= args.num_steps:
                done = True; break

    final = os.path.join(args.ckpt_dir, f"{args.name}.pth")
    torch.save(model.state_dict(), final)
    save_ckpt(latest, model, optimizer, scheduler, scaler, total_steps)
    writer.close()
    print(f"done at step {total_steps}, final: {final}, best val_epe_drone={best_val:.4f}")
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="videoflow-mof-nat3")
    p.add_argument("--restore_ckpt", default=None)
    p.add_argument("--datasets_path", default=os.environ.get("HOME", ""))
    p.add_argument("--root", default="Nature3")
    p.add_argument("--train_split_file", required=True)
    p.add_argument("--val_split_file", required=True)
    p.add_argument("--ckpt_dir", default="checkpoints_nat3")
    p.add_argument("--lr", type=float, default=6.25e-5)
    p.add_argument("--num_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--image_size", type=int, nargs=2, default=[384, 768],
                   help="training crop size (<= eval_size)")
    p.add_argument("--eval_size", type=int, nargs=2, default=[432, 960],
                   help="frames resized to this before cropping; validation full-frame size")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--wdecay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--add_noise", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min_scale", type=float, default=0.0)
    p.add_argument("--max_scale", type=float, default=0.4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--val_every", type=int, default=250)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
