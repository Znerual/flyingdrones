"""
Shared FlyingDrones (Nature3) evaluation helpers.

Copy this file next to every evaluator / trainer (like ``drone_split.py``). It has a
unique module name on purpose: container images bake an old ``drone_split.py`` next to
each script, which would shadow a shared module with a generic name.

Ground-truth convention (verified 2026-10-07, photometric check + two models):
    ``flow/<t>_flow.npy`` is the BACKWARD flow of frame t: for every pixel x of
    frame t, ``x + F[x]`` is its position in frame t-1, i.e.
    ``I_t(x) ~= I_{t-1}(x + F[x])``.
A forward-flow estimator must therefore be run on the pair ``(I_t, I_{t-1})`` with
``I_t`` as the reference frame. Temporal models get a time-reversed window, e.g.
VideoFlow MOF ``[t+2, t+1, t, t-1, t-2]``.

Contents
--------
list_samples(scene_root, offsets, ...)
    Path tuples ``(*frame_paths, flow_path, label_path)``; frame_paths follow
    ``offsets`` relative to the reference frame t. Frame 0000 is never a reference
    (no t-1). Offsets that fall outside the sequence are clamped to the nearest
    existing frame (edge replication); only frame 0001's t-2 is affected in the
    published split.
compute_metrics(pred, gt, drone_mask)
    The paper's metric definitions (Section "Metrics"), numpy or torch input.
PairDumper
    Writes one CSV row per evaluated pair (for bootstrap confidence intervals).
aggregate(rows)
    Frame-weighted means as reported in the paper tables.

Offsets for the models in the benchmark:
    two-frame models          (0, -1)
    VideoFlow BOF             (+1, 0, -1)        read forward slot 0  (t -> t-1)
    VideoFlow MOF             (+2, +1, 0, -1, -2) read forward slot 1 (t -> t-1)
    ARFlow three-frame        (+1, 0, -1)        read flows_fw[0]      (t -> t-1)
    MemFlow reversed clip     (+3, +2, +1, 0, -1) step through, keep last flow
"""

import csv
import glob
import math
import os
import os.path as osp

import numpy as np

TWO_FRAME = (0, -1)
BOF_WINDOW = (1, 0, -1)
MOF_WINDOW = (2, 1, 0, -1, -2)
ARFLOW_WINDOW = (1, 0, -1)
MEMFLOW_CLIP = (3, 2, 1, 0, -1)

METRIC_NAMES = ("epe", "epe_norm", "px1_err", "px3_err", "px5_err", "fl")
REGIONS = ("all", "drone", "bg")


# --------------------------------------------------------------------------- samples

def _scene_dirs(scene_root):
    """Accept either a flat scene dir (images/flow/labels) or a parent of scenes."""
    if osp.isdir(osp.join(scene_root, "images")) and osp.isdir(osp.join(scene_root, "flow")):
        return [scene_root]
    out = []
    for d in sorted(os.listdir(scene_root)):
        p = osp.join(scene_root, d)
        if osp.isdir(osp.join(p, "images")) and osp.isdir(osp.join(p, "flow")):
            out.append(p)
    if not out:
        raise FileNotFoundError(f"no images/ + flow/ under {scene_root}")
    return out


def list_samples(scene_root, offsets=TWO_FRAME, require_labels=True):
    """Return ``[(*frame_paths, flow_path, label_path), ...]``.

    ``offsets`` are frame offsets relative to the reference frame t; offset 0 must be
    present. Reference frames are every frame t >= 1 that has ``<t>_flow.npy`` (and a
    label when ``require_labels``). Frames are addressed by sorted position, which
    equals the frame number for the contiguous Nature3 sequence.
    """
    if 0 not in offsets:
        raise ValueError("offsets must contain 0 (the reference frame)")
    samples = []
    for scene in _scene_dirs(scene_root):
        imgs = sorted(glob.glob(osp.join(scene, "images", "*.png")))
        n = len(imgs)
        for i in range(1, n):  # frame 0 has no t-1 -> never a reference
            base = osp.splitext(osp.basename(imgs[i]))[0]
            ff = osp.join(scene, "flow", f"{base}_flow.npy")
            lf = osp.join(scene, "labels", f"{base}_label.png")
            if not osp.exists(ff):
                continue
            if require_labels and not osp.exists(lf):
                continue
            frames = [imgs[min(max(i + o, 0), n - 1)] for o in offsets]
            samples.append((*frames, ff, lf if osp.exists(lf) else None))
    return samples


# --------------------------------------------------------------------------- metrics

def _to_numpy_hw2(x):
    """Accept (H,W,2), (2,H,W), (1,2,H,W) numpy arrays or torch tensors."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    while x.ndim > 3:
        x = x[0]
    if x.shape[0] == 2 and x.shape[-1] != 2:
        x = np.transpose(x, (1, 2, 0))
    return x


def _to_numpy_mask(m, shape):
    if m is None:
        return np.zeros(shape, dtype=bool)
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    return m > 0.5 if m.dtype.kind == "f" else m > 0


def compute_metrics(pred, gt, drone_mask):
    """Paper metrics for one pair, keys ``<metric>_<region>``.

    * epe       mean ||F_pred - F_gt||_2 (px)
    * epe_norm  mean ||(du/W, dv/H)||_2 x 100 (u scaled by width, v by height)
    * pxT_err   % of pixels with EPE > T, T in {1, 3, 5}
    * fl        % of pixels with EPE > 3 and EPE / ||F_gt|| > 0.05
    Regions: all pixels, drone pixels (label > 0), background. A region with no
    pixels yields NaN for that pair and is skipped by ``aggregate``.
    """
    p = _to_numpy_hw2(pred)
    g = _to_numpy_hw2(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} vs gt {g.shape}")
    H, W = g.shape[:2]
    d = p - g
    epe = np.sqrt((d ** 2).sum(-1))
    epe_n = np.sqrt((d[..., 0] / W) ** 2 + (d[..., 1] / H) ** 2)
    gmag = np.sqrt((g ** 2).sum(-1))
    dm = _to_numpy_mask(drone_mask, (H, W))
    masks = {"all": np.ones((H, W), bool), "drone": dm, "bg": ~dm}
    out = {}
    for r, m in masks.items():
        if not m.any():
            for k in METRIC_NAMES:
                out[f"{k}_{r}"] = float("nan")
            continue
        e = epe[m]
        out[f"epe_{r}"] = float(e.mean())
        out[f"epe_norm_{r}"] = float(100.0 * epe_n[m].mean())
        out[f"px1_err_{r}"] = float(100.0 * (e > 1.0).mean())
        out[f"px3_err_{r}"] = float(100.0 * (e > 3.0).mean())
        out[f"px5_err_{r}"] = float(100.0 * (e > 5.0).mean())
        out[f"fl_{r}"] = float(100.0 * ((e > 3.0) & (e / np.maximum(gmag[m], 1e-6) > 0.05)).mean())
    out["n_drone_px"] = int(dm.sum())
    return out


# --------------------------------------------------------------------------- dumping

def ref_base(sample):
    for el in sample:
        if isinstance(el, str) and el.endswith("_flow.npy"):
            return osp.basename(el)[: -len("_flow.npy")]
    raise ValueError("sample has no *_flow.npy element")


class PairDumper:
    """Append one row per evaluated pair to ``<path>`` (CSV, header on creation)."""

    FIELDS = ["ref"] + [f"{k}_{r}" for r in REGIONS for k in METRIC_NAMES] + ["n_drone_px"]

    def __init__(self, path):
        self.path = path
        self.rows = []
        if path:
            os.makedirs(osp.dirname(osp.abspath(path)) or ".", exist_ok=True)
            self._fh = open(path, "w", newline="")
            self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS, extrasaction="ignore")
            self._w.writeheader()
        else:
            self._fh = None

    def add(self, ref, metrics):
        row = {"ref": ref, **metrics}
        self.rows.append(row)
        if self._fh:
            self._w.writerow(row)
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def aggregate(rows):
    """Frame-weighted mean of every metric, skipping NaN (pairs without drones)."""
    out = {}
    for r in REGIONS:
        for k in METRIC_NAMES:
            key = f"{k}_{r}"
            vals = [row[key] for row in rows if not math.isnan(row[key])]
            out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def summary_row(rows, checkpoint, dataset, image_size, skipped=0):
    """One summary row in the legacy results.csv column layout."""
    a = aggregate(rows)
    row = {"checkpoint": checkpoint, "dataset": dataset, "image_size": image_size}
    for k in METRIC_NAMES:
        row[k] = a[f"{k}_all"]
        row[f"{k}_drone"] = a[f"{k}_drone"]
    row["samples_processed"] = len(rows)
    row["samples_skipped"] = skipped
    return row


SUMMARY_FIELDS = ["checkpoint", "dataset", "image_size",
                  "epe", "epe_drone", "epe_norm", "epe_norm_drone",
                  "px1_err", "px1_err_drone", "px3_err", "px3_err_drone",
                  "px5_err", "px5_err_drone", "fl", "fl_drone",
                  "samples_processed", "samples_skipped"]


def write_summary(path, row):
    """Append ``row`` to ``path`` (header written if the file is new)."""
    new = not osp.exists(path)
    os.makedirs(osp.dirname(osp.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean over pairs (NaN entries dropped)."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def load_state_dict_checked(model, state_dict, strip_prefixes=("module.",), allow_missing=(), name="model"):
    """Load weights and fail loudly when keys do not match (the RAFT-bug guard).

    ``allow_missing`` lists key prefixes that may legitimately stay at init
    (e.g. a frozen backbone loaded elsewhere).
    """
    sd = dict(state_dict)
    for pre in strip_prefixes:
        if sd and all(k.startswith(pre) for k in sd):
            sd = {k[len(pre):]: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not any(k.startswith(a) for a in allow_missing)]
    unexpected = list(res.unexpected_keys)
    n_model = len(model.state_dict())
    print(f"[{name}] loaded {n_model - len(res.missing_keys)}/{n_model} keys; "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(f"[{name}] weight/key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return res
