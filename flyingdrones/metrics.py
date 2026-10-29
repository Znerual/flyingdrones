"""Paper metrics (Section "Metrics"), aggregation and bootstrap confidence intervals.

Per pair, for each region R in {all, drone, bg} (drone = ``label > 0``):

* ``epe``       mean ||F_pred - F_gt||_2 over R (pixels)
* ``epe_norm``  mean ||((u_p - u_g) / W, (v_p - v_g) / H)||_2 over R, x 100
                (resolution-normalised EPE; u scaled by the width, v by the height)
* ``pxT_err``   % of pixels in R with EPE > T, T in {1, 3, 5}
* ``fl``        % of pixels in R with EPE > 3 and EPE / ||F_gt|| > 0.05

A region with no pixels (drone-free frame) yields NaN for that pair; ``aggregate``
skips NaNs, so drone-region numbers are means over the pairs that contain drones.
Benchmark numbers are frame-weighted means over pairs (not pixel-weighted).

This module is numerically identical to ``evaluators/_common/fd_eval_common.py``,
the single-file copy that sits next to every evaluator (checked by
``tests/test_metrics.py``).
"""

import math

import numpy as np

METRIC_NAMES = ("epe", "epe_norm", "px1_err", "px3_err", "px5_err", "fl")
REGIONS = ("all", "drone", "bg")


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
    """Paper metrics for one pair; returns ``{"<metric>_<region>": float, "n_drone_px": int}``.

    ``pred`` and ``gt`` are flow fields at the same resolution, (H,W,2) or (2,H,W)
    (numpy or torch); ``drone_mask`` is a boolean/0-1 (H,W) mask or a label map
    (``> 0`` = drone).
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


def aggregate(rows):
    """Frame-weighted mean of every ``<metric>_<region>`` key, skipping NaN."""
    out = {}
    for r in REGIONS:
        for k in METRIC_NAMES:
            key = f"{k}_{r}"
            vals = [row[key] for row in rows if not math.isnan(row[key])]
            out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean over pairs (NaN entries dropped).

    Returns ``(mean, lo, hi)``. The paper uses 10,000 resamples, seed 0, alpha 0.05.
    """
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
