#!/usr/bin/env python3
"""Flow-alignment metrics for the real drone-tracking eval (no GT flow field).

We only have a point label of the drone centre per frame. The cameras are
(near-)static, so a good optical-flow model should produce a localised motion
blob on the drone. We score, per predicted flow field, how well the motion
signal aligns with the drone point, treating |flow| as a saliency map.

Metrics per (model, pair)
-------------------------
Localisation / saliency (need only the point label):
  nss        : z-score of |flow| at the drone pixel = (m(p)-mean)/std.
               How much the drone stands out as a motion outlier. >0 good,
               robust to a global (ego-motion) offset since it is relative.
  nss_disk   : same but averaged over a small disk around p (label-noise tolerant).
  ple_px     : distance from the (smoothed) |flow| argmax to p, in pixels.
  ple_norm   : ple_px / image diagonal.
  hit@r      : 1 if ple_px <= r  (r in HIT_RADII).
  contrast   : mean|flow| in a disk around p  /  mean|flow| in the background
               (outside a larger exclusion disk). >1 = motion concentrated on drone.

Motion-fidelity (uses the label displacement as a pseudo-GT drone motion vector,
valid because the camera is static — label_nxt - label_ref = drone image motion):
  pt_epe     : ||flow_pred(p) - gt_disp||, flow sampled on a small disk (median).
  pt_cos     : cosine similarity between predicted and gt displacement direction.
  gt_disp_mag: |label_nxt - label_ref| (px) — for stratifying by drone speed.

Aggregation: mean nss/nss_disk/contrast/pt_cos, median ple_px/ple_norm/pt_epe,
mean hit@r, over all pairs where the model produced a flow file.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HIT_RADII = (10, 25, 50, 100)
# drone image-motion strata (px/frame), mirroring prep_subset / build_split
MOTION_BINS = [("m1_subpx", 0.0, 1.0), ("m2_low", 1.0, 3.0),
               ("m3_mid", 3.0, 8.0), ("m4_high", 8.0, float("inf"))]


def motion_bin(mag):
    for name, lo, hi in MOTION_BINS:
        if lo <= mag < hi:
            return name
    return MOTION_BINS[-1][0]
SMOOTH_SIGMA = 3.0      # px, Gaussian smoothing of |flow| before argmax
BORDER_MARGIN = 20      # px, ignore this edge ring for peak-finding — optical-flow
                        # fields have unreliable border/corner values that otherwise
                        # dominate the global argmax (diagnosed: spurious peaks sit
                        # exactly at dist=0 with ~0.2px magnitude).
DISK_EVAL = 8           # px, radius for nss_disk / point sampling
DISK_FG = 15            # px, foreground disk for contrast
DISK_BG_EXCL = 40       # px, exclude this radius around drone from background


def parse_npy_name(name: str):
    """<model>__<safeckpt>[__<mode>]__<idx>.npy -> (system_key, idx)."""
    stem = name[:-4] if name.endswith(".npy") else name
    parts = stem.split("__")
    idx = int(parts[-1])
    system = "__".join(parts[:-1])   # model + ckpt (+ mode)
    return system, idx


def disk_mask(H, W, cx, cy, r):
    ys, xs = np.ogrid[:H, :W]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r


def metrics_for(flow, label_ref, label_nxt):
    """flow: (H,W,2) px displacement. labels: (x,y). Returns dict or None."""
    H, W = flow.shape[:2]
    x, y = label_ref
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < W and 0 <= cy < H):
        return None

    mag = np.sqrt((flow ** 2).sum(-1)).astype(np.float32)
    mag_s = cv2.GaussianBlur(mag, (0, 0), SMOOTH_SIGMA)

    mu, sd = float(mag.mean()), float(mag.std()) + 1e-6
    d_eval = disk_mask(H, W, cx, cy, DISK_EVAL)

    # localisation / saliency
    nss = (float(mag[cy, cx]) - mu) / sd
    nss_disk = (float(mag[d_eval].mean()) - mu) / sd
    # ignore the unreliable border ring before taking the global peak
    m = min(BORDER_MARGIN, (H - 1) // 2, (W - 1) // 2)
    mag_peak = mag_s.copy()
    if m > 0:
        edge = mag_s.min()
        mag_peak[:m, :] = edge; mag_peak[-m:, :] = edge
        mag_peak[:, :m] = edge; mag_peak[:, -m:] = edge
    py, px_ = np.unravel_index(int(np.argmax(mag_peak)), mag_peak.shape)
    ple_px = float(np.hypot(px_ - cx, py - cy))
    ple_norm = ple_px / float(np.hypot(H, W))

    fg = disk_mask(H, W, cx, cy, DISK_FG)
    bg = ~disk_mask(H, W, cx, cy, DISK_BG_EXCL)
    contrast = float(mag[fg].mean()) / (float(mag[bg].mean()) + 1e-6)

    # motion fidelity vs label-derived pseudo-GT displacement
    gt = np.array([label_nxt[0] - x, label_nxt[1] - y], dtype=np.float32)
    gt_mag = float(np.hypot(*gt))
    fu = float(np.median(flow[..., 0][d_eval]))
    fv = float(np.median(flow[..., 1][d_eval]))
    pred = np.array([fu, fv], dtype=np.float32)
    pt_epe = float(np.hypot(*(pred - gt)))
    denom = (np.linalg.norm(pred) * np.linalg.norm(gt)) + 1e-6
    pt_cos = float(np.dot(pred, gt) / denom)

    out = {"nss": nss, "nss_disk": nss_disk, "ple_px": ple_px,
           "ple_norm": ple_norm, "contrast": contrast, "pt_epe": pt_epe,
           "pt_cos": pt_cos, "gt_disp_mag": gt_mag}
    for r in HIT_RADII:
        out[f"hit@{r}"] = 1.0 if ple_px <= r else 0.0
    return out


def aggregate(rows):
    """rows: list of per-pair dicts -> one summary dict (mean/median as noted)."""
    if not rows:
        return {}
    med_keys = {"ple_px", "ple_norm", "pt_epe"}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    agg = {"n": len(rows)}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        agg[k] = float(np.median(vals)) if k in med_keys else float(np.mean(vals))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows-dir", required=True,
                    help="dir of <model>__<ckpt>__<idx>.npy flow files")
    ap.add_argument("--subset-dir", required=True,
                    help="prep output dir (selected.json, meta.json)")
    ap.add_argument("--out-prefix", default="results/dataset1_cam0")
    args = ap.parse_args()

    subset = Path(args.subset_dir)
    selected = json.loads((subset / "selected.json").read_text())
    labels = {s["idx"]: (s["label_ref"], s["label_nxt"]) for s in selected}

    flows_dir = Path(args.flows_dir)
    npys = sorted(flows_dir.glob("*.npy"))
    if not npys:
        raise SystemExit(f"no .npy in {flows_dir}")

    per_pair = []          # flat rows for CSV
    by_system = defaultdict(list)
    for f in npys:
        system, idx = parse_npy_name(f.name)
        if idx not in labels:
            continue
        flow = np.load(f)
        lr, ln = labels[idx]
        m = metrics_for(flow, lr, ln)
        if m is None:
            continue
        m["stratum"] = motion_bin(m["gt_disp_mag"])
        row = {"system": system, "idx": idx, **m}
        per_pair.append(row)
        by_system[system].append(m)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # per-pair CSV
    pp_csv = Path(f"{args.out_prefix}_per_pair.csv")
    if per_pair:
        with pp_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_pair[0]))
            w.writeheader()
            w.writerows(per_pair)

    # per-system summary CSV, ranked by nss_disk (primary)
    summ = [{"system": s, **aggregate(rows)} for s, rows in by_system.items()]
    summ.sort(key=lambda r: r.get("nss_disk", -1e9), reverse=True)
    sm_csv = Path(f"{args.out_prefix}_summary.csv")
    if summ:
        with sm_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summ[0]))
            w.writeheader()
            w.writerows(summ)

    # per-(system, stratum) summary — separates the hard sub-pixel regime
    by_ss = defaultdict(list)
    for r in per_pair:
        by_ss[(r["system"], r["stratum"])].append(r)
    strat_rows = [{"system": s, "stratum": st, **aggregate(rows)}
                  for (s, st), rows in by_ss.items()]
    strat_order = {b[0]: i for i, b in enumerate(MOTION_BINS)}
    strat_rows.sort(key=lambda r: (r["system"], strat_order.get(r["stratum"], 9)))
    if strat_rows:
        with Path(f"{args.out_prefix}_by_stratum.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(strat_rows[0]))
            w.writeheader()
            w.writerows(strat_rows)

    # console leaderboard
    print(f"\n{len(per_pair)} pairs scored across {len(by_system)} systems\n")
    hdr = f"{'system':42} {'n':>4} {'nss_d':>7} {'hit@50':>7} {'ple_px':>7} " \
          f"{'contr':>6} {'pt_cos':>7} {'pt_epe':>7}"
    print(hdr); print("-" * len(hdr))
    for r in summ:
        print(f"{r['system']:42.42} {r['n']:4d} {r.get('nss_disk',0):7.3f} "
              f"{r.get('hit@50',0):7.3f} {r.get('ple_px',0):7.1f} "
              f"{r.get('contrast',0):6.2f} {r.get('pt_cos',0):7.3f} "
              f"{r.get('pt_epe',0):7.3f}")
    print(f"\nwrote {sm_csv}\n      {pp_csv}")


if __name__ == "__main__":
    main()
