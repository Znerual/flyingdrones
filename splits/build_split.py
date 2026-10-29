"""
Build a random, stratified evaluation subset of the FlyingDrones (Nature3) dataset.

The split is keyed by REFERENCE FRAME INDEX t: the features of entry t are computed
from flow/<t>_flow.npy and labels/<t>_label.png, i.e. from the pair (I_t, I_{t-1})
that reference frame t denotes under the backward-flow convention (<t>_flow.npy maps
frame t to frame t-1). The split is therefore identical across all three resolution
variants (Small 480x270, Standard 960x432, Large 1920x1080) and every model.
Candidates were frames 0000..8999; frame 0000 has no predecessor and is dropped at
evaluation time (999 evaluated pairs).

Two-phase:
  phase 1  extract per-pair features (drone footprint size + drone-region motion)
           from the native 1080p labels/flow, cached to features CSV.
  phase 2  bin into strata (Table-1 aligned) and stratified-sample to N pairs
           with proportional-with-floor allocation, fixed seed.

Outputs (into --out_dir):
  nature3_features.csv       per-pair features (cache; reused if present)
  nature3_split.json         full manifest: seed, strata defs, allocation, selection
  nature3_split_frames.txt   selected reference-frame basenames, one per line (e.g. 0100)
  nature3_split_indices.txt  selected pair indices, one per line

Usage:
  # phase 1 (parallel read of flow+labels; ~10 min with -j 8)
  python build_split.py features --root <NATURE3_ROOT> --out_dir . -j 8
  # phase 2 (instant; uses cached features)
  python build_split.py sample   --out_dir . -n 1000 --seed 42
  # both at once
  python build_split.py all       --root <NATURE3_ROOT> --out_dir . -n 1000 --seed 42 -j 8
"""

import os
import os.path as osp
import sys
import csv
import json
import argparse
from glob import glob
from multiprocessing import Pool

import numpy as np
import cv2


# --------------------------------------------------------------------------
# Stratum definitions (aligned to paper Table 1 at native 1920x1080)
# --------------------------------------------------------------------------

# Drone footprint size (px area, per-frame total).
# NOTE: paper Table 1 percentiles (47/155/549/2367) are PER-INSTANCE over 19999
# instances; a "pair" here is a frame whose total drone footprint sums all drones
# present, so the per-frame distribution sits ~3x higher. Edges are therefore
# derived DATA-DRIVEN as quintiles of the with-drone population (see set_size_bins),
# giving five balanced size strata. Placeholder edges below are replaced at runtime.
SIZE_BINS = [
    ("s1_tiny",   1,     67),
    ("s2_small",  68,    319),
    ("s3_medium", 320,   947),
    ("s4_large",  948,   2294),
    ("s5_huge",   2295,  np.inf),
]

# Drone-region mean flow magnitude (px/frame). Fixed, physically meaningful edges:
# sub-pixel motion is the paper's key regime and is well populated (461 frames).
MOTION_BINS = [
    ("m1_subpx", 0.0,  1.0),    # sub-pixel drone motion
    ("m2_low",   1.0,  3.0),
    ("m3_mid",   3.0,  8.0),
    ("m4_high",  8.0,  np.inf),
]

EMPTY_STRATUM = "empty"  # reference frame has no drone pixels


def set_size_bins(drone_px_values):
    """Recompute SIZE_BINS as quintiles of the with-drone footprint distribution."""
    global SIZE_BINS
    d = np.asarray([v for v in drone_px_values if v > 0])
    e = [int(np.percentile(d, q)) for q in (20, 40, 60, 80)]
    names = ["s1_tiny", "s2_small", "s3_medium", "s4_large", "s5_huge"]
    lo = [1, e[0] + 1, e[1] + 1, e[2] + 1, e[3] + 1]
    hi = [e[0], e[1], e[2], e[3], np.inf]
    SIZE_BINS = list(zip(names, lo, hi))
    return SIZE_BINS


def size_bin(px):
    for name, lo, hi in SIZE_BINS:
        if lo <= px <= hi:
            return name
    return SIZE_BINS[-1][0]


def motion_bin(mag):
    for name, lo, hi in MOTION_BINS:
        if lo <= mag < hi:
            return name
    return MOTION_BINS[-1][0]


def stratum_of(drone_px, flow_mag_mean):
    if drone_px <= 0:
        return EMPTY_STRATUM
    return f"{size_bin(drone_px)}|{motion_bin(flow_mag_mean)}"


# --------------------------------------------------------------------------
# Phase 1: feature extraction
# --------------------------------------------------------------------------

def _feat_one(arg):
    """Compute features for pair i (reference frame i)."""
    root, base = arg
    lbl_p = osp.join(root, "labels", f"{base}_label.png")
    flow_p = osp.join(root, "flow", f"{base}_flow.npy")

    lbl = cv2.imread(lbl_p, cv2.IMREAD_GRAYSCALE)
    mask = lbl > 0
    drone_px = int(mask.sum())
    n_instances = int(len(np.unique(lbl[mask]))) if drone_px else 0

    flow = np.load(flow_p)                       # (H, W, 2)
    mag = np.sqrt((flow.astype(np.float32) ** 2).sum(axis=-1))
    if drone_px:
        dm = mag[mask]
        fm_mean = float(dm.mean())
        fm_max = float(dm.max())
    else:
        fm_mean = float(mag.mean())              # full-frame fallback for empty
        fm_max = float(mag.max())

    return {
        "idx": int(base),
        "ref_frame": base,
        "drone_px": drone_px,
        "n_instances": n_instances,
        "flow_mag_mean": round(fm_mean, 4),
        "flow_mag_max": round(fm_max, 4),
    }


def extract_features(root, out_dir, jobs):
    imgs = sorted(glob(osp.join(root, "images", "*.png")))
    if not imgs:
        sys.exit(f"ERROR: no images under {root}/images")
    # pairs = frames 0 .. N-2 (reference frame needs a successor)
    bases = [osp.splitext(osp.basename(p))[0] for p in imgs][:-1]
    print(f"phase 1: {len(bases)} pairs from {root}  (-j {jobs})")

    args = [(root, b) for b in bases]
    feats = []
    with Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_feat_one, args, chunksize=16)):
            feats.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(bases)}")
    feats.sort(key=lambda d: d["idx"])

    csv_p = osp.join(out_dir, "nature3_features.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(feats[0].keys()))
        w.writeheader()
        w.writerows(feats)
    print(f"phase 1 done -> {csv_p}")
    return feats


def load_features(out_dir):
    csv_p = osp.join(out_dir, "nature3_features.csv")
    if not osp.exists(csv_p):
        sys.exit(f"ERROR: {csv_p} not found -- run 'features' first")
    feats = []
    with open(csv_p) as f:
        for row in csv.DictReader(f):
            feats.append({
                "idx": int(row["idx"]),
                "ref_frame": row["ref_frame"],
                "drone_px": int(row["drone_px"]),
                "n_instances": int(row["n_instances"]),
                "flow_mag_mean": float(row["flow_mag_mean"]),
                "flow_mag_max": float(row["flow_mag_max"]),
            })
    return feats


# --------------------------------------------------------------------------
# Phase 2: stratified sampling (proportional with floor)
# --------------------------------------------------------------------------

def allocate(pop, n_target, floor):
    """Largest-remainder proportional allocation with a per-occupied-stratum floor."""
    strata = sorted(pop)
    sizes = {s: len(pop[s]) for s in strata}
    total = sum(sizes.values())

    # floor first (never exceed the stratum population)
    alloc = {s: min(floor, sizes[s]) for s in strata}
    remaining = n_target - sum(alloc.values())

    if remaining > 0:
        # distribute the rest proportionally over remaining capacity
        cap = {s: sizes[s] - alloc[s] for s in strata}
        cap_total = sum(cap.values())
        if cap_total > 0:
            raw = {s: remaining * cap[s] / cap_total for s in strata}
            base = {s: int(np.floor(raw[s])) for s in strata}
            for s in strata:
                add = min(base[s], cap[s])
                alloc[s] += add
            left = remaining - sum(min(base[s], cap[s]) for s in strata)
            # largest remainder for the leftover units
            order = sorted(strata, key=lambda s: raw[s] - np.floor(raw[s]), reverse=True)
            for s in order:
                if left <= 0:
                    break
                if alloc[s] < sizes[s]:
                    alloc[s] += 1
                    left -= 1
    return alloc, sizes


def sample_split(feats, out_dir, n_target, seed, floor):
    rng = np.random.default_rng(seed)

    # derive data-driven size quintile edges from the with-drone population
    set_size_bins([d["drone_px"] for d in feats])

    # group indices by stratum
    pop = {}
    for d in feats:
        s = stratum_of(d["drone_px"], d["flow_mag_mean"])
        pop.setdefault(s, []).append(d["idx"])
    for s in pop:
        pop[s] = sorted(pop[s])

    alloc, sizes = allocate(pop, n_target, floor)

    selected = []
    for s in sorted(pop):
        idxs = np.array(pop[s])
        k = alloc[s]
        pick = rng.choice(idxs, size=k, replace=False) if k > 0 else np.array([], int)
        selected.extend(int(i) for i in pick)
    selected.sort()

    by_idx = {d["idx"]: d for d in feats}
    sel_rows = [by_idx[i] for i in selected]

    # -------- manifest --------
    strata_meta = {
        s: {
            "population": sizes[s],
            "allocated": alloc[s],
            "pct_of_pop": round(100 * sizes[s] / len(feats), 2),
            "pct_of_split": round(100 * alloc[s] / len(selected), 2) if selected else 0,
        }
        for s in sorted(pop)
    }
    manifest = {
        "dataset": "FlyingDrones / Nature3",
        "native_resolution": [1080, 1920],
        "total_pairs": len(feats),
        "seed": seed,
        "n_target": n_target,
        "n_selected": len(selected),
        "floor_per_stratum": floor,
        "allocation": "proportional-with-floor (largest remainder)",
        "size_bins_px": [[n, lo, (None if np.isinf(hi) else hi)] for n, lo, hi in SIZE_BINS],
        "motion_bins_pxframe": [[n, lo, (None if np.isinf(hi) else hi)] for n, lo, hi in MOTION_BINS],
        "strata": strata_meta,
        "selection": [
            {
                "idx": d["idx"],
                "ref_frame": d["ref_frame"],
                "drone_px": d["drone_px"],
                "n_instances": d["n_instances"],
                "flow_mag_mean": d["flow_mag_mean"],
                "flow_mag_max": d["flow_mag_max"],
                "stratum": stratum_of(d["drone_px"], d["flow_mag_mean"]),
            }
            for d in sel_rows
        ],
    }

    json_p = osp.join(out_dir, "nature3_split.json")
    with open(json_p, "w") as f:
        json.dump(manifest, f, indent=2)

    frames_p = osp.join(out_dir, "nature3_split_frames.txt")
    with open(frames_p, "w") as f:
        f.write("\n".join(d["ref_frame"] for d in sel_rows) + "\n")

    idx_p = osp.join(out_dir, "nature3_split_indices.txt")
    with open(idx_p, "w") as f:
        f.write("\n".join(str(d["idx"]) for d in sel_rows) + "\n")

    # -------- report --------
    print(f"\nphase 2 done: {len(selected)} / {n_target} pairs selected  (seed {seed})")
    print(f"{'stratum':<16} {'pop':>6} {'pop%':>6} {'alloc':>6} {'split%':>7}")
    print("-" * 46)
    for s in sorted(strata_meta):
        m = strata_meta[s]
        print(f"{s:<16} {m['population']:>6} {m['pct_of_pop']:>6} "
              f"{m['allocated']:>6} {m['pct_of_split']:>7}")
    print("-" * 46)
    print(f"outputs -> {json_p}\n           {frames_p}\n           {idx_p}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["features", "sample", "all"])
    ap.add_argument("--root", default=os.environ.get("FD_DATA") and osp.join(os.environ["FD_DATA"], "Nature3"),
                    help="Nature3 scene dir (images/ flow/ labels/); default $FD_DATA/Nature3")
    ap.add_argument("--out_dir", default=osp.dirname(osp.abspath(__file__)))
    ap.add_argument("-n", "--n_target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--floor", type=int, default=15,
                    help="minimum samples per occupied stratum")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.phase in ("features", "all"):
        feats = extract_features(args.root, args.out_dir, args.jobs)
    if args.phase in ("sample", "all"):
        feats = feats if args.phase == "all" else load_features(args.out_dir)
        sample_split(feats, args.out_dir, args.n_target, args.seed, args.floor)


if __name__ == "__main__":
    main()
