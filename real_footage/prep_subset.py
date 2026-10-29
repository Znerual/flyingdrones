#!/usr/bin/env python3
"""Prepare a real drone-tracking subset for the 16-model flow-alignment eval.

Source: drone-tracking-datasets (static tripod cameras filming a hexacopter,
per-frame point labels of the drone centre in detections/<cam>.txt as
`frame_id x y`, with `0 0` meaning "not visible / not labelled").

Because the cameras are (near-)static and the drone is the moving object, an
optical-flow field should light up on the drone. We select consecutive frame
pairs where BOTH the reference frame and its successor carry a valid label,
stratified by the drone's label displacement (default) or spread evenly across
the flight, extract only the needed frames, and lay them out as
images/<idx:04d>.png (idx = 1-based video frame number, as in the label file).
A pair is (images/<idx>.png, images/<idx+1>.png); the label at the reference
frame idx is the alignment target.

Output layout (self-contained):
    <out>/images/<idx:04d>.png     # only frames referenced by the selected pairs
    <out>/labels.json              # {"<idx>": [x, y], ...}  (all valid labels)
    <out>/selected.json            # [{"idx": i, "label_ref":[x,y], "label_nxt":[x,y]}]
    <out>/frames.txt               # reference-frame basenames, one per line ("0100")
    <out>/meta.json                # dataset, camera, native H/W, counts
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_labels(path: Path):
    """frame_id x y -> {int frame_id: (x, y)} keeping only nonzero (valid) rows."""
    labels = {}
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            fid = int(round(float(parts[0])))
            x, y = float(parts[1]), float(parts[2])
        except ValueError:
            continue  # header row ("frame no. x y") or comment
        if x == 0.0 and y == 0.0:
            continue
        labels[fid] = (x, y)
    return labels


# Drone image-motion strata (px/frame), mirroring build_split.py MOTION_BINS.
# The label displacement |label[i+1]-label[i]| is the drone's true image motion
# (the camera is static). Sub-pixel motion is the paper's hard regime.
MOTION_BINS = [
    ("m1_subpx", 0.0, 1.0),
    ("m2_low",   1.0, 3.0),
    ("m3_mid",   3.0, 8.0),
    ("m4_high",  8.0, np.inf),
]


def motion_bin(mag):
    for name, lo, hi in MOTION_BINS:
        if lo <= mag < hi:
            return name
    return MOTION_BINS[-1][0]


def _valid_pairs(labels, first, last):
    fids = sorted(labels)
    valid = [i for i in fids if (i + 1) in labels]
    if first is not None:
        valid = [i for i in valid if i >= first]
    if last is not None:
        valid = [i for i in valid if i <= last]
    if not valid:
        raise SystemExit("no valid consecutive labelled pairs found")
    return valid


def disp_mag(labels, i):
    (x0, y0), (x1, y1) = labels[i], labels[i + 1]
    return float(np.hypot(x1 - x0, y1 - y0))


def _allocate(pop, n_target, floor):
    """Proportional-with-floor (largest remainder), never exceeding population."""
    strata = sorted(pop)
    sizes = {s: len(pop[s]) for s in strata}
    alloc = {s: min(floor, sizes[s]) for s in strata}
    remaining = n_target - sum(alloc.values())
    if remaining > 0:
        cap = {s: sizes[s] - alloc[s] for s in strata}
        cap_total = sum(cap.values())
        if cap_total > 0:
            raw = {s: remaining * cap[s] / cap_total for s in strata}
            for s in strata:
                alloc[s] += min(int(np.floor(raw[s])), cap[s])
            left = n_target - sum(alloc.values())
            for s in sorted(strata, key=lambda s: raw[s] - np.floor(raw[s]), reverse=True):
                if left <= 0:
                    break
                if alloc[s] < sizes[s]:
                    alloc[s] += 1
                    left -= 1
    return alloc, sizes


def select_pairs(labels, n, first=None, last=None, strategy="stratified",
                 seed=42, floor=10):
    """Select n reference indices of valid consecutive pairs.

    strategy='stratified' (default): bin by drone displacement magnitude and
    sample proportional-with-floor so rare fast-motion pairs are represented.
    strategy='even': evenly spread across the flight timeline.
    Returns (selected_indices, info) where info has strata/allocation (or None).
    """
    valid = _valid_pairs(labels, first, last)
    if strategy == "even" or n >= len(valid):
        pos = sorted(set(np.linspace(0, len(valid) - 1, min(n, len(valid)))
                         .round().astype(int).tolist()))
        return [valid[p] for p in pos], None

    rng = np.random.default_rng(seed)
    pop = {}
    for i in valid:
        pop.setdefault(motion_bin(disp_mag(labels, i)), []).append(i)
    alloc, sizes = _allocate(pop, n, floor)
    selected = []
    for s in sorted(pop):
        idxs = np.array(sorted(pop[s]))
        k = alloc[s]
        pick = rng.choice(idxs, size=k, replace=False) if k > 0 else np.array([], int)
        selected.extend(int(i) for i in pick)
    selected.sort()
    info = {"strategy": "stratified", "seed": seed, "floor": floor,
            "motion_bins_pxframe": [[n_, lo, (None if np.isinf(hi) else hi)]
                                    for n_, lo, hi in MOTION_BINS],
            "strata": {s: {"population": sizes[s], "allocated": alloc[s]}
                       for s in sorted(pop)}}
    return selected, info


def extract_frames(video: Path, frame_ids, out_img_dir: Path):
    """Decode exactly the requested (1-based) frame_ids to <idx:04d>.png.

    frame_id is 1-based (matches the label file); OpenCV frames are 0-based,
    so frame_id f -> POS_FRAMES f-1.
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    for f in sorted(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, frame = cap.read()
        if not ok:
            print(f"  WARN: failed to read frame {f}")
            continue
        cv2.imwrite(str(out_img_dir / f"{f:04d}.png"), frame)
        written += 1
    cap.release()
    return H, W, total, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="root of the drone-tracking-datasets checkout (dataset1/ ... dataset5/)")
    ap.add_argument("--dataset", default="dataset1")
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--n", type=int, default=100, help="number of frame pairs")
    ap.add_argument("--out", required=True, help="output camera dir, e.g. real_eval/all/dataset1_cam0")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--strategy", choices=["stratified", "even"], default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--floor", type=int, default=10,
                    help="min pairs per occupied motion stratum (stratified only)")
    args = ap.parse_args()

    droot = Path(args.dataset_root) / args.dataset
    label_file = droot / "detections" / f"{args.cam}.txt"
    zip_file = droot / args.cam / f"{args.cam}.zip"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = parse_labels(label_file)
    print(f"labels: {len(labels)} valid frames "
          f"(range {min(labels)}..{max(labels)})")

    pairs, strat_info = select_pairs(labels, args.n, args.first, args.last,
                                     strategy=args.strategy, seed=args.seed,
                                     floor=args.floor)
    needed = sorted({i for i in pairs} | {i + 1 for i in pairs})
    print(f"selected {len(pairs)} pairs ({args.strategy}) "
          f"-> {len(needed)} unique frames")
    if strat_info:
        for s, m in sorted(strat_info["strata"].items()):
            print(f"  {s:10} pop={m['population']:5d} alloc={m['allocated']}")

    # extract the mp4 out of the zip once (streamed) to a temp path
    video_path = out / f"{args.cam}.mp4"
    if not video_path.exists():
        print(f"extracting {zip_file.name} -> {video_path.name}")
        with video_path.open("wb") as fh:
            subprocess.run(["7z", "e", "-so", str(zip_file)], stdout=fh, check=True,
                           stderr=subprocess.DEVNULL)

    H, W, total, written = extract_frames(video_path, needed, out / "images")
    print(f"video {W}x{H}, {total} frames; wrote {written} pngs")
    video_path.unlink()  # drop the big mp4, keep only the needed frames

    (out / "labels.json").write_text(json.dumps(
        {str(k): list(v) for k, v in labels.items()}))
    selected = [{"idx": i, "label_ref": list(labels[i]),
                 "label_nxt": list(labels[i + 1]),
                 "gt_disp_mag": round(disp_mag(labels, i), 4),
                 "motion_stratum": motion_bin(disp_mag(labels, i))} for i in pairs]
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    (out / "frames.txt").write_text("\n".join(f"{i:04d}" for i in pairs) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "dataset": args.dataset, "cam": args.cam,
        "camera_model": None, "H": H, "W": W,
        "n_pairs": len(pairs), "n_frames": written,
        "selection": strat_info or {"strategy": args.strategy},
        "note": "static tripod camera; label = drone centre at reference frame",
    }, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
