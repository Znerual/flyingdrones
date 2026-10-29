#!/usr/bin/env python3
"""Add the frame window that multi-frame models (VideoFlow BOF/MOF) need.

2-frame models use only {idx, idx+1}; VideoFlow MOF needs [idx-2..idx+2].
This extracts any window frames still missing from <subset>/images/ for every
selected reference index, without touching existing frames or labels.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="camera dir written by prep_subset.py")
    ap.add_argument("--zip", required=True, help="<dataset-root>/<dataset>/<cam>/<cam>.zip")
    ap.add_argument("--radius", type=int, default=2, help="frames each side of idx")
    args = ap.parse_args()

    sub = Path(args.subset)
    img_dir = sub / "images"
    sel = json.loads((sub / "selected.json").read_text())
    idxs = [s["idx"] for s in sel]

    needed = set()
    for i in idxs:
        for off in range(-args.radius, args.radius + 2):  # covers idx-2..idx+2 (+ idx+1)
            f = i + off
            if f >= 1:
                needed.add(f)
    missing = sorted(f for f in needed if not (img_dir / f"{f:04d}.png").exists())
    print(f"{len(idxs)} refs -> {len(needed)} window frames, {len(missing)} missing")
    if not missing:
        print("nothing to extract")
        return

    video = sub / "cam0_tmp.mp4"
    with video.open("wb") as fh:
        subprocess.run(["7z", "e", "-so", args.zip], stdout=fh, check=True,
                       stderr=subprocess.DEVNULL)
    cap = cv2.VideoCapture(str(video))
    wrote = 0
    for f in missing:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(img_dir / f"{f:04d}.png"), frame)
            wrote += 1
    cap.release()
    video.unlink()
    print(f"extracted {wrote} window frames -> {img_dir}")


if __name__ == "__main__":
    main()
