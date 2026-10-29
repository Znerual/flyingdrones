#!/usr/bin/env python3
"""Prepare the displacement-stratified real-footage subsets for every labelled camera
of the drone-tracking dataset of Li et al. (datasets 1-4; dataset5 has no 2-D labels).

For each (dataset, cam) it runs prep_subset.py (100 pairs, stratified by the drone's
label displacement, seed 42), extract_window.py (the +-2 frame window needed by
VideoFlow MOF) and downscale_camera.py (cameras wider than 1920 px are scored at
1920 px width, labels scaled accordingly), writing a self-contained dir

    <out>/<dataset>_<cam>/{images/, labels.json, selected.json, frames.txt, meta.json}

usage: python prep_all.py --dataset-root <drone-tracking-datasets> [--out real_eval/all] [--n 100]
Needs OpenCV and the `7z` command line tool (the videos ship as .zip archives).
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# labelled cameras per dataset (detections/camN.txt present)
CAMS = {"dataset1": range(4), "dataset2": range(4),
        "dataset3": range(6), "dataset4": range(7)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="root of the drone-tracking-datasets checkout (dataset1/ ... dataset5/)")
    ap.add_argument("--out", default="real_eval/all")
    ap.add_argument("--n", type=int, default=100, help="frame pairs per camera")
    ap.add_argument("--max-width", type=int, default=1920,
                    help="downscale wider cameras to this width (0 = keep native)")
    a = ap.parse_args()
    raw, out_root = Path(a.dataset_root), Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    todo = [(d, f"cam{i}") for d, rng in CAMS.items() for i in rng]
    print(f"preparing {len(todo)} cameras, N={a.n} pairs each")
    for ds, cam in todo:
        out = out_root / f"{ds}_{cam}"
        zip_ = raw / ds / cam / f"{cam}.zip"
        if (out / "frames.txt").exists():
            print(f"[skip] {ds}/{cam} already prepared")
        else:
            print(f"[prep] {ds}/{cam}")
            r = subprocess.run([sys.executable, str(HERE / "prep_subset.py"),
                                "--dataset-root", str(raw), "--dataset", ds, "--cam", cam,
                                "--n", str(a.n), "--out", str(out)])
            if r.returncode != 0:
                print(f"  FAILED prep {ds}/{cam}")
                continue
        # VideoFlow +-2 window frames
        subprocess.run([sys.executable, str(HERE / "extract_window.py"),
                        "--subset", str(out), "--zip", str(zip_)], check=True)
        if a.max_width:
            subprocess.run([sys.executable, str(HERE / "downscale_camera.py"), str(out),
                            "--max-width", str(a.max_width)], check=True)
    print("done")


if __name__ == "__main__":
    main()
