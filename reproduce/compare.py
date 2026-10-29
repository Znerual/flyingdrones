#!/usr/bin/env python3
"""Compare your evaluator outputs with the paper numbers (results/cells.csv).

usage: python reproduce/compare.py $FD_OUT [--tol 1e-3]

For every ``$FD_OUT/pairs/<tag>.csv`` found, the per-pair rows are aggregated with
``flyingdrones.aggregate`` and compared with the reference cell of the same tag
(full-frame and drone-region EPE and the other metrics). Deterministic models
should agree to ~1e-4 px; GPU/library differences can move learned models
slightly, FlowDiffusion samples with unseeded noise.
"""
import argparse
import csv
import glob
import os
import os.path as osp
import sys

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)

import flyingdrones as fd  # noqa: E402

KEYS = ["epe_all", "epe_drone", "epe_norm_all", "epe_norm_drone", "px1_err_drone", "px3_err_drone", "fl_drone"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--tol", type=float, default=1e-3, help="abs. tolerance for EPE-type metrics")
    a = ap.parse_args()
    ref = {r["tag"]: r for r in csv.DictReader(open(osp.join(REPO, "results", "cells.csv")))}
    files = sorted(glob.glob(osp.join(a.out_dir, "pairs", "*.csv")))
    if not files:
        sys.exit(f"no per-pair CSVs under {a.out_dir}/pairs")
    n_bad = 0
    for p in files:
        tag = osp.basename(p)[:-4]
        rows = fd.read_pairs_csv(p)
        agg = fd.aggregate(rows)
        r = ref.get(tag)
        if r is None:
            print(f"?    {tag}: no reference cell")
            continue
        diffs = {k: agg[k] - float(r[k]) for k in KEYS}
        tol = {k: (a.tol if k.startswith("epe") else 100 * a.tol) for k in KEYS}
        bad = [k for k in KEYS if abs(diffs[k]) > tol[k]]
        n_ok = len(rows) == int(r["n_pairs"])
        flag = "OK  " if not bad and n_ok else "DIFF"
        n_bad += flag == "DIFF"
        print(f"{flag} {tag}: pairs {len(rows)}/{r['n_pairs']}  epe_drone {agg['epe_drone']:.4f} "
              f"(paper {float(r['epe_drone']):.4f})  epe_all {agg['epe_all']:.4f} (paper {float(r['epe_all']):.4f})"
              + (f"  off: {', '.join(bad)}" if bad else ""))
    print(f"\n{len(files)} cells compared, {n_bad} outside tolerance")


if __name__ == "__main__":
    main()
