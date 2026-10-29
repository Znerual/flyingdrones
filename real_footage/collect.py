#!/usr/bin/env python3
"""Collect the per-camera CSVs written by score_one.py into the table inputs.

usage: python collect.py <per_cam_dir> [--results results/all]

<per_cam_dir> holds <camera>__<system>.csv and <camera>__<system>__stratum.csv
(score_one.py --output-csv <per_cam_dir>/<camera>__<system>.csv). Writes
    <results>/per_camera_checkpoint.csv   one row per (camera, system)
    <results>/per_camera_stratum.csv      one row per (camera, system, motion stratum)
    <results>/strata_hit50.tsv            per system: Hit@50 per stratum, mean over the
                                          cameras that have pairs in that stratum
Then run aggregate_all.py --results <results> for the mean +- std over cameras
(paper tables "real-all" and "real-strata").
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

STRATA = ["m1_subpx", "m2_low", "m3_mid", "m4_high"]


def write(path, cols, rows, delim=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=delim)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_cam_dir")
    ap.add_argument("--results", default="results/all")
    a = ap.parse_args()
    pp, st = [], []
    for p in sorted(Path(a.per_cam_dir).glob("*.csv")):
        rows = list(csv.DictReader(p.open()))
        (st if p.name.endswith("__stratum.csv") else pp).extend(rows)
    if not pp:
        raise SystemExit(f"no per-camera CSVs in {a.per_cam_dir}")
    order = {s: i for i, s in enumerate(STRATA)}
    pp.sort(key=lambda r: (r["camera"], r["system"]))
    st.sort(key=lambda r: (r["camera"], r["system"], order.get(r["stratum"], 9)))
    res = Path(a.results)
    write(res / "per_camera_checkpoint.csv", list(pp[0]), pp)
    if st:
        write(res / "per_camera_stratum.csv", list(st[0]), st)
        strat = defaultdict(lambda: defaultdict(list))
        for r in st:
            strat[r["system"]][r["stratum"]].append(float(r["hit@50"]))
        rows = []
        for s in sorted(strat):
            row = {"system": s}
            for k in STRATA:
                v = strat[s][k]
                row[k] = round(float(np.mean(v)), 4) if v else ""
                row[f"ncam_{k}"] = len(v)
            rows.append(row)
        write(res / "strata_hit50.tsv", ["system"] + STRATA + [f"ncam_{k}" for k in STRATA], rows, delim="\t")
    cams = defaultdict(set)
    for r in pp:
        cams[r["system"]].add(r["camera"])
    print(f"{len(cams)} systems, cameras per system: {sorted({len(v) for v in cams.values()})}")
    print(f"wrote {res}/per_camera_checkpoint.csv" + (", per_camera_stratum.csv, strata_hit50.tsv" if st else ""))


if __name__ == "__main__":
    main()
