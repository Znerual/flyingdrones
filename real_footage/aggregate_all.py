#!/usr/bin/env python3
"""Aggregate per-camera checkpoint results into a mean-over-cameras leaderboard.

Reads <results>/per_camera_checkpoint*.csv (written by collect.py), averages each
metric over all cameras a checkpoint was scored on (mean and sample std, ddof=1),
and writes <results>/overall_checkpoint.csv (ranked by mean nss_disk) plus a
printed leaderboard.

usage: python aggregate_all.py [--results results/all]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all")
    RES = Path(ap.parse_args().results)
    rows = []
    for pp in sorted(RES.glob("per_camera_checkpoint*.csv")):
        rows.extend(csv.DictReader(pp.open()))
    if not rows:
        raise SystemExit("no rows yet")
    metric_cols = [c for c in rows[0] if c not in ("camera", "system")]
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)

    out = []
    for sysname, rs in by_sys.items():
        agg = {"system": sysname, "n_cameras": len(rs)}
        for c in metric_cols:
            vals = [float(r[c]) for r in rs if r[c] not in ("", "nan")]
            agg[c] = float(np.mean(vals)) if vals else float("nan")
            # std across cameras (population->sample); NaN if <2 cameras
            agg[c + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    # (loop body continues)
        out.append(agg)
    out.sort(key=lambda r: r.get("nss_disk", -1e9), reverse=True)

    cols = []
    for c in metric_cols:
        cols += [c, c + "_std"]
    ocsv = RES / "overall_checkpoint.csv"
    with ocsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "n_cameras"] + cols)
        w.writeheader(); w.writerows(out)

    print(f"{len(out)} checkpoints, mean +/- std across cameras "
          f"(max n={max(r['n_cameras'] for r in out)})\n")
    print(f"{'system':44} {'cams':>4} {'nss_disk (mean+/-std)':>22} {'hit@50':>16} {'pt_epe':>14}")
    print("-" * 104)
    for r in out:
        print(f"{r['system']:44.44} {r['n_cameras']:4d} "
              f"{r.get('nss_disk',0):9.2f} +/-{r.get('nss_disk_std',float('nan')):7.2f}   "
              f"{r.get('hit@50',0):6.2f} +/-{r.get('hit@50_std',float('nan')):5.2f}   "
              f"{r.get('pt_epe',0):6.2f} +/-{r.get('pt_epe_std',float('nan')):5.2f}")
    print(f"\nwrote {ocsv}")


if __name__ == "__main__":
    main()
