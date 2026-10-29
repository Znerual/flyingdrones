#!/usr/bin/env python3
"""Score ONE checkpoint's dumped flows for ONE camera and write a single-row CSV.

A flow model first dumps one flow field per selected pair to --flows-dir (see
README.md for the file contract); this script then writes --output-csv
(camera,system,<metrics...>) and <output-csv stem>__stratum.csv (per motion
stratum). Metric definitions live in align_metric.py.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align_metric as AM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows-dir", required=True)
    ap.add_argument("--subset-dir", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--output-csv", required=True)
    a = ap.parse_args()

    sel = json.loads((Path(a.subset_dir) / "selected.json").read_text())
    labels = {s["idx"]: (s["label_ref"], s["label_nxt"]) for s in sel}
    rows = []
    for npy in sorted(Path(a.flows_dir).glob("*.npy")):
        _, idx = AM.parse_npy_name(npy.name)
        if idx not in labels:
            continue
        lr, ln = labels[idx]
        m = AM.metrics_for(np.load(npy), lr, ln)
        if m is None:
            continue
        m["stratum"] = AM.motion_bin(m["gt_disp_mag"])
        rows.append(m)
    if not rows:
        raise SystemExit("no scorable flows")

    agg = AM.aggregate(rows)
    out = Path(a.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {"camera": a.camera, "system": a.system, **agg}
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader(); w.writerow(row)

    # also a per-stratum CSV alongside (optional; gathered later)
    by_s = defaultdict(list)
    for r in rows:
        by_s[r["stratum"]].append(r)
    st = out.with_name(out.stem + "__stratum.csv")
    with st.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["camera", "system", "stratum", "n", "nss_disk", "hit@50"])
        w.writeheader()
        for s, rs in by_s.items():
            b = AM.aggregate(rs)
            w.writerow({"camera": a.camera, "system": a.system, "stratum": s,
                        "n": b["n"], "nss_disk": b.get("nss_disk"), "hit@50": b.get("hit@50")})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
