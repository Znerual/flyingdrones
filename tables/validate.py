"""Validate every v2 cell; write <out>/validation.csv and print anomalies.

Checks per manifest row:
  * summary CSV present (else MISSING)
  * pairs CSV: expected row count (999 stratified / 500 window), unique refs, refs == split
  * status file state, the log of the run that produced the result (RESULT line), and the
    weight-loading line ("loaded N/M keys; missing=0 unexpected=0") for learned models
  * summary == aggregate(pairs) recomputed locally; tolerance = max(1e-6 relative,
    half a unit in the last printed decimal of the summary CSV) -- the evaluators round
    the summary to 4 decimals
  * NaN in any aggregate; drone EPE >= zero-flow drone EPE at the same resolution
  * byte-identical per-pair EPE vectors across different tags (same resolution)

Job logs/status files are optional; without them only the result files are checked.

Usage: FD_RESULTS=<results dir> python tables/validate.py [--out tables_out/data]
"""
import argparse
import collections
import csv
import hashlib
import math
import os
import os.path as osp
import re

import numpy as np

import v2lib as L

SUM_COLS = {  # summary column -> pairs aggregate key
    "epe": "epe_all", "epe_drone": "epe_drone", "epe_norm": "epe_norm_all",
    "epe_norm_drone": "epe_norm_drone", "px1_err": "px1_err_all", "px1_err_drone": "px1_err_drone",
    "px3_err": "px3_err_all", "px3_err_drone": "px3_err_drone", "px5_err": "px5_err_all",
    "px5_err_drone": "px5_err_drone", "fl": "fl_all", "fl_drone": "fl_drone",
}
EXPECTED_IDENTICAL = {  # same released weights
    frozenset({"gmflow__gmflow_with_refine_things-36579974", "unimatch__gmflow-scale2-things-36579974"}),
    frozenset({"gmflow__gmflow_things-e9887eda", "unimatch__gmflow-scale1-things-e9887eda"}),
    frozenset({"gmflow__gmflow_with_refine_sintel-3ed1cf48", "unimatch__gmflow-scale2-sintel-3ed1cf48"}),
}


def split_refs(window):
    if window:
        return {f"{i:04d}" for i in range(1, 501)}
    with open(osp.join(L.SPLIT_DIR, "nature3_split_frames.txt")) as fh:
        s = {l.strip() for l in fh if l.strip()}
    s.discard("0000")  # never a reference frame under backward pairing
    return s


def log_index():
    idx = collections.defaultdict(list)
    p = osp.join(L.ROOT, "logs", "index.tsv")
    if osp.exists(p):
        for l in open(p):
            f = l.rstrip("\n").split("\t")
            if len(f) >= 2:
                idx[f[1]].append(f[0])
    return idx


KEYS_RE = re.compile(r"loaded (\d+)/(\d+) keys; missing=(\d+) unexpected=(\d+)")


def find_log(tag, jobs):
    """Return (log file, keys line, result-line-present) for the run that produced the result."""
    best = (None, None, False)
    for j in dict.fromkeys(jobs):
        f = osp.join(L.ROOT, "logs", f"fdv2-{j}.out")
        if not osp.exists(f):
            continue
        txt = open(f, errors="replace").read().replace("\r", "\n")
        has_res = f"RESULT {tag} " in txt
        keys = [m.group(0) for m in KEYS_RE.finditer(txt)]
        if has_res:
            return f, (keys[-1] if keys else None), True
        if best[0] is None:
            best = (f, keys[-1] if keys else None, False)
    return best


def vec_hash(rows):
    a = np.round(np.array([r["epe_all"] for r in rows]), 9)
    b = np.round(np.array([r["epe_drone"] for r in rows]), 9)
    return hashlib.md5(a.tobytes() + b.tobytes()).hexdigest()


def tol_for(s):
    m = re.match(r"^-?\d+\.(\d+)$", s.strip())
    return 0.5 * 10 ** (-len(m.group(1))) + 1e-12 if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables_out/data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = L.read_manifest()
    lidx = log_index()
    out, hashes = [], collections.defaultdict(list)
    zero = {}
    cache = {}
    for r in man:
        tag = r["tag"]
        rec = dict(idx=r["idx"], tag=tag, family=r["family"], label=r["label"], W=r["W"], H=r["H"],
                   window=int(r["window"]), state="", n_pairs=0, n_drone_pairs=0, keys="", log="",
                   max_abs_diff=float("nan"), epe=float("nan"), epe_drone=float("nan"), flags="")
        flags = []
        st = osp.join(L.ROOT, "status", f"{tag}.status")
        rec["state"] = open(st).read().split("\t")[0] if osp.exists(st) else "NO_STATUS"
        summ = L.read_summary(tag)
        pairs = L.read_pairs(tag) if summ else None
        if not summ or not pairs:
            rec["state"] = "MISSING" if not summ else "MISSING_PAIRS"
            out.append(rec)
            continue
        cache[tag] = pairs
        want = split_refs(r["window"])
        refs = [p["ref"] for p in pairs]
        rec["n_pairs"] = len(pairs)
        if len(pairs) != len(want):
            flags.append(f"n_pairs={len(pairs)}!={len(want)}")
        if len(set(refs)) != len(refs):
            flags.append("dup_refs")
        if set(refs) != want:
            flags.append(f"refs!=split(missing={len(want - set(refs))},extra={len(set(refs) - want)})")
        if len(summ) != 1:
            flags.append(f"summary_rows={len(summ)}")
        agg = L.fdc.aggregate(pairs)
        rec["n_drone_pairs"] = int(sum(not math.isnan(p["epe_drone"]) for p in pairs))
        s = summ[-1]
        md = 0.0
        for sc, pk in SUM_COLS.items():
            sv = float(s[sc])
            d = abs(sv - agg[pk])
            md = max(md, d)
            if d > max(1e-6 * abs(agg[pk]), tol_for(s[sc])):
                flags.append(f"summary_mismatch:{sc}({s[sc]} vs {agg[pk]:.6f})")
        if int(float(s.get("samples_processed", -1))) != len(pairs):
            flags.append(f"samples_processed={s.get('samples_processed')}")
        rec["max_abs_diff"] = md
        rec["epe"], rec["epe_drone"] = agg["epe_all"], agg["epe_drone"]
        if any(math.isnan(v) for v in agg.values()):
            flags.append("NaN_aggregate:" + ",".join(k for k, v in agg.items() if math.isnan(v)))
        # log / keys
        f, keys, has_res = find_log(tag, lidx.get(tag, []))
        rec["log"] = osp.basename(f) if f else ""
        rec["keys"] = keys or ""
        learned = r["family"] not in ("pyflow", "zeroflow")
        if learned:
            if not has_res and osp.isdir(osp.join(L.ROOT, "logs")):
                flags.append("producing_log_lost(requeue overwrote log)")
            if keys:
                m = KEYS_RE.search(keys)
                n, tot, mi, un = map(int, m.groups())
                if mi or un:
                    flags.append("KEYS_BAD")
                if n != tot and r["family"] != "vcn":
                    flags.append("KEYS_PARTIAL")
            elif has_res:
                flags.append("no_keys_line")
        if r["family"] == "zeroflow":
            zero[(r["W"], r["window"])] = agg["epe_drone"]
        hashes[(r["W"], r["window"], vec_hash(pairs))].append(tag)
        rec["flags"] = ";".join(flags)
        out.append(rec)

    # zero-flow comparison
    for rec in out:
        z = zero.get((rec["W"], bool(rec["window"])))
        if rec["family"] != "zeroflow" and z is not None and not math.isnan(rec["epe_drone"]) and rec["epe_drone"] >= z:
            rec["flags"] = (rec["flags"] + ";" if rec["flags"] else "") + f"drone_EPE>=zero({z:.3f})"
    # identical vectors
    ident = []
    for (W, win, h), tags in hashes.items():
        if len(tags) > 1:
            stems = frozenset(t.rsplit("__", 1)[0] if not win else t.rsplit("__", 2)[0] for t in tags)
            expected = stems in EXPECTED_IDENTICAL
            ident.append((W, win, tags, expected))
            for rec in out:
                if rec["tag"] in tags:
                    rec["flags"] = (rec["flags"] + ";" if rec["flags"] else "") + \
                        ("identical_expected:" if expected else "IDENTICAL:") + "|".join(t for t in tags if t != rec["tag"])

    cols = list(out[0].keys())
    with open(osp.join(a.out, "validation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    n_ok = sum(1 for r in out if r["state"] not in ("MISSING", "MISSING_PAIRS"))
    print(f"cells: {len(out)}  with results: {n_ok}  missing: {len(out) - n_ok}")
    for r in out:
        if r["state"] in ("MISSING", "MISSING_PAIRS"):
            print(f"  MISSING  {r['tag']}  (status={r['state']})")
    print("max |summary - aggregate(pairs)| over all cells:",
          max((r["max_abs_diff"] for r in out if not math.isnan(r["max_abs_diff"])), default=float("nan")))
    print("flagged cells:")
    for r in out:
        if r["flags"]:
            print(f"  {r['tag']}: {r['flags']}")
    print("identical per-pair vectors:")
    for W, win, tags, exp in ident:
        print(f"  {W} win={win} expected={exp}: {tags}")


if __name__ == "__main__":
    main()
