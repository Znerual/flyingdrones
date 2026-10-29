"""Shared loading / metadata for the FlyingDrones v2 (backward-GT) benchmark results.

Results layout ($FD_RESULTS, i.e. the $FD_OUT of the evaluator commands):
    summary/<tag>.csv   one-row summary written by the evaluator (4-decimal rounded)
    pairs/<tag>.csv     one row per evaluated pair (full precision)
    status/<tag>.status, logs/fdv2-<job>.out + logs/index.tsv   optional job bookkeeping
Manifest: reproduce/manifest.tsv of this repository (override with $FD_MANIFEST).
Tags: <family>__<ckpt>__<W>x<H>[__win500].
"""
import csv
import math
import os
import os.path as osp
import re
import sys

import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(HERE)
ROOT = os.environ.get("FD_RESULTS") or os.environ.get("FD_OUT") or osp.join(os.getcwd(), "results")
SPLIT_DIR = osp.join(REPO, "splits")
sys.path.insert(0, osp.join(REPO, "evaluators", "_common"))
import fd_eval_common as fdc  # noqa: E402

RES = [(480, 270), (960, 432), (1920, 1080)]  # (W, H)
RES_TEX = {480: r"480$\times$270", 960: r"960$\times$432", 1920: r"1920$\times$1080"}
RES_KEY = {480: "480x270", 960: "960x432", 1920: "1920x1080"}

# family key -> (display, class letter, temporal dagger, tiled at 1080 in v2)
# Tiling at 1080 (v2 manifest / evaluator code): RAFT, WAFT, LiteFlowNet3, MemFlow (--tile 432 960),
# VideoFlow (--tile_size 432 960), FF++ / GMFlow / UniMatch (evaluator auto-tiles above 432x960).
# NOT tiled: VCN, SEA-RAFT, PWC-Net, FlowNet2, ARFlow, FastFlowNet, FlowDiffusion, PyFlow.
FAMILIES = {
    "arflow": ("ARFlow", "E", False, False),
    "fastflownet": ("FastFlowNet", "E", False, False),
    "flowdiffusion": ("FlowDiffusion (DDVM)", "M", False, False),
    "flowformerplusplus": ("FlowFormer++", "T", False, True),
    "flownet2": ("FlowNet2", "P", False, False),
    "gmflow": ("GMFlow", "T", False, True),
    "liteflownet3": ("LiteFlowNet3", "P", False, True),
    "memflow": ("MemFlow", "M", True, True),
    "pwcnet": ("PWC-Net", "P", False, False),
    "pyflow": ("PyFlow (C2F)", "P", False, False),
    "raft": ("RAFT", "C", False, True),
    "searaft": ("SEA-RAFT", "C", False, False),
    "unimatch": ("UniMatch", "T", False, True),
    "vcn": ("VCN", "P", False, False),
    "videoflow": ("VideoFlow", "M", True, True),
    "waft": ("WAFT", "C", False, True),
}
# paper ordering = alphabetical by display name
FAMILY_ORDER = sorted(FAMILIES, key=lambda f: FAMILIES[f][0].lower())

METRICS = ["epe", "epe_norm", "px1_err", "px3_err", "px5_err", "fl"]


def ckpt_label(family, ckpt_path, cfgv):
    """Checkpoint label as printed in the paper tables (unescaped)."""
    if family == "pyflow":
        return cfgv.split("=", 1)[1]
    if family == "zeroflow":
        return "zero flow"
    p = ckpt_path.rstrip("/")
    if family == "arflow":
        parts = p.split("/")
        return f"{parts[-2]}/{osp.splitext(parts[-1])[0]}"
    if family == "vcn":
        m = re.search(r"(?:models/|\$CKPT/)(.*)\.tar$", p)
        return m.group(1)
    base = osp.basename(p)
    if family == "searaft" and cfgv.startswith("variant=sintel"):
        return "Tartan-C-T-TSKH432x960-M (sintel)"  # the sintel release has no dataset tag in its file name
    for ext in (".pth.tar", ".pth", ".ckpt", ".tar"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def read_manifest(path=None):
    path = path or os.environ.get("FD_MANIFEST") or osp.join(REPO, "reproduce", "manifest.tsv")
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip()]
    hdr = lines[0].lstrip("#").split("\t")
    rows = []
    for l in lines[1:]:
        if l.startswith("#"):
            continue
        r = dict(zip(hdr, l.split("\t")))
        r["idx"] = int(r["idx"])
        r["W"] = int(r["W"])
        r["H"] = int(r["H"])
        r["window"] = r["tag"].endswith("__win500")
        r["label"] = ckpt_label(r["family"], r["ckpt"], r["cfg_variant"])
        rows.append(r)
    return rows


def ckpt_sort_key(family, label):
    # paper order = manifest order except MemFlow, which the paper sorts ASCII
    return label if family == "memflow" else None


def read_pairs(tag):
    p = osp.join(ROOT, "pairs", f"{tag}.csv")
    if not osp.exists(p):
        return None
    with open(p) as fh:
        rd = csv.DictReader(fh)
        rows = []
        for r in rd:
            out = {"ref": r["ref"]}
            for k, v in r.items():
                if k == "ref":
                    continue
                out[k] = float(v) if v not in ("", None) else float("nan")
            rows.append(out)
    return rows


def read_summary(tag):
    p = osp.join(ROOT, "summary", f"{tag}.csv")
    if not osp.exists(p):
        return None
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    return rows


def pairs_array(rows, key):
    return np.array([r[key] for r in rows], dtype=np.float64)


def fmt4(x):
    """Paper agg-table rule: 4 significant figures, 2 decimals if >=10 else 3 (trailing zeros kept).
    Values >=100 get 1 decimal (never occurs for EPE; Px can reach 100.0)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def fmt3(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.3f}"


def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
