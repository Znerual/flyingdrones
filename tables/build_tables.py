"""Bootstrap CIs, rank stability, LaTeX tables and DIGEST.md for the FlyingDrones v2 benchmark.

Reads $FD_RESULTS/{pairs,summary}/ + reproduce/manifest.tsv (+ <out>/data/validation.csv
from validate.py). All numbers are recomputed from the per-pair CSVs at full precision
(the summary CSVs are rounded to 4 decimals).

Outputs (default --out = tables_out):
    tab_agg_full.tex, tab_agg_drone.tex, tab_summary_best_drone_{270x480,432x960,1080x1920}.tex,
    tab_protocol.tex, DIGEST.md,
    data/cells.csv            every cell, all metrics (all + drone), drone/full EPE 95% CI
    data/leaderboard.csv      best-drone-EPE checkpoint per family per resolution
    data/rank_stability.csv   paired bootstrap between leaderboard families
    data/rank1_probability.csv  P(family is #1) under joint paired resampling
    data/protocol.csv         contiguous window vs stratified (protocol-sensitivity table)
Bootstrap: percentile, 10k resamples over pairs, seed 0 (fd_eval_common.bootstrap_ci); paired
tests resample the same reference frames for every model (rng seed 0, multinomial counts).
"""
import argparse
import collections
import csv
import math
import os
import os.path as osp

import numpy as np

import v2lib as L

N_BOOT = 10000
SEED = 0
RES_LB = {480: "270x480", 960: "432x960", 1920: "1080x1920"}
RES_TXT = {480: "480x270", 960: "960x432", 1920: "1920x1080"}

CLS_CAPTION = (r"\emph{Cls} groups families as in Sections~\ref{sec:corr}--\ref{sec:temporal}: "
               r"C~= correlation/iterative, P~= classical\,/\,pyramid, E~= efficient\,/\,reduced-supervision, "
               r"T~= transformer\,/\,global matching, M~= temporal\,/\,memory\,/\,generative.")


# --------------------------------------------------------------------------- data

class Cell:
    def __init__(self, r, pairs):
        self.r = r
        self.tag, self.family, self.label, self.W = r["tag"], r["family"], r["label"], r["W"]
        self.window = r["window"]
        self.pairs = pairs
        self.missing = pairs is None
        if pairs:
            self.agg = L.fdc.aggregate(pairs)
            self.refs = [p["ref"] for p in pairs]
            self.d = L.pairs_array(pairs, "epe_drone")
            self.a = L.pairs_array(pairs, "epe_all")
            self.ci_d = L.fdc.bootstrap_ci(self.d, n_boot=N_BOOT, seed=SEED)
            self.ci_a = L.fdc.bootstrap_ci(self.a, n_boot=N_BOOT, seed=SEED)
        else:
            self.agg = {f"{m}_{g}": float("nan") for m in L.METRICS for g in ("all", "drone")}

    def m(self, metric, region):
        return self.agg[f"{metric}_{region}"]

    @property
    def epe_d(self):
        return self.agg["epe_drone"]


def load_cells():
    man = L.read_manifest()
    cells = []
    for r in man:
        pairs = L.read_pairs(r["tag"]) if L.read_summary(r["tag"]) else None
        cells.append(Cell(r, pairs))
    return cells


def family_ckpt_order(cells, family):
    labs = []
    for c in sorted((c for c in cells if c.family == family and not c.window), key=lambda c: c.r["idx"]):
        if c.label not in labs:
            labs.append(c.label)
    return sorted(labs) if family == "memflow" else labs


def leaderboard(cells, W):
    best = {}
    for c in cells:
        if c.window or c.missing or c.W != W or c.family == "zeroflow":
            continue
        if c.family not in best or c.epe_d < best[c.family].epe_d:
            best[c.family] = c
    return sorted(best.values(), key=lambda c: c.epe_d)


def zero_cell(cells, W, window=False):
    for c in cells:
        if c.family == "zeroflow" and c.W == W and c.window == window:
            return c
    return None


# --------------------------------------------------------------------------- bootstrap

def aligned_drone_matrix(cs):
    """(n_ckpt, n_pairs) drone EPE aligned on reference frame; pairs without drones dropped."""
    refs = sorted(cs[0].refs)
    for c in cs:
        assert sorted(c.refs) == refs, f"ref set differs: {c.tag}"
    M = np.vstack([np.array([dict(zip(c.refs, c.d))[r] for r in refs]) for c in cs])
    keep = ~np.isnan(M).any(0)
    assert keep.sum() == (~np.isnan(M[0])).sum(), "drone-pair sets differ between checkpoints"
    return M[:, keep]


def boot_counts(n, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.float64)


def rank_stability(cells, out_rows, rank1_rows):
    for W, _ in L.RES:
        lb = leaderboard(cells, W)
        fams = [c.family for c in lb]
        fam_cells = {f: [c for c in cells if c.family == f and c.W == W and not c.window and not c.missing]
                     for f in fams}
        allc = [c for f in fams for c in fam_cells[f]]
        M = aligned_drone_matrix(allc)  # (n_ckpt, n_pairs)
        n = M.shape[1]
        rng = np.random.default_rng(SEED)
        C = boot_counts(n, rng)
        means = C @ M.T / n  # (N_BOOT, n_ckpt)
        col = {c.tag: i for i, c in enumerate(allc)}
        fixed = np.stack([means[:, col[c.tag]] for c in lb], 1)  # leaderboard ckpt per family
        resel = np.stack([means[:, [col[c.tag] for c in fam_cells[f]]].min(1) for f in fams], 1)
        for i in range(len(lb)):
            for j in range(i + 1, len(lb)):
                if not (j == i + 1 or i == 0):
                    continue
                out_rows.append(dict(
                    res=RES_TXT[W], rank_a=i + 1, family_a=lb[i].family, ckpt_a=lb[i].label,
                    epe_drone_a=lb[i].epe_d, rank_b=j + 1, family_b=lb[j].family, ckpt_b=lb[j].label,
                    epe_drone_b=lb[j].epe_d, mean_diff_b_minus_a=lb[j].epe_d - lb[i].epe_d,
                    diff_ci_lo=float(np.quantile(fixed[:, j] - fixed[:, i], 0.025)),
                    diff_ci_hi=float(np.quantile(fixed[:, j] - fixed[:, i], 0.975)),
                    p_a_beats_b_fixed_ckpt=float((fixed[:, i] < fixed[:, j]).mean()),
                    p_a_beats_b_reselect_ckpt=float((resel[:, i] < resel[:, j]).mean()),
                    n_pairs=n))
        win_f = np.bincount(fixed.argmin(1), minlength=len(lb)) / N_BOOT
        win_r = np.bincount(resel.argmin(1), minlength=len(lb)) / N_BOOT
        for k, c in enumerate(lb):
            rank1_rows.append(dict(res=RES_TXT[W], rank=k + 1, family=c.family, ckpt=c.label,
                                   p_rank1_fixed_ckpt=float(win_f[k]), p_rank1_reselect_ckpt=float(win_r[k])))


# --------------------------------------------------------------------------- latex helpers

def model_name(family, W, leaderboard_style=False):
    disp, _, dag, tiled = L.FAMILIES[family]
    s = disp + (r"$^\dagger$" if dag else "")
    if leaderboard_style and W == 1920 and tiled:
        s += r"$^\ddagger$"
    return s


def res_cell(family, W):
    tiled = family in L.FAMILIES and L.FAMILIES[family][3]
    return L.RES_TEX[W] + (r"$^\ddagger$" if W == 1920 and tiled else "")


def metric_cells(c, region):
    vals = []
    for m in L.METRICS:
        v = c.m(m, region)
        vals.append(L.fmt3(v) if m == "epe_norm" else L.fmt4(v))
    return vals


def ci_txt(lo, hi):
    f = (lambda x: f"{x:.1f}") if hi >= 10 else (lambda x: f"{x:.2f}")
    return r"{\scriptsize[" + f(lo) + "," + f(hi) + "]}"


AGG_CAPTION = {
    "all": "Full-frame",
    "drone": "Drone-region (drone-pixel-only)",
}


def agg_table(cells, region):
    lab = "tab:agg-full" if region == "all" else "tab:agg-drone"
    cap = (AGG_CAPTION[region] + r" evaluation results on FlyingDrones, all model families, on the stratified "
           r"subset (999 flow pairs; each reference frame $t$ is paired with $t-1$). Rows grouped by model family, "
           r"resolution, then checkpoint. \textit{Italic}: best result within the model family at that resolution "
           r"(lowest " + ("full-frame" if region == "all" else "drone-region") + r" EPE). EPE is end-point error; "
           r"EPE$_\text{n}$ is the resolution-normalized EPE; Px$k$ is the percentage of pixels with error $>k$; "
           r"Fl is the outlier rate (within 0.02 percentage points of Px3 here). $\dagger$~marks the two families "
           r"consuming more than two input frames: VideoFlow (\texttt{BOF\_*} three frames, \texttt{MOF\_*} five) "
           r"and MemFlow (recurrent memory over an extended window). $\ddagger$~marks $1920\times1080$ runs that "
           r"used tiled inference ($432\times960$ tiles). The three SEA-RAFT rows are the separately released "
           r"\texttt{kitti}, \texttt{sintel} and \texttt{spring} checkpoints. The last block is the zero-flow "
           r"reference (predicting $\mathbf{0}$ everywhere).")
    head = (r"\textbf{Model} & \textbf{Checkpoint} & \textbf{Res.} & \textbf{EPE} & \textbf{EPE$_\text{n}$} & "
            r"\textbf{Px1} & \textbf{Px3} & \textbf{Px5} & \textbf{Fl} \\")
    o = [r"\begingroup", r"\setlength{\LTleft}{0pt}", r"\setlength{\LTright}{0pt}", r"\setlength{\tabcolsep}{4pt}",
         r"\scriptsize", r"\renewcommand{\arraystretch}{1.08}",
         r"\begin{longtable}{>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{4.0cm}"
         r">{\raggedright\arraybackslash}p{1.7cm}rrrrrr}",
         r"\caption{" + cap + r"}\label{" + lab + r"}\\", r"\toprule", head, r"\midrule", r"\endfirsthead", "",
         r"\multicolumn{9}{l}{\textit{Table~\ref{" + lab + r"} continued from previous page.}} \\", r"\toprule",
         head, r"\midrule", r"\endhead", "", r"\midrule",
         r"\multicolumn{9}{r}{\textit{Continued on next page.}} \\", r"\endfoot", "", r"\bottomrule",
         r"\endlastfoot", ""]
    by = {(c.family, c.label, c.W): c for c in cells if not c.window}
    first = True
    for fam in L.FAMILY_ORDER + ["zeroflow"]:
        if not first:
            o.append(r"\midrule")
        first = False
        labs = family_ckpt_order(cells, fam)
        for W, _ in L.RES:
            group = [by[(fam, lb, W)] for lb in labs if (fam, lb, W) in by]
            avail = [c for c in group if not c.missing]
            best = min(avail, key=lambda c: c.m("epe", region)).tag if (avail and fam != "zeroflow") else None
            for c in group:
                if fam == "zeroflow":
                    cols = ["Zero flow (reference)", "--", L.RES_TEX[W]]
                else:
                    cols = [model_name(fam, W), L.tex_escape(c.label), res_cell(fam, W)]
                if c.missing:
                    cols += ["--"] * 6
                    o.append("  " + " & ".join(cols) + r" \\  % PENDING: " + c.tag + " (missing result)")
                    continue
                cols += metric_cells(c, region)
                if c.tag == best:
                    cols = [r"\textit{" + x + "}" for x in cols]
                o.append("  " + " & ".join(cols) + r" \\")
    o += [r"\end{longtable}", r"\endgroup", ""]
    return "\n".join(o)


def leaderboard_table(cells, W):
    lb = leaderboard(cells, W)
    z = zero_cell(cells, W)
    H = dict(L.RES)[W]
    tiled_note = r"; $\ddagger$~marks runs that used tiled inference ($432\times960$ tiles)" if W == 1920 else ""
    cap = (r"Best drone-region EPE per model family at " + f"{W}$\\times${H}" + r", on the stratified subset "
           r"(999 flow pairs). One row per family: the checkpoint minimising drone-region EPE. The three best EPE "
           r"values are in \textbf{bold}; brackets give the 95\% percentile-bootstrap confidence interval of the "
           r"drone-region EPE (10{,}000 resamples over pairs). " + CLS_CAPTION +
           r" $\dagger$~marks models consuming more than two input frames" + tiled_note +
           r". The last row is the zero-flow reference (predicting $\mathbf{0}$ everywhere).")
    o = [r"\begin{table}[h]", "  \\caption{" + cap + "}",
         "  \\label{tab:summary-best-drone-" + RES_LB[W] + "}", r"  \centering", r"  \small",
         r"  \resizebox{\linewidth}{!}{%", r"  \begin{tabular}{cllrrrrrr}", r"    \toprule",
         r"    Cls & Model & Checkpoint & EPE [95\% CI] & EPE$_\text{n}$ & Px1 & Px3 & Px5 & Fl \\",
         r"    \midrule"]
    for k, c in enumerate(lb):
        vals = metric_cells(c, "drone")
        e = vals[0] if k >= 3 else r"\textbf{" + vals[0] + "}"
        e += " " + ci_txt(c.ci_d[1], c.ci_d[2])
        o.append("    " + " & ".join([L.FAMILIES[c.family][1], model_name(c.family, W, True),
                                      L.tex_escape(c.label), e] + vals[1:]) + r" \\")
    if z and not z.missing:
        vals = metric_cells(z, "drone")
        o.append(r"    \midrule")
        o.append("    " + " & ".join(["--", "Zero flow (reference)", "--",
                                      vals[0] + " " + ci_txt(z.ci_d[1], z.ci_d[2])] + vals[1:]) + r" \\")
    missing = [c.tag for c in cells if c.missing and c.W == W and not c.window]
    o += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}"]
    if missing:
        o.insert(0, "% NOTE: selection excludes unfinished cells: " + ", ".join(missing))
    return "\n".join(o) + "\n"


def short_ckpt(family, label):
    s = label
    if family in ("gmflow", "unimatch"):
        s = s.rsplit("-", 1)[0]
        if family == "unimatch":
            s = s.replace("gmflow-", "", 1)
    elif family == "vcn":
        s = s.replace("vcn_weights/", "").replace("vcn_small_weights/", "small/").replace("finetune_", "")
    elif family == "fastflownet":
        s = s.replace("fastflownet_", "")
    elif family == "flownet2":
        s = s.replace("_checkpoint", "")
    elif family == "searaft":
        s = s.replace("Tartan-C-T-", "")
    return s


PROT_NAME = {"flowdiffusion": "FlowDiffusion", "pyflow": "PyFlow"}


def protocol_rows(cells, W):
    lb = leaderboard(cells, W)
    win = {c.tag[: -len("__win500")]: c for c in cells if c.window}
    rows = []
    for c in lb:
        w = win.get(c.tag)
        wv = w.epe_d if (w is not None and not w.missing) else float("nan")
        rows.append(dict(res=RES_TXT[W], family=c.family, ckpt=c.label, short=short_ckpt(c.family, c.label),
                         window=wv, strat=c.epe_d, delta=c.epe_d - wv,
                         window_ci_lo=w.ci_d[1] if (w is not None and not w.missing) else float("nan"),
                         window_ci_hi=w.ci_d[2] if (w is not None and not w.missing) else float("nan"),
                         n_window_pairs=len(w.pairs) if (w is not None and not w.missing) else 0))
    rows.sort(key=lambda r: (math.isnan(r["window"]), r["window"]))
    z = zero_cell(cells, W)
    zw = zero_cell(cells, W, True)
    zrow = dict(res=RES_TXT[W], family="zeroflow", ckpt="zero flow", short="--",
                window=zw.epe_d if (zw and not zw.missing) else float("nan"), strat=z.epe_d if z else float("nan"))
    zrow["delta"] = zrow["strat"] - zrow["window"]
    return rows, zrow


def prot_stats(rows):
    d = np.array([r["delta"] for r in rows if not math.isnan(r["delta"])])
    if d.size == 0:
        return dict(n=0, median_abs=float("nan"), mean_abs=float("nan"), mean=float("nan"), median=float("nan"))
    return dict(n=int(d.size), median_abs=float(np.median(np.abs(d))), mean_abs=float(np.mean(np.abs(d))),
                mean=float(np.mean(d)), median=float(np.median(d)))


def prot_line(r):
    name = PROT_NAME.get(r["family"], L.FAMILIES[r["family"]][0]) if r["family"] != "zeroflow" else "Zero flow"
    ck = r"\texttt{" + L.tex_escape(r["short"]) + "}" if r["family"] != "zeroflow" else "--"
    wv = "--" if math.isnan(r["window"]) else f"{r['window']:.3f}"
    dv = "--" if math.isnan(r["delta"]) else f"{r['delta']:+.2f}"
    return [name, ck, wv, f"{r['strat']:.3f}", dv]


def protocol_table(cells):
    R = {W: protocol_rows(cells, W) for W, _ in L.RES}
    st = {W: prot_stats(R[W][0]) for W in R}
    def fm(x):
        return "--" if math.isnan(x) else f"{x:.2f}"
    cap = (r"Drone-region EPE for the leaderboard checkpoint of each family, contrasting a contiguous window "
           r"(reference frames 1--500, each paired with its predecessor) against the stratified subset (999 pairs) "
           r"used throughout the paper (checkpoint held fixed). \emph{Window} is the contiguous window; "
           r"\emph{Strat.} is the stratified subset. $\Delta$ is Strat.\ $-$ Window. Median (mean) $|\Delta|$: "
           + ", ".join(f"{fm(st[W]['median_abs'])} ({fm(st[W]['mean_abs'])})\\,px at ${W}\\times{dict(L.RES)[W]}$"
                       + (f" (PENDING: {st[W]['n']}/16 families)" if st[W]['n'] < 16 else "")
                       for W, _ in L.RES) +
           r". The last row of each block is the zero-flow reference."
           + (r" \textbf{DRAFT: Window cells marked -- are still running.}"
              if any(st[W]['n'] < 16 for W in st) else ""))
    o = [r"\begin{table}[t]", r"\centering", r"\caption{" + cap + "}", r"\label{tab:protocol}", r"\small",
         r"\setlength{\tabcolsep}{5pt}", r"\resizebox{\linewidth}{!}{%",
         r"\begin{tabular}{llrrr@{\hskip 1.2em}llrrr}", r"\toprule",
         r"\multicolumn{5}{c}{$480\times270$} & \multicolumn{5}{c}{$960\times432$} \\",
         r"\cmidrule(lr){1-5}\cmidrule(lr){6-10}",
         r"Model & Ckpt & Window & Strat. & $\Delta$ & Model & Ckpt & Window & Strat. & $\Delta$ \\", r"\midrule"]
    a, b = R[480][0], R[960][0]
    for i in range(max(len(a), len(b))):
        la = prot_line(a[i]) if i < len(a) else [""] * 5
        lb_ = prot_line(b[i]) if i < len(b) else [""] * 5
        o.append(" & ".join(la + lb_) + r" \\")
    o.append(r"\midrule")
    o.append(" & ".join(prot_line(R[480][1]) + prot_line(R[960][1])) + r" \\")
    o += [r"\bottomrule", r"\end{tabular}%", "}", "", r"\vspace{0.8em}", r"\begin{tabular}{llrrr}", r"\toprule",
          r"\multicolumn{5}{c}{$1920\times1080$} \\", r"\midrule",
          r"Model & Ckpt & Window & Strat. & $\Delta$ \\", r"\midrule"]
    for r in R[1920][0]:
        o.append(" & ".join(prot_line(r)) + r" \\" + ("  % PENDING window cell" if math.isnan(r["window"]) else ""))
    o.append(r"\midrule")
    o.append(" & ".join(prot_line(R[1920][1])) + r" \\")
    o += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(o), R, st


# --------------------------------------------------------------------------- digest

def has(label, key):
    return key in label.lower()


def digest(cells, rs_rows, r1_rows, R, st, val_rows):
    o = ["# FlyingDrones v2 (backward-GT) benchmark digest", "",
         "Generated by `tables/build_tables.py` from the per-pair CSVs "
         "(full precision). Stratified subset = 999 pairs (reference frames of the 1000-frame split minus "
         "frame 0000). Drone EPE = mean over pairs that contain drone pixels "
         "(885 / 896 / 898 pairs at 270 / 432 / 1080). CI = 95% percentile bootstrap over pairs, "
         "10k resamples, seed 0.", ""]
    missing = [c.tag for c in cells if c.missing]
    o += ["**Missing cells:** " + (", ".join(f"`{t}`" for t in missing) if missing else "none"), ""]
    for W, H in L.RES:
        lb = leaderboard(cells, W)
        z = zero_cell(cells, W)
        o += [f"## Leaderboard {W}x{H}", "",
              "| # | family | ckpt | drone EPE [95% CI] | full EPE [95% CI] |", "|---|---|---|---|---|"]
        for k, c in enumerate(lb):
            o.append(f"| {k+1} | {L.FAMILIES[c.family][0]} | {c.label} | {c.epe_d:.3f} [{c.ci_d[1]:.3f}, "
                     f"{c.ci_d[2]:.3f}] | {c.m('epe','all'):.3f} [{c.ci_a[1]:.3f}, {c.ci_a[2]:.3f}] |")
        if z:
            o.append(f"| ref | zero flow | -- | {z.epe_d:.3f} [{z.ci_d[1]:.3f}, {z.ci_d[2]:.3f}] | "
                     f"{z.m('epe','all'):.3f} [{z.ci_a[1]:.3f}, {z.ci_a[2]:.3f}] |")
        best_full = min((c for c in cells if c.W == W and not c.window and not c.missing and c.family != "zeroflow"),
                        key=lambda c: c.m("epe", "all"))
        o += ["", f"Best overall (drone EPE): **{L.FAMILIES[lb[0].family][0]} {lb[0].label}** "
              f"{lb[0].epe_d:.3f}, {100*(1-lb[0].epe_d/z.epe_d):.1f}% below zero flow ({z.epe_d:.3f}). "
              f"Worst leaderboard family: {L.FAMILIES[lb[-1].family][0]} {lb[-1].epe_d:.3f}. "
              f"Best full-frame EPE: {L.FAMILIES[best_full.family][0]} {best_full.label} "
              f"{best_full.m('epe','all'):.3f}.", ""]
        n_above = [c.tag for c in cells if c.W == W and not c.window and not c.missing and c.family != "zeroflow"
                   and c.epe_d >= z.epe_d]
        o.append(f"Checkpoints at or above zero-flow drone EPE: {len(n_above)}"
                 + (": " + ", ".join(f"`{t}`" for t in n_above) if n_above else ""))
        o.append("")
        o.append("Rank stability (paired bootstrap over the same frames; fixed = leaderboard checkpoints, "
                 "reselect = best checkpoint re-chosen inside each resample):")
        o.append("")
        o.append("| pair | drone EPE diff [95% CI] | P(a beats b) fixed | P reselect |")
        o.append("|---|---|---|---|")
        for r in rs_rows:
            if r["res"] == RES_TXT[W] and r["rank_b"] == r["rank_a"] + 1 and r["rank_a"] <= 5:
                o.append(f"| #{r['rank_a']} {L.FAMILIES[r['family_a']][0]} vs #{r['rank_b']} "
                         f"{L.FAMILIES[r['family_b']][0]} | {r['mean_diff_b_minus_a']:.3f} "
                         f"[{r['diff_ci_lo']:.3f}, {r['diff_ci_hi']:.3f}] | {r['p_a_beats_b_fixed_ckpt']:.3f} | "
                         f"{r['p_a_beats_b_reselect_ckpt']:.3f} |")
        p1 = [r for r in r1_rows if r["res"] == RES_TXT[W] and (r["p_rank1_fixed_ckpt"] > 0.005 or
                                                              r["p_rank1_reselect_ckpt"] > 0.005)]
        o += ["", "P(rank 1): " + "; ".join(f"{L.FAMILIES[r['family']][0]} {r['p_rank1_fixed_ckpt']:.3f} "
                                            f"(reselect {r['p_rank1_reselect_ckpt']:.3f})" for r in p1), ""]

    # within-family spread
    o += ["## Within-family spread (drone EPE max/min over checkpoints)", "",
          "| family | n ckpt | 480x270 | 960x432 | 1920x1080 |", "|---|---|---|---|---|"]
    for fam in L.FAMILY_ORDER:
        row = [L.FAMILIES[fam][0]]
        nck = len(family_ckpt_order(cells, fam))
        row.append(str(nck))
        for W, _ in L.RES:
            v = [c for c in cells if c.family == fam and c.W == W and not c.window and not c.missing]
            if len(v) < 2:
                row.append("--")
                continue
            lo, hi = min(v, key=lambda c: c.epe_d), max(v, key=lambda c: c.epe_d)
            row.append(f"{hi.epe_d/lo.epe_d:.2f} ({lo.epe_d:.2f}-{hi.epe_d:.2f})")
        o.append("| " + " | ".join(row) + " |")
    o.append("")

    # Sintel vs KITTI
    o += ["## Sintel vs KITTI fine-tuning (families releasing both)", "",
          "Per family: best drone EPE among checkpoints whose name contains `sintel` vs `kitti` "
          "(MemFlow `T_things_kitti` counts as KITTI, UniMatch `sintelft` as Sintel). rel = (KITTI - Sintel) / KITTI "
          "(positive = Sintel better).", "",
          "| family | 480x270 S / K | 960x432 S / K | 1920x1080 S / K |", "|---|---|---|---|"]
    tally = {W: [] for W, _ in L.RES}
    for fam in L.FAMILY_ORDER:
        row = [L.FAMILIES[fam][0]]
        ok = False
        for W, _ in L.RES:
            v = [c for c in cells if c.family == fam and c.W == W and not c.window and not c.missing]
            s = [c for c in v if has(c.label, "sintel")]
            k = [c for c in v if has(c.label, "kitti")]
            if not s or not k:
                row.append("--")
                continue
            ok = True
            bs, bk = min(s, key=lambda c: c.epe_d), min(k, key=lambda c: c.epe_d)
            rel = (bk.epe_d - bs.epe_d) / bk.epe_d
            tally[W].append(rel)
            row.append(f"{bs.epe_d:.3f} / {bk.epe_d:.3f} ({'S' if rel > 0 else 'K'}, {100*rel:+.1f}%)")
        if ok:
            o.append("| " + " | ".join(row) + " |")
    o.append("")
    for W, H in L.RES:
        t = np.array(tally[W])
        o.append(f"- {W}x{H}: Sintel wins {int((t > 0).sum())}/{t.size}, KITTI wins {int((t < 0).sum())}/{t.size}; "
                 f"median rel. advantage of Sintel {100*np.median(t):+.1f}% (mean {100*np.mean(t):+.1f}%)")
    o.append("")

    # MOF vs BOF
    o += ["## VideoFlow MOF (5-frame) vs BOF (3-frame), drone EPE", "",
          "| training set | 480x270 MOF / BOF | 960x432 MOF / BOF | 1920x1080 MOF / BOF |", "|---|---|---|---|"]
    vf = {(c.label, c.W): c for c in cells if c.family == "videoflow" and not c.window and not c.missing}
    for ts in ("kitti", "sintel", "things", "things_288960noise"):
        row = [ts]
        for W, _ in L.RES:
            m, b = vf.get((f"MOF_{ts}", W)), vf.get((f"BOF_{ts}", W))
            row.append(f"{m.epe_d:.3f} / {b.epe_d:.3f} ({'MOF' if m.epe_d < b.epe_d else 'BOF'})" if m and b else "--")
        o.append("| " + " | ".join(row) + " |")
    row = ["best of each"]
    for W, _ in L.RES:
        ms = min((c for (l, w), c in vf.items() if w == W and l.startswith("MOF")), key=lambda c: c.epe_d)
        bs = min((c for (l, w), c in vf.items() if w == W and l.startswith("BOF")), key=lambda c: c.epe_d)
        row.append(f"{ms.label} {ms.epe_d:.3f} / {bs.label} {bs.epe_d:.3f}")
    o += ["| " + " | ".join(row) + " |", ""]

    # aerial / TartanAir-trained
    o += ["## TartanAir-trained families (SEA-RAFT, WAFT): leaderboard rank of 16", ""]
    for fam in ("searaft", "waft"):
        parts = []
        for W, H in L.RES:
            lb = leaderboard(cells, W)
            k = [c.family for c in lb].index(fam)
            parts.append(f"{W}x{H}: #{k+1} ({lb[k].label}, {lb[k].epe_d:.3f})")
        o.append(f"- {L.FAMILIES[fam][0]}: " + "; ".join(parts))
    o.append("")

    # protocol
    o += ["## Protocol sensitivity (window = reference frames 0001-0500 vs stratified 999 pairs)", "",
          "Delta = Strat - Window (drone EPE, leaderboard checkpoint held fixed).", ""]
    for W, H in L.RES:
        s = st[W]
        z = R[W][1]
        o.append(f"- {W}x{H}: n={s['n']}/16 families; median |Delta| {s['median_abs']:.3f}, mean |Delta| "
                 f"{s['mean_abs']:.3f}, mean signed Delta {s['mean']:+.3f}, median signed {s['median']:+.3f} px; "
                 f"zero flow window {z['window']:.3f} vs strat {z['strat']:.3f}")
    o.append("")
    for W, H in L.RES:
        rows = R[W][0]
        o += [f"### {W}x{H}", "", "| family | ckpt | window [95% CI] | strat | Delta |", "|---|---|---|---|---|"]
        for r in rows:
            wv = "--" if math.isnan(r["window"]) else f"{r['window']:.3f} [{r['window_ci_lo']:.3f}, {r['window_ci_hi']:.3f}]"
            o.append(f"| {L.FAMILIES[r['family']][0]} | {r['ckpt']} | {wv} | {r['strat']:.3f} | "
                     f"{'--' if math.isnan(r['delta']) else format(r['delta'], '+.3f')} |")
        # rank agreement
        ok = [r for r in rows if not math.isnan(r["window"])]
        if len(ok) >= 3:
            from_w = np.argsort(np.argsort([r["window"] for r in ok]))
            from_s = np.argsort(np.argsort([r["strat"] for r in ok]))
            rho = np.corrcoef(from_w, from_s)[0, 1]
            o += ["", f"Spearman rank correlation window vs strat: {rho:.3f} (n={len(ok)})"]
        o.append("")

    # validation
    flagged = [r for r in val_rows if r.get("flags")]
    o += ["## Validation flags (from validate.py)", ""]
    for r in flagged:
        o.append(f"- `{r['tag']}`: {r['flags']}")
    o.append("")
    return "\n".join(o)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables_out")
    a = ap.parse_args()
    D = osp.join(a.out, "data")
    os.makedirs(D, exist_ok=True)
    cells = load_cells()

    # cells.csv
    with open(osp.join(D, "cells.csv"), "w", newline="") as fh:
        cols = ["tag", "family", "ckpt", "res", "window", "missing", "n_pairs", "n_drone_pairs"] + \
               [f"{m}_{g}" for g in ("all", "drone") for m in L.METRICS] + \
               ["epe_drone_ci_lo", "epe_drone_ci_hi", "epe_all_ci_lo", "epe_all_ci_hi"]
        w = csv.writer(fh)
        w.writerow(cols)
        for c in cells:
            base = [c.tag, c.family, c.label, RES_TXT[c.W], int(c.window), int(c.missing)]
            if c.missing:
                w.writerow(base + [0, 0] + [""] * (len(cols) - 8))
                continue
            w.writerow(base + [len(c.pairs), int((~np.isnan(c.d)).sum())] +
                       [f"{c.m(m, g):.6f}" for g in ("all", "drone") for m in L.METRICS] +
                       [f"{c.ci_d[1]:.6f}", f"{c.ci_d[2]:.6f}", f"{c.ci_a[1]:.6f}", f"{c.ci_a[2]:.6f}"])

    # leaderboard.csv
    with open(osp.join(D, "leaderboard.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["res", "rank", "family", "class", "ckpt", "tag", "epe_drone", "ci_lo", "ci_hi", "epe_all",
                    "epe_all_ci_lo", "epe_all_ci_hi", "tiled_1080"])
        for W, _ in L.RES:
            for k, c in enumerate(leaderboard(cells, W)):
                w.writerow([RES_TXT[W], k + 1, c.family, L.FAMILIES[c.family][1], c.label, c.tag,
                            f"{c.epe_d:.6f}", f"{c.ci_d[1]:.6f}", f"{c.ci_d[2]:.6f}", f"{c.m('epe','all'):.6f}",
                            f"{c.ci_a[1]:.6f}", f"{c.ci_a[2]:.6f}", int(W == 1920 and L.FAMILIES[c.family][3])])
            z = zero_cell(cells, W)
            w.writerow([RES_TXT[W], "ref", "zeroflow", "-", "zero flow", z.tag, f"{z.epe_d:.6f}",
                        f"{z.ci_d[1]:.6f}", f"{z.ci_d[2]:.6f}", f"{z.m('epe','all'):.6f}", f"{z.ci_a[1]:.6f}",
                        f"{z.ci_a[2]:.6f}", 0])

    rs_rows, r1_rows = [], []
    rank_stability(cells, rs_rows, r1_rows)
    for name, rows in (("rank_stability.csv", rs_rows), ("rank1_probability.csv", r1_rows)):
        with open(osp.join(D, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    with open(osp.join(a.out, "tab_agg_full.tex"), "w") as fh:
        fh.write(agg_table(cells, "all"))
    with open(osp.join(a.out, "tab_agg_drone.tex"), "w") as fh:
        fh.write(agg_table(cells, "drone"))
    for W, _ in L.RES:
        with open(osp.join(a.out, f"tab_summary_best_drone_{RES_LB[W]}.tex"), "w") as fh:
            fh.write(leaderboard_table(cells, W))
    ptex, R, st = protocol_table(cells)
    with open(osp.join(a.out, "tab_protocol.tex"), "w") as fh:
        fh.write(ptex)
    with open(osp.join(D, "protocol.csv"), "w", newline="") as fh:
        rows = [r for W in R for r in R[W][0] + [R[W][1]]]
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    vp = osp.join(D, "validation.csv")
    val_rows = list(csv.DictReader(open(vp))) if osp.exists(vp) else []
    with open(osp.join(a.out, "DIGEST.md"), "w") as fh:
        fh.write(digest(cells, rs_rows, r1_rows, R, st, val_rows))
    print("wrote tables to", a.out)
    for W in st:
        print(W, st[W])


if __name__ == "__main__":
    main()
