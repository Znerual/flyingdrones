#!/usr/bin/env python3
"""Remaining paper tables: real footage, fine-tuning, main-text leaderboard.

usage: python tables/build_extras.py --tables tables_out --real <real results dir> \
           --ft-table <ft_results_dir>/ft_table.tsv --ft-real ft_vs_baseline.tsv
Writes into --tables (next to the output of build_tables.py):

  tab_real_all.tex        replaces tab:real-all     (41 ckpts, 21 cams, sample std)
  tab_real_strata.tex     replaces tab:real-strata
  tab_ft.tex              replaces tab:ft-curves + tab:ft-test (recipe table + results table)
  tab_ft_main.tex         small main-text fine-tune table (tab:ft-main)
  tab_leaderboard_main.tex  main-text per-family leaderboard (tab:leaderboard-main)

Inputs (read-only):
  <tables>/data/{cells,leaderboard}.csv          from build_tables.py
  <real>/overall_checkpoint.csv,
  <real>/strata_hit50.tsv                        from real_footage/collect.py + aggregate_all.py
                                                 (--real = their --results dir)
  ft_table.tsv                                   from ft_table.py
  ft_vs_baseline.tsv                             real-footage scores of the init and fine-tuned
      checkpoints, one row per (arm, variant in {baseline, finetuned}); columns
      arm (RAFT | FF++ | GMFlow | VideoFlow MOF), variant, system, nss_d, nss_d_std, hit50,
      contrast, cos, pt_epe, m1_subpx, m2_low, m3_mid, m4_high (Hit@50 per stratum)
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

OUT = DATA = REAL = FT_TABLE = FT_REAL = None  # set in main()

RES = ["480x270", "960x432", "1920x1080"]
RES_TEX = {"480x270": r"480$\times$270", "960x432": r"960$\times$432",
           "1920x1080": r"1920$\times$1080"}

FAM_TEX = {  # synthetic family key -> display (as in the v2 appendix leaderboards)
    "arflow": "ARFlow", "fastflownet": "FastFlowNet", "flowdiffusion": "FlowDiffusion (DDVM)",
    "flowformerplusplus": "FlowFormer++", "flownet2": "FlowNet2", "gmflow": "GMFlow",
    "liteflownet3": "LiteFlowNet3", "memflow": "MemFlow", "pwcnet": "PWC-Net",
    "pyflow": "PyFlow (C2F)", "raft": "RAFT", "searaft": "SEA-RAFT", "unimatch": "UniMatch",
    "vcn": "VCN", "videoflow": "VideoFlow", "waft": "WAFT",
}
MULTIFRAME = {"memflow", "videoflow"}

# real system id -> (synthetic family key, synthetic ckpt name, real-table display ckpt)
REAL_MAP = {
    "FastFlowNet__fastflownet_chairs": ("fastflownet", "fastflownet_chairs", "fastflownet_chairs"),
    "FastFlowNet__fastflownet_ft_kitti": ("fastflownet", "fastflownet_ft_kitti", "fastflownet_ft_kitti"),
    "FastFlowNet__fastflownet_ft_sintel": ("fastflownet", "fastflownet_ft_sintel", "fastflownet_ft_sintel"),
    "FastFlowNet__fastflownet_things3d": ("fastflownet", "fastflownet_things3d", "fastflownet_things3d"),
    "arflow__pwclite_ar": ("arflow", "Sintel/pwclite_ar", "Sintel/pwclite_ar"),
    "flowdiffusion__pipeline-305000": ("flowdiffusion", "pipeline-305000", "pipeline-305000"),
    "flowformerplusplus__kitti": ("flowformerplusplus", "kitti", "kitti"),
    "flowformerplusplus__sintel": ("flowformerplusplus", "sintel", "sintel"),
    "flowformerplusplus__things_288960": ("flowformerplusplus", "things_288960", "things_288960"),
    "flownet2__FlowNet2-C_checkpoint": ("flownet2", "FlowNet2-C_checkpoint", "FlowNet2-C_checkpoint"),
    "flownet2__FlowNet2-S_checkpoint": ("flownet2", "FlowNet2-S_checkpoint", "FlowNet2-S_checkpoint"),
    "gmflow__gmflow_chairs-1d776046": ("gmflow", "gmflow_chairs-1d776046", "gmflow_chairs-1d776046"),
    "gmflow__gmflow_kitti-285701a8": ("gmflow", "gmflow_kitti-285701a8", "gmflow_kitti-285701a8"),
    "gmflow__gmflow_sintel-0c07dcb3": ("gmflow", "gmflow_sintel-0c07dcb3", "gmflow_sintel-0c07dcb3"),
    "gmflow__gmflow_things-e9887eda": ("gmflow", "gmflow_things-e9887eda", "gmflow_things-e9887eda"),
    "gmflow__gmflow_with_refine_chairs-020cc9be": ("gmflow", "gmflow_with_refine_chairs-020cc9be", "gmflow_with_refine_chairs-020cc9be"),
    "gmflow__gmflow_with_refine_kitti-8d3b9786": ("gmflow", "gmflow_with_refine_kitti-8d3b9786", "gmflow_with_refine_kitti-8d3b9786"),
    "liteflownet3__liteflownet3s-kitti": ("liteflownet3", "liteflownet3s-kitti", "liteflownet3s-kitti"),
    "liteflownet3__liteflownet3s-sintel": ("liteflownet3", "liteflownet3s-sintel", "liteflownet3s-sintel"),
    "memflow__MemFlowNet_kitti": ("memflow", "MemFlowNet_kitti", "MemFlowNet_kitti"),
    "memflow__MemFlowNet_sintel": ("memflow", "MemFlowNet_sintel", "MemFlowNet_sintel"),
    "memflow__MemFlowNet_spring": ("memflow", "MemFlowNet_spring", "MemFlowNet_spring"),
    "memflow__MemFlowNet_things": ("memflow", "MemFlowNet_things", "MemFlowNet_things"),
    "pwc-net__pwc_net": ("pwcnet", "pwc_net", "pwc_net"),
    "pwc-net__pwc_net_chairs": ("pwcnet", "pwc_net_chairs", "pwc_net_chairs"),
    "pyflow__fast": ("pyflow", "fast", "fast"),
    "raft__raft-chairs": ("raft", "raft-chairs", "raft-chairs"),
    "raft__raft-kitti": ("raft", "raft-kitti", "raft-kitti"),
    "raft__raft-sintel": ("raft", "raft-sintel", "raft-sintel"),
    "raft__raft-things": ("raft", "raft-things", "raft-things"),
    "sea-raft__Tartan-C-T-TSKH432x960-M": ("searaft", "Tartan-C-T-TSKH432x960-M (sintel)", "Tartan-C-T-TSKH432x960-M (sintel)"),
    "unimatch__gmflow-scale1-things-e9887eda": ("unimatch", "gmflow-scale1-things-e9887eda", "gmflow-scale1-things-e9887eda"),
    "unimatch__gmflow-scale2-regrefine6-kitti15-25b554d7": ("unimatch", "gmflow-scale2-regrefine6-kitti15-25b554d7", "gmflow-scale2-regrefine6-kitti15-25b554d7"),
    "vcn__finetune_139999": ("vcn", "vcn_small_weights/chairs/finetune_139999", "vcn_small_weights/chairs/finetune_139999"),
    "vcn__finetune_141999": ("vcn", "vcn_weights/chairs/finetune_141999", "vcn_weights/chairs/finetune_141999"),
    "vcn__finetune_149999": ("vcn", "vcn_weights/kitti-ft-trainval/finetune_149999", "vcn_weights/kitti-ft-trainval/finetune_149999"),
    "vcn__finetune_67999": ("vcn", "vcn_weights/sintel-ft-trainval/finetune_67999", "vcn_weights/sintel-ft-trainval/finetune_67999"),
    "vcn__finetune_239999": ("vcn", "vcn_weights/things/finetune_239999", "vcn_weights/things/finetune_239999"),
    "videoflow__MOF_sintel__MOF": ("videoflow", "MOF_sintel", "MOF_sintel"),
    "waft__tar-c-t-kitti": ("waft", "tar-c-t-kitti", "tar-c-t-kitti"),
    "waft__tar-c-t-sintel": ("waft", "tar-c-t-sintel", "tar-c-t-sintel"),
}
# real families tiled at 1280x720 (uniform-weighted overlapping tiles, see caption)
REAL_TILED = {"gmflow", "unimatch", "flowformerplusplus"}


def tt(s):
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def esc(s):
    return s.replace("_", r"\_")


def fam_tex(fam, dagger=True):
    s = FAM_TEX[fam]
    if dagger and fam in MULTIFRAME:
        s += r"$^\dagger$"
    return s


# ------------------------------------------------------------------ load
def load_cells():
    cells = {}
    for r in csv.DictReader((DATA / "cells.csv").open()):
        if r["window"] != "0" or r["missing"] != "0":
            continue
        cells[(r["family"], r["ckpt"], r["res"])] = float(r["epe_drone"])
    return cells


def load_leaderboard():
    lb = defaultdict(dict)
    for r in csv.DictReader((DATA / "leaderboard.csv").open()):
        lb[r["res"]][r["family"]] = r
    return lb


def load_real():
    rows = list(csv.DictReader((REAL / "overall_checkpoint.csv").open()))
    for r in rows:
        for k in list(r):
            if k != "system":
                r[k] = float(r[k])
        fam, sck, disp = REAL_MAP[r["system"]]
        r["fam"], r["syn_ckpt"], r["disp"] = fam, sck, disp
    rows.sort(key=lambda r: (-round(r["nss_disk"], 6), FAM_TEX[r["fam"]], r["disp"]))
    assert len(rows) == 41, len(rows)
    return rows


def load_tsv(p):
    return list(csv.DictReader(p.open(), delimiter="\t"))


# ------------------------------------------------------------------ real tables
def write_real_all(real):
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Real-footage motion-alignment metrics for all 41 published checkpoints, "
             r"mean $\pm$ sample standard deviation (ddof\,=\,1) over all 21 labelled cameras of datasets 1--4 "
             r"(100 displacement-stratified frame pairs per camera), ordered by drone-disk NSS. "
             r"Inference at $1280\times720$ (VideoFlow MOF at $960\times540$), every field rescaled to $1920\times1080$. "
             r"GMFlow, UniMatch and FlowFormer++ exceed their $432\times960$ tile size at this resolution "
             r"(FlowFormer++ \texttt{things\_288960}: $288\times960$) and are run with overlapping tiles "
             r"blended with uniform weights; all other models run in a single pass. "
             r"UniMatch \texttt{gmflow-scale1-things} and GMFlow \texttt{gmflow\_things} share identical weights "
             r"and hence identical scores. VCN paths are relative to \texttt{vcn\_weights/} (\texttt{small/} = \texttt{vcn\_small\_weights/}). "
             r"Higher is better except pt-EPE.}")
    L.append(r"\label{tab:real-all}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{llrrrrrr}")
    L.append(r"\toprule")
    L.append(r"Family & Checkpoint & $n$ & $\mathrm{NSS}_D$ & Hit@50 & Contr. & cos & pt-EPE \\")
    L.append(r"\midrule")
    for r in real:
        fam = FAM_TEX[r["fam"]].replace(" (DDVM)", "").replace(" (C2F)", "")
        ck = tt(r["disp"].replace("vcn_small_weights/", "small/").replace("vcn_weights/", ""))
        if r["fam"] == "videoflow":
            ck = tt("MOF_sintel") + " (MOF)"
        if r["fam"] in REAL_TILED:
            ck += r"$^\ddagger$"
        L.append(f"{fam:<13} & {ck} & {int(r['n_cameras'])} & "
                 f"{r['nss_disk']:5.1f}$\\pm${r['nss_disk_std']:4.1f} & {r['hit@50']:.2f} & "
                 f"{r['contrast']:6.1f} & {r['pt_cos']:5.2f} & {r['pt_epe']:5.2f} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}%")
    L.append(r"}")
    L.append(r"\par\smallskip{\footnotesize $^\ddagger$\,tiled inference ($432\times960$ tiles, $288\times960$ for "
             r"\texttt{things\_288960}), uniform tile blending.}")
    L.append(r"\end{table}")
    (OUT / "tab_real_all.tex").write_text("\n".join(L) + "\n")


STRATA_ROWS = [
    ("memflow__MemFlowNet_spring", r"MemFlow / \texttt{spring}"),
    ("pyflow__fast", r"PyFlow / \texttt{fast}"),
    ("flowformerplusplus__things_288960", r"FlowFormer++ / \texttt{things}"),
    ("gmflow__gmflow_with_refine_kitti-8d3b9786", r"GMFlow / \texttt{with\_refine\_kitti}"),
    ("videoflow__MOF_sintel__MOF", r"VideoFlow MOF / \texttt{sintel}"),
    ("raft__raft-sintel", r"RAFT / \texttt{sintel}"),
]


def write_real_strata():
    st = {r["system"]: r for r in load_tsv(REAL / "strata_hit50.tsv")}
    ncam = st["memflow__MemFlowNet_spring"]
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Localisation (Hit@50) versus drone image-motion stratum for representative checkpoints, "
             r"mean over the labelled cameras that contain pairs in the stratum "
             f"({ncam['ncam_m1_subpx']} cameras for sub-px, low and mid; {ncam['ncam_m4_high']} for high). "
             r"Localisation rises with motion for the transferable models, including RAFT; "
             r"the one failing checkpoint shown here (VideoFlow \texttt{sintel}) stays low at every level.}")
    L.append(r"\label{tab:real-strata}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{6pt}")
    L.append(r"\begin{tabular}{lrrrr}")
    L.append(r"\toprule")
    L.append(r"Checkpoint & sub-px & low & mid & high \\")
    L.append(r" & ($<1$\,px) & ($1$--$3$) & ($3$--$8$) & ($>8$) \\")
    L.append(r"\midrule")
    for sysid, lab in STRATA_ROWS:
        s = st[sysid]
        v = [float(s[k]) for k in ("m1_subpx", "m2_low", "m3_mid", "m4_high")]
        L.append(f"{lab:<38} & " + " & ".join(f"{x:.2f}" for x in v) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    (OUT / "tab_real_strata.tex").write_text("\n".join(L) + "\n")
    return st


# ------------------------------------------------------------------ fine-tune tables
FT_ARMS = [  # arm key in ft_table, arm key in ft_vs_baseline, display, recipe
    ("ffpp", "FF++", "FlowFormer++", dict(init="things_288960", res=(960, 432), crop=(768, 384),
                                         bs=1, accum=6, lr=r"$1.25\times10^{-4}$", gamma="0.85")),
    ("gmflow", "GMFlow", "GMFlow", dict(init="gmflow_chairs", res=(1280, 720), crop=(896, 512),
                                       bs=1, accum=1, lr=r"$5.0\times10^{-5}$", gamma="0.90")),
    ("raft", "RAFT", "RAFT", dict(init="raft-things", res=(960, 432), crop=(768, 384),
                                 bs=4, accum=1, lr=r"$2.5\times10^{-5}$", gamma="0.85")),
    ("videoflow_mof", "VideoFlow MOF", "VideoFlow MOF", dict(init="MOF_sintel", res=(960, 432), crop=(768, 384),
                                                             bs=1, accum=4, lr=r"$6.25\times10^{-5}$", gamma="0.85")),
]


def load_ft():
    ft = {r["arm"]: r for r in load_tsv(FT_TABLE)}
    rv = defaultdict(dict)
    for r in load_tsv(FT_REAL):
        rv[r["arm"]][r["variant"]] = r
    out = []
    for arm, rarm, disp, rec in FT_ARMS:
        f = ft[arm]
        bd, fd = float(f["base_drone"]), float(f["ft_drone"])
        bf, ff = float(f["base_full"]), float(f["ft_full"])
        d = dict(arm=arm, disp=disp, rec=rec, bd=bd, fd=fd, bf=bf, ff=ff,
                 rel_d=100 * (fd - bd) / bd, rel_f=100 * (ff - bf) / bf,
                 best=int(f["best_step"]), val0=float(f["val0_drone"]),
                 val_best=float(f["val_best"].split(":")[1]), val_last=float(f["val_last"].split(":")[1]),
                 tr_first=float(f["train_first"].split(":")[1]), tr_last=float(f["train_last"].split(":")[1]),
                 final_drone=float(f["final_drone"]),
                 b_nss=float(rv[rarm]["baseline"]["nss_d"]), f_nss=float(rv[rarm]["finetuned"]["nss_d"]),
                 b_nss_sd=float(rv[rarm]["baseline"]["nss_d_std"]), f_nss_sd=float(rv[rarm]["finetuned"]["nss_d_std"]),
                 b_hit=float(rv[rarm]["baseline"]["hit50"]), f_hit=float(rv[rarm]["finetuned"]["hit50"]),
                 b_strata=[float(rv[rarm]["baseline"][k]) for k in ("m1_subpx", "m2_low", "m3_mid", "m4_high")],
                 f_strata=[float(rv[rarm]["finetuned"][k]) for k in ("m1_subpx", "m2_low", "m3_mid", "m4_high")],
                 b_pt=float(rv[rarm]["baseline"]["pt_epe"]), f_pt=float(rv[rarm]["finetuned"]["pt_epe"]))
        out.append(d)
    return out


def arrow(a, b, fmt):
    return f"${fmt.format(a)}\\rightarrow{fmt.format(b)}$"


def pct(x):
    return f"${'+' if x >= 0 else '-'}{abs(x):.1f}\\%$"


def write_ft(ft):
    L = []
    # ---- recipe / optimisation table (replaces tab:ft-curves)
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Fine-tuning recipe and optimisation behaviour per arm (800-pair FlyingDrones training split, "
             r"1500 AdamW steps, one-cycle schedule, mixed precision; frames resized to the evaluation scale, then randomly cropped). "
             r"\emph{Batch} is per-step batch $\times$ gradient-accumulation steps; $\gamma$ is the sequence-loss weight decay. "
             r"\emph{Val drone-EPE} is the canonical drone-region EPE on the 100-pair validation split at step~0 (public initialisation), "
             r"at the selected checkpoint, and at step 1500; validation runs every 150 steps and the checkpoint minimising it "
             r"(\emph{Best}: its step) is reported.}")
    L.append(r"\label{tab:ft-curves}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{llcccccrrrr}")
    L.append(r"\toprule")
    L.append(r" & & & & & & & & \multicolumn{3}{c}{Val drone-EPE} \\")
    L.append(r"\cmidrule(lr){9-11}")
    L.append(r"Model & Init & Eval res. & Crop & Batch & LR & $\gamma$ & Best & step 0 & best & 1500 \\")
    L.append(r"\midrule")
    for d in ft:
        r = d["rec"]
        L.append(f"{d['disp']} & {tt(r['init'])} & ${r['res'][0]}\\times{r['res'][1]}$ & "
                 f"${r['crop'][0]}\\times{r['crop'][1]}$ & ${r['bs']}\\times{r['accum']}$ & {r['lr']} & {r['gamma']} & "
                 f"{d['best']} & {d['val0']:.3f} & {d['val_best']:.3f} & {d['val_last']:.3f} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}%")
    L.append(r"}")
    L.append(r"\end{table}")
    L.append("")
    # ---- results table (replaces tab:ft-test)
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Fine-tuning results, public initialisation (base) $\rightarrow$ fine-tuned (best validation checkpoint). "
             r"\emph{Synthetic}: held-out FlyingDrones test split (100 pairs never seen in training or validation) at the arm's evaluation "
             r"resolution, canonical evaluator; drone-region and full-frame EPE. "
             r"\emph{Real}: the same two checkpoints on the real-footage motion-alignment protocol of Section~\ref{sec:real} "
             r"(21 cameras, mean; $1280\times720$, VideoFlow MOF $960\times540$).}")
    L.append(r"\label{tab:ft-test}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\resizebox{\linewidth}{!}{%")
    L.append(r"\begin{tabular}{lcrrrrr}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{3}{c}{Synthetic test (FlyingDrones)} & \multicolumn{2}{c}{Real footage} \\")
    L.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}")
    L.append(r"Model & Res. & Drone EPE & $\Delta$ & Full EPE & $\mathrm{NSS}_D$ & Hit@50 \\")
    L.append(r"\midrule")
    for d in ft:
        r = d["rec"]
        L.append(f"{d['disp']} & ${r['res'][0]}\\times{r['res'][1]}$ & {arrow(d['bd'], d['fd'], '{:.3f}')} & {pct(d['rel_d'])} & "
                 f"{arrow(d['bf'], d['ff'], '{:.3f}')} & {arrow(d['b_nss'], d['f_nss'], '{:.1f}')} & "
                 f"{arrow(d['b_hit'], d['f_hit'], '{:.2f}')} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}%")
    L.append(r"}")
    L.append(r"\end{table}")
    (OUT / "tab_ft.tex").write_text("\n".join(L) + "\n")

    # ---- main-text version
    M = []
    M.append(r"\begin{table}[t]")
    M.append(r"\centering")
    M.append(r"\caption{Fine-tuning on 800 FlyingDrones pairs: public initialisation $\rightarrow$ fine-tuned. "
             r"Synthetic: drone-region EPE on the 100-pair held-out test split ($\downarrow$). "
             r"Real: motion alignment on 21 real cameras ($\uparrow$). Per-arm recipes and full-frame EPE in "
             r"Tables~\ref{tab:ft-curves} and~\ref{tab:ft-test}.}")
    M.append(r"\label{tab:ft-main}")
    M.append(r"\small")
    M.append(r"\setlength{\tabcolsep}{4pt}")
    M.append(r"\begin{tabular}{lccc}")
    M.append(r"\toprule")
    M.append(r"Model & Synth.\ drone EPE ($\Delta$) & Real $\mathrm{NSS}_D$ & Real Hit@50 \\")
    M.append(r"\midrule")
    for d in ft:
        M.append(f"{d['disp']} & {arrow(d['bd'], d['fd'], '{:.2f}')} ({pct(d['rel_d'])}) & "
                 f"{arrow(d['b_nss'], d['f_nss'], '{:.1f}')} & {arrow(d['b_hit'], d['f_hit'], '{:.2f}')} \\\\")
    M.append(r"\bottomrule")
    M.append(r"\end{tabular}")
    M.append(r"\end{table}")
    (OUT / "tab_ft_main.tex").write_text("\n".join(M) + "\n")


# ------------------------------------------------------------------ main leaderboard
def write_leaderboard_main(lb):
    fams = [f for f in lb["480x270"] if f != "zeroflow"]
    rank = {f: {res: int(lb[res][f]["rank"]) for res in RES} for f in fams}
    mean_rank = {f: np.mean([rank[f][r] for r in RES]) for f in fams}
    # tie-break: mean drone EPE normalised by the zero-flow EPE at each resolution
    zf = {res: float(lb[res]["zeroflow"]["epe_drone"]) for res in RES}
    rel = {f: np.mean([float(lb[r][f]["epe_drone"]) / zf[r] for r in RES]) for f in fams}
    order = sorted(fams, key=lambda f: (mean_rank[f], rel[f]))
    top3 = {res: sorted(fams, key=lambda f: float(lb[res][f]["epe_drone"]))[:3] for res in RES}
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Best drone-region EPE (px) per model family at each resolution on the stratified subset (999 flow pairs; "
             r"lower is better). The best checkpoint is "
             r"selected per family and resolution, so it can differ across columns. The three best values per column are in "
             r"\textbf{bold}; families are sorted by their mean rank $\bar r$ over the three resolutions. "
             r"\emph{Cls}: C~= correlation/iterative, P~= classical\,/\,pyramid, E~= efficient\,/\,reduced-supervision, "
             r"T~= transformer\,/\,global matching, M~= temporal\,/\,memory\,/\,generative. "
             r"$\dagger$~more than two input frames; $\ddagger$~tiled inference ($432\times960$ tiles). "
             r"Selected checkpoints, 95\% bootstrap confidence intervals and further metrics are in "
             r"Tables~\ref{tab:summary-best-drone-270x480}--\ref{tab:summary-best-drone-1080x1920}.}")
    L.append(r"\label{tab:leaderboard-main}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{6pt}")
    L.append(r"\begin{tabular}{clrrrr}")
    L.append(r"\toprule")
    L.append(r"Cls & Family & " + " & ".join(RES_TEX[r] for r in RES) + r" & $\bar r$ \\")
    L.append(r"\midrule")
    for f in order:
        cells = []
        for res in RES:
            v = float(lb[res][f]["epe_drone"])
            s = f"{v:.3f}"
            if f in top3[res]:
                s = r"\textbf{" + s + "}"
            if res == "1920x1080" and lb[res][f]["tiled_1080"] == "1":
                s += r"$^\ddagger$"
            else:
                s += r"\phantom{$^\ddagger$}" if res == "1920x1080" else ""
            cells.append(s)
        L.append(f"{lb['480x270'][f]['class']} & {fam_tex(f)} & " + " & ".join(cells) + f" & {mean_rank[f]:.1f} \\\\")
    L.append(r"\midrule")
    zcells = [f"{zf[r]:.3f}" for r in RES]
    zcells[-1] += r"\phantom{$^\ddagger$}"
    L.append(r"-- & Zero flow (reference) & " + " & ".join(zcells) + r" & -- \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    (OUT / "tab_leaderboard_main.tex").write_text("\n".join(L) + "\n")
    return order, mean_rank, rank


# ------------------------------------------------------------------ numbers.md
def corr(a, b):
    rho, prho = stats.spearmanr(a, b)
    tau, ptau = stats.kendalltau(a, b)   # tau-b
    return rho, prho, tau, ptau


def fmt_corr(c, n):
    rho, prho, tau, ptau = c
    return f"rho = {rho:+.3f} (p = {prho:.2g}); tau_b = {tau:+.3f} (p = {ptau:.2g}); n = {n}"


def main():
    global OUT, DATA, REAL, FT_TABLE, FT_REAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tables_out", help="output dir of build_tables.py (data/ inside)")
    ap.add_argument("--real", required=True, help="real-footage results dir (overall_checkpoint.csv, strata_hit50.tsv)")
    ap.add_argument("--ft-table", required=True)
    ap.add_argument("--ft-real", required=True)
    a = ap.parse_args()
    OUT = Path(a.tables)
    DATA = OUT / "data"
    REAL = Path(a.real)
    FT_TABLE, FT_REAL = Path(a.ft_table), Path(a.ft_real)
    cells = load_cells()
    lb = load_leaderboard()
    real = load_real()
    write_real_all(real)
    write_real_strata()
    ft = load_ft()
    write_ft(ft)
    write_leaderboard_main(lb)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*")) if p.name in {
        "tab_real_all.tex", "tab_real_strata.tex", "tab_ft.tex", "tab_ft_main.tex",
        "tab_leaderboard_main.tex"}))


if __name__ == "__main__":
    main()
