"""Aggregate the fine-tuning results into one table (stdout, TSV).

usage: python ft_table.py <ft_results_dir> > <ft_results_dir>/ft_table.tsv
<ft_results_dir> holds, per arm (raft, gmflow, ffpp, videoflow_mof):
    summary/<arm>__init__{test,val}.csv   canonical evaluator on the init checkpoint
    summary/<arm>__{best,final}__test.csv canonical evaluator on <name>_best.pth / <name>.pth
    <arm>_train_summary.tsv               from ft_parse_log.py <arm> <ft_results_dir>
"""
import csv
import os
import sys

R = sys.argv[1]
INIT = {"raft": "raft-things", "gmflow": "gmflow_chairs", "ffpp": "things_288960", "videoflow_mof": "MOF_sintel"}
RUN = {}   # optional arm -> run-tag mapping (e.g. for a re-run under a different tag)
RES = {"raft": "960x432", "gmflow": "1280x720", "ffpp": "960x432", "videoflow_mof": "960x432"}


def row(arm, v, s):
    p = os.path.join(R, "summary", f"{arm}__{v}__{s}.csv")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return next(csv.DictReader(fh))


def summ(arm):
    p = os.path.join(R, f"{arm}_train_summary.tsv")
    return dict(l.rstrip("\n").split("\t", 1) for l in open(p)) if os.path.exists(p) else {}


cols = ["arm", "init", "res", "base_drone", "base_full", "ft_drone", "ft_full", "rel_drone_%", "rel_full_%",
        "ft_epe_n_drone", "ft_px1_drone", "ft_px3_drone", "base_epe_n_drone", "base_px1_drone", "base_px3_drone",
        "final_drone", "final_full", "best_step", "val0_drone", "val_first", "val_best", "val_last",
        "train_first", "train_last", "test_pairs"]
print("\t".join(cols))
for arm in INIT:
    run = RUN.get(arm, arm)
    b, f, fin, s = row(arm, "init", "test"), row(run, "best", "test"), row(run, "final", "test"), summ(run)
    g = lambda r, k: (r or {}).get(k, "")
    rel = lambda k: f"{100 * (float(f[k]) / float(b[k]) - 1):+.1f}" if b and f else ""
    out = [run, INIT[arm], RES[arm], g(b, "epe_drone"), g(b, "epe"), g(f, "epe_drone"), g(f, "epe"),
           rel("epe_drone"), rel("epe"), g(f, "epe_norm_drone"), g(f, "px1_err_drone"), g(f, "px3_err_drone"),
           g(b, "epe_norm_drone"), g(b, "px1_err_drone"), g(b, "px3_err_drone"),
           g(fin, "epe_drone"), g(fin, "epe"), s.get("best_step", ""), g(row(arm, "init", "val"), "epe_drone"),
           f"{s.get('val_first_step', '')}:{s.get('val_drone_first', '')}",
           f"{s.get('best_step', '')}:{s.get('val_drone_best', '')}",
           f"{s.get('val_last_step', '')}:{s.get('val_drone_last', '')}",
           f"{s.get('train_first_step', '')}:{s.get('train_epe_first', '')}",
           f"{s.get('train_last_step', '')}:{s.get('train_epe_last', '')}",
           f"{g(b, 'samples_processed')}/{g(f, 'samples_processed')}"]
    print("\t".join(str(x) for x in out))
