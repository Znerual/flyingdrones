# Table builders

All table numbers are recomputed from the per-pair CSVs (`$FD_OUT/pairs/<tag>.csv`, full
precision); the evaluators' summary CSVs are rounded to 4 decimals and only cross-checked.

| script | input | output |
|---|---|---|
| `validate.py` | `$FD_RESULTS/{summary,pairs}/`, `reproduce/manifest.tsv` | `<out>/validation.csv`: pair counts vs split, summary == aggregate(pairs), NaNs, cells worse than zero flow, unexpected identical per-pair vectors (GMFlow and UniMatch share three released weight files, which is expected) |
| `build_tables.py` | same | `tab_agg_full.tex`, `tab_agg_drone.tex` (all checkpoints), `tab_summary_best_drone_{270x480,432x960,1080x1920}.tex` (leaderboards with 95 % CI), `tab_protocol.tex` (window vs stratified), `DIGEST.md`, `data/{cells,leaderboard,rank_stability,rank1_probability,protocol}.csv` |
| `ft_parse_log.py`, `ft_table.py` | fine-tuning trainer logs + canonical test/val evaluations | `ft_table.tsv` |
| `build_extras.py` | `data/{cells,leaderboard}.csv`, real-footage results, `ft_table.tsv`, `ft_vs_baseline.tsv` | `tab_real_all.tex`, `tab_real_strata.tex`, `tab_ft.tex`, `tab_ft_main.tex`, `tab_leaderboard_main.tex` |
| `v2lib.py` | -- | shared helpers (family metadata, checkpoint labels, number formatting) |

```bash
export FD_RESULTS=$FD_OUT          # the directory the evaluator commands wrote to
python tables/validate.py    --out tables_out/data
python tables/build_tables.py --out tables_out

# fine-tuning: one results dir with summary/<arm>__{init,best,final}__{test,val}.csv and <arm>_train_log.txt
for arm in raft gmflow ffpp videoflow_mof; do python tables/ft_parse_log.py $arm $FT_RESULTS $arm; done
python tables/ft_table.py $FT_RESULTS > $FT_RESULTS/ft_table.tsv

python tables/build_extras.py --tables tables_out --real results/all \
    --ft-table $FT_RESULTS/ft_table.tsv --ft-real ft_vs_baseline.tsv
```

Conventions (as in the paper): CI = 95 % percentile bootstrap over pairs, 10,000 resamples,
seed 0; drone-region numbers are means over the pairs that contain drone pixels (885 / 896 / 898
of the 999 stratified pairs at 480x270 / 960x432 / 1920x1080); leaderboards keep the checkpoint
with the lowest drone-region EPE per family; italic = best within family and resolution.

`results/cells.csv` and `results/leaderboard.csv` at the repository root are the `data/` files
produced from the paper runs; rebuilding from those runs reproduces the paper's synthetic tables
byte for byte.
