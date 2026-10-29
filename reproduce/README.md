# Reproducing the benchmark tables

`manifest.tsv` lists every evaluation run of the paper: 88 checkpoints x 3 resolutions
(264 cells) + 3 zero-flow rows on the stratified subset (`protocol = stratified`), and the
51 leaderboard cells re-run on the contiguous window of frames 0001-0500
(`protocol = window500`, protocol-sensitivity table). Columns:

| column | meaning |
|---|---|
| `idx`, `tag` | row id; `tag` = `<family>__<checkpoint>__<W>x<H>[__win500]` names the output files |
| `family`, `evaluator_dir` | model family and the folder under `evaluators/` |
| `H`, `W` | evaluation resolution |
| `script`, `ckpt`, `cfg_variant` | evaluator script (relative to the upstream root), checkpoint, config |
| `command` | the exact command line, run from the upstream checkout root |

Commands use four environment variables: `FD_DATA` (contains `Nature3/`), `FD_REPO` (this
repository), `FD_OUT` (results; `summary/<tag>.csv` and `pairs/<tag>.csv` are written there)
and `CKPT` (the family's checkpoint folder). The same commands are available per model as
`evaluators/<dir>/commands.sh`, or filtered with

```bash
python reproduce/make_commands.py --family raft --H 270
```

## Steps

1. Download FlyingDrones from Kaggle and set `FD_DATA` to the folder that contains `Nature3/`.
2. For each family, prepare the upstream checkout and environment as described in
   `evaluators/<dir>/README.md` and download the released checkpoints into `$CKPT`.
3. Run the commands (`bash $FD_REPO/evaluators/<dir>/commands.sh` from the upstream root, or a job
   array: `slurm_array_example.sh`). Every run writes one summary row and one per-pair CSV.
4. Compare with the paper numbers: `python reproduce/compare.py $FD_OUT`
   (reference values in `results/cells.csv`, best checkpoint per family in `results/leaderboard.csv`).
5. Rebuild the LaTeX tables (bootstrap CIs, leaderboards, appendix tables, protocol table):

   ```bash
   FD_RESULTS=$FD_OUT python tables/validate.py --out tables_out/data
   FD_RESULTS=$FD_OUT python tables/build_tables.py --out tables_out
   ```

   See `tables/README.md` for the real-footage and fine-tuning tables.

Runtime: the 1920x1080 cells of the large models need a 24-48 GB GPU (tiled where noted);
PyFlow is CPU-only and slow at 1920x1080 (the `accurate` preset at 1920x1080 is listed in the
manifest but not reported in the paper).
