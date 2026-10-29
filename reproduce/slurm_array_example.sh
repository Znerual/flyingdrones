#!/bin/bash
# Generic example: run every benchmark cell as one task of a Slurm job array.
# Adapt partition/account/resources to your cluster. One manifest row per task:
#
#   N=$(grep -vc '^#' reproduce/manifest.tsv)
#   sbatch --array=1-$N reproduce/slurm_array_example.sh
#
# Required environment (export before sbatch, or edit below):
#   FD_REPO      this repository
#   FD_DATA      directory that contains Nature3/
#   FD_OUT       results directory (summary/ and pairs/ are created)
#   UPSTREAM_DIR directory holding one prepared upstream checkout per evaluator dir,
#                named like evaluators/<dir> (raft, sea-raft, ..., pyflow)
#   CKPT_DIR     directory holding one checkpoint folder per evaluator dir (same names)
# Each checkout must be prepared as described in evaluators/<dir>/README.md
# (patched evaluator + fd_eval_common.py + drone_split.py copied in, environment installed).
# If every model lives in its own Python environment or container, wrap the `python`
# call below accordingly (e.g. `conda run -n <env>` or `apptainer exec --nv <image>`).
#SBATCH --job-name=flyingdrones
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=flyingdrones-%A_%a.out
set -euo pipefail
: "${FD_REPO:?}" "${FD_DATA:?}" "${FD_OUT:?}" "${UPSTREAM_DIR:?}" "${CKPT_DIR:?}"
ROW=${SLURM_ARRAY_TASK_ID:?run as a job array}
MAN=$FD_REPO/reproduce/manifest.tsv

line=$(awk -F'\t' -v i="$ROW" '$1==i' "$MAN")
[ -n "$line" ] || { echo "no manifest row $ROW"; exit 2; }
IFS=$'\t' read -r IDX TAG FAMILY EVDIR PROTO H W SCRIPT CKPT_REL CFGV COMMAND <<< "$line"

export CKPT=$CKPT_DIR/$EVDIR
mkdir -p "$FD_OUT/summary" "$FD_OUT/pairs"
if [ -s "$FD_OUT/summary/$TAG.csv" ]; then echo "skip $TAG (exists)"; exit 0; fi

echo "[$TAG] $(date -u) host=$(hostname)"
cd "$UPSTREAM_DIR/$EVDIR"
# the manifest command uses $FD_DATA, $FD_REPO, $FD_OUT and $CKPT
eval "$COMMAND"
