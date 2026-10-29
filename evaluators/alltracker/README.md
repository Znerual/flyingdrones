# AllTracker evaluator (extra, not part of the paper benchmark)

> **AllTracker is not one of the 16 families or 88 checkpoints in the paper's tables**,
> and `reproduce/manifest.tsv` has no AllTracker rows. It is included as an additional
> example of plugging a dense point tracker into the same FlyingDrones protocol. It is
> treated as a two-frame flow model on `(I_t, I_{t-1})`.

| | |
|---|---|
| Upstream | https://github.com/aharley/alltracker (MIT, Copyright (c) 2025 Adam W. Harley) |
| Tested upstream commit | `e7553135e7b361590dbccd10e2b274b024f41cd6` |
| Checkpoints | `alltracker.pth`, `alltracker_tiny.pth` from https://huggingface.co/aharley/alltracker (`tiny` in the file name selects `Net(seqlen=16, use_basicencoder=True, no_split=True)`) |
| Pairing | two-frame clip `[I_t, I_{t-1}]` through `Net.forward`; the prediction is the flow from the query frame `I_t` to `I_{t-1}` |
| Image resize | PIL `Image.resize((W, H))` (bicubic) |
| Device | CUDA only: upstream `Net.get_fmaps` calls `.cuda()`, so the script rejects `--device cpu` |

## Files

| this folder | goes to (`$UPSTREAM` = alltracker checkout) |
|---|---|
| `scripts/analyze_drone_epe.py` | `$UPSTREAM/scripts/` (verbatim; only the usage example in the docstring was edited) |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/scripts/` |

## Install and run

```bash
git clone https://github.com/aharley/alltracker.git && cd alltracker
git checkout e7553135e7b361590dbccd10e2b274b024f41cd6
pip install -r requirements.txt            # upstream: python 3.12
mkdir -p ckpts && wget -P ckpts https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth
cp $FD_REPO/evaluators/alltracker/scripts/analyze_drone_epe.py scripts/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py scripts/

for HW in "270 480" "432 960" "1080 1920"; do
  set -- $HW
  python scripts/analyze_drone_epe.py --ckpt ckpts/alltracker.pth \
      --datasets_path $FD_DATA --root Nature3 --image_size $1 $2 \
      --split_file $FD_REPO/splits/nature3_split_frames.txt \
      --output_csv $FD_OUT/summary/alltracker__alltracker__$2x$1.csv \
      --pairs_csv  $FD_OUT/pairs/alltracker__alltracker__$2x$1.csv
done
```

Options: `--iters` (default 4) and `--num_workers`. Weights are loaded with
`fd_eval_common.load_state_dict_checked`, which raises on any key mismatch. The script
does not tile. If the GPU runs out of memory it aborts instead of skipping pairs.
