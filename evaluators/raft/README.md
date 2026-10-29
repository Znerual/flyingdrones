# RAFT evaluator

| | |
|---|---|
| Upstream | https://github.com/princeton-vl/RAFT (BSD 3-Clause) |
| Tested upstream commit | `3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02` |
| Paper checkpoints | `raft-chairs`, `raft-kitti`, `raft-sintel`, `raft-small`, `raft-things` (upstream `download_models.sh`) |
| Pairing | two-frame, `(I_t, I_{t-1})` via `fd_eval_common.list_samples(root, TWO_FRAME)` |
| Image resize | PIL `Image.resize((W, H))` (bicubic) |
| 1920x1080 | tiled, `--tile 432 960` (Gaussian-blended tiles, 20 px overlap) |

## Files in this folder

| this folder | goes to (`$UPSTREAM` = RAFT checkout) | what it is |
|---|---|---|
| `scripts/analyze_drone_epe.py` | `$UPSTREAM/scripts/analyze_drone_epe.py` | the evaluator used for the paper (verbatim; only the usage example in the docstring was edited) |
| `config/parser.py`, `config/raft-large.json`, `config/raft-small.json` | `$UPSTREAM/config/` | model configs read by `--cfg` (`raft-small.json` for `raft-small`, `raft-large.json` for all other checkpoints) |
| `raft/__init__.py`, `raft/utils/__init__.py`, `raft/utils/utils.py` | `$UPSTREAM/raft/` | compatibility shim, see below |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/scripts/` | shared loader / metrics / split filter (must sit next to the script) |
| `commands.sh` | run from `$UPSTREAM` | every paper cell (generated from `reproduce/manifest.tsv`) |

**Compatibility shim.** The paper runs used an in-house packaging of RAFT in which the
model lives in a `raft/` package, the constructor takes keyword arguments and
`forward(..., test_mode=True)` returns a dict with key `final`. The network code of
that packaging is identical to upstream `core/`. To keep the evaluator verbatim, the
`raft/` shim here wraps the *unmodified* upstream `core/raft.py` with exactly that
interface. We verified on CPU that shim + upstream and the in-house package produce
bit-identical flow (max abs difference 0.0) for `raft-things` and `raft-small`
with the same inputs. Weights are always loaded by the evaluator through
`fd_eval_common.load_state_dict_checked` (the `module.` prefix is stripped, and any missing or
unexpected key raises), so a checkpoint can never be evaluated at random init.

## Install

```bash
git clone https://github.com/princeton-vl/RAFT.git && cd RAFT
git checkout 3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02
export UPSTREAM=$PWD
./download_models.sh                        # -> models/raft-*.pth
pip install torch torchvision numpy scipy pillow opencv-python-headless   # PyTorch >= 1.6
cp -r $FD_REPO/evaluators/raft/{scripts,config,raft} $UPSTREAM/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py $UPSTREAM/scripts/
```

The paper numbers were produced with a `pytorch/pytorch` (CUDA) Docker base image plus
`numpy==1.26.4` and `opencv-python-headless` on Ampere GPUs. `alt_cuda_corr` is not
needed (`alternate_corr: false`).

## Run

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/{images,flow,labels}
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/models
cd $UPSTREAM
bash $FD_REPO/evaluators/raft/commands.sh                     # all 15 + 3 window cells
python $FD_REPO/reproduce/make_commands.py --family raft --H 270   # or print a subset
```

Each run writes a one-row summary CSV (`--output_csv`, columns `job_id` +
`fd_eval_common.SUMMARY_FIELDS`, 4 decimals) and a per-pair CSV (`--pairs_csv`, one row
per reference frame, full precision; the paper tables and bootstrap CIs are computed
from these). With `--split_file splits/nature3_split_frames.txt` 999 pairs are
evaluated, because frame `0000` has no predecessor and is dropped.

Notes
* `scale` is 0 in both configs, so inference runs at the evaluation resolution.
  Predictions are compared at that resolution against the GT flow, which is bilinearly
  resized and rescaled (`u*W/1920`, `v*H/1080`).
* `--image_size H W` overrides the config's `image_size`; the manifest always passes it.
