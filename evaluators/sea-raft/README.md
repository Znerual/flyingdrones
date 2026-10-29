# SEA-RAFT evaluator

| | |
|---|---|
| Upstream | https://github.com/princeton-vl/SEA-RAFT (BSD 3-Clause) |
| Tested upstream commit | `f41d0a7ebff0e09def0f8ba9ebfd961686013b41` |
| Paper checkpoints | `Tartan-C-T-TSKH432x960-M` (sintel), `Tartan-C-T-TSKH-kitti432x960-M`, `Tartan-C-T-TSKH-spring540x960-M` (upstream Google Drive / HuggingFace links in the SEA-RAFT README) |
| Configs | upstream `config/eval/{sintel,kitti,spring}-M.json` (one per checkpoint, chosen with `--variant`) |
| Pairing | two-frame, `(I_t, I_{t-1})` via `fd_eval_common.list_samples(root, TWO_FRAME)` |
| Image resize | PIL `Image.resize((W, H))` (bicubic) |
| 1920x1080 | **not tiled** (full-frame inference, as in the paper) |

## Files in this folder

| this folder | goes to (`$UPSTREAM` = SEA-RAFT checkout) | what it is |
|---|---|---|
| `scripts/analyze_drone_epe.py` | `$UPSTREAM/scripts/` | the evaluator used for the paper (verbatim; only the usage example in the docstring was edited) |
| `searaft/` (`__init__.py`, `_upstream.py`, `raft.py`, `utils/`) | `$UPSTREAM/searaft/` | compatibility shim, see below |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/scripts/` | shared loader / metrics / split filter |
| `commands.sh` | run from `$UPSTREAM` | every paper cell |

**Compatibility shim.** The paper runs used an in-house packaging of SEA-RAFT
(`searaft/` package, keyword-argument constructor `RAFT(**cfg)`). Its RAFT network
and forward pass are the same as upstream `core/`. The `searaft/` shim here builds the
*unmodified* upstream `core/raft.py` model from the same config keywords and re-exports
upstream `core/utils/utils.py` (`InputPadder`) as `searaft.utils.utils`. On CPU, shim +
upstream and the in-house package gave bit-identical flow (max abs difference 0.0) for
all three paper checkpoints with their eval configs. `ModifiedRAFT` / `FocusRAFT`
(imported by the evaluator, never used for released checkpoints) raise
`NotImplementedError`.

## Install

```bash
git clone https://github.com/princeton-vl/SEA-RAFT.git && cd SEA-RAFT
git checkout f41d0a7ebff0e09def0f8ba9ebfd961686013b41
export UPSTREAM=$PWD
pip install -r requirements.txt huggingface_hub     # upstream: python 3.10, PyTorch 2.2
# download the three *-M.pth checkpoints into $UPSTREAM/models (links in upstream README)
cp -r $FD_REPO/evaluators/sea-raft/{scripts,searaft} $UPSTREAM/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py $UPSTREAM/scripts/
```

The paper numbers were produced on `pytorch/pytorch:2.8.0-cuda12.8-cudnn9` with
`numpy==1.26.4`, `opencv-python-headless`, and `safetensors==0.5.3`.

## Run

```bash
export FD_DATA=/path/to/datasets FD_REPO=/path/to/flyingdrones-code FD_OUT=/path/to/results
export CKPT=$UPSTREAM/models
cd $UPSTREAM
bash $FD_REPO/evaluators/sea-raft/commands.sh
```

Every command passes `--variant <sintel|kitti|spring> --cfg config/eval/<variant>-M.json
--ckpt $CKPT/<file>`. An explicit `--cfg`/`--ckpt` wins over the `--variant` defaults,
and a checkpoint/variant mismatch prints a warning. Outputs are the summary CSV
(`--output_csv`) and the per-pair CSV (`--pairs_csv`). Weights are loaded with
`fd_eval_common.load_state_dict_checked`, which raises on any key mismatch (472/472 keys
for all three files).

Notes
* `spring-M.json` has `"scale": -1`. For that checkpoint the evaluator downsamples the
  input by 2 (bilinear, `align_corners=False`), runs the model, then upsamples the
  prediction and multiplies it by 2. `sintel-M` and `kitti-M` use `scale: 0`.
* Use the eval config that belongs to each checkpoint (`sintel-M` for
  `Tartan-C-T-TSKH432x960-M`). Pairing the sintel weights with `spring-M` changes the
  inference scale.
* `--tile TH TW` exists but is not used for any paper cell.
