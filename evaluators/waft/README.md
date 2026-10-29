# WAFT evaluator

| | |
|---|---|
| Upstream | https://github.com/princeton-vl/WAFT (BSD 3-Clause); uses the DepthAnythingV2 encoder (https://github.com/DepthAnything/Depth-Anything-V2, code Apache 2.0, ViT-S weights Apache 2.0) |
| Tested upstream commit | `8dd41723f566a5cd5e2d715284dc06a4c96dfabc` |
| Paper checkpoints | `tar-c-t`, `tar-c-t-kitti`, `tar-c-t-sintel`, `tar-c-t-spring-1080p`, `tar-c-t-spring-540p` (upstream Google Drive model zoo) with the matching upstream `config/<name>.json` |
| Pairing | two-frame, `(I_t, I_{t-1})` via `fd_eval_common.list_samples(root, TWO_FRAME)` |
| Image resize | PIL `Image.resize((W, H))` (bicubic) |
| 1920x1080 | tiled, `--tile 432 960` (upstream `InferenceWrapper(tiling=True, pad_to_train_size=True)`) |

## Files in this folder

| this folder | goes to (`$UPSTREAM` = WAFT checkout) | what it is |
|---|---|---|
| `scripts/analyze_drone_epe.py` | `$UPSTREAM/scripts/` | the evaluator used for the paper (verbatim; only the usage example in the docstring was edited) |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/scripts/` | shared loader / metrics / split filter |
| `commands.sh` | run from `$UPSTREAM` | every paper cell |

No other file is needed. The evaluator only uses upstream APIs (`config.parser.json_to_args`,
`model.fetch_model`, `inference_tools.InferenceWrapper`). On CPU, it gave
bit-identical flow on an unmodified upstream checkout and on the in-house fork used for
the paper, both untiled and tiled (max abs difference 0.0, `tar-c-t`).

## Install

```bash
git clone https://github.com/princeton-vl/WAFT.git && cd WAFT
git checkout 8dd41723f566a5cd5e2d715284dc06a4c96dfabc
export UPSTREAM=$PWD
pip install -r requirements.txt        # upstream: python 3.12, PyTorch 2.7, CUDA 12.8; xformers optional
# DepthAnythingV2 ViT-S weights, needed when the model is constructed (path relative to the CWD):
mkdir -p depth-anything-ckpts && wget -O depth-anything-ckpts/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
# download the WAFT checkpoints into $UPSTREAM/models (Google Drive link in upstream README)
mkdir -p scripts && cp $FD_REPO/evaluators/waft/scripts/analyze_drone_epe.py scripts/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py scripts/
```

The paper numbers were produced on `pytorch/pytorch:2.8.0-cuda12.8-cudnn9` with
`numpy==1.26.4`, `opencv-python-headless`, and `safetensors==0.5.3`.

## Run

```bash
export FD_DATA=/path/to/datasets FD_REPO=/path/to/flyingdrones-code FD_OUT=/path/to/results
export CKPT=$UPSTREAM/models
cd $UPSTREAM            # must be the CWD: upstream loads depth-anything-ckpts/ relative to it
bash $FD_REPO/evaluators/waft/commands.sh
```

Outputs are the summary CSV (`--output_csv`) and the per-pair CSV (`--pairs_csv`).
Weights go through `fd_eval_common.load_state_dict_checked` (604/604 keys; raises on
any mismatch).

Notes
* The evaluator always builds `InferenceWrapper(scale=0, ...)`, so the config's `scale`
  (`-1` in `tar-c-t-spring-540p.json`) is **not** applied. All WAFT cells run at the
  evaluation resolution. This matches the paper runs.
* Untiled runs use `train_size = image_size`, `pad_to_train_size=False`.
* On CPU without xformers, upstream DepthAnythingV2 falls back to standard attention when
  xformers is not installed. If xformers is installed but you run on CPU, uninstall
  it or patch `MemEffAttention.forward` to fall back when `x.device.type != "cuda"`.
