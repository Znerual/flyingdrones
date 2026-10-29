# FlowFormer++ evaluator

FlyingDrones evaluation wrapper for **FlowFormer++** (paper class T, transformer).

| | |
|---|---|
| Upstream | https://github.com/XiaoyuShi97/FlowFormerPlusPlus |
| License | Apache 2.0 (upstream). See `THIRD_PARTY.md`. |
| Upstream base commit | `c33de90f35af3fac1a55de6eac58036dd8ffb3b3` (2024-01-26) |
| Paper checkpoints | 4: `kitti.pth`, `sintel.pth`, `things.pth`, `things_288960.pth` |
| Checkpoint source | Google Drive folder linked in the upstream README (https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI) |

## Files

| File in this folder | Destination in the upstream checkout (`$UPSTREAM`) | Notes |
|---|---|---|
| `evaluate.py` | `$UPSTREAM/evaluate.py` | new file; upstream ships `evaluate_FlowFormer_tile.py` but no `evaluate.py`, so nothing is overwritten (if your checkout has one, back it up first) |
| `configs/things_288960.py` | `$UPSTREAM/configs/things_288960.py` | config for the released `things_288960.pth` (image size 288x960); upstream ships no config of that name |
| `datasets/__init__.py`, `datasets/drones.py` | `$UPSTREAM/datasets/` | legacy loader that `evaluate.py` imports (`from datasets import SyntheticDrones`); it is **not** used for the evaluation (the pairs come from `fd_eval_common.list_samples`), but the import must resolve. The package shadows upstream `core/datasets.py` only for this script. |
| `../_common/fd_eval_common.py` | `$UPSTREAM/fd_eval_common.py` | shared pairing / metrics / per-pair dump |
| `../_common/drone_split.py` | `$UPSTREAM/drone_split.py` | split filter |
| `commands.sh` | (run from `$UPSTREAM`) | exact command lines for every paper cell |

## Install

```bash
git clone https://github.com/XiaoyuShi97/FlowFormerPlusPlus.git && cd FlowFormerPlusPlus
git checkout c33de90f35af3fac1a55de6eac58036dd8ffb3b3
export UPSTREAM=$PWD
# environment used for the paper: Python 3.10+, PyTorch 2.x; timm MUST be pinned to 0.4.12
pip install "torch>=2.0" "torchvision>=0.15" "numpy>=1.24" opencv-python-headless==4.10.0.84 \
    timm==0.4.12 "yacs>=0.1.8" "loguru>=0.7" einops imageio scipy matplotlib tqdm tensorboard
cp -r $FD_REPO/evaluators/flowformerplusplus/{evaluate.py,datasets,configs} "$UPSTREAM/"
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py "$UPSTREAM/"
mkdir -p checkpoints   # download kitti.pth, sintel.pth, things.pth, things_288960.pth into it
```

(`cp -r .../configs` only adds `configs/things_288960.py`; the upstream configs are kept.)

## Run

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/{images,flow,labels}
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/checkpoints
cd "$UPSTREAM"
bash $FD_REPO/evaluators/flowformerplusplus/commands.sh   # 12 paper cells + 3 window cells
```

A single cell:

```bash
python evaluate.py --cfg configs.submissions --model $CKPT/sintel.pth \
    --datasets_path $FD_DATA --root Nature3 --image_size 1080 1920 \
    --split_file $FD_REPO/splits/nature3_split_frames.txt \
    --job_id flowformerplusplus__sintel__1920x1080 \
    --output_csv $FD_OUT/summary/flowformerplusplus__sintel__1920x1080.csv \
    --pairs_csv  $FD_OUT/pairs/flowformerplusplus__sintel__1920x1080.csv
```

Checkpoint -> config used in the paper:

| checkpoint | `--cfg` |
|---|---|
| `kitti.pth` | `configs.kitti` |
| `sintel.pth` | `configs.submissions` |
| `things.pth` | `configs.things` |
| `things_288960.pth` | `configs.things_288960` |

## Notes

* **`sintel.pth` needs `configs.submissions`** (the GMA-style decoder). With `configs.sintel` only
  389/475 keys match and part of the decoder would stay randomly initialised; with `configs.submissions`
  all 415/415 keys load. The evaluator now loads weights through
  `fd_eval_common.load_state_dict_checked`, which raises on any missing or unexpected key, so a wrong
  config fails loudly instead of silently evaluating a partly random network.
* **Pairing.** `flow/<t>_flow.npy` is backward flow t -> t-1; the model is run on `(I_t, I_{t-1})`
  with `I_t` as the reference. Frame 0000 is never a reference (999 pairs on the stratified split).
* **Resizing.** OpenCV read, `cv2.INTER_AREA` image resize to `--image_size H W`; GT flow bilinear
  (`align_corners=False`) and rescaled by `W/1920`, `H/1080`; drone mask (label > 0) nearest-neighbour.
* **Tiling.** The model input size stays at the config's `image_size` (432x960, or 288x960 for
  `things_288960`); any evaluation image larger than that is processed in overlapping tiles of that size
  with uniform blending (no resize of tiles). This is automatic: all 1920x1080 cells are tiled, and
  `things_288960` is also tiled at 432x960.
* **Outputs.** One summary row in `--output_csv`, one row per pair in `--pairs_csv`.
* `train_nature3.py` (fine-tuning, `finetune/flowformerplusplus/`) imports this `evaluate.py`
  (`DroneFlowDataset`, `infer_flow`, `_compute_metrics`), so validation uses exactly this evaluation path.
