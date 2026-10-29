# LiteFlowNet3 evaluator

| | |
|---|---|
| Upstream | https://github.com/twhui/LiteFlowNet3 (base commit `e9f21c8281fc1328f969e40b19dcb46532091def`) |
| Upstream license | Custom non-commercial license (Tak-Wai Hui). Commercial use requires a prior arrangement with the author. |
| Model implementation | PyTorch port in [PTLFlow](https://github.com/hmorimitsu/ptlflow) **v0.4.2** (`ptlflow.get_model`) |
| Paper checkpoints | `liteflownet3-sintel`, `liteflownet3-kitti`, `liteflownet3s-sintel`, `liteflownet3s-kitti` |
| Resolutions | 480x270, 960x432 untiled; 1920x1080 with `--tile 432 960` (Gaussian-blended tiles, 20 px min overlap) |

The upstream LiteFlowNet3 repository is Caffe-only. The benchmark runs the PTLFlow
PyTorch implementation with PTLFlow's converted weights. The upstream checkout is only
needed for its license and documentation.

## Checkpoints

These are the PTLFlow v0.4.2 release files, renamed by dropping the hash suffix. The
SHA-256 prefix of each file equals the suffix in its original name.

| file used (`$CKPT/...`) | PTLFlow release asset | `--model_name` |
|---|---|---|
| `liteflownet3-sintel.ckpt` | `weights1/liteflownet3-sintel-d985929f.ckpt` | `liteflownet3` |
| `liteflownet3-kitti.ckpt` | `weights1/liteflownet3-kitti-b5d32443.ckpt` | `liteflownet3_pseudoreg` |
| `liteflownet3s-sintel.ckpt` | `weights1/liteflownet3s-sintel-89793e34.ckpt` | `liteflownet3s` |
| `liteflownet3s-kitti.ckpt` | `weights1/liteflownet3s-kitti-5dffb261.ckpt` | `liteflownet3s_pseudoreg` |

Release URL prefix: `https://github.com/hmorimitsu/ptlflow/releases/download/`.

## Contents

| file | role |
|---|---|
| `evaluate.py` | FlyingDrones evaluator (backward-flow pairing, paper metrics, per-pair CSV, optional tiling) |
| `config/drones-{S,M,L}.json` | Evaluation resolution: S = 270x480, M = 432x960, L = 1080x1920 (H x W) |
| `commands.sh` | The exact command for every paper cell (from `reproduce/manifest.tsv`) |

## Install

Python 3.10 or newer (the script uses `str | None`). The paper environment used torch 2.11,
ptlflow 0.4.2, lightning 2.6 and opencv-python-headless 4.10.0.84.

```bash
pip install torch torchvision "ptlflow==0.4.2" numpy pillow tqdm opencv-python-headless==4.10.0.84
```

## Setup and run

```bash
export UPSTREAM=/path/to/LiteFlowNet3               # or any working directory
cp $FD_REPO/evaluators/liteflownet3/evaluate.py $UPSTREAM/
cp -r $FD_REPO/evaluators/liteflownet3/config $UPSTREAM/
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py $UPSTREAM/
export FD_DATA=/path/to/datasets FD_OUT=/path/to/results CKPT=/path/to/ptlflow-ckpts
cd $UPSTREAM && bash $FD_REPO/evaluators/liteflownet3/commands.sh
```

## Notes

- **Pairing.** The model is run on `(I_t, I_{t-1})` with `I_t` as reference. Frame 0000 is skipped, which leaves 999 pairs.
- **Resizing.** The resolution comes from `--cfg` (there is no `--image_size` flag). Images use PIL `Image.resize` (default BICUBIC). GT flow is resized bilinearly with per-axis magnitude scaling, and the mask with nearest-neighbour. Images are scaled to [0,1], and `preds['flows'][:, 0]` is scored.
- **Weight guard.** After `ptlflow.get_model(name, ckpt)` the script reloads the checkpoint file and checks that every model tensor is present and equal to it, and raises otherwise.
- **OOM.** A pair that runs out of CUDA memory is skipped and counted in `samples_skipped`. After 5 OOMs the resolution is aborted and an all-NaN summary row is written; check `samples_processed` = 999.
- **Outputs.** A summary CSV (`--output_csv`) and a per-pair CSV (`--pairs_csv`).
