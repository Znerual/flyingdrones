# FastFlowNet evaluator

| | |
|---|---|
| Upstream | https://github.com/ltkong218/FastFlowNet (base commit `484108870e4ced569d42ecc33faa32c5a46e5ac8`) |
| Upstream license | MIT (Copyright 2022 Lingtong Kong), see `models/LICENSE` |
| Paper checkpoints | upstream `checkpoints/`: `fastflownet_chairs`, `fastflownet_things3d`, `fastflownet_ft_mix`, `fastflownet_ft_sintel`, `fastflownet_ft_kitti` (`.pth`) |
| Resolutions | 480x270, 960x432, 1920x1080 (all untiled) |

## Contents

| file | role |
|---|---|
| `evaluate.py` | FlyingDrones evaluator (backward-flow pairing, paper metrics, per-pair CSV) |
| `models/FastFlowNet_pt.py` | FastFlowNet with a pure-PyTorch correlation layer, equivalent to `SpatialCorrelationSampler(1, 9, 1, 0, 1)`. It replaces upstream `models/FastFlowNet_v2.py`, so no CUDA extension has to be built. The layer names are the upstream ones and the released weights load unchanged. MIT, derived from upstream. |
| `commands.sh` | The exact command for every paper cell (from `reproduce/manifest.tsv`) |

## Install and setup

The paper environment used Python 3.10+, torch 2.11, numpy 2.x and opencv-python-headless 4.13.

```bash
git clone https://github.com/ltkong218/FastFlowNet.git $UPSTREAM
git -C $UPSTREAM checkout 484108870e4ced569d42ecc33faa32c5a46e5ac8
pip install torch torchvision numpy opencv-python-headless tqdm
cp $FD_REPO/evaluators/fastflownet/evaluate.py $UPSTREAM/
cp $FD_REPO/evaluators/fastflownet/models/FastFlowNet_pt.py $UPSTREAM/models/
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py $UPSTREAM/
```

| from this repo | to `$UPSTREAM/` |
|---|---|
| `evaluate.py` | `evaluate.py` |
| `models/FastFlowNet_pt.py` | `models/FastFlowNet_pt.py` |
| `_common/fd_eval_common.py`, `_common/drone_split.py` | repo root (next to `evaluate.py`) |

## Run

```bash
export FD_DATA=/path/to/datasets FD_OUT=/path/to/results CKPT=$UPSTREAM/checkpoints
cd $UPSTREAM && bash $FD_REPO/evaluators/fastflownet/commands.sh
```

## Notes

- **Pairing.** `(I_t, I_{t-1})` against the backward ground truth. Frame 0000 is skipped, which leaves 999 pairs.
- **Resizing.** Images are read with `cv2.imread` (BGR to RGB) and resized with **`cv2.INTER_AREA`**. GT flow is resized bilinearly (`align_corners=False`) with per-axis magnitude scaling, and the mask with nearest-neighbour.
- **Inference** follows upstream `demo.py`: subtract the joint RGB mean, pad to a multiple of 64, upsample the output and multiply by `DIV_FLOW = 20`.
- **Weight guard.** `fd_eval_common.load_state_dict_checked` raises on any missing or unexpected key (104/104).
- **OOM.** A pair that runs out of memory is skipped and counted in `samples_skipped`.
- **Outputs.** A summary CSV (`--output_csv`) and a per-pair CSV (`--pairs_csv`).
