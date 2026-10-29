# PWC-Net evaluator

| | |
|---|---|
| Upstream | https://github.com/NVlabs/PWC-Net (weights: `PyTorch/pwc_net.pth.tar`, `PyTorch/pwc_net_chairs.pth.tar`) |
| Upstream license | CC BY-NC-SA 4.0 (Copyright 2018 NVIDIA Corporation), non-commercial use only |
| Paper checkpoints | `pwc_net` (FlyingChairs + FlyingThings3D), `pwc_net_chairs` (FlyingChairs) |
| Resolutions | 480x270, 960x432, 1920x1080 (all untiled) |

This folder is self-contained: the upstream repository only supplies the two released
weight files. No upstream code has to be checked out or built.

## Contents

| file | role |
|---|---|
| `scripts/analyze_drone_epe.py` | FlyingDrones evaluator (backward-flow pairing, paper metrics, per-pair CSV) |
| `pwcnet/model.py`, `pwcnet/__init__.py` | Pure-PyTorch re-implementation of the PWC-DC network with unfold-based correlation, so no CUDA extension has to be compiled. Layer names match the NVlabs PyTorch model, so the released `.pth.tar` files load unchanged. It is a derivative of the NVlabs PWC-Net code and is therefore distributed under CC BY-NC-SA 4.0 (`pwcnet/LICENSE.md`), not under the license of this repository. |
| `config/drones-{S,M,L}.json` | Evaluation resolution: S = 270x480, M = 432x960, L = 1080x1920 (H x W). `datasets_path` is left empty because the command line always passes `--datasets_path`. |
| `commands.sh` | The exact command for every paper cell, generated from `reproduce/manifest.tsv` |

## Install

Python 3.10-3.13. The paper environment used torch 2.8, numpy 2.x, Pillow 12 and
opencv-python-headless 4.10.0.84.

```bash
pip install "torch>=2.7,<3" torchvision numpy pillow opencv-python-headless==4.10.0.84 tqdm scipy
```

## Setup

```bash
export UPSTREAM=$FD_REPO/evaluators/pwc-net      # run from this folder
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py \
   $UPSTREAM/scripts/
export CKPT=/path/to/pwc-net-weights             # pwc_net.pth.tar, pwc_net_chairs.pth.tar
```

| from this repo | to |
|---|---|
| `evaluators/_common/fd_eval_common.py`, `evaluators/_common/drone_split.py` | `$UPSTREAM/scripts/` |
| `$UPSTREAM` itself | on `PYTHONPATH`. The script also inserts its parent dir, so `pwcnet` is importable either way. |

## Run

```bash
export FD_DATA=/path/to/datasets   # contains Nature3/{images,flow,labels}
export FD_OUT=/path/to/results
cd $UPSTREAM && PYTHONPATH=$UPSTREAM bash $FD_REPO/evaluators/pwc-net/commands.sh
```

A GPU is used when available (`--cpu` forces CPU). The 1920x1080 cells run untiled at full resolution.

## Notes

- **Pairing.** `Nature3/flow/<t>_flow.npy` is backward flow t -> t-1, so the network is run on `(I_t, I_{t-1})`. Frame 0000 is never a reference, which leaves 999 pairs from the 1000-frame split.
- **Resizing.** Images use PIL `Image.resize` (default BICUBIC). GT flow is resized bilinearly (`align_corners=False`) with u scaled by W/1920 and v by H/1080. The drone mask uses nearest-neighbour resizing.
- **Inference.** Inputs are scaled to [0,1] and zero-padded to a multiple of 64. The x20 de-normalised level-2 flow is bilinearly upsampled.
- **Weight guard.** Weights load through `fd_eval_common.load_state_dict_checked`, which raises on any missing or unexpected key. The released files carry a `deconv2.*` layer that PWC-DC never uses, and exactly those keys are dropped first (126/128 checkpoint tensors loaded).
- **Outputs.** `--output_csv` is a one-row summary (paper metrics, all pixels + drone pixels). `--pairs_csv` has one row per reference frame, which the CI/bootstrap tables use.
