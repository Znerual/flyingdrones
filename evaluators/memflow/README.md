# MemFlow evaluator (also used for the zero-flow reference)

| | |
|---|---|
| Upstream | https://github.com/DQiaole/MemFlow |
| License | Apache 2.0 (see `THIRD_PARTY.md`) |
| Upstream commit used | `5486fb88a55b04c2e2b292b3abaa7641d9988831` (upstream HEAD, "typos") |
| Upstream files modified | none. The evaluator only adds files. |
| Paper checkpoints | `MemFlowNet_{things,sintel,kitti,spring}.pth` and `MemFlowNet_T_{things,things_kitti,sintel,kitti}.pth` from the upstream release https://github.com/DQiaole/MemFlow/releases/tag/v1.0.0 |
| Cells | 8 checkpoints x 3 resolutions, plus 3 zero-flow rows and 4 window-protocol rows (`commands.sh`) |

## Files in this folder

| this folder | copy to |
|---|---|
| `scripts/analyze_drone_epe.py` | `$UPSTREAM/scripts/analyze_drone_epe.py` |
| `scripts/download_checkpoints.sh` | `$UPSTREAM/scripts/download_checkpoints.sh` (optional; it downloads the 8 release files to `$UPSTREAM/ckpts/`) |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/scripts/` (next to the evaluator) |

The script puts its own directory first on `sys.path`, so the shared
`fd_eval_common.py` / `drone_split.py` next to it always win.

## Install

```bash
git clone https://github.com/DQiaole/MemFlow && cd MemFlow
git checkout 5486fb88a55b04c2e2b292b3abaa7641d9988831
export UPSTREAM=$PWD
# Python >= 3.10, PyTorch >= 2.0 (CUDA build for GPU runs). We tested a modern stack;
# upstream documents python 3.8 + torch 1.13.
pip install torch torchvision numpy scipy matplotlib tensorboard h5py tqdm einops yacs \
            loguru imageio pillow "timm==0.4.12" opencv-python-headless
# FlashAttention is optional (upstream README); it is not needed for evaluation.
cp $FD_REPO/evaluators/memflow/scripts/*.{py,sh} scripts/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py scripts/
bash scripts/download_checkpoints.sh        # -> ckpts/
```

`timm==0.4.12` is required: newer timm versions change the Twins internals that the
MemFlow encoder relies on.

## Run the paper cells

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/ckpts
cd $UPSTREAM
bash $FD_REPO/evaluators/memflow/commands.sh     # or copy single lines out of it
```

Each run writes one summary row to `$FD_OUT/summary/<tag>.csv` and one row per
evaluated pair to `$FD_OUT/pairs/<tag>.csv`. The per-pair CSV is what the table
builders in `tables/` read.

## Protocol details

* **Pairing.** `flow/<t>_flow.npy` is backward flow t -> t-1. MemFlow sees the
  time-reversed clip `[I_{t+3}, I_{t+2}, I_{t+1}, I_t, I_{t-1}]`
  (`fd_eval_common.MEMFLOW_CLIP`). Edge frames are clamped, and frame 0000 is never
  a reference.
* **Memory.** The script runs one `InferenceCore` per sample and steps through the
  consecutive pairs exactly like upstream `evaluate_MemFlowNet.py`
  (`step(images[:, ti:ti+2], end=(ti == T-2))`), so memory is written on every
  non-final step. Only the last flow, (I_t, I_{t-1}) -> t -> t-1, is scored.
* **Resizing.** Images are resized with PIL's default resampler (bicubic) to
  `--image_size H W`. GT flow is resized bilinearly (`align_corners=False`) and
  scaled by W/W0 and H/H0. The drone mask is resized with nearest neighbour.
* **Tiling.** 1920x1080 uses Gaussian-blended tiled inference (`--tile 432 960`). The
  tiled path keeps one memory core per tile across the clip. 480x270 and 960x432 run
  untiled.
* **Weights.** Checkpoints are loaded through
  `fd_eval_common.load_state_dict_checked`, which aborts on any missing or unexpected
  key.
* **Split.** `--split_file $FD_REPO/splits/nature3_split_frames.txt` gives 999
  evaluated pairs, because frame 0000 is dropped. The window protocol uses
  `nature3_window500_frames.txt`.

## Zero-flow reference

The zero-flow rows of the paper tables (`zeroflow__zero_flow__<W>x<H>`) also come from
this evaluator, run with `--zero_flow --device cpu`. In that mode no checkpoint is
loaded, no image is read, and the prediction is identically 0. The summary row is
labelled `zero_flow`. The commands are the `zeroflow__*` entries in `commands.sh`.
They need the MemFlow checkout only for its imports. The repository's standalone
`examples/zero_flow.py` computes the same numbers without MemFlow.
