# VideoFlow evaluator (BOF and MOF)

| | |
|---|---|
| Upstream | https://github.com/XiaoyuShi97/VideoFlow |
| License | Apache 2.0 (see `THIRD_PARTY.md`) |
| Upstream commit used | `51489304db6e75fbdd9ff64d4656c1d120b5a673` (upstream HEAD, "fix typo", 2023-09-20). Our bundled copy of `core/`, `configs/` and `alt_cuda_corr/` is byte-identical to this commit, apart from the patch below. |
| Upstream files modified | `core/Networks/BOFNet/corr.py`. Its unconditional `import alt_cuda_corr` is made optional (`patches/videoflow_bofnet_corr.patch`). The paper configs use `corr_fn = "default"`, so the CUDA extension is never called. |
| Paper checkpoints | `{BOF,MOF}_{kitti,sintel,things,things_288960noise}.pth` (8 files) from the upstream model folder linked in the VideoFlow README |
| Cells | 8 checkpoints x 3 resolutions, plus 3 window-protocol rows (`commands.sh`) |

## Layout

The evaluator expects the VideoFlow source in a sibling folder called `VideoFlow/`.
It resolves `scripts/../VideoFlow`, so `$UPSTREAM` is a small work directory, not the
VideoFlow clone itself:

```
$UPSTREAM/
  scripts/analyze_drone_epe.py      <- this folder
  scripts/fd_eval_common.py         <- ../_common/
  scripts/drone_split.py            <- ../_common/
  VideoFlow/                        <- git clone of XiaoyuShi97/VideoFlow (+ patch)
```

## Install

```bash
mkdir videoflow-eval && cd videoflow-eval && export UPSTREAM=$PWD
git clone https://github.com/XiaoyuShi97/VideoFlow VideoFlow
git -C VideoFlow checkout 51489304db6e75fbdd9ff64d4656c1d120b5a673
git -C VideoFlow apply $FD_REPO/evaluators/videoflow/patches/videoflow_bofnet_corr.patch
mkdir scripts
cp $FD_REPO/evaluators/videoflow/scripts/analyze_drone_epe.py scripts/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py scripts/
# Python >= 3.10, PyTorch >= 2.0
pip install torch torchvision "numpy<2" scipy matplotlib tensorboard tqdm pillow \
            opencv-python-headless yacs loguru einops "timm==0.4.12" imageio
```

`timm==0.4.12` is required. VideoFlow's custom forward pass reaches into the internals
of `twins_svt_large` (`patch_embeds`, `pos_drops`, `blocks`, `pos_block`), and newer
timm versions changed those. Download the checkpoints into, for example,
`$UPSTREAM/VideoFlow_ckpt/`.

## Run the paper cells

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/VideoFlow_ckpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # optional, helps MOF at 1080p
cd $UPSTREAM
bash $FD_REPO/evaluators/videoflow/commands.sh
```

Outputs are `$FD_OUT/summary/<tag>.csv` (one summary row) and `$FD_OUT/pairs/<tag>.csv`
(one row per pair).

## Protocol details

* **Pairing.** `flow/<t>_flow.npy` is backward flow t -> t-1. The model gets a genuine
  time-reversed window from the full sequence. Edge frames are clamped, and frame
  0000 is never a reference.
  * **BOF** (`--mode BOF`, 3 frames): window `[t+1, t, t-1]`; read `flow_pre[:, 0]`.
    BOFNet outputs `stack([flow_23, flow_21])`, so slot 0 is centre -> third frame,
    which is t -> t-1.
  * **MOF** (`--mode MOF`, 5 frames): window `[t+2, t+1, t, t-1, t-2]`; read forward
    slot `(N-3)//2 = 1`, i.e. centre -> next = t -> t-1. Upstream
    `evaluate_MOFNet.py` uses the same `gt_index`.
* **Configs.** All BOF checkpoints use `configs/sintel_submission.py`, and all MOF
  checkpoints use `configs/multiframes_sintel_submission.py` with `input_frames=5`.
  `image_size` is set to the evaluation size.
* **Resizing.** Images are resized with PIL's default resampler (bicubic). GT flow is
  resized bilinearly (`align_corners=False`) and scaled by W/W0 and H/H0. The drone
  mask is resized with nearest neighbour.
* **Tiling.** 1920x1080 uses Gaussian-blended tiles (`--tile_size 432 960 --tile_overlap 40`).
  The lower resolutions run untiled.
* **Weights.** Checkpoints are loaded through
  `fd_eval_common.load_state_dict_checked` (strips `module.`), which aborts on any
  key mismatch.
* **Split.** `--split_file $FD_REPO/splits/nature3_split_frames.txt` gives 999 pairs.
  The window protocol uses `nature3_window500_frames.txt`.
