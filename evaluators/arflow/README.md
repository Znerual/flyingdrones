# ARFlow evaluator

| | |
|---|---|
| Upstream | https://github.com/lliuz/ARFlow |
| License | MIT (see `THIRD_PARTY.md`) |
| Upstream commit used | `e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e` (upstream HEAD, "Specify align_corners=True for compatibility #15") |
| Upstream files modified | `models/pwclite.py` and `utils/torch_utils.py` (`patches/arflow_eval.patch`, see below) |
| Paper checkpoints | the 9 files shipped in upstream `checkpoints/`: `CityScapes/pwclite_ar`, `KITTI12/pwclite_ar`, `KITTI12/pwclite_ar_mv`, `KITTI15/pwclite_ar`, `KITTI15/pwclite_ar_mv`, `KITTI15/pwclite_raw`, `Sintel/pwclite_ar`, `Sintel/pwclite_ar_mv`, `Sintel/pwclite_raw` (`.tar`) |
| Cells | 9 checkpoints x 3 resolutions, plus 3 window-protocol rows (`commands.sh`) |

## Files in this folder

| this folder | copy to |
|---|---|
| `scripts/analyze_drone_epe.py` | `$UPSTREAM/scripts/analyze_drone_epe.py` |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/scripts/` |
| `patches/arflow_eval.patch` | apply in `$UPSTREAM` with `git apply` |

### What the patch changes

1. `models/pwclite.py`:
   * It imports the pure-PyTorch `correlation_native` instead of the CUDA
     `correlation_package`, so nothing has to be compiled. Upstream ships this
     alternative and mentions it in its README.
   * The pyramid and final upsampling now go to the exact target size, with
     per-axis flow scaling, instead of `scale_factor=2/4`. The evaluator does not pad
     its inputs, and when the input size is not a multiple of the pyramid stride
     (480x270, 1920x1080), the upstream scale-factor path produces feature maps of
     the wrong size.
2. `utils/torch_utils.py`: `torch.load(..., map_location='cpu', weights_only=False)` so
   the original `.tar` checkpoints load on recent PyTorch and on CPU-only machines.

## Install

```bash
git clone https://github.com/lliuz/ARFlow && cd ARFlow
git checkout e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e
export UPSTREAM=$PWD
git apply $FD_REPO/evaluators/arflow/patches/arflow_eval.patch
cp $FD_REPO/evaluators/arflow/scripts/analyze_drone_epe.py scripts/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py scripts/
# Python >= 3.10, PyTorch >= 2.0 (no CUDA extension needed)
pip install torch torchvision numpy scipy pillow opencv-python-headless easydict imageio scikit-image tqdm
```

## Run the paper cells

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/checkpoints     # the checkpoints ship with the upstream repo
cd $UPSTREAM
bash $FD_REPO/evaluators/arflow/commands.sh
```

This evaluator takes the scene folder directly (`--root $FD_DATA/Nature3`) and has
no `--datasets_path`. Outputs are `$FD_OUT/summary/<tag>.csv` and
`$FD_OUT/pairs/<tag>.csv`.

## Protocol details

* **Pairing.** `flow/<t>_flow.npy` is backward flow t -> t-1. The evaluator
  auto-detects the number of input frames and `reduce_dense` from the checkpoint
  weight shapes.
  * 2-frame checkpoints: input `(I_t, I_{t-1})`; read `flows_fw[0]`.
  * 3-frame (`_mv`) checkpoints: window `[I_{t+1}, I_t, I_{t-1}]`
    (`fd_eval_common.ARFLOW_WINDOW`); read `flows_fw[0]`, which is middle -> last
    = t -> t-1 at full resolution.

  Edge frames are clamped, and frame 0000 is never a reference.
* **Resizing.** Images are resized with PIL **bilinear** (set explicitly). GT flow is
  resized bilinearly (`align_corners=False`) and scaled by W/W0 and H/H0. The drone
  mask is resized with nearest neighbour.
* **Tiling.** None at any resolution.
* **Weights.** Checkpoints are loaded through
  `fd_eval_common.load_state_dict_checked`, which aborts on any key mismatch.
* **Split.** `--split_file $FD_REPO/splits/nature3_split_frames.txt` gives 999 pairs.
  The window protocol uses `nature3_window500_frames.txt`.
