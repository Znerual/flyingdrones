# FlowDiffusion (DDVM) evaluator

| | |
|---|---|
| Upstream code | https://github.com/DQiaole/FlowDiffusion_pytorch ("open DDVM", an unofficial PyTorch implementation of DDVM, which is cited in the paper as google-research DDVM) |
| License | Apache 2.0 (see `THIRD_PARTY.md`) |
| Upstream commit used | `f834a1e8d70ec6999a976bc2ada11aa142b6bb1b` (upstream HEAD, "Merge pull request #5") |
| Upstream files modified | `local_diffusers/` (`patches/flowdiffusion_local_diffusers.patch`, see below) |
| Paper checkpoints | `pipeline-305000` and `pipeline-900000` from https://github.com/DQiaole/FlowDiffusion_pytorch/releases/tag/v1.0.0 (unzip; each is a diffusers pipeline folder) |
| Cells | 2 checkpoints x 3 resolutions, plus 3 window-protocol rows (`commands.sh`) |

## Files in this folder

| this folder | copy to |
|---|---|
| `evaluate_drones.py` | `$UPSTREAM/evaluate_drones.py` |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/` (next to `evaluate_drones.py`) |
| `patches/flowdiffusion_local_diffusers.patch` | apply in `$UPSTREAM` with `git apply` |

### What the patch changes

1. It adds `local_diffusers/models/platte_unet.py`, a three-line shim that re-exports
   `SRUnet256`. The released `pipeline-900000/model_index.json` names the module
   `local_diffusers.models.platte_unet`, which does not exist upstream, so without
   the shim that checkpoint cannot be loaded.
2. `local_diffusers/pipelines/DDPM/pipeline_ddpm.py` falls back to importing
   `randn_tensor` from `diffusers.utils.torch_utils`, its location in newer diffusers.
3. `local_diffusers/models/imagen_unet.py` uses `torch.nn.functional.scaled_dot_product_attention`
   (torch >= 2.1) instead of the naive O(N^2)-memory attention when neither
   xformers nor flash-attn is installed. This is mathematically the same operation,
   but it avoids out-of-memory errors at large resolutions.

## Install

```bash
git clone https://github.com/DQiaole/FlowDiffusion_pytorch && cd FlowDiffusion_pytorch
git checkout f834a1e8d70ec6999a976bc2ada11aa142b6bb1b
export UPSTREAM=$PWD
git apply $FD_REPO/evaluators/flowdiffusion/patches/flowdiffusion_local_diffusers.patch
cp $FD_REPO/evaluators/flowdiffusion/evaluate_drones.py .
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py .
# Python >= 3.10, PyTorch >= 2.1 with CUDA (bfloat16 inference on GPU)
pip install torch torchvision "diffusers>=0.25" "accelerate>=0.25" safetensors huggingface-hub \
            numpy opencv-python-headless tqdm einops scipy
mkdir -p checkpoints && cd checkpoints
# download pipeline-305000.zip and pipeline-900000.zip from the v1.0.0 release, then:
unzip pipeline-305000.zip && unzip pipeline-900000.zip && cd ..
```

## Run the paper cells

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/checkpoints     # holds pipeline-305000/ and pipeline-900000/
cd $UPSTREAM
bash $FD_REPO/evaluators/flowdiffusion/commands.sh
```

`--dataset_path` must point at the scene folder itself (`$FD_DATA/Nature3`), as in
`commands.sh`. A parent folder holding several scenes also works. Outputs are
`$FD_OUT/summary/<tag>.csv` and `$FD_OUT/pairs/<tag>.csv`.

## Protocol details

* **Pairing.** `flow/<t>_flow.npy` is backward flow t -> t-1. The model is run on
  `(I_t, I_{t-1})` with I_t as the reference frame. Frame 0000 is never a reference.
* **Sampling.** `--ddpm_num_steps 4 --normalize_range`, as in every paper cell. The
  pipeline runs in bfloat16 on GPU (float32 on CPU). DDPM sampling starts from
  unseeded Gaussian noise, so repeated runs differ slightly, in the second or third
  decimal of the per-cell means.
* **Resizing.**
  * Images are read with OpenCV and resized to `--image_size` with `cv2.INTER_AREA`.
  * The model wrapper then resizes both frames bicubically to the training size
    320x448, predicts flow there, and resizes the flow bicubically back to the
    evaluation size. Flow vectors are scaled by W_eval/448 and H_eval/320.
  * GT flow is resized bilinearly (`align_corners=False`) and scaled by W/W0 and H/H0.
    The drone mask is resized with nearest neighbour.
* **Tiling.** None at any resolution.
* **Weights.** After `from_pretrained`, every UNet tensor is compared with the
  `unet/diffusion_pytorch_model.safetensors` file (`_check_unet_weights`). The run
  aborts on any missing, unexpected or differing key. diffusers itself only warns.
* **Out-of-memory.** After the first GPU out-of-memory error, the remaining pairs run
  on the CPU. A pair that runs out of memory again is skipped and counted in
  `samples_skipped`, and it gets no per-pair row.
* **Split.** `--split_file $FD_REPO/splits/nature3_split_frames.txt` gives 999 pairs.
  The window protocol uses `nature3_window500_frames.txt`.
