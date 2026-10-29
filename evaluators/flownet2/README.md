# FlowNet2 evaluator

| | |
|---|---|
| Model (paper Table asset_licenses) | FlowNet2, https://github.com/lmb-freiburg/flownet2 (BSD 2-Clause, non-commercial research use only per the repository documentation) |
| Code actually run | PyTorch port https://github.com/NVIDIA/flownet2-pytorch (Apache 2.0) at commit `bce69c881ad2b11bef1e6ce9fd0dfd45bd8a1756`, plus `flownet2-pytorch.patch` from this folder |
| Paper checkpoints | `FlowNet2`, `FlowNet2-C`, `FlowNet2-CS`, `FlowNet2-CSS-ft-sd`, `FlowNet2-S`, `FlowNet2-SD` (`*_checkpoint.pth.tar`, Google Drive links in the flownet2-pytorch README; converted Caffe weights, so the Freiburg licence terms apply) |
| Pairing | two-frame, `(I_t, I_{t-1})` via `fd_eval_common.list_samples(root, TWO_FRAME)` |
| Image resize | PIL `Image.resize((W, H), Image.BILINEAR)` |
| 1920x1080 | not tiled |

## Files in this folder

| this folder | goes to (`$UPSTREAM` = flownet2-pytorch checkout) | what it is |
|---|---|---|
| `scripts/analyze_drone_epe.py` | `$UPSTREAM/scripts/` | the evaluator used for the paper (verbatim; only the usage example in the docstring was edited) |
| `config/parser.py`, `config/drones-*.json` | `$UPSTREAM/config/` | `--cfg` files; key `model` selects the class in `models.py` (`FlowNet2`, `FlowNet2C`, `FlowNet2CS`, `FlowNet2CSS`, `FlowNet2S`, `FlowNet2SD`), `rgb_max: 255`, `fp16: false` |
| `flownet2-pytorch.patch` | apply in `$UPSTREAM` (`git apply`) | changes to upstream `models.py` and `networks/`, see below |
| `../_common/fd_eval_common.py`, `../_common/drone_split.py` | `$UPSTREAM/scripts/` | shared loader / metrics / split filter |
| `commands.sh` | run from `$UPSTREAM` | every paper cell |

### What `flownet2-pytorch.patch` changes

1. **`models.py`, `FlowNet2SD.forward` (eval branch): `flow2 * div_flow` becomes `flow2 / div_flow`.**
   This is an upstream bug (NVIDIA/flownet2-pytorch #37): the fused FlowNet2 graph divides
   the SD output by `div_flow`, but the stand-alone `FlowNet2SD` multiplies, which makes
   predictions about 400x too large. **The paper's FlowNet2-SD numbers use the corrected
   version.** The patch also uses plain absolute imports instead of the try/except
   relative-import block.
2. **CUDA extensions (`correlation`, `resample2d`, `channelnorm`).** These build with
   current PyTorch: C++17, `data_ptr<T>()` in place of the removed `.data<T>()`, and
   updated `-gencode` lists (sm_60 to sm_90 plus compute_90 PTX).
   `networks/cuda_stubs/nv/target` is a header stub for builds without the full CUDA
   toolkit headers.
3. **Fallbacks.** Each op wrapper first tries the installed extension, then a JIT build
   (`torch.utils.cpp_extension.load`), and finally the pure-PyTorch implementations in
   the new file `networks/pytorch_fallbacks.py`. The fallbacks run on CPU too and are
   slower. The paper numbers were produced with the compiled CUDA extensions.

The patch was checked to reproduce the files used for the paper byte for byte on a clean
checkout of the commit above.

## Install

```bash
git clone https://github.com/NVIDIA/flownet2-pytorch.git && cd flownet2-pytorch
git checkout bce69c881ad2b11bef1e6ce9fd0dfd45bd8a1756
export UPSTREAM=$PWD
git apply $FD_REPO/evaluators/flownet2/flownet2-pytorch.patch
pip install torch torchvision numpy scipy pillow opencv-python-headless tqdm
for p in correlation_package resample2d_package channelnorm_package; do
  (cd networks/$p && python setup.py install); done        # needs nvcc; optional (fallbacks)
# download the six *_checkpoint.pth.tar files into $UPSTREAM/models
cp -r $FD_REPO/evaluators/flownet2/{scripts,config} $UPSTREAM/
cp $FD_REPO/evaluators/_common/{fd_eval_common,drone_split}.py $UPSTREAM/scripts/
```

The paper numbers were produced on `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` with the
three extensions built by `setup.py install` (with `TORCH_CUDA_ARCH_LIST="12.0"`) and
`numpy==1.26.4`, on an RTX 5090.

## Run

```bash
export FD_DATA=/path/to/datasets FD_REPO=/path/to/flyingdrones-code FD_OUT=/path/to/results
export CKPT=$UPSTREAM/models
cd $UPSTREAM
bash $FD_REPO/evaluators/flownet2/commands.sh
```

`--device` defaults to `cuda`; pass `--device cpu` to use the CPU fallbacks. Checkpoints
are loaded from `state['state_dict']` through `fd_eval_common.load_state_dict_checked`
(raises on any key mismatch). Inputs are padded to a multiple of 64 (replicate) and the
prediction is cropped back. Outputs are the summary CSV (`--output_csv`) and the
per-pair CSV (`--pairs_csv`).
