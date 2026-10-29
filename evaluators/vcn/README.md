# VCN evaluator

| | |
|---|---|
| Upstream | https://github.com/gengshan-y/VCN (model code ported from commit `00c4befdbdf4e42050867996a6f686f52086e01a`) |
| Upstream license | MIT (Copyright 2019 Carnegie Mellon University), see `vcnflow/models/LICENSE` |
| Paper checkpoints | VCN: `vcn_weights/{chairs/finetune_141999, things/finetune_211999, things/finetune_239999, kitti-ft-trainval/finetune_149999, sintel-ft-trainval/finetune_67999}`<br>VCN-small: `vcn_small_weights/{chairs/finetune_139999, things/finetune_183999}` |
| Weights | the Google-Drive folders linked in the upstream README ("weights files (VCN)", "weights files (VCN-small)") |
| Resolutions | 480x270, 960x432, 1920x1080 (all untiled; the evaluator has no tiling option) |

This folder is self-contained. No upstream checkout is needed.

## Contents

| file | role |
|---|---|
| `scripts/analyze_drone_epe.py` | FlyingDrones evaluator (backward-flow pairing, paper metrics, per-pair CSV, checked weight loading) |
| `vcnflow/inference.py` | Inference wrapper: BGR/mean preprocessing, padding to a multiple of 64, rebuilding the size-dependent modules for each resolution, rescaling the flow |
| `vcnflow/models/{VCN,VCN_small,conv4d,submodule}.py` | Upstream `models/` patched for current PyTorch (no `Variable`, `F.interpolate`; correlation via upstream's pure-PyTorch `corrf`). The network definition is unchanged, so the released `.tar` files load. MIT, upstream copyright. |
| `config/drones-{S,M,L}.json` | Evaluation resolution: S = 270x480, M = 432x960, L = 1080x1920 (H x W) |
| `commands.sh` | The exact command for every paper cell (from `reproduce/manifest.tsv`) |

## Install

Python 3.10-3.13. The paper environment used torch 2.8, numpy 2.x, Pillow 12 and
opencv-python-headless 4.10.0.84.

```bash
pip install "torch>=2.7,<3" torchvision numpy pillow opencv-python-headless==4.10.0.84 tqdm scipy
```

## Setup

```bash
export UPSTREAM=$FD_REPO/evaluators/vcn
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py $UPSTREAM/scripts/
export CKPT=/path/to/vcn-weights      # must contain vcn_weights/... and vcn_small_weights/... as in the table above
```

**Keep the upstream directory names**, e.g. `kitti-ft-trainval/finetune_149999.tar`. The
loader takes the search range from the parent directory name: `kitti` gives maxdisp
512 / fac 2, `sintel` gives 448 / 1.4, and anything else gives 256 / 1. This is the
upstream evaluation setting.

## Run

```bash
export FD_DATA=/path/to/datasets FD_OUT=/path/to/results
cd $UPSTREAM && PYTHONPATH=$UPSTREAM bash $FD_REPO/evaluators/vcn/commands.sh
```

The paper ran every cell on a GPU, where the model is wrapped in `nn.DataParallel`.
`--device cpu` also works now: weights load with the same key check on both paths.

## Notes

- **Pairing.** The network is run on `(I_t, I_{t-1})` because `<t>_flow.npy` is backward flow (t -> t-1). Frame 0000 is skipped, which leaves 999 pairs.
- **Resizing.** Images use PIL `Image.resize` (default BICUBIC). GT flow is resized bilinearly with per-axis magnitude scaling, and the drone mask with nearest-neighbour.
- **Weight guard.** The evaluator builds the model itself (`_load_vcn_checked`) and calls `fd_eval_common.load_state_dict_checked`. Only two kinds of key are tolerated:
  - missing size-dependent buffers (`flow_reg*.flowx/flowy`, `warp*.grid`), which are rebuilt for every input size;
  - unused training-only heads in the checkpoint (`dc*_convo`, `bf*`, `bfo*`).

  Anything else raises. All 579/579 (VCN) and 468/468 (VCN-small) parameters load. `vcnflow.inference.load_vcn` (`strict=False`) is kept for reference but is not used by the evaluator.
- **Outputs.** As for every evaluator: a summary CSV (`--output_csv`) and a per-pair CSV (`--pairs_csv`).
