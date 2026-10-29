# Third-party code and assets

The models are run through their official repositories, which are **not** redistributed here
(each `evaluators/<dir>/README.md` names the upstream repository and commit). Files under
`evaluators/` are evaluation wrappers written for this benchmark on top of those code bases;
several are adapted from the upstream evaluation scripts, and a few folders contain small
upstream-derived files or patches that are needed to run them. Those files remain under the
license of the upstream project they derive from (table below), regardless of the license chosen
for the rest of this repository. No model weights and no third-party datasets are included.

## Model code (paper Table "Licenses of external assets")

| Model | Official source | License / terms | Upstream-derived content in this repo |
|---|---|---|---|
| FlowNet2 | https://github.com/lmb-freiburg/flownet2 | BSD 2-Clause (Caffe-derived); repository documentation restricts use to non-commercial research and requires author consent for commercial use | `evaluators/flownet2/` (wrapper, configs, `flownet2-pytorch.patch`). The code actually run is the PyTorch port https://github.com/NVIDIA/flownet2-pytorch (Apache 2.0); the converted Caffe weights keep the Freiburg terms |
| PyFlow | https://github.com/pathak22/pyflow | MIT (Python wrapper, (c) 2017 Deepak Pathak); the underlying Ce Liu C++ optical-flow code carries separate terms (https://people.csail.mit.edu/celiu/OpticalFlow/) | `evaluators/pyflow/` (wrapper only) |
| PWC-Net | https://github.com/NVlabs/PWC-Net | CC BY-NC-SA 4.0 ((c) 2018 NVIDIA Corporation); non-commercial, share-alike, attribution | `evaluators/pwc-net/pwcnet/` (PyTorch PWC-DC network definition, licensed CC BY-NC-SA 4.0, see `pwcnet/LICENSE.md`) |
| LiteFlowNet3 | https://github.com/twhui/LiteFlowNet3 | Custom non-commercial license (Tak-Wai Hui); commercial use requires prior arrangement | `evaluators/liteflownet3/` (wrapper, configs; the model runs through the `ptlflow` package, https://github.com/hmorimitsu/ptlflow, Apache 2.0) |
| VCN | https://github.com/gengshan-y/VCN | MIT | `evaluators/vcn/vcnflow/models/` (adapted VCN model code, MIT, see `models/LICENSE`) |
| ARFlow | https://github.com/lliuz/ARFlow | MIT | `evaluators/arflow/patches/` |
| FastFlowNet | https://github.com/ltkong218/FastFlowNet | MIT | `evaluators/fastflownet/models/` (pure-PyTorch FastFlowNet, MIT, see `models/LICENSE`) |
| RAFT | https://github.com/princeton-vl/RAFT | BSD 3-Clause | `evaluators/raft/` (wrapper, configs, `raft/` interface shim around upstream `core/`) |
| GMFlow | https://github.com/haofeixu/gmflow | Apache 2.0 | `evaluators/gmflow/` (wrapper) |
| UniMatch | https://github.com/autonomousvision/unimatch | MIT | `evaluators/unimatch/` (wrapper) |
| FlowFormer++ | https://github.com/XiaoyuShi97/FlowFormerPlusPlus | Apache 2.0 | `evaluators/flowformerplusplus/` (wrapper, config) |
| VideoFlow | https://github.com/XiaoyuShi97/VideoFlow | Apache 2.0 | `evaluators/videoflow/patches/` |
| FlowDiffusion (DDVM) | https://github.com/google-research/google-research | Apache 2.0 (monorepo-level license) | `evaluators/flowdiffusion/` (wrapper, `patches/`). The code actually run is the PyTorch implementation https://github.com/DQiaole/FlowDiffusion_pytorch (Apache 2.0) |
| MemFlow | https://github.com/DQiaole/MemFlow | Apache 2.0 | `evaluators/memflow/` (wrapper) |
| SEA-RAFT | https://github.com/princeton-vl/SEA-RAFT | BSD 3-Clause | `evaluators/sea-raft/` (wrapper, `searaft/` interface shim around upstream `core/`) |
| WAFT | https://github.com/princeton-vl/WAFT | BSD 3-Clause | `evaluators/waft/` (wrapper) |
| DepthAnythingV2 encoder (used by WAFT) | https://github.com/DepthAnything/Depth-Anything-V2 | Code Apache 2.0; weights: Small Apache 2.0, Base/Large/Giant CC BY-NC 4.0 | none |
| AllTracker (extra, not part of the paper benchmark) | https://github.com/aharley/alltracker | MIT | `evaluators/alltracker/` (wrapper) |

The PWC-Net, LiteFlowNet3 and FlowNet2 code bases (and the Base/Large/Giant DepthAnythingV2
weights used by some WAFT checkpoints) are restricted to non-commercial use; this applies to any
use of the corresponding evaluators.

## Datasets

| Asset | Source | Terms |
|---|---|---|
| FlyingDrones (this benchmark) | https://www.kaggle.com/datasets/flyingdrones/flyingdrones/data | see the dataset page |
| Real-footage study: drone-tracking dataset of Li et al. (arXiv:2003.04784) | released by the authors (see `real_footage/README.md`) | see the authors' release; not redistributed |
| Pre-training datasets of the public checkpoints (FlyingChairs, FlyingThings3D, MPI Sintel, KITTI, TartanAir, Spring, HD1K) | see the paper's license table | not used or redistributed here |
