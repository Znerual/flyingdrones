# PyFlow (Coarse2Fine) evaluator

| | |
|---|---|
| Upstream | https://github.com/pathak22/pyflow (base commit `8f8ab9e90845f1b819b3833f995f481cb5d92166`) |
| Upstream license | MIT for the Python wrapper (Copyright 2017 Deepak Pathak). The bundled C++ code by Ce Liu has its own terms, see https://people.csail.mit.edu/celiu/OpticalFlow/ |
| "Checkpoints" | No learned weights. The three parameter presets `fast`, `default` and `accurate` defined in `evaluate.py` (`CONFIGS`) take the place of checkpoints. |
| Resolutions | 480x270, 960x432, 1920x1080 (untiled, CPU only) |

## Contents

| file | role |
|---|---|
| `evaluate.py` | FlyingDrones evaluator (backward-flow pairing, paper metrics, per-pair CSV, presets) |
| `commands.sh` | The exact command for every manifest cell (from `reproduce/manifest.tsv`) |

## Install and setup

`setup.py` imports `distutils`, so build with Python 3.11 or older, or have setuptools
installed so that it provides the `distutils` shim. The paper container used torch
2.2.1 with **numpy 1.26.4**: torch 2.2 wheels are built against numpy 1.x, and with
numpy 2 `torch.from_numpy` fails with "Numpy is not available". A recent torch with
numpy 2 works too. Torch is only used for resizing the GT flow and mask.

```bash
git clone https://github.com/pathak22/pyflow.git $UPSTREAM
git -C $UPSTREAM checkout 8f8ab9e90845f1b819b3833f995f481cb5d92166
pip install "numpy==1.26.4" "torch==2.2.1" opencv-python-headless pillow tqdm cython setuptools
cd $UPSTREAM && python setup.py build_ext -i        # builds the pyflow extension in place
cp $FD_REPO/evaluators/pyflow/evaluate.py $UPSTREAM/
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py $UPSTREAM/
```

## Run

```bash
export FD_DATA=/path/to/datasets FD_OUT=/path/to/results
cd $UPSTREAM && bash $FD_REPO/evaluators/pyflow/commands.sh
```

PyFlow is CPU-only and single-threaded per pair. At 1920x1080 a preset takes on the order
of a day for 999 pairs, so run cells in parallel. The `accurate` preset at 1920x1080 is
in the manifest but is not reported in the paper.

## Notes

- **Pairing.** `pyflow.coarse2fine_flow(I_t, I_{t-1}, ...)` with `I_t` as reference, scored against the backward GT `<t>_flow.npy`. Frame 0000 is skipped, which leaves 999 pairs.
- **Resizing.** Images are read with `cv2.imread` (BGR to RGB), resized with **`cv2.INTER_AREA`** and scaled to [0,1] as float64. GT flow is resized bilinearly (`align_corners=False`) with per-axis magnitude scaling, and the mask with nearest-neighbour.
- **Outputs.** A summary CSV (`--output_csv`, with the preset name as checkpoint) and a per-pair CSV (`--pairs_csv`).
