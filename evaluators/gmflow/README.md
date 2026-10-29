# GMFlow evaluator

FlyingDrones evaluation wrapper for **GMFlow** (paper class T, transformer / global matching).

| | |
|---|---|
| Upstream | https://github.com/haofeixu/gmflow |
| License | Apache 2.0 (upstream). The files in this folder are derived from it and keep its license; see `THIRD_PARTY.md`. |
| Upstream base commit | `b5123431164d01ec14526a1c3d22218aecb62024` (2024-02-20) |
| Paper checkpoints | 8: `gmflow_{chairs-1d776046, kitti-285701a8, sintel-0c07dcb3, things-e9887eda}`, `gmflow_with_refine_{chairs-020cc9be, kitti-8d3b9786, sintel-3ed1cf48, things-36579974}` |
| Checkpoint source | `gmflow_pretrained.zip` from the Google Drive link in the upstream README (https://drive.google.com/file/d/1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2/view) |

## Files

| File in this folder | Destination in the upstream checkout (`$UPSTREAM`) | Notes |
|---|---|---|
| `evaluate_drones.py` | `$UPSTREAM/evaluate_drones.py` | new file (does not replace an upstream file) |
| `datasets/__init__.py`, `datasets/drones.py` | `$UPSTREAM/datasets/` | new package (upstream uses `data/`, so there is no clash) |
| `../_common/fd_eval_common.py` | `$UPSTREAM/fd_eval_common.py` | shared pairing / metrics / per-pair dump |
| `../_common/drone_split.py` | `$UPSTREAM/drone_split.py` | split filter |
| `commands.sh` | (run from `$UPSTREAM`) | exact command lines for every paper cell |

`fd_eval_common.py` and `drone_split.py` must sit next to `evaluate_drones.py` (the
upstream root); `datasets/drones.py` imports `fd_eval_common` from there.

## Install

```bash
git clone https://github.com/haofeixu/gmflow.git && cd gmflow
git checkout b5123431164d01ec14526a1c3d22218aecb62024
export UPSTREAM=$PWD
# environment used for the paper: Python 3.10+, PyTorch 2.x
pip install "torch>=2.0" "torchvision>=0.15" "numpy>=1.24" opencv-python-headless==4.10.0.84 einops tqdm scipy
cp -r $FD_REPO/evaluators/gmflow/evaluate_drones.py $FD_REPO/evaluators/gmflow/datasets "$UPSTREAM/"
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py "$UPSTREAM/"
mkdir -p pretrained && unzip gmflow_pretrained.zip -d pretrained   # downloaded checkpoints
```

## Run

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/{images,flow,labels}
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/pretrained
cd "$UPSTREAM"
bash $FD_REPO/evaluators/gmflow/commands.sh          # all 24 paper cells + 3 window cells
```

A single cell, e.g. the GMFlow-refine `things` checkpoint at 480x270:

```bash
python evaluate_drones.py --resume $CKPT/gmflow_with_refine_things-36579974.pth \
    --datasets_path $FD_DATA --root Nature3 --image_size 270 480 \
    --split_file $FD_REPO/splits/nature3_split_frames.txt \
    --job_id gmflow__gmflow_with_refine_things-36579974__480x270 \
    --output_csv $FD_OUT/summary/gmflow__gmflow_with_refine_things-36579974__480x270.csv \
    --pairs_csv  $FD_OUT/pairs/gmflow__gmflow_with_refine_things-36579974__480x270.csv
```

`commands.sh` is generated from `reproduce/manifest.tsv`; `python reproduce/make_commands.py --family gmflow`
prints the same lines.

## Notes

* **Pairing.** `flow/<t>_flow.npy` is backward flow t -> t-1, so the model is run on
  `(I_t, I_{t-1})` with `I_t` as the reference (`fd_eval_common.list_samples(root, (0, -1))`).
  Frame 0000 is never a reference, so the stratified split yields 999 pairs.
* **Resizing.** Images are read with OpenCV and resized with `cv2.INTER_AREA` to `--image_size H W`;
  GT flow is resized bilinearly (`align_corners=False`) and scaled by `W/1920`, `H/1080`; the drone mask
  (label > 0) is resized nearest-neighbour. 960x432 is a direct (anisotropic) resize of the 1920x1080 frame.
* **Tiling.** Inputs larger than 432x960 (i.e. the 1920x1080 cells) are processed in overlapping
  432x960 tiles with uniform blending; no flag is needed.
* **Model config** is inferred from the file name (`with_refine` -> 2 scales + refinement, padding 32;
  otherwise 1 scale, padding 16).
* **Weight loading** goes through `fd_eval_common.load_state_dict_checked`, which strips a `module.`
  prefix and raises on any missing or unexpected key (it prints `loaded N/N keys`).
* **Identical weights.** Three GMFlow files are byte-identical to UniMatch releases and score identically:
  `gmflow_things-e9887eda` = UniMatch `gmflow-scale1-things-e9887eda`,
  `gmflow_with_refine_things-36579974` = `gmflow-scale2-things-36579974`,
  `gmflow_with_refine_sintel-3ed1cf48` = `gmflow-scale2-sintel-3ed1cf48`.
* **Outputs.** `--output_csv` gets one summary row (frame-weighted means, all + drone region);
  `--pairs_csv` gets one row per evaluated pair (input to the bootstrap CIs in `tables/`).
  `--max_samples` (default 500) is ignored when `--split_file` is given.
