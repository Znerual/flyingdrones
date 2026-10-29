# UniMatch evaluator

FlyingDrones evaluation wrapper for **UniMatch** (flow task; paper class T, transformer / global matching).

| | |
|---|---|
| Upstream | https://github.com/autonomousvision/unimatch |
| License | MIT (upstream). See `THIRD_PARTY.md`. |
| Upstream base commit | `943a99006950ab6c84ea72a72beb147fdd03c19a` (2025-01-04) |
| Paper checkpoints | 9: `gmflow-scale1-mixdata-train320x576-4c3a6e9a`, `gmflow-scale1-things-e9887eda`, `gmflow-scale2-mixdata-train320x576-9ff1c094`, `gmflow-scale2-regrefine6-kitti15-25b554d7`, `gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d`, `gmflow-scale2-regrefine6-sintelft-6e39e2b9`, `gmflow-scale2-regrefine6-things-776ed612`, `gmflow-scale2-sintel-3ed1cf48`, `gmflow-scale2-things-36579974` |
| Checkpoint source | upstream `MODEL_ZOO.md`, e.g. `https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/<name>.pth` |

## Files

| File in this folder | Destination in the upstream checkout (`$UPSTREAM`) | Notes |
|---|---|---|
| `evaluate.py` | `$UPSTREAM/evaluate.py` | new file; upstream has `evaluate_flow.py` but no `evaluate.py`, so nothing is overwritten (if your checkout has one, back it up first) |
| `../_common/fd_eval_common.py` | `$UPSTREAM/fd_eval_common.py` | shared pairing / metrics / per-pair dump |
| `../_common/drone_split.py` | `$UPSTREAM/drone_split.py` | split filter |
| `commands.sh` | (run from `$UPSTREAM`) | exact command lines for every paper cell |

## Install

```bash
git clone https://github.com/autonomousvision/unimatch.git && cd unimatch
git checkout 943a99006950ab6c84ea72a72beb147fdd03c19a
export UPSTREAM=$PWD
# environment used for the paper: Python 3.10+, PyTorch 2.x
pip install "torch>=2.0" "torchvision>=0.15" "numpy>=1.24" opencv-python-headless==4.10.0.84 tqdm scipy pillow imageio scikit-image matplotlib
cp $FD_REPO/evaluators/unimatch/evaluate.py "$UPSTREAM/"
cp $FD_REPO/evaluators/_common/fd_eval_common.py $FD_REPO/evaluators/_common/drone_split.py "$UPSTREAM/"
mkdir -p checkpoints   # download the 9 checkpoints listed above into it
```

## Run

```bash
export FD_DATA=/path/to/datasets      # contains Nature3/{images,flow,labels}
export FD_REPO=/path/to/flyingdrones-code
export FD_OUT=/path/to/results
export CKPT=$UPSTREAM/checkpoints
cd "$UPSTREAM"
bash $FD_REPO/evaluators/unimatch/commands.sh        # 27 paper cells + 3 window cells
```

A single cell:

```bash
python evaluate.py --model $CKPT/gmflow-scale2-regrefine6-things-776ed612.pth \
    --datasets_path $FD_DATA --root Nature3 --image_size 432 960 \
    --split_file $FD_REPO/splits/nature3_split_frames.txt \
    --job_id unimatch__gmflow-scale2-regrefine6-things-776ed612__960x432 \
    --output_csv $FD_OUT/summary/unimatch__gmflow-scale2-regrefine6-things-776ed612__960x432.csv \
    --pairs_csv  $FD_OUT/pairs/unimatch__gmflow-scale2-regrefine6-things-776ed612__960x432.csv
```

## Notes

* **Pairing.** `flow/<t>_flow.npy` is backward flow t -> t-1; the model is run on `(I_t, I_{t-1})`
  with `I_t` as the reference. Frame 0000 is never a reference (999 pairs on the stratified split).
* **Resizing.** OpenCV read, `cv2.INTER_AREA` image resize to `--image_size H W`; GT flow bilinear
  (`align_corners=False`) and rescaled by `W/1920`, `H/1080`; drone mask (label > 0) nearest-neighbour.
* **Tiling.** Inputs larger than 432x960 (the 1920x1080 cells) are processed in overlapping
  432x960 tiles with uniform blending, automatically.
* **Model config** is inferred from the checkpoint file name (`scale1` / `scale2` / `scale2` + `regrefine6`),
  matching upstream `MODEL_ZOO.md` (scale1 uses padding 16 so that the 1/8 feature map is divisible by the
  attention split).
* **Weight loading** uses `fd_eval_common.load_state_dict_checked` and raises on any missing or unexpected key.
* **Identical weights.** `gmflow-scale1-things-e9887eda`, `gmflow-scale2-things-36579974` and
  `gmflow-scale2-sintel-3ed1cf48` are byte-identical to GMFlow releases (`gmflow_things`,
  `gmflow_with_refine_things`, `gmflow_with_refine_sintel`) and score identically.
* **Outputs.** One summary row in `--output_csv`, one row per pair in `--pairs_csv`.
