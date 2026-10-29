# FlyingDrones: A Dataset and Benchmark for Optical Flow Estimation from UAV motion

Code release for the NeurIPS 2026 paper (Laurenz Ruzicka, Roman Pflugfelder, Daniel Cremers): the dataset loader with the backward-flow pairing convention, the normalized and drone-region metrics, the stratified evaluation and fine-tuning splits, the per-model evaluation wrappers for the 16 benchmarked model families, the fine-tuning trainers, the real-footage motion-alignment study and the table builders.

Dataset: [https://www.kaggle.com/datasets/flyingdrones/flyingdrones/data](https://www.kaggle.com/datasets/flyingdrones/flyingdrones/data)

## Dataset convention

```
Nature3/
  images/<t>.png          RGB frame t, 1920x1080, t = 0000 ... 9000
  flow/<t>_flow.npy       float32 (1080, 1920, 2): (u, v) in pixels
  labels/<t>_label.png    uint8 (1080, 1920): 0 = background, 1..6 = drone (render pass index)
```

**`flow/<t>_flow.npy`** **is the BACKWARD flow of frame t (t -\> t-1): for every pixel x of frame t,** **`I_t(x) ≈ I_{t-1}(x + F_t(x))`. Evaluate a (forward) flow model on the pair** **`(I_t, I_{t-1})`, with** **`I_t`** **as the reference/first frame, and compare its output directly with** **`<t>_flow.npy`** **(no sign flip). Frame 0000 has no predecessor and is never a reference frame.**

**Label PNG values 1-6 are drone pass indices (one value per drone); the drone region used by all drone-specific metrics is** **`label > 0`** **(union of all drones in the reference frame).**

Temporal models get the time-reversed window around t: VideoFlow BOF `[t+1, t, t-1]`, VideoFlow MOF `[t+2, t+1, t, t-1, t-2]`, ARFlow (three-frame) `[t+1, t, t-1]`, MemFlow `[t+3, t+2, t+1, t, t-1]` (edge frames are replicated). The three benchmark resolutions are produced on the fly from the native frames: 480x270, 960x432 and 1920x1080 (W x H). Ground-truth flow is resized bilinearly (`align_corners=False`) and its u / v components are scaled by W/1920 and H/1080; the drone mask is resized with nearest neighbour; images are resized with each model's own convention (PIL bicubic, PIL bilinear or OpenCV `INTER_AREA`, see `flyingdrones/loader.py`). 960x432 is a plain resize of the full 1920x1080 frame.

## Repository layout

```
flyingdrones/        python package: loader.py (list_samples, read_flow/label/image, resize conventions),
                     metrics.py (compute_metrics, aggregate, bootstrap_ci), splits.py, results_io.py, checkpoints.py
splits/              stratified evaluation subset, contiguous window, fine-tuning train/val/test + builders
examples/zero_flow.py  minimal end-to-end evaluation (zero-flow reference), template for new models
evaluators/<model>/  the patched evaluator(s) used for the paper, README (upstream repo + commit, install,
                     file placement, checkpoints) and commands.sh (exact command per checkpoint x resolution)
evaluators/_common/  fd_eval_common.py + drone_split.py: single-file copies placed next to every evaluator
reproduce/           manifest.tsv (all 318 paper runs), make_commands.py, compare.py, cluster example
results/             paper numbers per cell (cells.csv) and per-family leaderboard (leaderboard.csv)
tables/              validation + LaTeX table builders (bootstrap CIs, leaderboards, protocol, real, fine-tune)
finetune/<model>/    train_nature3.py for FlowFormer++, GMFlow, RAFT, VideoFlow MOF
real_footage/        real-footage motion-alignment study: data prep, scoring, aggregation
tests/               pytest unit tests (metrics, pairing convention, resize equivalence)
```

## Install and quick start

```
pip install -e .            # numpy + Pillow; optional: opencv-python, torch, scipy, pytest
pytest tests/

# zero-flow reference on the stratified subset (999 pairs) at 480x270
python examples/zero_flow.py --data /path/to/Nature3 --size 270 480 --split stratified
```

The zero-flow reference predicts F = 0, so its drone-region EPE is the mean ground-truth drone motion. Paper values (stratified subset, drone-region / full-frame EPE): 1.744 / 0.516 px at 480x270, 3.498 / 0.973 px at 960x432, 7.549 / 2.065 px at 1920x1080.

### Evaluating your own model

```
import flyingdrones as fd

samples = fd.list_samples("/path/to/Nature3", fd.TWO_FRAME)          # (I_t, I_{t-1}, <t>_flow, <t>_label)
samples = fd.filter_samples(samples, fd.load_split_frames("stratified"))
rows = []
for s in samples:
    smp = fd.load_sample(s, size=(270, 480), resize="pil_bicubic")    # frames, flow, label, mask
    I_t, I_tm1 = smp["frames"]
    pred = my_model(I_t, I_tm1)                                       # (H, W, 2) flow  t -> t-1
    rows.append(fd.compute_metrics(pred, smp["flow"], smp["mask"]))
print(fd.aggregate(rows)["epe_drone"], fd.bootstrap_ci([r["epe_drone"] for r in rows]))
```

## Metrics

Per pair and region (all pixels, drone = `label > 0`, background): EPE (px); EPE\_norm = mean \|\|(Δu / W, Δv / H)\|\| x 100; Px\_τ = % pixels with EPE \> τ, τ ∈ {1, 3, 5}; Fl = % pixels with EPE \> 3 px and EPE / \|\|F\_gt\|\| \> 5 %. Benchmark numbers are frame-weighted means over pairs; drone-region means skip drone-free pairs (885 / 896 / 898 of the 999 stratified pairs at 480x270 / 960x432 / 1920x1080). Confidence intervals: 95 % percentile bootstrap over pairs, 10,000 resamples, seed 0. Implementation: `flyingdrones/metrics.py` (identical to `evaluators/_common/fd_eval_common.py`, checked by the tests).

## Splits

`splits/nature3_split_frames.txt` (stratified evaluation subset, used for all benchmark tables), `nature3_window500_frames.txt` (frames 0001-0500, protocol-sensitivity study) and `nature3_ft_{train,val,test}_frames.txt` (fine-tuning) list reference frames t. See `splits/README.md` for the stratification and how to rebuild them.

## Reproducing the leaderboard

1. Download the dataset; `export FD_DATA=<dir containing Nature3/>`, `FD_REPO=$PWD`, `FD_OUT=<results dir>`.
2. For each model family, follow `evaluators/<model>/README.md`: clone the upstream repository at the listed commit, install it, copy the patched evaluator together with `evaluators/_common/{fd_eval_common,drone_split}.py` into place, and download the released checkpoints into `$CKPT`.
3. From the upstream root run `bash $FD_REPO/evaluators/<model>/commands.sh` (or single lines from `reproduce/manifest.tsv`; `reproduce/slurm_array_example.sh` runs them as a job array). Each run writes `$FD_OUT/summary/<tag>.csv` and the per-pair `$FD_OUT/pairs/<tag>.csv`.
4. `python reproduce/compare.py $FD_OUT` compares every finished cell with `results/cells.csv`.
5. `FD_RESULTS=$FD_OUT python tables/validate.py && FD_RESULTS=$FD_OUT python tables/build_tables.py` rebuilds the leaderboards with confidence intervals and the appendix tables (`tables/README.md`).

The evaluators load every checkpoint with a strict key check and stop if any network weight is missing or unexpected.

## Fine-tuning and real footage

- `finetune/`: trainers and exact commands for the four fine-tuned arms (800 / 100 / 100 split).
- `real_footage/`: preparation of the 21 labelled cameras of the drone-tracking dataset of Li et al., the motion-alignment metrics (NSS\_D, Hit@r, contrast, pt-EPE / pt-cos) and the aggregation over cameras.

## Citation

See `CITATION.cff`.

## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en), see LICENSE
