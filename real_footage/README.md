# Real-footage motion-alignment study (paper appendix "Real-Footage Ranking Transfer")

Ground-truth-free check of whether a flow model's motion response lands on the drone in real
video. Data: the multi-view drone-tracking dataset of Li et al. (*Reconstruction of 3D flight
trajectories from ad-hoc camera networks*, arXiv:2003.04784; data released by the authors as
`drone-tracking-datasets`, e.g. <https://github.com/CenekAlbl/drone-tracking-datasets>). Static,
tripod-mounted cameras film a hexacopter; `detections/camN.txt` holds the manually labelled 2-D
drone centre per frame (`frame_id x y`, `0 0` = not visible). The dataset is **not**
redistributed here; download it from the authors and respect its terms.

## 1. Prepare the 21 camera subsets

```bash
python real_footage/prep_all.py --dataset-root /path/to/drone-tracking-datasets --out real_eval/all
```

For every labelled camera of datasets 1-4 (21 cameras; dataset 5 has no 2-D labels):

* `prep_subset.py` selects 100 consecutive frame pairs `(idx, idx+1)` whose two frames are both
  labelled, stratified by the label displacement `|label[idx+1] - label[idx]|` (the drone's image
  motion; valid as pseudo ground truth because the camera is static) into `m1_subpx` < 1,
  `m2_low` 1-3, `m3_mid` 3-8, `m4_high` > 8 px/frame (proportional with a floor of 10 per
  occupied stratum, seed 42), and decodes only the needed frames;
* `extract_window.py` adds the frames `idx-2 .. idx+2` needed by VideoFlow MOF;
* `downscale_camera.py` rescales cameras wider than 1920 px (two 4K cameras of dataset 3) to
  1920 px width and scales the labels accordingly.

Each camera dir contains `images/<idx:04d>.png`, `selected.json` (pair list with `label_ref`,
`label_nxt`, displacement and stratum), `labels.json`, `frames.txt` and `meta.json`. Needs
OpenCV and the `7z` command line tool.

## 2. Run a flow model (file contract)

For every entry `idx` of `selected.json`, run the model on the **forward** pair
`(images/<idx>.png, images/<idx+1>.png)` (unlike the synthetic benchmark, there is no backward
ground truth here; the label displacement is `label_nxt - label_ref`). Paper protocol:

* inference at 1280x720 (VideoFlow MOF at 960x540 for memory), frames resized with PIL bicubic;
* the predicted flow is upsampled to the camera's (prepared) native resolution with bilinear
  interpolation (`align_corners=False`) and the vectors scaled by the size ratio;
* every checkpoint is loaded with a strict key check (all network weights must match the file).

Save one array per pair as `<flows_dir>/<system>__<idx:04d>.npy`, shape `(H, W, 2)`, float32,
`(u, v)` in native pixels; `<system>` is any name without a trailing `__<digits>`
(e.g. `raft__raft-sintel`). In the paper runs this was done with each model's own single-pair
inference script on top of the same upstream checkouts as in `evaluators/`.

## 3. Score, collect, aggregate

```bash
# per (camera, system): metrics over that camera's 100 pairs
python real_footage/score_one.py --flows-dir flows/dataset1_cam0/raft__raft-sintel \
    --subset-dir real_eval/all/dataset1_cam0 --camera dataset1_cam0 --system raft__raft-sintel \
    --output-csv per_cam/dataset1_cam0__raft__raft-sintel.csv

# after all cameras / systems
python real_footage/collect.py per_cam --results results/all
python real_footage/aggregate_all.py --results results/all     # -> results/all/overall_checkpoint.csv
```

Metrics (`align_metric.py`, per pair; |flow| treated as a motion-saliency map):

| metric | definition | aggregation over pairs |
|---|---|---|
| `nss`, `nss_disk` (NSS_D) | (|F| at the label - mean |F|) / std |F|; `nss_disk` averages over an 8 px disk | mean |
| `hit@r` | smoothed (Gaussian sigma 3 px) |F| peak, ignoring a 20 px border ring, within r px of the label, r in {10, 25, 50, 100} | mean |
| `ple_px`, `ple_norm` | distance of that peak to the label (px, / image diagonal) | median |
| `contrast` | mean |F| in a 15 px disk / mean |F| outside a 40 px disk | mean |
| `pt_epe`, `pt_cos` | median flow vector in the 8 px disk vs the label displacement: endpoint error, cosine | median / mean |

The paper reports, per checkpoint, the mean and sample standard deviation (ddof = 1) over the 21
cameras (`overall_checkpoint.csv`), and Hit@50 per displacement stratum averaged over the cameras
that have pairs in that stratum (`strata_hit50.tsv`).
