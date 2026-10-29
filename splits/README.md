# Splits

Every `*_frames.txt` file lists **reference-frame basenames** (`0100`), one per line.
Reference frame `t` denotes the pair `(I_t, I_{t-1})`, scored against `flow/<t>_flow.npy`
(backward flow, see the top-level README). Because the files are keyed by frame number, the
same split applies to all three resolutions and to every model.

| file | used for | listed | evaluated pairs |
|---|---|---|---|
| `nature3_split_frames.txt` | all benchmark tables (stratified subset) | 1000 | 999 |
| `nature3_window500_frames.txt` | protocol-sensitivity table (contiguous frames 0001-0500) | 500 | 500 |
| `nature3_ft_train_frames.txt` | fine-tuning, training | 800 | 799 |
| `nature3_ft_val_frames.txt` | fine-tuning, checkpoint selection | 100 | 100 |
| `nature3_ft_test_frames.txt` | fine-tuning, held-out test | 100 | 100 |

Frame `0000` is listed in the stratified subset and in the fine-tuning training split (the
splits were drawn over frame indices 0..8999), but it has no predecessor and is therefore
never produced by `flyingdrones.list_samples`; hence 999 and 799 pairs.

## Stratified evaluation subset (`nature3_split.json`)

`build_split.py` bins every reference frame of the 9000-pair sequence by

* total drone footprint in the frame (px at 1920x1080, union of all drones): five quintile
  bins of the frames that contain drones (`s1_tiny` ... `s5_huge`, edges in the JSON);
* mean ground-truth flow magnitude inside the drone mask (px/frame): `m1_subpx` < 1,
  `m2_low` 1-3, `m3_mid` 3-8, `m4_high` > 8;
* plus an `empty` stratum for frames without drone pixels,

and draws N = 1000 frames without replacement, proportionally to the stratum population
with a floor of 15 per occupied stratum (largest-remainder allocation, seed 42).

```bash
python splits/build_split.py features --root $FD_DATA/Nature3 --out_dir splits -j 8   # writes nature3_features.csv
python splits/build_split.py sample   --out_dir splits -n 1000 --seed 42             # uses the cached features
```

`nature3_features.csv` (per-frame features) is shipped, so the `sample` phase alone
reproduces `nature3_split.json` / `nature3_split_frames.txt` byte for byte.

## Fine-tuning split (`nature3_ft_split.json`)

`build_ft_split_buffered.py` partitions the 1000-frame stratified subset into 800 / 100 / 100
train / val / test frames by a nested stratification that keeps every stratum represented in
all parts. Frames that lie four or fewer frames apart are first chained into blocks and whole
blocks are assigned to one partition, so every validation and test frame is at least five frames
away from any training frame and no five-frame window (`t-2 .. t+2`) crosses the partition.

```bash
python splits/build_ft_split_buffered.py    # reads nature3_split.json next to it, deterministic
```

## Using a split

```python
from flyingdrones import list_samples, load_split_frames, filter_samples
samples = filter_samples(list_samples("/data/Nature3"), load_split_frames("stratified"))
```

The evaluators do the same through `--split_file splits/nature3_split_frames.txt`
(`evaluators/_common/drone_split.py`).
