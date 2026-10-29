# Per-model evaluators

One folder per model family, each with the evaluator script(s) exactly as used for the paper,
a `README.md` (upstream repository and commit, installation, where to put each file, checkpoint
sources, model-specific notes) and `commands.sh` (every paper cell of that family, generated
from `reproduce/manifest.tsv`).

| folder | family | evaluator | pairing / window | image resize | 1920x1080 |
|---|---|---|---|---|---|
| `arflow` | ARFlow | `scripts/analyze_drone_epe.py` | 3-frame `[t+1, t, t-1]` or 2-frame (from ckpt) | PIL bilinear | full frame |
| `fastflownet` | FastFlowNet | `evaluate.py` | `(I_t, I_{t-1})` | cv2 INTER_AREA | full frame |
| `flowdiffusion` | FlowDiffusion (DDVM) | `evaluate_drones.py` | `(I_t, I_{t-1})` | cv2 INTER_AREA (+ model resize to 320x448) | full frame |
| `flowformerplusplus` | FlowFormer++ | `evaluate.py` | `(I_t, I_{t-1})` | cv2 INTER_AREA | tiled (cfg image size) |
| `flownet2` | FlowNet2 | `scripts/analyze_drone_epe.py` | `(I_t, I_{t-1})` | PIL bilinear | full frame |
| `gmflow` | GMFlow | `evaluate_drones.py` (+ `datasets/drones.py`) | `(I_t, I_{t-1})` | cv2 INTER_AREA | tiled 432x960 |
| `liteflownet3` | LiteFlowNet3 | `evaluate.py` | `(I_t, I_{t-1})` | PIL bicubic | tiled 432x960 |
| `memflow` | MemFlow (+ zero-flow reference) | `scripts/analyze_drone_epe.py` | clip `[t+3, ..., t, t-1]`, last flow | PIL bicubic | tiled 432x960 |
| `pwc-net` | PWC-Net | `scripts/analyze_drone_epe.py` | `(I_t, I_{t-1})` | PIL bicubic | full frame |
| `pyflow` | PyFlow (Coarse2Fine, CPU) | `evaluate.py` | `(I_t, I_{t-1})` | cv2 INTER_AREA | full frame |
| `raft` | RAFT | `scripts/analyze_drone_epe.py` | `(I_t, I_{t-1})` | PIL bicubic | tiled 432x960 |
| `sea-raft` | SEA-RAFT | `scripts/analyze_drone_epe.py` | `(I_t, I_{t-1})` | PIL bicubic | full frame |
| `unimatch` | UniMatch | `evaluate.py` | `(I_t, I_{t-1})` | cv2 INTER_AREA | tiled 432x960 |
| `vcn` | VCN | `scripts/analyze_drone_epe.py` | `(I_t, I_{t-1})` | PIL bicubic | full frame |
| `videoflow` | VideoFlow BOF / MOF | `scripts/analyze_drone_epe.py` | BOF `[t+1, t, t-1]` slot 0; MOF `[t+2 .. t-2]` forward slot 1 | PIL bicubic | tiled 432x960 |
| `waft` | WAFT | `scripts/analyze_drone_epe.py` | `(I_t, I_{t-1})` | PIL bicubic | tiled 432x960 |
| `alltracker` | AllTracker (extra, **not** part of the paper benchmark) | `scripts/analyze_drone_epe.py` | `(I_t, I_{t-1})` | PIL bicubic | -- |

Common to all evaluators:

* Copy `_common/fd_eval_common.py` and `_common/drone_split.py` **next to the evaluator script**
  (each script puts its own directory first on `sys.path`). `fd_eval_common.py` provides
  `list_samples` (backward pairing, time-reversed windows), `compute_metrics`, the per-pair dumper
  and `load_state_dict_checked`; it is numerically identical to the `flyingdrones` package.
* `--split_file $FD_REPO/splits/nature3_split_frames.txt` restricts the run to the stratified
  subset (999 pairs); frame 0000 is never evaluated.
* Every run appends one row to `--output_csv` (summary, 4 decimals) and writes one row per pair to
  `--pairs_csv` (full precision; all paper tables are computed from these).
* Checkpoints are loaded through `load_state_dict_checked`: any missing or unexpected network key
  aborts the run (VCN exempts only its input-size-dependent meshgrid buffers; PyFlow has no weights).
