# Fine-tuning on FlyingDrones (paper appendix "Fine-Tuning on FlyingDrones")

Four stand-alone trainers, one per arm. Each one is dropped into the root of the upstream
checkout of its model, next to that model's patched evaluator (see `evaluators/<model>/`),
because validation calls the evaluator's own data path and metric code. The validation
drone-EPE printed during training is therefore exactly the number the evaluator reports on
the same split.

| arm | trainer | goes to | evaluator it imports | init checkpoint |
|---|---|---|---|---|
| RAFT | `raft/train_nature3.py` | `$UPSTREAM/train_nature3.py` | `scripts/analyze_drone_epe.py` | `raft-things.pth` |
| GMFlow | `gmflow/train_nature3.py` | `$UPSTREAM/train_nature3.py` | `evaluate_drones.py` (+ `datasets/drones.py`) | `gmflow_chairs-1d776046.pth` |
| FlowFormer++ | `flowformerplusplus/train_nature3.py` | `$UPSTREAM/train_nature3.py` | `evaluate.py` | `things_288960.pth` (`--cfg configs.things_288960`) |
| VideoFlow MOF | `videoflow/train_nature3.py` | evaluator work dir (next to `scripts/`) | `scripts/analyze_drone_epe.py` | `MOF_sintel.pth` |

Also copy `evaluators/_common/fd_eval_common.py` and `evaluators/_common/drone_split.py` next to
the trainer (and next to the evaluator script). Install the upstream code base exactly as for
evaluation; training additionally needs `tensorboard`.

## Data and pairing

* Training split `splits/nature3_ft_train_frames.txt` (800 frames, 799 pairs: frame 0000 has no
  predecessor), validation `nature3_ft_val_frames.txt` (100), test `nature3_ft_test_frames.txt` (100).
  The splits are block-buffered (>= 5 frames between partitions; see `splits/README.md`).
* Every training sample is the backward pair `(I_t, I_{t-1})` with ground truth `<t>_flow.npy`
  (`fd_eval_common.list_samples(root, TWO_FRAME)`). VideoFlow MOF is trained on the genuine
  time-reversed window `[I_{t+2}, I_{t+1}, I_t, I_{t-1}, I_{t-2}]` with ground truth on its three
  forward slots (`<t+1>_flow`, `<t>_flow`, `<t-1>_flow`), backward slots masked.
* Frames are resized to the evaluation scale and randomly cropped; weights are loaded with the
  strict key check (`load_state_dict_checked`).

## Commands (paper recipe)

Common arguments for all four arms:

```bash
COMMON="--datasets_path $FD_DATA --root Nature3 \
  --train_split_file $FD_REPO/splits/nature3_ft_train_frames.txt \
  --val_split_file   $FD_REPO/splits/nature3_ft_val_frames.txt \
  --num_steps 1500 --val_every 150"
```

```bash
# RAFT (run from the RAFT checkout)
python -u train_nature3.py --name raft-nat3-bwd --restore_ckpt $CKPT/raft-things.pth $COMMON \
  --image_size 384 768 --eval_size 432 960 --batch_size 4 --lr 2.5e-5 --gamma 0.85 --mixed_precision \
  --ckpt_dir $FD_OUT/ft/raft

# GMFlow (run from the GMFlow checkout)
python -u train_nature3.py --name gmflow-nat3-bwd --restore_ckpt $CKPT/gmflow_chairs-1d776046.pth $COMMON \
  --image_size 512 896 --eval_size 720 1280 --batch_size 1 --lr 5e-5 --mixed_precision \
  --ckpt_dir $FD_OUT/ft/gmflow

# FlowFormer++ (run from the FlowFormer++ checkout)
python -u train_nature3.py --name ffpp-nat3-bwd --cfg configs.things_288960 \
  --restore_ckpt $CKPT/things_288960.pth $COMMON \
  --image_size 384 768 --eval_size 432 960 --batch_size 1 --accum 6 --lr 1.25e-4 --mixed_precision \
  --ckpt_dir $FD_OUT/ft/ffpp

# VideoFlow MOF (run from the VideoFlow evaluator work dir)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u train_nature3.py --name videoflow-mof-nat3-bwd --restore_ckpt $CKPT/MOF_sintel.pth $COMMON \
  --image_size 384 768 --eval_size 432 960 --batch_size 1 --accum 4 --lr 6.25e-5 --mixed_precision \
  --ckpt_dir $FD_OUT/ft/videoflow_mof
```

Each trainer writes `<name>_best.pth` (lowest validation drone-EPE; the reported model),
`<name>.pth` (final, step 1500) and `<name>_latest.pth` (resume point) to `--ckpt_dir`.
**Use a fresh `--ckpt_dir`**: a trainer resumes from `<name>_latest.pth` if it exists.

Test-split scores are produced with the canonical evaluator of each model, e.g. for RAFT:

```bash
python scripts/analyze_drone_epe.py --cfg config/raft-large.json --ckpt $FD_OUT/ft/raft/raft-nat3-bwd_best.pth \
  --datasets_path $FD_DATA --root Nature3 --image_size 432 960 \
  --split_file $FD_REPO/splits/nature3_ft_test_frames.txt \
  --output_csv $FD_OUT/ft/summary/raft__best__test.csv --pairs_csv $FD_OUT/ft/pairs/raft__best__test.csv
```

(GMFlow: `evaluate_drones.py --resume <ckpt> --image_size 720 1280`; FlowFormer++:
`evaluate.py --cfg configs.things_288960 --model <ckpt> --image_size 432 960`; VideoFlow:
`scripts/analyze_drone_epe.py --mode MOF --ckpt <ckpt> --image_size 432 960`.) The baseline row
is the same command with the init checkpoint.

## Paper results (held-out test split, 100 pairs)

| arm | eval res. | drone EPE base -> fine-tuned | full-frame EPE | best step |
|---|---|---|---|---|
| FlowFormer++ | 960x432 | 1.614 -> 1.305 (-19.2 %) | 0.619 -> 0.072 | 600 |
| GMFlow | 1280x720 | 3.649 -> 2.723 (-25.4 %) | 1.394 -> 0.794 | 300 |
| RAFT | 960x432 | 1.857 -> 1.933 (+4.1 %) | 0.604 -> 0.510 | 150 |
| VideoFlow MOF | 960x432 | 1.727 -> 1.089 (-36.9 %) | 0.573 -> 0.069 | 1500 |

Training is not bit-reproducible across GPUs (mixed precision, non-deterministic kernels), so
expect small deviations.
