#!/usr/bin/env python3
"""Zero-flow reference, end to end (needs only NumPy and Pillow).

Predicts F = 0 for every pair, so its drone-region EPE equals the mean ground-truth
drone motion. This is the "zero flow" row of the paper tables and a template for
plugging in any flow model: replace ``predict()`` with a call to your model on
``(I_t, I_{t-1})`` (``sample["frames"]``) that returns the flow t -> t-1 at the
evaluation resolution.

Examples::

    # paper protocol: stratified subset (999 pairs) at 480x270
    python examples/zero_flow.py --data $FD_DATA/Nature3 --size 270 480 --split stratified

    # quick check on the first 5 evaluable pairs, per-pair CSV written
    python examples/zero_flow.py --data $FD_DATA/Nature3 --size 270 480 --max-pairs 5 \
        --pairs-csv /tmp/zero_pairs.csv
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import flyingdrones as fd  # noqa: E402


def predict(sample):
    """Zero flow at the resolution of the ground truth."""
    return np.zeros_like(sample["flow"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Nature3 scene dir (images/ flow/ labels/)")
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=None,
                    help="evaluation size, e.g. 270 480 | 432 960 | 1080 1920 (default: native)")
    ap.add_argument("--split", default=None,
                    help="split name (stratified, window500, ft_train, ft_val, ft_test) or file; default: all pairs")
    ap.add_argument("--max-pairs", type=int, default=None, help="evaluate only the first N pairs")
    ap.add_argument("--pairs-csv", default=None, help="write one row per pair here")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples for the 95%% CI")
    a = ap.parse_args()

    samples = fd.list_samples(a.data, fd.TWO_FRAME)          # (I_t, I_{t-1}, <t>_flow, <t>_label)
    samples = fd.filter_samples(samples, fd.load_split_frames(a.split))
    if a.max_pairs:
        samples = samples[: a.max_pairs]
    if not samples:
        sys.exit("no samples")

    dumper = fd.PairDumper(a.pairs_csv)
    t0 = time.time()
    for s in samples:
        # zero flow does not look at the images; a real model needs load_images=True
        sample = fd.load_sample(s, size=a.size, load_images=False)
        m = fd.compute_metrics(predict(sample), sample["flow"], sample["mask"])
        dumper.add(sample["ref"], m)
    dumper.close()

    agg = fd.aggregate(dumper.rows)
    H, W = a.size if a.size else fd.NATIVE_SIZE
    n_drone = sum(1 for r in dumper.rows if r["n_drone_px"] > 0)
    print(f"\nzero flow @ {W}x{H}: {len(dumper.rows)} pairs ({n_drone} with drones), {time.time() - t0:.1f}s")
    print(f"{'metric':10} {'all':>10} {'drone':>10} {'bg':>10}")
    for k in fd.METRIC_NAMES:
        print(f"{k:10} " + " ".join(f"{agg[f'{k}_{r}']:10.4f}" for r in fd.REGIONS))
    for r in ("all", "drone"):
        mean, lo, hi = fd.bootstrap_ci([row[f"epe_{r}"] for row in dumper.rows], n_boot=a.n_boot)
        print(f"EPE ({r}) 95% CI: {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    if a.pairs_csv:
        print(f"per-pair rows -> {a.pairs_csv}")


if __name__ == "__main__":
    main()
