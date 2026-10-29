"""
Helper to restrict any drone eval to the stratified split.

The split is a set of REFERENCE-FRAME basenames (e.g. "0100"). Every dataset loader
in this repo stores samples whose first path is the reference image `<...>/<base>.png`,
so filtering on that basename works uniformly for 2-frame and multi-frame loaders.

Usage in an eval script:

    from drone_split import load_split_frames, filter_samples
    frames = load_split_frames(args.split_file)          # None if flag unset -> no filtering
    dataset.samples = filter_samples(dataset.samples, frames)

`filter_samples` accepts the sample tuple/list used by every loader here; it looks at
the FIRST string element that ends in `.png` and matches its basename (sans extension).
"""

import os.path as osp


def load_split_frames(path):
    """Read a split_frames.txt (one basename per line). Returns a set, or None."""
    if not path:
        return None
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def _ref_base(sample):
    # The flow file <ref>_flow.npy is ALWAYS the reference frame -- unambiguous even
    # for multi-frame windows (videoflow/memflow) that list several images.
    for el in sample:
        if isinstance(el, str) and el.endswith("_flow.npy"):
            return osp.basename(el)[: -len("_flow.npy")]
    # fall back: first .png basename (2-frame loaders)
    for el in sample:
        if isinstance(el, str) and el.endswith(".png"):
            return osp.splitext(osp.basename(el))[0]
    return osp.splitext(osp.basename(str(sample[0])))[0]


def filter_samples(samples, frames):
    """Keep only samples whose reference-frame basename is in `frames`.

    If `frames` is None, returns `samples` unchanged.
    """
    if frames is None:
        return samples
    kept = [s for s in samples if _ref_base(s) in frames]
    print(f"drone_split: {len(kept)}/{len(samples)} samples kept")
    return kept
