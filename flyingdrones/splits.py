"""Evaluation and fine-tuning splits.

All split files list REFERENCE-FRAME basenames (``"0100"``), one per line. A
reference frame t denotes the pair ``(I_t, I_{t-1})`` scored against
``flow/<t>_flow.npy``. Because the files are keyed by frame number they apply
unchanged to every resolution and every model.

Shipped splits (``splits/`` at the repository root):

==================  =========================================  =================
name                file                                       evaluated pairs
==================  =========================================  =================
``stratified``      ``nature3_split_frames.txt``               999 (1000 listed)
``window500``       ``nature3_window500_frames.txt``           500 (frames 1-500)
``ft_train``        ``nature3_ft_train_frames.txt``            799 (800 listed)
``ft_val``          ``nature3_ft_val_frames.txt``              100
``ft_test``         ``nature3_ft_test_frames.txt``             100
==================  =========================================  =================

Frame ``0000`` appears in ``stratified`` and ``ft_train`` (the splits were drawn
over frame indices 0..8999) but has no predecessor, so ``list_samples`` never
produces it; this is why 999 / 799 pairs are evaluated / trained on.
"""

import os.path as osp

SPLIT_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "splits")

SPLIT_FILES = {
    "stratified": "nature3_split_frames.txt",
    "window500": "nature3_window500_frames.txt",
    "ft_train": "nature3_ft_train_frames.txt",
    "ft_val": "nature3_ft_val_frames.txt",
    "ft_test": "nature3_ft_test_frames.txt",
}


def split_path(name_or_path):
    """Resolve a split name (``"stratified"``, ...) or return a path unchanged."""
    if name_or_path in SPLIT_FILES:
        return osp.join(SPLIT_DIR, SPLIT_FILES[name_or_path])
    return name_or_path


def load_split_frames(name_or_path):
    """Set of reference-frame basenames, or ``None`` if ``name_or_path`` is falsy."""
    if not name_or_path:
        return None
    with open(split_path(name_or_path)) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def _ref_base(sample):
    for el in sample:
        if isinstance(el, str) and el.endswith("_flow.npy"):
            return osp.basename(el)[: -len("_flow.npy")]
    for el in sample:
        if isinstance(el, str) and el.endswith(".png"):
            return osp.splitext(osp.basename(el))[0]
    return osp.splitext(osp.basename(str(sample[0])))[0]


def filter_samples(samples, frames, verbose=True):
    """Keep samples whose reference frame (the ``<t>_flow.npy`` element) is in ``frames``.

    ``frames=None`` returns ``samples`` unchanged.
    """
    if frames is None:
        return samples
    kept = [s for s in samples if _ref_base(s) in frames]
    if verbose:
        print(f"split: {len(kept)}/{len(samples)} samples kept")
    return kept
