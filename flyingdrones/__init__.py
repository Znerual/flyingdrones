"""FlyingDrones: dataset loader, metrics and splits for the optical-flow benchmark.

Ground truth ``flow/<t>_flow.npy`` is BACKWARD flow (t -> t-1): evaluate a flow
model on the pair ``(I_t, I_{t-1})`` with ``I_t`` as the reference frame.
See ``flyingdrones.loader`` for the full convention.
"""

from .loader import (ARFLOW_WINDOW, BOF_WINDOW, IMAGE_RESIZE_BY_FAMILY, MEMFLOW_CLIP, MOF_WINDOW,
                     NATIVE_SIZE, RESOLUTIONS, TWO_FRAME, drone_mask, list_samples, load_sample,
                     read_flow, read_image, read_label, ref_base, resize_bilinear, resize_nearest)
from .metrics import METRIC_NAMES, REGIONS, aggregate, bootstrap_ci, compute_metrics
from .results_io import PairDumper, read_pairs_csv, summary_row, write_summary
from .splits import SPLIT_FILES, filter_samples, load_split_frames, split_path

__version__ = "1.0.0"

__all__ = [
    "ARFLOW_WINDOW", "BOF_WINDOW", "IMAGE_RESIZE_BY_FAMILY", "MEMFLOW_CLIP", "MOF_WINDOW",
    "NATIVE_SIZE", "RESOLUTIONS", "TWO_FRAME", "drone_mask", "list_samples", "load_sample",
    "read_flow", "read_image", "read_label", "ref_base", "resize_bilinear", "resize_nearest",
    "METRIC_NAMES", "REGIONS", "aggregate", "bootstrap_ci", "compute_metrics",
    "PairDumper", "read_pairs_csv", "summary_row", "write_summary",
    "SPLIT_FILES", "filter_samples", "load_split_frames", "split_path",
]
