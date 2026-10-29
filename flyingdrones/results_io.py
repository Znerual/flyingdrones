"""Result files written by the evaluators.

* per-pair CSV (``--pairs_csv``): one row per evaluated pair, columns
  ``ref, <metric>_<region>..., n_drone_px`` (full precision). Bootstrap CIs and all
  paper tables are computed from these files.
* summary CSV (``--output_csv``): one row per run in the legacy column layout
  ``SUMMARY_FIELDS`` (``epe`` = full frame, ``epe_drone`` = drone region, ...).
"""

import csv
import os
import os.path as osp

from .metrics import METRIC_NAMES, REGIONS, aggregate

PAIR_FIELDS = ["ref"] + [f"{k}_{r}" for r in REGIONS for k in METRIC_NAMES] + ["n_drone_px"]

SUMMARY_FIELDS = ["checkpoint", "dataset", "image_size",
                  "epe", "epe_drone", "epe_norm", "epe_norm_drone",
                  "px1_err", "px1_err_drone", "px3_err", "px3_err_drone",
                  "px5_err", "px5_err_drone", "fl", "fl_drone",
                  "samples_processed", "samples_skipped"]


class PairDumper:
    """Collect per-pair metric rows; optionally stream them to a CSV file."""

    FIELDS = PAIR_FIELDS

    def __init__(self, path=None):
        self.path = path
        self.rows = []
        self._fh = None
        if path:
            os.makedirs(osp.dirname(osp.abspath(path)) or ".", exist_ok=True)
            self._fh = open(path, "w", newline="")
            self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS, extrasaction="ignore")
            self._w.writeheader()

    def add(self, ref, metrics):
        row = {"ref": ref, **metrics}
        self.rows.append(row)
        if self._fh:
            self._w.writerow(row)
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def read_pairs_csv(path):
    """Read a per-pair CSV back into a list of dicts (floats, NaN for empty cells)."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            row = {"ref": r["ref"]}
            for k, v in r.items():
                if k != "ref":
                    row[k] = float(v) if v not in ("", None) else float("nan")
            rows.append(row)
    return rows


def summary_row(rows, checkpoint, dataset, image_size, skipped=0):
    """One summary row (``SUMMARY_FIELDS``) from per-pair rows."""
    a = aggregate(rows)
    row = {"checkpoint": checkpoint, "dataset": dataset, "image_size": image_size}
    for k in METRIC_NAMES:
        row[k] = a[f"{k}_all"]
        row[f"{k}_drone"] = a[f"{k}_drone"]
    row["samples_processed"] = len(rows)
    row["samples_skipped"] = skipped
    return row


def write_summary(path, row):
    """Append ``row`` to ``path`` (header written if the file is new)."""
    new = not osp.exists(path)
    os.makedirs(osp.dirname(osp.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
