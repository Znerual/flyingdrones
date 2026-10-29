"""FlyingDrones compatibility shim for upstream RAFT (princeton-vl/RAFT).

``scripts/analyze_drone_epe.py`` was written against a packaged RAFT variant that
(a) is importable as ``raft``, (b) takes keyword arguments in the constructor
(``RAFT(small=..., dropout=..., alternate_corr=..., mixed_precision=...)``) and
(c) returns a dict ``{'final': flow_up, ...}`` in ``test_mode``. This shim provides
exactly that interface on top of the unmodified upstream ``core/raft.py``; the network
definition and forward computation are upstream's (no numerical change).

Place this folder at ``<RAFT checkout>/raft/`` (next to upstream ``core/``).
"""
import argparse
import importlib.util
import os
import sys

_CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")
if _CORE not in sys.path:
    # upstream core/ uses flat imports (``from update import ...``, ``from utils.utils import ...``)
    sys.path.append(_CORE)

_spec = importlib.util.spec_from_file_location("_raft_upstream_core", os.path.join(_CORE, "raft.py"))
_upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_upstream)


class RAFT(_upstream.RAFT):
    """Upstream RAFT with a keyword constructor and dict outputs."""

    def __init__(self, small=False, dropout=0, alternate_corr=False, mixed_precision=False, **kwargs):
        super().__init__(argparse.Namespace(small=bool(small), dropout=float(dropout),
                                            alternate_corr=bool(alternate_corr),
                                            mixed_precision=bool(mixed_precision)))

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        out = super().forward(image1, image2, iters=iters, flow_init=flow_init,
                              upsample=upsample, test_mode=test_mode)
        if test_mode:
            flow_low, flow_up = out
            return {"final": flow_up, "flow_low": flow_low}
        return {"final": out[-1], "flow": out}


__all__ = ["RAFT"]
