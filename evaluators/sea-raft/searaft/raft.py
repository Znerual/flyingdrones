"""Shim imported by the evaluator as top-level ``raft`` (it puts ``searaft/`` on sys.path).

``RAFT(**cfg)`` builds the unmodified upstream ``core/raft.py`` RAFT from the keyword
arguments of an upstream ``config/eval/*.json``. ``ModifiedRAFT`` / ``FocusRAFT`` are
names the evaluator imports but never instantiates for released checkpoints.
"""
import argparse

from searaft._upstream import load_core

_up = load_core("raft.py")


class RAFT(_up.RAFT):
    def __init__(self, **kwargs):
        super().__init__(argparse.Namespace(**kwargs))


class ModifiedRAFT(RAFT):
    def __init__(self, **kwargs):
        raise NotImplementedError("frame-differencing variant is not part of the FlyingDrones benchmark")


class FocusRAFT(RAFT):
    def __init__(self, **kwargs):
        raise NotImplementedError("sub-region variant is not part of the FlyingDrones benchmark")
