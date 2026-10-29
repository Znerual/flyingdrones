"""FlyingDrones compatibility shim for upstream SEA-RAFT (princeton-vl/SEA-RAFT).

``scripts/analyze_drone_epe.py`` was written against a packaged SEA-RAFT variant
(``searaft/``) whose ``RAFT`` takes keyword arguments. This shim exposes that interface
on top of the unmodified upstream ``core/`` modules; network definition and forward
computation are upstream's. Place this folder at ``<SEA-RAFT checkout>/searaft/``.
"""
