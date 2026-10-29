"""Load upstream SEA-RAFT ``core/`` modules by file path (avoids module-name clashes:
the evaluator imports this shim's ``raft.py`` as top-level ``raft``)."""
import importlib.util
import os
import sys

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")
if CORE not in sys.path:
    # upstream core/ uses flat imports (``from layer import ...``, ``from update import ...``)
    sys.path.append(CORE)

_cache = {}


def load_core(relpath):
    """Import ``core/<relpath>`` once under a private module name and return it."""
    if relpath not in _cache:
        name = "_searaft_upstream_" + relpath.replace("/", "_").replace(".py", "")
        spec = importlib.util.spec_from_file_location(name, os.path.join(CORE, relpath))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _cache[relpath] = mod
    return _cache[relpath]
