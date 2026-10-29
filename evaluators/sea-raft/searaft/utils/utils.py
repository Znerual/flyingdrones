"""Re-export upstream SEA-RAFT ``core/utils/utils.py`` (InputPadder, coords_grid, ...).

Upstream ``core/*.py`` import ``utils.utils``; because the evaluator puts ``searaft/`` on
sys.path, that name may resolve to this file, so it must expose the upstream content."""
from searaft._upstream import load_core

_m = load_core("utils/utils.py")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
