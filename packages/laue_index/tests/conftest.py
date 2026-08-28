"""Shared pytest setup for the laue_index test suite.

Puts the legacy ``scripts/`` directory on ``sys.path`` so the characterization
tests can import the *current* implementation (``laue_stream_utils``,
``laue_config``, …) as the behaviour baseline.  The ``laue_index`` package is
imported the same way and these golden anchors must keep passing unchanged.

``scripts/`` lives at the REPO root, while the package now lives at
``packages/laue_index/``.  Both are discovered rather than hardcoded by depth,
so the suite keeps working if the tree is rearranged again.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = Path(__file__).resolve()

# The distribution root: packages/laue_index/ (contains the laue_index package).
_PKG_ROOT = _HERE.parents[1]

# The repo root: identified by scripts/ + the C build, which the
# characterization tests need. None when running against an installed package.
_REPO_ROOT = None
for _candidate in _HERE.parents:
    if (_candidate / "scripts").is_dir() and (_candidate / "CMakeLists.txt").is_file():
        _REPO_ROOT = _candidate
        break

# Package root first so ``import laue_index`` resolves the package; scripts/
# next so the legacy modules (laue_stream_utils, laue_config) import flat.
_paths = [str(_PKG_ROOT)]
if _REPO_ROOT is not None:
    _paths.append(str(_REPO_ROOT / "scripts"))

for _p in reversed(_paths):
    if _p not in sys.path:
        sys.path.insert(0, _p)
