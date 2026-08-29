"""Shared pytest setup for the laue_index test suite.

The orchestration modules (``RunImage``, ``laue_config``, ``laue_stream_utils``,
…) import each other FLAT, so the directory holding them has to be on
``sys.path``. That directory used to be the repo's ``scripts/``; it is now
``laue_index/pipeline/`` inside the package, which is why the tests keep
passing unchanged after the move and why they now also run against an
installed package with no checkout in sight.

``scripts/`` must NOT go on ``sys.path`` any more: it holds one-line shims with
the same filenames, and importing those instead of the real modules would give
tests a module object with none of the functions they patch.

Paths are discovered, not hardcoded by depth, so the suite survives the tree
being rearranged again.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = Path(__file__).resolve()

# The distribution root: packages/laue_index/ (contains the laue_index package).
_PKG_ROOT = _HERE.parents[1]

# Where the orchestration modules live. Prefer this checkout's copy; fall back
# to whatever `import laue_index` resolves to (an installed package).
_PIPELINE = _PKG_ROOT / "laue_index" / "pipeline"
if not _PIPELINE.is_dir():
    sys.path.insert(0, str(_PKG_ROOT))
    from laue_index.pipeline import PIPELINE_DIR as _PIPELINE  # noqa: E402

# The repo root: identified by scripts/ + the C build. None when running
# against an installed package; tests that need the repo skip themselves.
_REPO_ROOT = None
for _candidate in _HERE.parents:
    if (_candidate / "scripts").is_dir() and (_candidate / "CMakeLists.txt").is_file():
        _REPO_ROOT = _candidate
        break

# Package root first so ``import laue_index`` resolves the package; the
# pipeline directory next so the orchestration modules import flat.
for _p in (str(_PIPELINE), str(_PKG_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
