"""The orchestration scripts, shipped with the package.

These modules used to live only in the repo's ``scripts/`` directory, which
meant ``pip install laue-index`` gave you the library and (since 0.2.0) the C
binaries, but nothing that could actually run an image through them. They now
live here and ship in the wheel; ``scripts/`` keeps a thin executable shim for
each so every documented ``python scripts/RunImage.py ...`` invocation, and the
shell pipeline that calls them, keep working from a checkout.

They import each other FLAT -- ``import laue_config``, ``import
laue_stream_utils`` -- and shell out to siblings by path (RunImage runs
GenerateHKLs.py; laue_simulation runs GenerateSimulation.py). Both work as long
as this directory is on ``sys.path``, which is what :func:`add_to_path` and
:func:`run_module` arrange. That is deliberate: it kept the move to a rename,
with no import rewriting in ~8000 lines of working code.

Import them flat, from one place, or not at all:

    from laue_index.pipeline import add_to_path
    add_to_path()
    import RunImage

Importing the same file under two names (``RunImage`` and
``laue_index.pipeline.RunImage``) makes two module objects, and patching one
does not affect the other.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

__all__ = ["PIPELINE_DIR", "add_to_path", "run_module", "SCRIPTS"]

#: Directory holding the orchestration modules (this package's own directory).
PIPELINE_DIR = Path(__file__).resolve().parent

#: The modules that are entry points -- each has a shim in the repo's scripts/.
SCRIPTS = (
    "RunImage",
    "GenerateHKLs",
    "GenerateSimulation",
    "GenerateOrientations",
    "ImageCleanup",
    "annotate_orientation_db",
    "laue_orchestrator",
    "laue_image_server",
    "laue_postprocess",
)


def add_to_path() -> str:
    """Put this directory first on ``sys.path`` so the modules import flat."""
    p = str(PIPELINE_DIR)
    if sys.path and sys.path[0] == p:
        return p
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    return p


def run_module(name: str, argv: list[str] | None = None) -> None:
    """Run one of :data:`SCRIPTS` as ``__main__``, as if invoked directly.

    Args:
        name: module name, without ``.py``.
        argv: replacement for ``sys.argv[1:]``; ``None`` keeps the caller's.
    """
    target = PIPELINE_DIR / f"{name}.py"
    if not target.is_file():
        raise FileNotFoundError(
            f"{name}.py is not part of laue_index.pipeline (looked in {PIPELINE_DIR}). "
            f"Available: {', '.join(sorted(p.stem for p in PIPELINE_DIR.glob('*.py')))}")
    add_to_path()
    if argv is not None:
        sys.argv = [str(target), *argv]
    runpy.run_path(str(target), run_name="__main__")
