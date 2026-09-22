"""laue_index — the LaueMatching Python orchestration, packaged.

REFACTOR_PLAN §3: the loose ``scripts/`` are being migrated into this package,
laid out like the sibling ``laue_torch`` (curated public API, single-
responsibility modules, typed records).  This package stays independent of
``laue_torch`` and ``laue_jax``; shared pure math is duplicated with
``# TODO(unify-after-publish)`` until a common leaf can be extracted.

NOTE: those ``unify-after-publish`` TODOs are now actionable -- ``laue_torch``
and ``laue_jax`` are packaged for release rather than private and paper-tied,
so extracting a shared leaf is no longer blocked on publication.

Public API: typed solution records (§6.1), the ``PostProcessor`` stage, detector
calibration, XMAS geometry conversion and the build manifest (see ``__all__``).
The pipeline orchestrators live in ``laue_index.pipeline``; the indexer wrapper,
filtering, preprocessing and worker sizing are importable as submodules.
"""

__version__ = "0.7.3"

from .records import Solution, SolutionFormat, SOLUTION_FORMATS, parse_solutions
from .postprocess import PostProcessor, PostProcessResult
from .calibrate import (
    Anchor, CalibrationResult, DetectorSpec, calibrate,
)
from .buildmeta import build_info
from .xmas import (
    LaueGeometry, XmasCalibration, convert as xmas_to_laue,
    enumerate_candidates as xmas_candidates,
)

__all__ = [
    "Solution", "SolutionFormat", "SOLUTION_FORMATS", "parse_solutions",
    "PostProcessor", "PostProcessResult",
    "Anchor", "CalibrationResult", "DetectorSpec", "calibrate",
    "XmasCalibration", "LaueGeometry", "xmas_to_laue", "xmas_candidates",
    "build_info",
]
