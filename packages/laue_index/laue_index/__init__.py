"""laue_index — the LaueMatching Python orchestration, packaged.

REFACTOR_PLAN §3: the loose ``scripts/`` are being migrated into this package,
laid out like the sibling ``laue_torch`` (curated public API, single-
responsibility modules, typed records).  This package stays independent of
``laue_torch`` and ``laue_jax``; shared pure math is duplicated with
``# TODO(unify-after-publish)`` until a common leaf can be extracted.

NOTE: those ``unify-after-publish`` TODOs are now actionable -- ``laue_torch``
and ``laue_jax`` are packaged for release rather than private and paper-tied,
so extracting a shared leaf is no longer blocked on publication.

Public API grows as modules land.  Today: typed solution records (§6.1).
"""

__version__ = "0.4.0"

from .records import Solution, SolutionFormat, SOLUTION_FORMATS, parse_solutions
from .postprocess import PostProcessor, PostProcessResult
from .calibrate import (
    Anchor, CalibrationResult, DetectorSpec, calibrate,
)

__all__ = [
    "Solution", "SolutionFormat", "SOLUTION_FORMATS", "parse_solutions",
    "PostProcessor", "PostProcessResult",
    "Anchor", "CalibrationResult", "DetectorSpec", "calibrate",
]
