"""laue_index — the LaueMatching Python orchestration, packaged.

REFACTOR_PLAN §3: the loose ``scripts/`` are being migrated into this package,
laid out like the in-repo ``laue_torch`` (curated public API, single-
responsibility modules, typed records).  This package stays independent of the
paper-tied ``laue_torch`` / ``laue_jax`` / ``jax_cpfem`` (see the §1.5
constraint box); shared pure math is duplicated with ``# TODO(unify-after-
publish)`` until a common leaf can be extracted.

Public API grows as modules land.  Today: typed solution records (§6.1).
"""
from .records import Solution, SolutionFormat, SOLUTION_FORMATS, parse_solutions
from .postprocess import PostProcessor, PostProcessResult

__all__ = [
    "Solution", "SolutionFormat", "SOLUTION_FORMATS", "parse_solutions",
    "PostProcessor", "PostProcessResult",
]
