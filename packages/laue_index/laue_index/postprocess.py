"""PostProcessor stage — solutions -> unique-spots -> filter -> spot-filter.

REFACTOR_PLAN §3 / §6.5.  The result-processing core that was inlined in
RunImage._process_indexing_results (and duplicated in laue_postprocess): given
parsed orientation + spot arrays and the segmentation labels, compute per-
orientation unique spots, sort by quality, apply the config-selected
OrientationFilter, and filter spots to the kept grains.

Pure transform (no file I/O — that stays in the orchestrator / output.py).
Column access via a SolutionFormat (no magic numbers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set

import numpy as np

from .filtering import (calculate_unique_spots, LegacyUniqueSpotFilter,
                        RobustCSLAwareFilter)
from .records import SOLUTION_FORMATS, SolutionFormat

__all__ = ["PostProcessResult", "PostProcessor", "sort_by_quality"]


def sort_by_quality(orientations: np.ndarray, quality_col: int) -> np.ndarray:
    """Sort rows by quality (descending).  Faithful to the legacy
    lsu.sort_orientations_by_quality guards (empty / 1-D / bad column)."""
    if orientations.size == 0 or orientations.ndim == 1:
        return orientations.copy() if orientations.size else orientations
    try:
        return orientations[np.argsort(orientations[:, quality_col])[::-1]].copy()
    except IndexError:
        return orientations.copy()


@dataclass
class PostProcessResult:
    orientations_sorted: np.ndarray
    filtered_orientations: np.ndarray
    filtered_spots: np.ndarray
    unique_spot_info: Dict[int, Dict]
    kept_grain_nrs: Set[int] = field(default_factory=set)


class PostProcessor:
    """Config-selected result post-processing (REFACTOR_PLAN §4c/§6.5)."""

    def __init__(self, *, robust: bool = True, min_unique: int = 2,
                 min_total_spots: int = 5, max_angle_deg: float = 5.0,
                 space_group: int = 225, csl_sigmas=(3,), csl_tol_deg: float = 3.0,
                 fmt: SolutionFormat = SOLUTION_FORMATS["runimage"]):
        self.fmt = fmt
        self.is_cubic = 195 <= space_group <= 230
        if robust:
            self.ofilter = RobustCSLAwareFilter(
                min_unique=min_unique, min_total_spots=min_total_spots,
                max_angle_deg=max_angle_deg, csl_sigmas=csl_sigmas,
                csl_tol_deg=csl_tol_deg, cubic=self.is_cubic, fmt=fmt)
        else:
            self.ofilter = LegacyUniqueSpotFilter(min_unique=min_unique, fmt=fmt)

    def __call__(self, orientations: np.ndarray, spots: np.ndarray,
                 labels: np.ndarray) -> PostProcessResult:
        f = self.fmt
        usi = calculate_unique_spots(
            orientations, spots, labels,
            grain_col=f.grain, quality_col=f.quality,
            spot_grain_col=f.spot_grain, spot_x_col=f.spot_x, spot_y_col=f.spot_y)

        osort = sort_by_quality(orientations, f.quality)
        if osort.size > 0 and osort.ndim == 1:
            osort = np.expand_dims(osort, 0)

        kept: Set[int] = set()
        if osort.size == 0:
            filtered = osort.copy()
        else:
            filtered = self.ofilter(osort, usi)
            if filtered.size > 0:
                kept = set(filtered[:, f.grain].astype(int))

        if spots.size and kept:
            fspots = spots[np.isin(spots[:, f.spot_grain].astype(int), list(kept))]
        else:
            ncols = spots.shape[1] if spots.ndim == 2 else 0
            fspots = np.empty((0, ncols))

        return PostProcessResult(osort, filtered, fspots, usi, kept)
