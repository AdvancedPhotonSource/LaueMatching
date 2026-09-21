"""Orientation filtering — the single source of truth.

REFACTOR_PLAN §4b / §6.2.  Previously the unique-spot filter lived inline in
``RunImage._process_indexing_results`` *and* in ``lsu.filter_orientations`` *and*
was driven by ``laue_postprocess`` — three copies that drifted (the Sigma3-twin
fix had to be wired into RunImage's inline copy separately).  This module is the
one home; ``lsu`` re-exports it for back-compat and the callers use it directly.

Two strategies (selected from config, REFACTOR_PLAN §4c):
  * :class:`LegacyUniqueSpotFilter` -- winner-take-all (exclusive) spot count
    (deletes real Sigma3 twins; back-compat only).
  * :class:`RobustCSLAwareFilter`   — twin/CSL-aware dedup (the 2026-06-21 fix).

The functions keep their explicit column-kwarg signatures (so ``lsu`` stays a
thin shim); the classes wrap them using a :class:`~laue_index.records.SolutionFormat`
column map so callers never pass magic numbers.
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol

import numpy as np

from .geometry import CUBIC_OPS, disorientation_deg_axis, is_csl_related
from .records import SOLUTION_FORMATS, SolutionFormat

__all__ = [
    "calculate_unique_spots", "filter_orientations", "filter_orientations_robust",
    "OrientationFilter", "LegacyUniqueSpotFilter", "RobustCSLAwareFilter",
]


# ---------------------------------------------------------------------------
# Core algorithms (lifted verbatim from laue_stream_utils; behaviour pinned by
# tests/test_char_filtering.py + tests/test_char_geometry.py).
# ---------------------------------------------------------------------------

def calculate_unique_spots(
    orientations: np.ndarray,
    spots: np.ndarray,
    labels: np.ndarray,
    grain_col: int = 0,
    quality_col: int = 4,
    spot_grain_col: int = 0,
    spot_x_col: int = 5,
    spot_y_col: int = 6,
) -> Dict[int, Dict]:
    """Unique spots per orientation, prioritised by quality (winner-take-all).

    Orientations are processed best-quality first (stable sort); once a label or
    pixel position is claimed, lower-quality orientations cannot reuse it.

    "Unique" here means EXCLUSIVE WITHIN THIS FRAME'S SOLUTION LIST, not
    "distinct observed peak": an orientation whose every spot is also claimed by
    a better one scores 0 however many real peaks it explains. ``count`` is the
    claimed pixel positions (a pixel inside a label a better orientation
    already claimed is not claimed), ``unique_label_count`` the claimed
    segmentation labels. This is what ``unique_spots_per_orientation`` / ``Unique_Spots``
    store downstream. For the total evidence use the solution's ``NMatches``.
    """
    results: Dict[int, Dict] = {}
    if orientations.size == 0:
        return results
    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, 0)

    sorted_idx = np.argsort(orientations[:, quality_col], kind='stable')[::-1]
    assigned_labels: set = set()
    assigned_positions: set = set()

    for idx in sorted_idx:
        orient = orientations[idx]
        gn = int(orient[grain_col])
        results[gn] = {"count": 0, "unique_label_count": 0,
                       "unique_labels": set(), "positions": set()}
        orient_spots = spots[spots[:, spot_grain_col].astype(int) == gn]
        if orient_spots.size == 0:
            continue
        for spot in orient_spots:
            try:
                x, y = int(spot[spot_x_col]), int(spot[spot_y_col])
            except (IndexError, ValueError):
                continue
            pos = (x, y)
            if pos in assigned_positions:
                continue
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                lbl = labels[y, x]
                if lbl > 0 and lbl in assigned_labels:
                    continue
                results[gn]["positions"].add(pos)
                assigned_positions.add(pos)
                if lbl > 0:
                    results[gn]["unique_labels"].add(lbl)
                    assigned_labels.add(lbl)
            else:
                results[gn]["positions"].add(pos)
                assigned_positions.add(pos)
        results[gn]["count"] = len(results[gn]["positions"])
        results[gn]["unique_label_count"] = len(results[gn]["unique_labels"])

    return results


def filter_orientations(
    orientations: np.ndarray,
    unique_spot_info: Dict[int, Dict],
    min_unique: int = 2,
    grain_col: int = 0,
    quality_col: int = 4,
) -> np.ndarray:
    """Legacy filter: keep orientations with >= min_unique winner-take-all
    (exclusive) label spots. Deletes a real Sigma3 twin whose reflections the
    parent claimed first."""
    if orientations.size == 0:
        return orientations.copy()
    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, 0)

    sorted_orient = orientations[np.argsort(orientations[:, quality_col])[::-1]]
    keep = []
    for row in sorted_orient:
        gn = int(row[grain_col])
        info = unique_spot_info.get(gn, {})
        if info.get("unique_label_count", 0) >= min_unique:
            keep.append(row)
    if keep:
        return np.array(keep)
    return np.empty((0, orientations.shape[1]))


def filter_orientations_robust(
    orientations: np.ndarray,
    unique_spot_info: Dict[int, Dict],
    *,
    min_unique: int = 2,
    grain_col: int = 0,
    quality_col: int = 4,
    om_start_col: int = 22,
    nmatches_col: int = 5,
    max_angle_deg: float = 5.0,
    min_total_spots: int = 5,
    csl_sigmas=(3,),
    csl_tol_deg: float = 3.0,
    cubic: bool = True,
    ops: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Twin/CSL-aware orientation filter (the 2026-06-21 fix).

    Keeps a solution (best-quality first) when it clears the evidence floor
    (>= min_total_spots matched pixels of its OWN: the winner-take-all
    ``count`` -- pixels no better orientation claimed, and not inside a
    segmentation label a better orientation claimed) AND one of: it is the
    first kept, it has >= min_unique exclusive labels (``unique_label_count``),
    or it is CSL/twin-related to an already-kept one. ``count`` can exceed the
    label count (several own pixels in one new label, or own pixels on
    unlabelled background), which is the case the CSL exemption rescues.

    MONOTONIC IN OWN EVIDENCE (changed 0.7.2). The floor used to fall back to
    ``NMatches`` whenever ``count`` was 0, so a Sigma3 twin with ALL of its
    spots claimed by the parent (count 0 -> NMatches, e.g. 10) was kept while
    one with 1-4 positions of its own (count 1-4 < floor) was dropped: more
    evidence, worse outcome. The fallback now applies only when
    *unique_spot_info* has no entry for the grain at all (no winner-take-all
    information supplied), never to a computed count of 0. A solution with no
    pixel of its own is therefore dropped -- it cannot be told apart from a
    phantom symmetry variant riding on the parent's reflections.

    Why not floor on ``NMatches`` outright: the C writes only solutions with
    NMatches >= MinNrSpots, so that floor is vacuous and the CSL exemption would
    then keep EVERY Sigma3 variant of a kept grain. On the real frame101
    fixture it keeps 6 solutions instead of 2 -- four extra Sigma3 variants of
    the two kept grains, each at NMatches 5-6 with 0-1 own labels and 2-4 own
    pixels.

    Symmetry:
      * ``cubic=True`` -- near-duplicates (disorientation < max_angle_deg under
        the cubic group) are dropped, and the Sigma-CSL exemption applies.
      * ``cubic=False`` with ``ops`` (e.g. ``geometry.HEX_OPS``) -- near-
        duplicates are dropped under those ops; there is NO CSL exemption (the
        CSL table is cubic only).
      * ``cubic=False`` and ``ops=None`` -- neither: the floor plus the
        exclusive-spot test only (what every non-cubic group got before ``ops``).
    """
    if orientations.size == 0:
        return orientations.copy()
    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, 0)

    sym_ops = CUBIC_OPS if cubic else ops
    has_nmatches = orientations.shape[1] > nmatches_col
    order = np.argsort(orientations[:, quality_col])[::-1]
    kept_rows, kept_oms = [], []
    for idx in order:
        row = orientations[idx]
        gn = int(row[grain_col])
        info = unique_spot_info.get(gn)
        if info is not None:
            uniq = int(info.get("unique_label_count", 0))
            ntot = int(info.get("count", 0))
        else:
            # No winner-take-all information for this grain: the only evidence
            # available is the solution's own NMatches.
            uniq = 0
            ntot = 0
            if has_nmatches:
                try:
                    ntot = int(round(float(row[nmatches_col])))
                except (ValueError, TypeError):
                    ntot = 0
        om = None
        if sym_ops is not None and orientations.shape[1] >= om_start_col + 9:
            try:
                om = np.asarray(row[om_start_col:om_start_col + 9], dtype=float).reshape(3, 3)
            except (ValueError, TypeError):
                om = None

        if om is not None and kept_oms and \
                any(disorientation_deg_axis(om, k, sym_ops)[0] < max_angle_deg
                    for k in kept_oms):
            continue

        csl_exempt = (
            cubic and om is not None and bool(kept_oms) and ntot >= min_total_spots
            and any(is_csl_related(om, k, csl_sigmas, csl_tol_deg) for k in kept_oms)
        )

        if ntot >= min_total_spots and (csl_exempt or uniq >= min_unique or not kept_rows):
            kept_rows.append(row)
            if om is not None:
                kept_oms.append(om)

    if kept_rows:
        out = np.array(kept_rows)
        return out[np.argsort(out[:, quality_col])[::-1]]
    return np.empty((0, orientations.shape[1]))


# ---------------------------------------------------------------------------
# Strategy objects (REFACTOR_PLAN §4b) — config-selected, column-map driven.
# ---------------------------------------------------------------------------

class OrientationFilter(Protocol):
    def __call__(self, orientations: np.ndarray,
                 unique_spot_info: Dict[int, Dict]) -> np.ndarray: ...


class LegacyUniqueSpotFilter:
    """Keep solutions with >= min_unique winner-take-all (exclusive) spots.

    Deletes real Sigma3 twins; kept for back-compat (RobustFilter 0)."""

    def __init__(self, min_unique: int = 2, fmt: SolutionFormat = SOLUTION_FORMATS["runimage"]):
        self.min_unique = min_unique
        self.fmt = fmt

    def __call__(self, orientations, unique_spot_info):
        return filter_orientations(
            orientations, unique_spot_info, min_unique=self.min_unique,
            grain_col=self.fmt.grain, quality_col=self.fmt.quality)


class RobustCSLAwareFilter:
    """Twin/CSL-aware filter (the 2026-06-21 fix). See filter_orientations_robust."""

    def __init__(self, min_unique: int = 2, min_total_spots: int = 5,
                 max_angle_deg: float = 5.0, csl_sigmas=(3,), csl_tol_deg: float = 3.0,
                 cubic: bool = True, fmt: SolutionFormat = SOLUTION_FORMATS["runimage"],
                 ops: Optional[np.ndarray] = None):
        self.min_unique = min_unique
        self.min_total_spots = min_total_spots
        self.max_angle_deg = max_angle_deg
        self.csl_sigmas = csl_sigmas
        self.csl_tol_deg = csl_tol_deg
        self.cubic = cubic
        self.fmt = fmt
        self.ops = ops

    def __call__(self, orientations, unique_spot_info):
        return filter_orientations_robust(
            orientations, unique_spot_info,
            min_unique=self.min_unique, grain_col=self.fmt.grain,
            quality_col=self.fmt.quality, om_start_col=self.fmt.om_start,
            nmatches_col=self.fmt.n_matches, max_angle_deg=self.max_angle_deg,
            min_total_spots=self.min_total_spots, csl_sigmas=self.csl_sigmas,
            csl_tol_deg=self.csl_tol_deg, cubic=self.cubic, ops=self.ops)
