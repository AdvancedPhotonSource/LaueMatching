"""Typed solution records and the single solution parser.

REFACTOR_PLAN §5 / §6.1.  The C indexer writes a fixed-width text table whose
columns the Python side previously read by positional "magic numbers"
(``OM = cols 22:31``, ``NMatches = col 5``, ``grain = col 0 or 1``) scattered
across ``RunImage`` and ``laue_postprocess`` with ``n_cols > 30`` heuristics.

This module centralises that knowledge:

* :class:`SolutionFormat` — the column map for one on-disk layout.
* :data:`SOLUTION_FORMATS` — the two layouts the indexer emits:
    - ``"runimage"`` : 34 columns, ``GrainNr`` at col 0.
    - ``"stream"``   : 35 columns, ``ImageNr`` prepended (every field ``+1``).
* :class:`Solution` — a typed record (the ``ForwardAux`` analogue) with named
  fields, so downstream code never indexes by column again.
* :func:`parse_solutions` — the one parser, from a path or an already-loaded
  array, selected by an explicit ``fmt`` (no fragile column-count heuristic).

Pure: numpy only, no I/O state, no torch.
``# TODO(unify-after-publish)``: the orientation/recip/lattice math here is
duplicated from the paper-tied ``laue_torch`` on purpose (see REFACTOR_PLAN
§1.5 constraint box); extract a shared leaf once those packages publish.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Solution", "SolutionFormat", "SOLUTION_FORMATS", "parse_solutions"]


@dataclass(frozen=True)
class SolutionFormat:
    """Column map for one on-disk solution-table layout (0-based indices)."""
    name: str
    n_cols: int          # expected column count (used for validation)
    grain: int
    intensity: int
    quality: int         # NMatches * sqrt(Intensity)
    n_matches: int
    n_spots_calc: int
    recip_start: int     # 9 contiguous: reciprocal matrix (row-major 3x3)
    lattice_start: int   # 6 contiguous: a, b, c, alpha, beta, gamma
    om_start: int        # 9 contiguous: orientation matrix (row-major 3x3)
    coarse_quality: int
    misorientation: int
    row_nr: int          # orientationRowNr (index into the orientation DB)
    image_nr: int = -1   # ImageNr column, or -1 if the layout has none

    # --- spots-file companion columns (the .spots.txt for this layout) ---
    spot_grain: int = 0
    spot_x: int = 5
    spot_y: int = 6


# RunImage layout: GrainNr first, 34 columns.
_RUNIMAGE = SolutionFormat(
    name="runimage", n_cols=34,
    grain=0, intensity=2, quality=4, n_matches=5, n_spots_calc=6,
    recip_start=7, lattice_start=16, om_start=22,
    coarse_quality=31, misorientation=32, row_nr=33, image_nr=-1,
    spot_grain=0, spot_x=5, spot_y=6,
)

# Stream layout: ImageNr prepended -> every field shifted +1, 35 columns.
_STREAM = SolutionFormat(
    name="stream", n_cols=35,
    grain=1, intensity=3, quality=5, n_matches=6, n_spots_calc=7,
    recip_start=8, lattice_start=17, om_start=23,
    coarse_quality=32, misorientation=33, row_nr=34, image_nr=0,
    spot_grain=1, spot_x=6, spot_y=7,
)

SOLUTION_FORMATS = {"runimage": _RUNIMAGE, "stream": _STREAM}


@dataclass
class Solution:
    """One indexed orientation, parsed from a solution-table row.

    Fields are the named replacement for positional column access.  ``raw``
    keeps the original row verbatim so a solution can be written back without
    reconstructing formatting (behaviour-preserving round-trips).
    """
    grain_nr: int
    n_matches: int
    quality: float                 # NMatches * sqrt(Intensity)
    intensity: float
    orientation: np.ndarray        # (3, 3) row-major
    recip: np.ndarray              # (3, 3) row-major
    lattice: np.ndarray            # (6,)  a, b, c, alpha, beta, gamma
    misorientation_post_refine: float
    # provenance
    orientation_row_nr: int        # row index into the orientation database
    source_row: int                # index of this row in the parsed array
    n_spots_calc: int = 0
    coarse_quality: float = 0.0
    image_nr: int | None = None
    raw: np.ndarray = field(default_factory=lambda: np.empty(0))

    @classmethod
    def from_row(cls, row: np.ndarray, fmt: SolutionFormat, source_row: int) -> "Solution":
        return cls(
            grain_nr=int(row[fmt.grain]),
            n_matches=int(round(float(row[fmt.n_matches]))),
            quality=float(row[fmt.quality]),
            intensity=float(row[fmt.intensity]),
            orientation=np.asarray(row[fmt.om_start:fmt.om_start + 9],
                                   dtype=float).reshape(3, 3),
            recip=np.asarray(row[fmt.recip_start:fmt.recip_start + 9],
                             dtype=float).reshape(3, 3),
            lattice=np.asarray(row[fmt.lattice_start:fmt.lattice_start + 6],
                               dtype=float),
            misorientation_post_refine=float(row[fmt.misorientation]),
            orientation_row_nr=int(round(float(row[fmt.row_nr]))),
            source_row=source_row,
            n_spots_calc=int(round(float(row[fmt.n_spots_calc]))),
            coarse_quality=float(row[fmt.coarse_quality]),
            image_nr=(int(row[fmt.image_nr]) if fmt.image_nr >= 0 else None),
            raw=np.asarray(row, dtype=float).copy(),
        )


def _load_array(source) -> np.ndarray:
    """Load a solution table from a path, or pass through an array."""
    if isinstance(source, np.ndarray):
        data = source
    elif isinstance(source, (str, Path)):
        try:
            data = np.loadtxt(source, skiprows=1)
        except ValueError:
            data = np.genfromtxt(source, skip_header=1)
    else:
        raise TypeError(f"parse_solutions: unsupported source type {type(source)!r}")
    if data.size == 0:
        return np.empty((0, 0))
    if data.ndim == 1:
        data = data[None, :]
    return data


def parse_solutions(source, fmt: str = "runimage") -> list[Solution]:
    """Parse a solution table into typed :class:`Solution` records.

    Args:
        source: path to a ``*.solutions.txt`` file, or a pre-loaded 2-D array
            (e.g. from ``read_solutions`` / an H5 dataset).
        fmt: ``"runimage"`` (34 cols, GrainNr first) or ``"stream"`` (35 cols,
            ImageNr prepended).  Explicit — no column-count guessing.

    Returns:
        One :class:`Solution` per row, in source order.
    """
    if fmt not in SOLUTION_FORMATS:
        raise ValueError(f"Unknown solution format {fmt!r}; "
                         f"choose from {sorted(SOLUTION_FORMATS)}")
    spec = SOLUTION_FORMATS[fmt]
    data = _load_array(source)
    if data.size == 0:
        return []
    if data.shape[1] < spec.n_cols:
        raise ValueError(
            f"Solution table has {data.shape[1]} columns but format "
            f"{fmt!r} needs at least {spec.n_cols}.")
    return [Solution.from_row(data[i], spec, i) for i in range(data.shape[0])]
