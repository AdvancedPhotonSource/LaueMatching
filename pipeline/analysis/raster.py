"""Shared raster conventions for the analysis scripts.

GRAIN CONNECTIVITY. Every script that splits a set of beam positions into
connected pieces must use the same neighbourhood, or two scripts describing the
same grain disagree about how many pieces it has. Before 2026-09 they did:
``regrain.py`` used 8-connectivity while its own usage string said 4,
``collect_scan_metrics.py`` and ``substrate_deposit.py`` used 8, and
``big_grain_*`` and ``variant_coherence.py`` used ``ndimage.label``'s default,
which is 4.

8 is the value here because it is what ``regrain.py`` -- the script that
produced every reported grain count -- actually ran with. Changing it changes
those counts; do that deliberately, through ``LAUE_CONNECTIVITY``, and say so
beside the number.

RASTER POSITION. See :func:`raster_positions`. Before 2026-09 about a dozen
scripts each parsed the frame number out of the file name and divided by a
hard-coded 201, so any scan that was not 201 columns wide (81x81, 201x101 read
as square, ...) was mis-placed or crashed on an out-of-range index.

ONE ORIENTATION PER POSITION. See :func:`winner_per_position`. Five figure
scripts each kept "the highest-``nhit`` instance" at a position with their own
loop, ranking by a count that stacks harmonics and breaking ties by input order.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage as ndi


def connectivity() -> int:
    """4 or 8, from ``LAUE_CONNECTIVITY`` (default 8). Anything else exits."""
    raw = os.environ.get("LAUE_CONNECTIVITY", "8").strip()
    if raw not in ("4", "8"):
        raise SystemExit(f"LAUE_CONNECTIVITY must be 4 or 8, got {raw!r}")
    return int(raw)


def structure():
    """The ``ndimage.label`` structuring element for :func:`connectivity`."""
    return ndi.generate_binary_structure(2, 2 if connectivity() == 8 else 1)


# ---------------------------------------------------------------------------
# raster position
# ---------------------------------------------------------------------------
def raster_shape() -> tuple:
    """``(n_rows, n_cols)`` of the scan raster, from the environment. No default.

    ``LAUE_NR``     frames per raster row = number of COLUMNS (fast axis)
    ``LAUE_NROWS``  number of raster rows (slow axis)

    Both are required: a default is exactly how a 201x101 scan came to be read as
    201x201. Exits with a message naming the variable if either is unset or not a
    positive integer.
    """
    out = []
    for var, what in (("LAUE_NROWS", "number of raster rows (slow axis)"),
                      ("LAUE_NR", "frames per raster row = number of columns (fast axis)")):
        raw = os.environ.get(var, "").strip()
        if not raw:
            sys.exit(f"{var} is not set: it must give the {what} of this scan. "
                     f"There is no default -- read it from the scan's own record.")
        try:
            v = int(raw)
        except ValueError:
            sys.exit(f"{var} must be a positive integer ({what}), got {raw!r}")
        if v <= 0:
            sys.exit(f"{var} must be a positive integer ({what}), got {raw!r}")
        out.append(v)
    return tuple(out)


def _required_float(var, what, positive=True):
    raw = os.environ.get(var, "").strip()
    if not raw:
        sys.exit(f"{var} is not set: it must give {what}. There is no default.")
    try:
        v = float(raw)
    except ValueError:
        sys.exit(f"{var} must be a number ({what}), got {raw!r}")
    if not np.isfinite(v) or (positive and v <= 0):
        sys.exit(f"{var} must be a {'positive ' if positive else ''}finite number ({what}), got {raw!r}")
    return v


def step_um() -> float:
    """Raster step in micrometres, from ``LAUE_STEP_UM``. Required, no default.

    Every conversion of a raster index to micrometres (plot extents, scale bars,
    optical registration) goes through this. The scripts used to assume 1 um per
    position, which is wrong for any coarser scan (a 10 um survey is drawn 10x too
    small). One value for both axes: it is the STAGE step; on a 45-degree mount the
    slow-axis distance in the surface frame is not the same as the stage step (see
    :func:`raster_positions`).
    """
    return _required_float("LAUE_STEP_UM", "the raster step in micrometres (stage step "
                           "between adjacent positions)")


def mount_deg() -> float:
    """Sample mount angle in degrees, from ``LAUE_MOUNT_DEG``. Required, no default.

    The angle between the sample SURFACE and the stage's slow (Z) axis: a slow-axis
    stage step ``dz`` covers ``dz / cos(mount_deg)`` of surface, so surface-frame
    slow coordinates are the stage values divided by ``cos(radians(mount_deg()))``.
    45 is the 34-ID-E reflection mount (the factor sqrt(2) the scripts used to
    hard-code); 0 means the surface lies along the slow axis (no de-projection).
    Must be in [0, 90).
    """
    v = _required_float("LAUE_MOUNT_DEG", "the sample mount angle in degrees between the "
                        "sample surface and the stage slow axis (45 for the 34-ID-E "
                        "reflection mount, 0 for none)", positive=False)
    if not 0.0 <= v < 90.0:
        sys.exit(f"LAUE_MOUNT_DEG must be in [0, 90) degrees, got {v:g}")
    return v


def centred_extent(nrows, ncols, step):
    """imshow extent ``[-hx, hx, -hy, hy]`` in um, centred on the raster centre."""
    hx, hy = (ncols - 1) / 2 * step, (nrows - 1) / 2 * step
    return [-hx, hx, -hy, hy]


def optical_flip_y() -> int:
    """``LAUE_OPTICAL_FLIP_Y``: +1 if the optical imager is vertically flipped vs the scan, -1 if not."""
    raw = os.environ.get("LAUE_OPTICAL_FLIP_Y", "").strip()
    if raw not in ("1", "+1", "-1"):
        sys.exit("LAUE_OPTICAL_FLIP_Y must be set to +1 (optical image is vertically flipped "
                 f"relative to the scan) or -1 (not flipped); got {raw!r}. It is a property "
                 "of the micrograph and has no default.")
    return int(raw)


def optical_registration():
    """``(cx, cy, px_per_um, flip_y)`` registering the scan to an optical micrograph.

    All four are measured on ONE micrograph (the scan-centre markers' pixel, the
    scale bar, and whether the imager is flipped), so they are properties of that
    image, not of the analysis, and are required whenever a registration is used:

    ``LAUE_OPTICAL_CX``, ``LAUE_OPTICAL_CY``  scan centre in optical pixels
    ``LAUE_OPTICAL_PX_PER_UM``            image scale, pixels per micrometre
    ``LAUE_OPTICAL_FLIP_Y``               +1 flipped vertically, -1 not

    They were hard-coded (398, 284, 0.6, +1) for one campaign's image; applying
    those to another image registers it wrongly with no sign of error.
    """
    cx = _required_float("LAUE_OPTICAL_CX", "the scan centre's x pixel in the optical image",
                         positive=False)
    cy = _required_float("LAUE_OPTICAL_CY", "the scan centre's y pixel in the optical image",
                         positive=False)
    ppu = _required_float("LAUE_OPTICAL_PX_PER_UM",
                          "the optical image scale in pixels per micrometre (from its scale bar)")
    return cx, cy, ppu, optical_flip_y()


def frame_number(name) -> int:
    """``'<prefix>_000123.h5'`` -> 123: the last ``_``-separated token before the extension.

    This is the parse every script used before; kept so the 201-column results are
    reproduced exactly. Raises ``ValueError`` naming the file if there is no number.
    """
    base = os.path.basename(str(name))
    tok = base.split("_")[-1].split(".")[0]
    if not tok.isdigit():
        raise ValueError(f"cannot parse a frame number from {name!r}")
    return int(tok)


def _grid_index(v):
    """Integer grid index along one stage axis, relative to the lowest value present."""
    v = np.asarray(v, float)
    u = np.unique(np.round(v, 4))
    d = np.diff(u)
    d = d[d > 1e-6]
    if not len(d):
        return np.zeros(len(v), int)
    step = float(np.median(d))
    return np.rint((v - u[0]) / step).astype(int)


def raster_positions(frames=None, X=None, Z=None, shape=None):
    """``(row, col)`` integer raster indices for each instance.

    CONVENTION. Row-major, 1-based frame numbers: frame ``N`` sits at
    ``row = (N-1) // n_cols``, ``col = (N-1) % n_cols``. The FAST axis is the
    column (consecutive frames step along it; the stage's ``sampleX``); the SLOW
    axis is the row (``sampleZ``). On the 45-degree-mounted reflection specimens
    the slow axis is the 45-degree axis: a scatterer at depth ``d`` below the
    surface appears displaced by ``0.70711*d`` along the slow axis and by 0 along
    the fast axis, so any extent measured along the slow axis is inflated by the
    depth projection of the illuminated column, and a fast-vs-slow anisotropy is
    not by itself a sample property.

    SOURCE, in order of preference:

    1. The frame number with an EXPLICIT raster shape (``shape=(n_rows, n_cols)``,
       else :func:`raster_shape` from ``LAUE_NROWS``/``LAUE_NR``, required). This
       comes first, ahead of the recorded stage coordinates, because on a
       unidirectional raster the frame index IS the position, whereas the 34-ID-E
       fast-axis readback races the move and labels ~0.9% of frames one step ahead
       (``fix_positions.py``; on sampleH 180 of 20,301 frames). If stage
       coordinates are passed too they are used as a CHECK: frames that share a
       frame-derived row must share one slow-axis coordinate. A wrong ``LAUE_NR``
       mixes slow-axis values within most rows, and this exits rather than draw a
       sheared map.
    2. Only when no frame names are given (or none parse): the stage coordinates
       ``X`` (fast -> col) and ``Z`` (slow -> row), gridded by their median step.
       Indices are then relative to the LOWEST coordinate present among these
       instances, so they are not guaranteed to line up with a map built from all
       frames; a warning says so, and another if ``X`` is a raw readback.

    Frame numbers outside ``1..n_rows*n_cols`` exit: the shape is wrong.
    """
    if frames is not None:
        try:
            n = np.array([frame_number(f) for f in frames], int)
        except ValueError:
            n = None
        if n is not None and len(n):
            nrows, ncols = shape if shape is not None else raster_shape()
            lo, hi = int(n.min()), int(n.max())
            if lo < 1 or hi > nrows * ncols:
                sys.exit(f"frame numbers {lo}..{hi} do not fit a {nrows} x {ncols} raster "
                         f"(LAUE_NROWS x LAUE_NR): the raster shape is wrong for this scan")
            row, col = (n - 1) // ncols, (n - 1) % ncols
            if Z is not None:
                _check_rows_against_slow_axis(row, Z, ncols)
            return row, col
        if n is not None:                           # empty input
            return np.zeros(0, int), np.zeros(0, int)
    if X is None or Z is None:
        sys.exit("raster_positions: no parseable frame names and no stage coordinates "
                 "(X, Z) -- cannot place the instances on the raster")
    print("WARNING: raster position taken from stage coordinates (no frame numbers); "
          "indices are relative to the lowest coordinate present and may not align "
          "with maps built from every frame", file=sys.stderr)
    return _grid_index(Z), _grid_index(X)


def _check_rows_against_slow_axis(row, Z, ncols, max_mixed=0.5):
    """Exit if frame-derived rows do not correspond to single slow-axis positions."""
    Z = np.asarray(Z, float)
    u = np.unique(np.round(Z, 4))
    d = np.diff(u)
    d = d[d > 1e-6]
    if not len(d):
        return
    dz = float(np.median(d))
    order = np.argsort(row, kind="stable")
    r, z = row[order], Z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    zmax = np.maximum.reduceat(z, starts)
    zmin = np.minimum.reduceat(z, starts)
    multi = np.diff(np.r_[starts, len(r)]) >= 2
    if not multi.any():
        return
    mixed = float(((zmax - zmin) > 0.5 * dz)[multi].mean())
    if mixed > max_mixed:
        sys.exit(f"LAUE_NR={ncols} is inconsistent with the stage coordinates: "
                 f"{mixed * 100:.0f}% of frame-derived rows span more than one slow-axis "
                 f"position. Check the number of columns of this scan.")


# ---------------------------------------------------------------------------
# one orientation per position
# ---------------------------------------------------------------------------
def ranking_counts(z):
    """``(primary, secondary, name)`` counts to rank instances by, from a validated npz.

    ``nhit_distinct`` (distinct observed peaks matched, ``frame_peaks.count_matched_peaks``)
    when the file carries it, with ``nhit`` as the secondary; otherwise ``nhit`` alone,
    with a printed warning, because ``nhit`` counts predicted reflections within
    tolerance and so STACKS harmonics (one row per hkl, no q-hat dedup): it favours
    harmonic-rich low-index orientations.
    """
    keys = z.files if hasattr(z, "files") else list(z.keys())
    nh = np.asarray(z["nhit"]).astype(int)
    if "nhit_distinct" in keys:
        return np.asarray(z["nhit_distinct"]).astype(int), nh, "nhit_distinct"
    print("WARNING: this npz has no 'nhit_distinct' (written before it was added); "
          "ranking positions by 'nhit', which stacks harmonics and favours "
          "harmonic-rich low-index orientations. Re-validate to rank by distinct peaks.",
          file=sys.stderr)
    return nh, None, "nhit"


def winner_per_position(row, col, primary, secondary=None, tiebreak=None) -> dict:
    """``{(row, col): instance_index}`` -- the single top-ranked instance at each position.

    Ranking, highest first: ``primary`` (use ``nhit_distinct`` -- see
    :func:`ranking_counts`), then ``secondary`` (``nhit``), then ``tiebreak``
    ASCENDING -- an ``(n,)`` or ``(n, k)`` array compared lexicographically by
    column; the scripts pass the flattened orientation matrix, so the winner is a
    property of the instances and not of the order they arrive in -- and finally
    the instance index (only reached for rows identical in every key, where the
    choice cannot change the figure). Without ``tiebreak`` the result depends on
    input order among exact ties.

    This is WINNER-TAKE-ALL: every other orientation indexed at a position is
    dropped from the figure. The audit tracks replacing these single-winner maps
    with occupancy maps; this function only makes the choice of winner principled
    and deterministic.
    """
    row = np.asarray(row, int)
    col = np.asarray(col, int)
    n = len(row)
    if n == 0:
        return {}
    keys = [np.arange(n)]                              # least significant first
    if tiebreak is not None:
        tb = np.asarray(tiebreak, float).reshape(n, -1)
        keys += [tb[:, j] for j in range(tb.shape[1] - 1, -1, -1)]
    if secondary is not None:
        keys.append(-np.asarray(secondary, float))
    keys.append(-np.asarray(primary, float))
    keys += [col, row]
    order = np.lexsort(keys)
    r, c = row[order], col[order]
    first = np.r_[True, (r[1:] != r[:-1]) | (c[1:] != c[:-1])]
    return {(int(a), int(b)): int(i) for a, b, i in zip(r[first], c[first], order[first])}
