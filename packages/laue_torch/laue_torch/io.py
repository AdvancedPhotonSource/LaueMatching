"""Parameter-file parsing, HKL generation via midas-hkls, orientation-table
loading, and the detector axis-order convention (``[X, Y]`` vs ``[row, col]``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import math
import numpy as np
import torch
from torch import Tensor


@dataclass
class LaueParams:
    """Parsed contents of a LaueMatching params_sim.txt-style file.

    All lengths in nm, angles in degrees (matches the existing convention).
    """
    sg_num: int
    symmetry: str
    lattice: tuple[float, float, float, float, float, float]   # nm, deg
    P: tuple[float, float, float]                              # m
    R: tuple[float, float, float]                              # rad (Rodrigues)
    px_x: float                                                # m
    px_y: float                                                # m
    n_pix_x: int
    n_pix_y: int
    E_lo: float                                                # keV
    E_hi: float                                                # keV
    psf_sigma: float                                           # px
    hkl_file: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def to_tensors(self, dtype: torch.dtype = torch.float64, device: str = "cpu"):
        return {
            "lattice": torch.tensor(self.lattice, dtype=dtype, device=device),
            "P": torch.tensor(self.P, dtype=dtype, device=device),
            "R": torch.tensor(self.R, dtype=dtype, device=device),
            "px_size": (self.px_x, self.px_y),
            "n_pix": (self.n_pix_x, self.n_pix_y),
            "E_range": (self.E_lo, self.E_hi),
            "psf_sigma": self.psf_sigma,
        }


def parse_params(path: str | Path) -> LaueParams:
    """Parse a LaueMatching params_sim.txt-style file."""
    path = Path(path)
    text = path.read_text()
    extras: dict[str, str] = {}

    def grab(prefix: str) -> Optional[str]:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped.startswith(prefix):
                rest = stripped[len(prefix):].strip()
                if "#" in rest:
                    rest = rest.split("#", 1)[0].strip()
                return rest
        return None

    def fget(prefix: str, default=None) -> Optional[float]:
        s = grab(prefix)
        if s is None:
            return default
        return float(s.split()[0])

    def iget(prefix: str, default=None) -> Optional[int]:
        s = grab(prefix)
        if s is None:
            return default
        return int(s.split()[0])

    sg = iget("SpaceGroup")
    sym = (grab("Symmetry") or "F").split()[0]
    lat_s = grab("LatticeParameter")
    if lat_s is None:
        raise ValueError(f"{path} missing LatticeParameter")
    lat_vals = tuple(float(x) for x in lat_s.split()[:6])
    P_vals = tuple(float(x) for x in grab("P_Array").split()[:3])
    R_vals = tuple(float(x) for x in grab("R_Array").split()[:3])
    px_x = fget("PxX")
    px_y = fget("PxY")
    n_x = iget("NrPxX")
    n_y = iget("NrPxY")
    E_lo = fget("Elo", None)
    E_hi = fget("Ehi", None)
    # The forward CLI keeps its historical 5-30 keV default, but records that
    # the band was NOT in the file so the real-data refiners (which must fit
    # in the experiment's band) can refuse it; see experiment_band().
    missing = [k for k, v in (("Elo", E_lo), ("Ehi", E_hi)) if v is None]
    if missing:
        extras["energy_band_defaulted"] = ",".join(missing)
    E_lo = 5.0 if E_lo is None else E_lo
    E_hi = 30.0 if E_hi is None else E_hi
    psf = fget("SimulationSmoothingWidth", 2.0)
    hkl_file = grab("HKLFile")

    return LaueParams(
        sg_num=sg,
        symmetry=sym,
        lattice=lat_vals,
        P=P_vals,
        R=R_vals,
        px_x=px_x,
        px_y=px_y,
        n_pix_x=n_x,
        n_pix_y=n_y,
        E_lo=E_lo,
        E_hi=E_hi,
        psf_sigma=psf,
        hkl_file=hkl_file,
        extras=extras,
    )


def experiment_band(params: LaueParams) -> tuple[float, float]:
    """The experiment's energy band ``(E_lo, E_hi)`` in keV, or raise.

    Used by the real-data refiners, which must render in the band the data
    were taken in. Raises if the band is missing, was defaulted by
    :func:`parse_params` because the file had no ``Elo``/``Ehi``, or is not
    ``0 < E_lo < E_hi``. There is deliberately no (5, 30) fallback.
    """
    lo = getattr(params, "E_lo", None)
    hi = getattr(params, "E_hi", None)
    defaulted = (getattr(params, "extras", None) or {}).get("energy_band_defaulted")
    if lo is None or hi is None or defaulted:
        raise ValueError(
            f"no experiment energy band: params has E_lo={lo!r}, E_hi={hi!r}"
            + (f" (defaulted, {defaulted} missing from the parameter file)" if defaulted else "")
            + "; set Elo/Ehi (keV) for the measurement")
    lo, hi = float(lo), float(hi)
    if not (0.0 < lo < hi):
        raise ValueError(f"invalid energy band E_lo={lo}, E_hi={hi} keV")
    return lo, hi


# ── HKL generation via midas-hkls ──────────────────────────────────────────

_HC_KEV_A = 12.398419739     # keV·Å (note Å, not nm — midas_hkls uses Å)


def generate_hkls(
    sg_num: int,
    lattice_nm: tuple[float, float, float, float, float, float],
    E_hi_keV: float,
    *,
    d_min_nm: Optional[float] = None,
) -> Tensor:
    """Integer (h, k, l) reflection list via :mod:`midas_hkls`.

    Parameters
    ----------
    sg_num : int
        Space-group number (1-230).
    lattice_nm : (a, b, c, α, β, γ)
        Lattice in **nm** and degrees (the LaueMatching convention).
    E_hi_keV : float
        Upper energy bound. Sets ``d_min = hc / (2 · E_hi)`` if not given.
    d_min_nm : float, optional
        Override d_min (nm). Use a smaller value to include more reflections.

    Returns
    -------
    LongTensor of shape (H, 3).
    """
    try:
        from midas_hkls.lattice import Lattice
        from midas_hkls.space_group import SpaceGroup
        from midas_hkls.hkl_gen import generate_hkls as _gen
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "midas-hkls is required for HKL generation. "
            "pip install midas-hkls (PyPI) or "
            "pip install -e ~/opt/MIDAS/packages/midas_hkls/"
        ) from exc

    a, b, c, al, be, ga = lattice_nm
    # midas_hkls works in Angstroms.
    lat_A = Lattice(a * 10.0, b * 10.0, c * 10.0, al, be, ga)
    sg = SpaceGroup.from_number(sg_num)

    if d_min_nm is None:
        # White-beam Bragg cutoff: d_min = λ_min / 2 = hc / (2·E_hi).
        d_min_A = _HC_KEV_A / (2.0 * E_hi_keV)
    else:
        d_min_A = d_min_nm * 10.0

    refs = _gen(sg, lat_A, d_min=d_min_A)

    # midas_hkls returns one ASU representative per family. The C/Python
    # forward simulation iterates over *all* reflection vectors (signed).
    # Generate the equivalent set by enumerating sign permutations within
    # the search range — equivalent to looping over {h, k, l} ∈ ℤ³ with
    # |h|, |k|, |l| bounded.
    h_max = max(abs(r.h) for r in refs) if refs else 0
    k_max = max(abs(r.k) for r in refs) if refs else 0
    l_max = max(abs(r.l) for r in refs) if refs else 0
    H = max(h_max, k_max, l_max)
    if H == 0:
        return torch.zeros((0, 3), dtype=torch.long)

    # Cubic-extent fallback: enumerate the box that contains all ASU reps,
    # then accept everything midas_hkls considers Bragg-allowed.
    # We re-test with the SpaceGroup helpers to filter systematic absences.
    out: list[tuple[int, int, int]] = []
    for h in range(-H, H + 1):
        for k in range(-H, H + 1):
            for l in range(-H, H + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                d = lat_A.d_spacing(h, k, l)
                if not math.isfinite(d) or d < d_min_A:
                    continue
                if sg.is_systematically_absent(h, k, l):
                    continue
                out.append((h, k, l))
    return torch.tensor(out, dtype=torch.long)


# ── Detector axis order ────────────────────────────────────────────────────
#
# The forward model splats into img[X, Y] (X = detector column / fast axis
# first). Real detector frames, the backgrounds built from them and the C
# indexer are image[row, col] = image[Y, X] -- the TRANSPOSE. Getting this
# wrong does not raise on a square detector; it silently compares every spot
# with the wrong pixel (handbook invariant 38). Files carry the convention as
# ``<entry>/axis_order``: b"XY" = model layout (written by ``cli.py`` and
# ``coded_aperture.io_h5``), b"YX" = detector layout. LaueMatching indexer
# output carries no marker and is detector layout.

AXIS_ORDER_MODEL = "XY"      # img[X, Y], what LaueForwardModel returns
AXIS_ORDER_DETECTOR = "YX"   # image[row, col], a real frame


def read_axis_order(hf, key: str, default: Optional[str] = None) -> Optional[str]:
    """Return the ``axis_order`` marker stored at ``key`` in an open h5py file.

    Returns ``default`` when the dataset is absent. Raises ``ValueError`` on a
    value that is neither ``"XY"`` nor ``"YX"`` rather than guessing.
    """
    if key not in hf:
        return default
    raw = np.asarray(hf[key][()]).reshape(-1)[0]
    val = raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
    val = val.strip().upper()
    if val not in (AXIS_ORDER_MODEL, AXIS_ORDER_DETECTOR):
        raise ValueError(f"{key} = {val!r}; expected 'XY' (model layout) "
                         f"or 'YX' (detector row, col)")
    return val


def to_model_layout(image: Tensor, axis_order: Optional[str],
                    n_pix: tuple[int, int]) -> Tensor:
    """Return ``image`` (last two dims) in the forward model's ``[X, Y]`` layout.

    ``axis_order`` is the layout ``image`` is in (``"YX"`` for a real frame,
    ``"XY"`` if it is already model layout). There is no default: ``None``
    raises, because on a square detector a wrong guess cannot be detected.
    ``n_pix = (Nx, Ny)``. The result is shape-checked against ``n_pix`` so a
    frame in the wrong layout fails loudly on a non-square detector.
    """
    if axis_order is None:
        raise ValueError(
            "axis_order is None: say which layout the image is in -- 'YX' for "
            "a real detector frame image[row, col] (LaueScanLoader sets this "
            "from the file, 'YX' for indexer output), or 'XY' for an image "
            "already in the forward model's img[X, Y] layout (a laue_torch "
            "render). On a square detector a wrong guess would be silent.")
    if axis_order == AXIS_ORDER_DETECTOR:
        image = image.transpose(-1, -2).contiguous()
    elif axis_order != AXIS_ORDER_MODEL:
        raise ValueError(f"axis_order must be 'XY' or 'YX', got {axis_order!r}")
    Nx, Ny = int(n_pix[0]), int(n_pix[1])
    if tuple(image.shape[-2:]) != (Nx, Ny):
        raise ValueError(
            f"image in model layout has shape {tuple(image.shape[-2:])}, but the "
            f"forward model renders (Nx, Ny) = ({Nx}, {Ny}); declared axis_order "
            f"{axis_order!r} is wrong for this image (a real frame is 'YX' = "
            f"[row, col] = (NrPxY, NrPxX))")
    return image


# ── Indexer solution-table layouts ─────────────────────────────────────────
#
# DUPLICATED from ``laue_index.records.SOLUTION_FORMATS`` (the source of
# truth), because laue_torch does not depend on laue_index. Only the two
# fields laue_torch needs are copied. If the indexer's column map changes,
# change it there first, then here.
#   runimage: 34 columns, GrainNr at col 0, OrientMatrix at cols 22..30
#   stream  : 35 columns, ImageNr prepended, OrientMatrix at cols 23..31
SOLUTION_OM_START = {34: 22, 35: 23}     # n_cols -> first orientation column


def solution_orientation_columns(n_cols: int) -> tuple[int, int]:
    """(lo, hi) slice of the row-major orientation matrix in a solutions table.

    Selected by column count (34 = runimage, 35 = stream layout). Any other
    count raises: there is no safe default.
    """
    if n_cols not in SOLUTION_OM_START:
        raise ValueError(
            f"solutions table has {n_cols} columns; expected 34 (runimage) or "
            f"35 (stream) -- see laue_index.records.SOLUTION_FORMATS. Pass "
            f"the orientation columns explicitly for any other layout.")
    lo = SOLUTION_OM_START[n_cols]
    return lo, lo + 9


def _is_numeric_line(line: str) -> bool:
    toks = line.replace(",", " ").split()
    if not toks:
        return False
    try:
        [float(t) for t in toks]
    except ValueError:
        return False
    return True


def load_orientations(path: str | Path) -> Tensor:
    """Load orientation matrices from a text file.

    Accepts either a plain table whose first 9 columns are a row-major 3×3
    per row, or an indexer ``solutions.txt`` (34-column runimage or 35-column
    stream layout, with or without its ``%GrainNr``/``#`` header line), in
    which case the matrix is taken from the layout's OrientMatrix columns.
    Header lines are detected from the TEXT: any leading line that does not
    parse as numbers is skipped.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    rows = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith(("%", "#"))]
    # Drop any remaining non-numeric header line (e.g. a bare column-name row).
    while rows and not _is_numeric_line(rows[0]):
        rows = rows[1:]
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    arr = np.array([[float(t) for t in ln.replace(",", " ").split()] for ln in rows])
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 34:
        lo, hi = solution_orientation_columns(arr.shape[1])
        arr = arr[:, lo:hi]
    elif arr.shape[1] < 9:
        raise ValueError(f"{path}: {arr.shape[1]} columns; need 9 (row-major 3x3)")
    arr = arr[:, :9].reshape(-1, 3, 3)
    return torch.tensor(arr, dtype=torch.float64)
