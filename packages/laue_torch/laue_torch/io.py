"""Parameter-file parsing and HKL generation via midas-hkls."""

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
    E_lo = fget("Elo", 5.0)
    E_hi = fget("Ehi", 30.0)
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


def load_orientations(path: str | Path) -> Tensor:
    """Load orientation matrices from CSV. Each row is a flattened 3×3 (9 floats)."""
    path = Path(path)
    arr = np.genfromtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 31 and str(arr[0]).startswith(("%", "#")):
        # Old-style with %GrainNr header — caller can post-process.
        arr = arr[:, 22:31]
    arr = arr[:, :9].reshape(-1, 3, 3)
    return torch.tensor(arr, dtype=torch.float64)
