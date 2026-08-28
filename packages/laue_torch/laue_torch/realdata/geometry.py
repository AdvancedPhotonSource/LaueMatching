"""Parse APS 34-ID-E ``geoN`` geometry XML and crystal structure files
into a :class:`~laue_torch.io.LaueParams` object.

The 34-ID-E geometry XML stores up to ``Ndetectors`` detectors; we
take ``Detector N="0"`` (the main 2D detector used for Laue indexing).
Pixel size is derived from ``size`` and ``Npixels``.

Crystal structure can be given as either a small CIF-flavored
``.xml`` (the LaueMatching/Argonne convention --- ``Al.xml`` in the
test data) or a ``.xtal`` file (BraggPanel format).  Both are
parsed for: space group number, ``a, b, c`` (nm) and ``alpha, beta,
gamma`` (deg).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import re
import xml.etree.ElementTree as ET

from ..io import LaueParams


# ── geoN parsing ───────────────────────────────────────────────────────────

@dataclass
class GeoN:
    """Subset of geoN XML fields used by the Laue forward model."""
    detector_id: str
    P_mm: Tuple[float, float, float]                         # detector translation, mm
    R_rad: Tuple[float, float, float]                        # detector Rodrigues, radians
    npx_x: int
    npx_y: int
    px_size_mm_x: float
    px_size_mm_y: float
    sample_R_rad: Tuple[float, float, float]
    sample_origin_um: Tuple[float, float, float]


def parse_geon_xml(path: str | Path, detector_index: int = 0) -> GeoN:
    """Parse a 34-ID-E ``geoN`` XML and return the chosen detector's geometry."""
    tree = ET.parse(path)
    root = tree.getroot()
    # Strip namespace from tags so we can find by local name.
    def _ln(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _find_local(elem, name):
        return next((c for c in elem.iter() if _ln(c.tag) == name), None)

    detectors = [c for c in root.iter() if _ln(c.tag) == "Detector"]
    if not detectors:
        raise ValueError(f"{path}: no <Detector> elements")
    if detector_index >= len(detectors):
        raise IndexError(
            f"{path}: requested detector {detector_index} but only "
            f"{len(detectors)} present")
    det = detectors[detector_index]

    def _floats(text: str) -> list[float]:
        return [float(t) for t in re.split(r"\s+", text.strip()) if t]

    npix_text = _find_local(det, "Npixels").text
    npix_floats = _floats(npix_text)
    npx_x, npx_y = int(npix_floats[0]), int(npix_floats[1])

    size_text = _find_local(det, "size").text
    size_floats = _floats(size_text)
    sx, sy = size_floats[0], size_floats[1]                  # mm
    P_floats = _floats(_find_local(det, "P").text)           # mm
    R_floats = _floats(_find_local(det, "R").text)           # rad
    P = (P_floats[0], P_floats[1], P_floats[2])
    R = (R_floats[0], R_floats[1], R_floats[2])
    det_id = (_find_local(det, "ID").text or "").strip()

    sample = _find_local(root, "Sample")
    if sample is not None:
        sR = _floats(_find_local(sample, "R").text)
        sO = _floats(_find_local(sample, "Origin").text)
        sample_R = (sR[0], sR[1], sR[2])
        sample_origin = (sO[0], sO[1], sO[2])
    else:
        sample_R = (0.0, 0.0, 0.0)
        sample_origin = (0.0, 0.0, 0.0)

    return GeoN(
        detector_id=det_id,
        P_mm=P,
        R_rad=R,
        npx_x=npx_x,
        npx_y=npx_y,
        px_size_mm_x=sx / npx_x,
        px_size_mm_y=sy / npx_y,
        sample_R_rad=sample_R,
        sample_origin_um=sample_origin,
    )


# ── Crystal parsing ────────────────────────────────────────────────────────

@dataclass
class Crystal:
    sg_num: int
    lattice_nm: Tuple[float, float, float, float, float, float]   # (a,b,c,α,β,γ); nm + deg
    name: str = ""


def _parse_lattice_xml_block(root) -> Crystal:
    """Parse a CIF-flavored XML structure (LaueMatching ``Al.xml`` /
    BraggPanel ``.xtal``)."""
    def _ln(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    sg_elt = next((c for c in root.iter()
                   if _ln(c.tag) in ("space_group_IT_number", "IT_number")), None)
    if sg_elt is None:
        raise ValueError("crystal XML missing space-group IT number")
    sg = int(sg_elt.text.strip())

    cell = next((c for c in root.iter() if _ln(c.tag) == "cell"), None)
    if cell is None:
        raise ValueError("crystal XML missing <cell>")
    def _val(name) -> float:
        x = next((c for c in cell if _ln(c.tag) == name), None)
        if x is None:
            raise ValueError(f"<cell> missing {name}")
        return float(x.text.strip())
    a, b, c = _val("a"), _val("b"), _val("c")
    alpha, beta, gamma = _val("alpha"), _val("beta"), _val("gamma")

    name_elt = next(
        (c for c in root.iter()
         if _ln(c.tag) in ("chemical_name_common", "chemical_formula")), None)
    name = name_elt.text.strip() if name_elt is not None else ""

    return Crystal(sg_num=sg,
                   lattice_nm=(a, b, c, alpha, beta, gamma),
                   name=name)


def parse_crystal(path: str | Path) -> Crystal:
    """Parse a crystal structure file (either ``.xml`` LaueMatching format
    or ``.xtal`` BraggPanel format).  Both are XML; the parser is
    structure-agnostic so the same routine works for either."""
    tree = ET.parse(path)
    return _parse_lattice_xml_block(tree.getroot())


# ── Bridge to LaueParams ──────────────────────────────────────────────────

def make_lauematching_params(
    geon: GeoN,
    crystal: Crystal,
    *,
    E_lo: float = 5.0,
    E_hi: float = 30.0,
    psf_sigma: float = 2.0,
    symmetry: str = "F",
) -> LaueParams:
    """Bundle a parsed geometry + crystal into a :class:`LaueParams`.

    Notes
    -----
    LaueMatching's ``params_sim.txt`` convention:
      * P in **m** (we convert from mm).
      * R in **rad** (already correct).
      * Pixel sizes (``PxX``, ``PxY``) in **m**.
      * Lattice in **nm** + degrees.
    The energy bandpass and PSF Gaussian width default to the values
    used in the 34-ID-E setup; callers should override for non-standard
    runs.
    """
    P_m = tuple(p * 1e-3 for p in geon.P_mm)                  # mm → m
    px_x_m = geon.px_size_mm_x * 1e-3
    px_y_m = geon.px_size_mm_y * 1e-3
    return LaueParams(
        sg_num=crystal.sg_num,
        symmetry=symmetry,
        lattice=crystal.lattice_nm,
        P=P_m,
        R=geon.R_rad,
        px_x=px_x_m,
        px_y=px_y_m,
        n_pix_x=geon.npx_x,
        n_pix_y=geon.npx_y,
        E_lo=E_lo,
        E_hi=E_hi,
        psf_sigma=psf_sigma,
    )


def write_lauematching_config(
    params: LaueParams,
    out_path: str | Path,
    *,
    extras: Optional[dict] = None,
) -> Path:
    """Write a ``params_sim.txt``-style config file from a parsed
    :class:`LaueParams`, suitable for feeding to ``RunImage.py`` /
    LaueMatching's CPU/GPU binaries.

    ``extras`` is a dict of additional key-value lines to inject (e.g.
    ``OrientationFile``, ``MaxNrLaueSpots``, ``ResultDir``, etc.) ---
    see the LaueMatching documentation for the full list.
    """
    extras = extras or {}
    lines: list[str] = []

    lines.append(f"SpaceGroup {params.sg_num}")
    lines.append(f"Symmetry {params.symmetry}")
    lat = params.lattice
    lines.append(
        f"LatticeParameter {lat[0]:.6f} {lat[1]:.6f} {lat[2]:.6f} "
        f"{lat[3]:.6f} {lat[4]:.6f} {lat[5]:.6f}"
    )
    lines.append(f"P_Array {params.P[0]:.6f} {params.P[1]:.6f} {params.P[2]:.6f}")
    lines.append(f"R_Array {params.R[0]:.8f} {params.R[1]:.8f} {params.R[2]:.8f}")
    lines.append(f"PxX {params.px_x:.6f}")
    lines.append(f"PxY {params.px_y:.6f}")
    lines.append(f"NrPxX {params.n_pix_x}")
    lines.append(f"NrPxY {params.n_pix_y}")
    lines.append(f"Elo {params.E_lo}")
    lines.append(f"Ehi {params.E_hi}")
    lines.append(f"SimulationSmoothingWidth {params.psf_sigma}")
    for key, val in extras.items():
        lines.append(f"{key} {val}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
