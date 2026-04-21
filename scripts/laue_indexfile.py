"""laue_indexfile.py — write indexing results in the Tischler ``$filetype
IndexFile`` text format.

The schema mirrors the output of the Tischler ``indexing`` tool
(``.indexFile`` / ``.txt``): a header block of ``$keyword value`` lines
describing the indexing parameters and beamline state, followed by one
``$patternN`` section per indexed grain. Each pattern section carries
Euler angles, goodness, rms error, rotation matrix, reciprocal lattice,
and an ``$arrayN`` table with per-spot (G-hat, hkl, intensity, energy,
angular error, peak-index).

Usage
-----

High-level::

    import laue_indexfile as lif
    header, patterns = lif.build_from_h5(h5_path, cfg, mapping_entry)
    lif.write_indexfile(out_path, header, patterns)

Integration points:

* :file:`scripts/laue_postprocess.py` — emits per-image ``.indexing.txt``
  alongside each ``image_XXXXX.output.h5``.
* :file:`scripts/RunImage.py` — emits ``<image>.indexing.txt`` next to the
  output HDF5 from the single-image pipeline.
* :file:`scripts/laue_orchestrator.py` — passes the ``--indexfile-out`` /
  ``--no-indexfile`` flags down to postprocess.

Design decisions (see plan file ``cozy-kindling-meadow.md``):

* Per-spot energy is computed in Python from the fit's reciprocal lattice
  matrix and the hkl indices — matches the C formula at
  ``LaueMatchingHeaders.h:609`` but avoids touching the indexer binaries.
* Per-spot ``err(deg)`` uses the angle between the fit-predicted Qhat
  (stored in ``spots.txt`` cols 8–10) and the Qhat obtained by inverting
  the observed pixel coordinates through the detector geometry — so a
  perfect fit would give err ≈ 0.
* Per-spot ``PkIndex`` comes from a nearest-neighbour lookup against
  ``/entry/data/component_centers`` in the HDF5 output.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ``hc`` in keV·nm — used both here and in the C indexer.
HC_KEVNM = 1.2398419739


# ---------------------------------------------------------------------------
# Dataclasses describing the IndexFile contents
# ---------------------------------------------------------------------------

@dataclass
class IndexFileSpot:
    g_hat: Tuple[float, float, float]
    hkl: Tuple[int, int, int]
    intensity: float
    energy_kev: float
    err_deg: float
    peak_index: int


@dataclass
class IndexFilePattern:
    euler_deg: Tuple[float, float, float]       # (phi1, Phi, phi2) — Bunge ZXZ
    goodness: float
    rms_error_deg: float
    rotation_matrix: np.ndarray                  # (3, 3)
    recip_lattice: np.ndarray                    # (3, 3)
    spots: List[IndexFileSpot] = field(default_factory=list)


@dataclass
class IndexFileHeader:
    peak_file: str = ""
    input_image: str = ""
    kev_max_calc: float = 0.0                    # $keVmaxCalc  <- Ehi
    kev_max_test: float = 0.0                    # $keVmaxTest  <- usually Ehi or 2*Ehi
    angle_tolerance_deg: float = 0.0             # $angleTolerance  <- MaxAngle
    hkl_prefer: str = "{0,0,2}"
    cone_deg: float = 72.0
    n_patterns_found: int = 0
    n_indexed: int = 0
    n_input_data: int = 0
    execution_time_sec: float = 0.0
    structure_desc: str = ""                     # $structureDesc (optional, e.g. "Ni")
    space_group: int = 0
    lattice_params_nm_deg: Tuple[float, ...] = (0.0, 0.0, 0.0, 90.0, 90.0, 90.0)
    length_unit: str = "nm"
    atom_descriptions: List[str] = field(default_factory=list)
    xtal_file_name: str = ""
    x_dim: int = 0
    y_dim: int = 0
    x_dim_det: int = 0
    y_dim_det: int = 0
    start_x: int = 0
    end_x: int = 0
    group_x: int = 1
    start_y: int = 0
    end_y: int = 0
    group_y: int = 1
    beamline_meta: Dict[str, Any] = field(default_factory=dict)
    peak_search_params: Dict[str, Any] = field(default_factory=dict)
    geo_file_name: str = ""
    program_name: str = "LaueMatching"


# ---------------------------------------------------------------------------
# Per-spot helpers
# ---------------------------------------------------------------------------

def energy_from_recip_and_hkl(
    recip_lattice: np.ndarray,
    hkl: Sequence[float],
) -> float:
    """Energy (keV) of a Laue spot at orientation-rotated Q = recip_lattice · hkl.

    Mirrors the C formula at ``LaueMatchingHeaders.h:609``:

        E = hc_keVnm * |Q| / (4π sinθ),  sinθ = -Q̂_z

    Valid only for the standard detector geometry (incoming beam along +Z).
    Returns NaN for unphysical (behind-source) configurations.
    """
    q = np.asarray(recip_lattice, dtype=np.float64) @ np.asarray(hkl, dtype=np.float64)
    qlen = float(np.linalg.norm(q))
    if qlen == 0.0:
        return float("nan")
    qhat_z = q[2] / qlen
    sin_theta = -qhat_z
    if sin_theta <= 0.0:
        return float("nan")
    return HC_KEVNM * qlen / (4.0 * math.pi * sin_theta)


def _rodriguez_rotation(r: np.ndarray) -> np.ndarray:
    """Same Rodrigues formula used by ``GenerateHKLs.DetectorType`` so that
    our pixel↔direction maths stays numerically consistent with the
    indexer's geometry convention."""
    rotang = float(np.linalg.norm(r))
    if rotang < 1e-12:
        return np.eye(3)
    n = r / rotang
    c, s = math.cos(rotang), math.sin(rotang)
    nx, ny, nz = n
    return np.array([
        [c + (1 - c) * nx * nx,      (1 - c) * nx * ny - s * nz, (1 - c) * nx * nz + s * ny],
        [(1 - c) * ny * nx + s * nz, c + (1 - c) * ny * ny,      (1 - c) * ny * nz - s * nx],
        [(1 - c) * nz * nx - s * ny, (1 - c) * nz * ny + s * nx, c + (1 - c) * nz * nz],
    ])


def pixel_to_qhat(
    px: float,
    py: float,
    *,
    r_array: Sequence[float],      # rotation vector in radians (matches LaueMatching indexer)
    p_array: Sequence[float],      # (X, Y, Z) detector position in the C convention
    px_x: float,                   # pixel size X (m or whatever unit matches p_array)
    px_y: float,                   # pixel size Y
    nr_px_x: int,
    nr_px_y: int,
) -> np.ndarray:
    """Invert the forward-projection in ``LaueMatchingHeaders.h`` to get the
    unit Q-hat at a given pixel.

    The C forward pipeline is::

        xyz_det = (xp + P[0], yp + P[1], P[2])   with xp,yp from pixel
        kf_detframe = scaled by P[2]/xyz_det[2] (unused here — we only need direction)
        xyz_beam = R @ xyz_det
        kf = xyz_beam / |xyz_beam|
        Q = kf - ki,  ki = (0,0,1)
        Qhat = Q / |Q|
    """
    r = _rodriguez_rotation(np.asarray(r_array, dtype=np.float64))
    p = np.asarray(p_array, dtype=np.float64)
    xp = (float(px) - 0.5 * (nr_px_x - 1)) * px_x + p[0]
    yp = (float(py) - 0.5 * (nr_px_y - 1)) * px_y + p[1]
    xyz_det = np.array([xp, yp, p[2]], dtype=np.float64)
    xyz_beam = r @ xyz_det
    kf_mag = float(np.linalg.norm(xyz_beam))
    if kf_mag == 0.0:
        return np.array([np.nan, np.nan, np.nan])
    kf_hat = xyz_beam / kf_mag
    q = kf_hat - np.array([0.0, 0.0, 1.0])
    qmag = float(np.linalg.norm(q))
    if qmag == 0.0:
        return np.array([np.nan, np.nan, np.nan])
    return q / qmag


def angular_error_deg(
    qhat_predicted: Sequence[float],
    qhat_observed: Sequence[float],
) -> float:
    """Return arccos(clip(qhat_p · qhat_o, -1, 1)) in degrees."""
    a = np.asarray(qhat_predicted, dtype=np.float64)
    b = np.asarray(qhat_observed, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    cos_theta = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


# ---------------------------------------------------------------------------
# Orientation-matrix → Euler (faithful port of the C code so the resulting
# angles can round-trip back through Euler2OrientMat byte-for-byte)
# ---------------------------------------------------------------------------

_EPS = 1e-9


def _sin_cos_to_angle(s: float, c: float) -> float:
    c = max(-1.0, min(1.0, c))
    return math.acos(c) if s >= 0.0 else 2.0 * math.pi - math.acos(c)


def orient_matrix_to_euler_deg(m: np.ndarray) -> Tuple[float, float, float]:
    """Port of ``OrientMat2Euler`` from ``LaueMatchingHeaders.h`` returning
    (psi, phi, theta) in degrees, wrapped into (-180, 180] for the first
    and third angles so the output matches the Tischler-style convention
    the user's example file follows.
    """
    m = np.asarray(m, dtype=np.float64)
    if abs(m[2, 2] - 1.0) < _EPS:
        phi = 0.0
    else:
        phi = math.acos(max(-1.0, min(1.0, m[2, 2])))
    sph = math.sin(phi)
    if abs(sph) < _EPS:
        psi = 0.0
        if abs(m[2, 2] - 1.0) < _EPS:
            theta = _sin_cos_to_angle(m[1, 0], m[0, 0])
        else:
            theta = _sin_cos_to_angle(-m[1, 0], m[0, 0])
    else:
        if abs(-m[1, 2] / sph) <= 1.0:
            psi = _sin_cos_to_angle(m[0, 2] / sph, -m[1, 2] / sph)
        else:
            psi = _sin_cos_to_angle(m[0, 2] / sph, 1.0)
        if abs(m[2, 1] / sph) <= 1.0:
            theta = _sin_cos_to_angle(m[2, 0] / sph, m[2, 1] / sph)
        else:
            theta = _sin_cos_to_angle(m[2, 0] / sph, 1.0)
    psi_deg = math.degrees(psi)
    phi_deg = math.degrees(phi)
    theta_deg = math.degrees(theta)

    def _wrap(a: float) -> float:
        while a > 180.0:
            a -= 360.0
        while a <= -180.0:
            a += 360.0
        return a

    return (_wrap(psi_deg), phi_deg, _wrap(theta_deg))


# ---------------------------------------------------------------------------
# Peak-index lookup
# ---------------------------------------------------------------------------

def nearest_peak_index(
    px: float,
    py: float,
    component_centers: np.ndarray,
    *,
    max_distance_px: float = 5.0,
) -> int:
    """Map an observed spot (px, py) to the nearest connected-component
    center in ``component_centers`` (shape ``(N, 4)`` with columns
    ``[label, cx, cy, area]``). Returns -1 if no center is within
    ``max_distance_px``.
    """
    if component_centers is None or len(component_centers) == 0:
        return -1
    cx = component_centers[:, 1]
    cy = component_centers[:, 2]
    dist_sq = (cx - px) ** 2 + (cy - py) ** 2
    idx = int(np.argmin(dist_sq))
    if math.sqrt(dist_sq[idx]) > max_distance_px:
        return -1
    # Return label - 1 for Tischler-style 0-indexed peak numbering.
    return int(component_centers[idx, 0]) - 1


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _fmt_matrix3(m: np.ndarray, fmt: str = "{:.7f}") -> str:
    """Format a 3x3 matrix the way Tischler IndexFiles format it:
    ``{{a,b,c}{d,e,f}{g,h,i}}`` (row-major, no spaces)."""
    m = np.asarray(m)
    rows = []
    for r in range(3):
        rows.append("{" + ",".join(fmt.format(m[r, c]) for c in range(3)) + "}")
    return "{" + "".join(rows) + "}"


def _matrix_commentary(m: np.ndarray, label: str, scale_fmt: str = "{:12.5f}") -> List[str]:
    """Emit the three-line ``// column vectors`` commentary block. In
    Tischler's format the `{{row}{row}{row}}` form is transposed relative
    to the commentary, which lists columns. Line *i* prints the *i*-th
    column of the matrix (i.e. ``m[:, i]``) as a row of numbers.
    """
    m = np.asarray(m)
    tags = {
        "rotation": ("//   rotation matrix   ", "//   column vectors    ", "//                     "),
        "recip":    ("//   reciprocal matrix ", "//   column vectors    ", "//                     "),
    }[label]
    lines = []
    for i in range(3):
        col_vals = [scale_fmt.format(m[r, i]) for r in range(3)]
        lines.append(tags[i] + "   ".join(col_vals))
    return lines


def write_indexfile(
    path: str | os.PathLike,
    header: IndexFileHeader,
    patterns: Sequence[IndexFilePattern],
) -> Path:
    """Write the full IndexFile and return the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("$filetype\tIndexFile")
    lines.append(
        f"// Found {header.n_patterns_found} patterns, "
        f"indexed {header.n_indexed} out of {header.n_input_data} spots "
        f"in {_format_exec_time(header.execution_time_sec)}"
    )
    lines.append("// " + "-" * 60)

    def _kv(k: str, v: Any, comment: str = "") -> None:
        c = f"\t\t// {comment}" if comment else ""
        lines.append(f"${k}\t\t{v}{c}")

    _kv("peakFile", f"'{header.peak_file}'", "input data file")
    _kv("keVmaxCalc", f"{header.kev_max_calc:.2f}", "max energy (keV) for calculated hkl")
    _kv("angleTolerance", f"{header.angle_tolerance_deg:.3g}",
        "how close to vectors have to be considered to have the correct angle (deg)")
    _kv("keVmaxTest", f"{header.kev_max_test:.2f}",
        "max energy (keV) matching a spot (for calculating Gtest[][3])")
    _kv("hklPrefer", f"'{header.hkl_prefer}'", "preferred hkl, this should be hkl near center of pattern")
    _kv("cone", f"{header.cone_deg:g}",
        "angle from the preferred hkl to look for acceptable hkl when calculating (deg)")
    _kv("NpatternsFound", header.n_patterns_found, "number of patterns found")
    _kv("Nindexed", header.n_indexed, "number of spots indexed")
    _kv("NiData", header.n_input_data, "total number of data spots")
    _kv("executionTime", f"{header.execution_time_sec:.2f}", "execution time (sec)")
    lines.append("// " + "-" * 60)

    # Structure / lattice block
    lines.append("// these are parameters from header of $peakFile")
    if header.structure_desc:
        _kv("structureDesc", header.structure_desc)
    _kv("SpaceGroup", header.space_group)
    if len(header.lattice_params_nm_deg) == 6:
        a, b, c, al, be, ga = header.lattice_params_nm_deg
        _kv("latticeParameters",
            "{ " + ", ".join(f"{v:.5f}" for v in (a, b, c)) + ", "
            + ", ".join(f"{v:g}" for v in (al, be, ga)) + " }")
    _kv("lengthUnit", header.length_unit)
    for i, atom in enumerate(header.atom_descriptions):
        # Preserve the original (sic) spelling ``AtomDesctiption`` from the
        # Tischler tool so consumers' parsers keep working.
        _kv(f"AtomDesctiption{i + 1}", f"{{{atom}}}")
    if header.xtal_file_name:
        _kv("xtalFileName", header.xtal_file_name)
    if header.input_image:
        _kv("inputImage", header.input_image)
    _kv("xdim", header.x_dim, "number of binned pixels along X")
    _kv("ydim", header.y_dim, "number of binned pixels along Y")
    _kv("xDimDet", header.x_dim_det, "total number of un-binned pixels in detector along X")
    _kv("yDimDet", header.y_dim_det, "total number of un-binned pixels in detector along Y")
    _kv("startx", header.start_x, "starting X of ROI (un-binned pixels)")
    _kv("endx", header.end_x, "last X of ROI (un-binned pixels)")
    _kv("groupx", header.group_x, "binning along X for the ROI (un-binned pixels)")
    _kv("starty", header.start_y, "starting Y of ROI (un-binned pixels)")
    _kv("endy", header.end_y, "last Y of ROI (un-binned pixels)")
    _kv("groupy", header.group_y, "binning along Y for the ROI (un-binned pixels)")

    # Beamline metadata — emitted only if the user supplied values, to stay
    # honest about what's UNKNOWN vs. known.
    for key in (
        "exposure", "CCDshutterIN", "Xsample", "Ysample", "Zsample", "depth",
        "scanNum", "beamBad", "lightOn", "energy", "hutchTemperature",
        "sampleDistance", "monoMode", "dateExposed", "userName", "detector_ID",
    ):
        if key in header.beamline_meta:
            _kv(key, header.beamline_meta[key])

    # Peak-search params (emitted only when present)
    for key in ("boxsize", "minwidth", "maxwidth", "maxCentToFit", "maxRfactor",
                "threshold", "minSeparation", "smooth", "peakShape", "totalSum",
                "sumAboveThreshold", "numAboveThreshold", "NpeakMax"):
        if key in header.peak_search_params:
            _kv(key, header.peak_search_params[key])

    if header.program_name:
        _kv("programName", f"'{header.program_name}'")
    if header.geo_file_name:
        _kv("geoFileName", header.geo_file_name)

    lines.append("// " + "-" * 60)

    # Per-pattern blocks
    for idx, pat in enumerate(patterns):
        lines.append("")
        lines.append(f"$pattern{idx}")
        psi, phi, theta = pat.euler_deg
        lines.append(
            f"$EulerAngles{idx} {{ {psi:>12.8f}, {phi:>12.8f}, {theta:>12.8f}}}"
            f"\t// Euler angles for this pattern (deg)"
        )
        lines.append(f"$goodness{idx}\t\t{pat.goodness:g}\t\t\t\t\t\t// goodness of the this pattern")
        lines.append(f"$rms_error{idx}\t\t{pat.rms_error_deg:g}\t\t\t\t\t// rms error of (measured-predicted) (deg)")
        lines.append(f"$rotation_matrix{idx}\t\t{_fmt_matrix3(pat.rotation_matrix)}")
        lines.extend(_matrix_commentary(pat.rotation_matrix, "rotation"))
        lines.append(f"$recip_lattice{idx}\t\t{_fmt_matrix3(pat.recip_lattice)}")
        lines.extend(_matrix_commentary(pat.recip_lattice, "recip"))
        lines.append("//")
        n_spots = len(pat.spots)
        lines.append(f"$array{idx}\t {len(pat.spots):2d}   {n_spots:<8d}"
                     "     G^                         (hkl)       intens      E(keV)      err(deg)   PkIndex")
        for row, spot in enumerate(pat.spots):
            g = spot.g_hat
            h, k, l = spot.hkl
            lines.append(
                f"    [{row:3d}]   ({g[0]:>10.7f} {g[1]:>10.7f} {g[2]:>10.7f})"
                f"     ({h:>3d} {k:>3d} {l:>3d})    "
                f"{spot.intensity:>.4f},   {spot.energy_kev:>8.4f},   "
                f"{spot.err_deg:>.5f}      {spot.peak_index}"
            )

    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return out


def _format_exec_time(sec: float) -> str:
    """Return "HH:MM:SS = (X.XX sec)" to match the example file preamble."""
    if sec < 0 or not math.isfinite(sec):
        sec = 0.0
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = int(sec % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d} = ({sec:.2f} sec)"


# ---------------------------------------------------------------------------
# HDF5 → IndexFile builder
# ---------------------------------------------------------------------------

# Column indices in the streaming-mode solutions.txt / filtered_orientations
# HDF5 dataset (no leading ImageNr because the HDF5 stores per-image slices
# already deduplicated of the image number).
_COL_GRAIN = 0           # depends on whether ImageNr column is present
_COL_QUALITY = 4         # NMatches * sqrt(Intensity)
# The exact indexing depends on the presence of ImageNr. See
# ``_slice_orientation_row`` below.


def _slice_orientation_row(row: np.ndarray, has_image_nr: bool) -> Dict[str, Any]:
    """Extract fields from one row of filtered_orientations.

    Streaming format (``_save_image_h5`` uses this when ImageNr is present):

        cols 0..31:
            0 ImageNr        (optional)
            1 GrainNr        (or 0 if no ImageNr)
            2 NumberOfSolutions
            3 Intensity
            4 NMatches*Intensity
            5 NMatches*sqrt(Intensity)
            6 NMatches
            7 NSpotsCalc
            8..16  Recip (3x3)
            17..22 LatticeParam
            23..31 OrientMatrix (3x3)
            32 CoarseScore
            33 Misorientation (deg)
            34 orientationRowNr
    """
    off = 1 if has_image_nr else 0
    return {
        "grain_nr":    int(row[off + 0]) if has_image_nr else int(row[0]),
        "quality":     float(row[off + 4]) if has_image_nr else float(row[4]),
        "n_matches":   int(row[off + 5]) if has_image_nr else int(row[5]),
        "recip":       np.asarray(row[off + 7:off + 16]).reshape(3, 3),
        "lat_params":  tuple(float(v) for v in row[off + 16:off + 22]),
        "rotation":    np.asarray(row[off + 22:off + 31]).reshape(3, 3),
        "misorient":   float(row[off + 32]) if (off + 32) < row.size else 0.0,
    }


def _slice_spot_row(row: np.ndarray, has_image_nr: bool) -> Dict[str, Any]:
    """Format columns of filtered_spots: same layout as the stream output,
    with optional leading ImageNr.
    """
    off = 1 if has_image_nr else 0
    return {
        "grain_nr":  int(row[off + 0]),
        "spot_nr":   int(row[off + 1]),
        "hkl":       (int(row[off + 2]), int(row[off + 3]), int(row[off + 4])),
        "px":        float(row[off + 5]),
        "py":        float(row[off + 6]),
        "qhat":      (float(row[off + 7]), float(row[off + 8]), float(row[off + 9])),
        "intensity": float(row[off + 10]),
    }


def build_from_h5(
    h5_path: str | os.PathLike,
    cfg: Any,                                    # ConfigurationManager, LaueConfig, or dict
    mapping_entry: Optional[Dict[str, Any]] = None,
    *,
    beamline_meta: Optional[Dict[str, Any]] = None,
    peak_search_params: Optional[Dict[str, Any]] = None,
    execution_time_sec: float = 0.0,
) -> Tuple[IndexFileHeader, List[IndexFilePattern]]:
    """Build (header, patterns) from a LaueMatching ``image_XXXXX.output.h5``.

    Parameters mirror the data flowing through ``laue_postprocess._save_image_h5``.
    """
    import h5py

    mapping_entry = mapping_entry or {}

    def _cfg_get(key: str, default=None):
        if cfg is None:
            return default
        get = getattr(cfg, "get", None)
        if callable(get):
            try:
                val = get(key, default)
                if val is not None:
                    return val
            except Exception:
                pass
        # Fall back to attribute access
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return default

    ehi = float(_cfg_get("ehi", 30.0) or 30.0)
    elo = float(_cfg_get("elo", 5.0) or 5.0)
    max_angle = float(_cfg_get("maxAngle", _cfg_get("max_angle", 2.0)) or 2.0)
    space_group = int(_cfg_get("space_group", 0) or 0)

    # Lattice parameter: stored as "a b c alpha beta gamma" string in LaueConfig.
    lat_str = _cfg_get("lattice_parameter", "0 0 0 90 90 90")
    if isinstance(lat_str, str):
        lat_tuple = tuple(float(v) for v in lat_str.split()[:6])
    else:
        lat_tuple = tuple(float(v) for v in lat_str[:6])
    while len(lat_tuple) < 6:
        lat_tuple = lat_tuple + (0.0,)

    nr_px_x = int(_cfg_get("nr_px_x", 2048) or 2048)
    nr_px_y = int(_cfg_get("nr_px_y", 2048) or 2048)

    # Detector geometry (r_array / p_array are whitespace-separated strings)
    def _vec3(val, default=(0.0, 0.0, 0.0)):
        if val is None:
            return default
        if isinstance(val, str):
            parts = val.split()
            return tuple(float(v) for v in parts[:3]) if len(parts) >= 3 else default
        try:
            return tuple(float(v) for v in list(val)[:3])
        except Exception:
            return default

    r_array = _vec3(_cfg_get("r_array"))
    p_array = _vec3(_cfg_get("p_array"))
    px_x = float(_cfg_get("px_x", 0.0002) or 0.0002)
    px_y = float(_cfg_get("px_y", 0.0002) or 0.0002)

    structure_desc = _cfg_get("structure_desc", "") or ""
    xtal_file = _cfg_get("xtal_file", "") or ""
    atom_desc = _cfg_get("atom_description", "") or ""
    atom_descriptions = [atom_desc] if atom_desc else []

    header = IndexFileHeader(
        peak_file=mapping_entry.get("file", "") or os.path.basename(str(h5_path)),
        input_image=mapping_entry.get("file", ""),
        kev_max_calc=ehi,
        kev_max_test=ehi,  # LaueMatching doesn't distinguish these
        angle_tolerance_deg=max_angle,
        space_group=space_group,
        lattice_params_nm_deg=lat_tuple,
        structure_desc=structure_desc,
        xtal_file_name=xtal_file,
        atom_descriptions=atom_descriptions,
        x_dim=nr_px_x, y_dim=nr_px_y,
        x_dim_det=nr_px_x, y_dim_det=nr_px_y,
        end_x=nr_px_x - 1, end_y=nr_px_y - 1,
        beamline_meta=dict(beamline_meta or {}),
        peak_search_params=dict(peak_search_params or {}),
        execution_time_sec=execution_time_sec,
    )

    patterns: List[IndexFilePattern] = []

    with h5py.File(h5_path, "r") as hf:
        if "/entry/results/filtered_orientations" not in hf:
            # Nothing indexed; return an empty index file.
            header.n_patterns_found = 0
            header.n_indexed = 0
            return header, patterns

        filt_orient = hf["/entry/results/filtered_orientations"][()]
        filt_spots = hf["/entry/results/filtered_spots"][()]

        # Determine column layout by checking the number of columns. Stream
        # format prepends ImageNr for a total of 35 cols (orientations) and
        # 12 cols (spots). Batch format is 34/11.
        has_imgnr_orient = filt_orient.shape[1] >= 35 if filt_orient.ndim == 2 else False
        has_imgnr_spots = filt_spots.shape[1] >= 12 if filt_spots.ndim == 2 else False

        component_centers = None
        if "/entry/data/component_centers" in hf:
            component_centers = hf["/entry/data/component_centers"][()]

        # Group spots by grain_nr for O(N+M) matching.
        spots_by_grain: Dict[int, List[np.ndarray]] = {}
        if filt_spots.ndim == 2 and filt_spots.size > 0:
            for row in filt_spots:
                s = _slice_spot_row(row, has_imgnr_spots)
                spots_by_grain.setdefault(s["grain_nr"], []).append(s)

        if filt_orient.ndim == 2 and filt_orient.size > 0:
            for row in filt_orient:
                o = _slice_orientation_row(row, has_imgnr_orient)
                rot = o["rotation"]
                recip = o["recip"]
                euler = orient_matrix_to_euler_deg(rot)

                spots_out: List[IndexFileSpot] = []
                errs = []
                for sp in spots_by_grain.get(o["grain_nr"], []):
                    qhat_pred = sp["qhat"]
                    qhat_obs = pixel_to_qhat(
                        sp["px"], sp["py"],
                        r_array=r_array, p_array=p_array,
                        px_x=px_x, px_y=px_y,
                        nr_px_x=nr_px_x, nr_px_y=nr_px_y,
                    )
                    err = angular_error_deg(qhat_pred, qhat_obs)
                    if math.isnan(err):
                        err = 0.0
                    errs.append(err)
                    energy = energy_from_recip_and_hkl(recip, sp["hkl"])
                    if math.isnan(energy):
                        energy = 0.0
                    pk = nearest_peak_index(sp["px"], sp["py"], component_centers) if component_centers is not None else -1
                    spots_out.append(IndexFileSpot(
                        g_hat=tuple(qhat_pred),
                        hkl=sp["hkl"],
                        intensity=sp["intensity"],
                        energy_kev=energy,
                        err_deg=err,
                        peak_index=pk,
                    ))

                rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else o["misorient"]
                patterns.append(IndexFilePattern(
                    euler_deg=euler,
                    goodness=o["quality"],
                    rms_error_deg=rms,
                    rotation_matrix=rot,
                    recip_lattice=recip,
                    spots=spots_out,
                ))

    header.n_patterns_found = len(patterns)
    header.n_indexed = sum(len(p.spots) for p in patterns)
    return header, patterns


def write_from_h5(
    h5_path: str | os.PathLike,
    out_path: str | os.PathLike,
    cfg: Any,
    mapping_entry: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Path:
    """Convenience: build and write in one call."""
    header, patterns = build_from_h5(h5_path, cfg, mapping_entry, **kwargs)
    return write_indexfile(out_path, header, patterns)
