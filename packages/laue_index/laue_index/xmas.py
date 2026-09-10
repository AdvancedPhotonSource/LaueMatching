"""XMAS detector calibration -> LaueMatching ``P_Array`` / ``R_Array``.

XMAS (the ALS/TPS Laue analysis code) describes the detector with a
sample-detector distance, a "center channel" pixel, and roll/pitch/yaw tilts.
LaueMatching describes it with ``P_Array`` (metres) and ``R_Array`` (a rotation
vector in RADIANS, theta*axis -- not Rodrigues, not degrees).  This module
converts one to the other for a REFLECTION-geometry station: panel edge-on
above the sample, 2theta = 90 deg, ki = (0, 0, 1).

The target convention is transcribed from ``laue_index.calibrate.project`` and
``c_src/LaueMatchingCPU.c:589-620``.  The C there no longer spells ``qhat``
explicitly -- it folds it into ``|q|^2`` (``sFac = 2*q_z/|q|^2``, so
``kf = ki - sFac*q``), which is the same arithmetic as ``ki - 2(qhat.ki)qhat``::

    q  = U B h ;  kf = ki - 2(qhat.ki) qhat ,  ki = (0,0,1)
    xyz = R_mat^T kf
    px = (xyz0 * P2/xyz2 - P0)/dx + 0.5*(Nx-1)     # ipx is the FAST axis
    py = (xyz1 * P2/xyz2 - P1)/dy + 0.5*(Ny-1)     # image[ipy*nrPxX + ipx]

so ``NrPxX`` counts COLUMNS and ``NrPxY`` counts ROWS of the image array.

Two things the arithmetic cannot settle
---------------------------------------
1. **The panel's in-plane orientation and handedness.**  Knowing the panel
   normal is +/- lab Y leaves 8 right-handed mountings (4 in-plane rotations x
   2 normal directions).  ``MOUNTS`` enumerates all 8; the indexer decides.
2. **What XMAS's roll/pitch/yaw are named after, and their signs.**
   ``TILT_CONVENTIONS`` enumerates 4 readings.  They are sub-degree here, so
   they are refined afterwards by ``laue-index calibrate`` on the scan's own
   frames -- these variants exist to get close enough to index, not to be
   trusted.

Nothing in this module measures the rotation about the beam.  That is an exact
gauge freedom (``calibrate.py`` measures it at 4.5e-13 px), so the frame comes
in with the XMAS calibration and must be declared, never claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .calibrate import matrix_to_rodrigues, rodrigues_to_matrix

__all__ = [
    "XmasCalibration", "LaueGeometry", "MOUNTS", "TILT_CONVENTIONS",
    "poni_pixel", "convert", "enumerate_candidates",
]


@dataclass(frozen=True)
class XmasCalibration:
    """One XMAS "Calibration parameters setting" panel, as written down.

    ``n_first``/``n_second`` are the detector dimensions in XMAS's own order --
    the pair printed as "CCD camera dimensions (pixels)".  ``xcent`` is measured
    along ``n_first``, ``ycent`` along ``n_second``.  For the TPS 21A PILATUS3
    6M that is 2527 (image ROWS) then 2463 (image COLUMNS).
    """
    name: str
    distance_mm: float
    xcent: float
    ycent: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    n_first: int = 2527
    n_second: int = 2463
    px_size_m: float = 172e-6
    energy_lo_keV: float = 6.0
    energy_hi_keV: float = 26.0
    provenance: str = ""


@dataclass(frozen=True)
class LaueGeometry:
    """A LaueMatching detector pose, ready to write into a parameter file."""
    P: tuple[float, float, float]          # metres
    R: tuple[float, float, float]          # rotation vector, radians
    nr_px_x: int                           # COLUMNS (fast axis)
    nr_px_y: int                           # ROWS
    px_x: float
    px_y: float
    mount: str
    tilt_convention: str
    row_parity: int = +1        # -1 means the IMAGE must be written flipped in rows
    source: str = ""

    def params_lines(self) -> list[str]:
        return [
            f"P_Array {self.P[0]:.6f} {self.P[1]:.6f} {self.P[2]:.6f}",
            f"R_Array {self.R[0]:.8f} {self.R[1]:.8f} {self.R[2]:.8f}",
            f"PxX {self.px_x:.6f}",
            f"PxY {self.px_y:.6f}",
            f"NrPxX {self.nr_px_x}",
            f"NrPxY {self.nr_px_y}",
        ]


# ---------------------------------------------------------------------------
# The 8 right-handed mountings with the panel normal along +/- lab Y.
#
# A mounting is given as the detector-frame axes expressed in the LAB frame,
# i.e. the COLUMNS of R_mat (which maps detector -> lab; ``project`` applies
# R_mat^T to kf).  "34IDE" is the one decoded from the shipped 34-ID-E
# parameter files: det x -> lab +Z, det y -> lab +X, det z -> lab +Y, which is a
# 120 deg rotation about -(1,1,1)/sqrt(3), R_Array = (-1.2092, -1.2092, -1.2092).
# ---------------------------------------------------------------------------
_X, _Y, _Z = np.eye(3)


def _mount(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """R_mat from the detector x and y axes in lab; z = x cross y."""
    dz = np.cross(dx, dy)
    return np.column_stack([dx, dy, dz])


MOUNTS: dict[str, np.ndarray] = {
    # panel normal (det z) = +lab Y  -- the 34-ID-E family
    "34IDE":       _mount(+_Z, +_X),
    "34IDE_rot90": _mount(+_X, -_Z),
    "34IDE_rot180": _mount(-_Z, -_X),
    "34IDE_rot270": _mount(-_X, +_Z),
    # panel normal (det z) = -lab Y  -- the same four, flipped
    "flip":        _mount(+_Z, -_X),
    "flip_rot90":  _mount(-_X, -_Z),
    "flip_rot180": _mount(-_Z, +_X),
    "flip_rot270": _mount(+_X, +_Z),
}


#: How to read XMAS's (roll, pitch, yaw) as rotations of the detector frame.
#: Each entry maps a name to the (axis_index, sign) it acts on, in the
#: detector frame, applied roll -> pitch -> yaw.  Sub-degree here, so these
#: only have to get close enough to index; ``laue-index calibrate`` fits them.
TILT_CONVENTIONS: dict[str, tuple[tuple[int, float], ...]] = {
    "rpy_zxy_pos": ((2, +1.0), (0, +1.0), (1, +1.0)),
    "rpy_zxy_neg": ((2, -1.0), (0, -1.0), (1, -1.0)),
    "rpy_zyx_pos": ((2, +1.0), (1, +1.0), (0, +1.0)),
    "rpy_zyx_neg": ((2, -1.0), (1, -1.0), (0, -1.0)),
}


def poni_pixel(cal: XmasCalibration) -> tuple[float, float]:
    """XMAS center channel -> (px, py) 0-based LaueMatching pixel indices.

    px indexes COLUMNS (the fast axis), py indexes ROWS.

    The mapping is not a guess: each TPS 21A ``Condition.txt`` states the centre
    in both XMAS and Albula conventions, and

        2527 - xcent == albula_second   and   ycent - 1 == albula_first

    holds exactly to both decimals on both scans (A5: 1298.02/1204.76 ->
    1203.76/1228.98 ; Nb1_3: 1298.27/1212.62 -> 1211.62/1228.73).  So XMAS x
    runs along the 2527-pixel axis counting from the far end -- that axis is the
    image's ROWS -- and XMAS y runs along the 2463-pixel axis, 1-based.
    """
    py = cal.n_first - cal.xcent          # row, 0-based
    px = cal.ycent - 1.0                  # column, 0-based
    return px, py


def convert(cal: XmasCalibration, mount: str = "34IDE",
            tilt_convention: str = "rpy_zxy_pos",
            row_parity: int = +1) -> LaueGeometry:
    """One XMAS calibration + one convention choice -> a LaueMatching pose.

    ``row_parity = -1`` is the hypothesis that the readout's row direction runs
    OPPOSITE to the panel's detector-y axis.  Because ``R_Array`` must be a
    proper rotation, that cannot be written into the parameter file: it is
    equivalent to flipping the image rows (``data[::-1, :]``) and negating
    ``P1``, and the caller must do the flip.  ``LaueGeometry.row_parity``
    records which, so a geometry and the images it describes cannot drift apart.

    It is not a free choice and not a gauge -- but it is *nearly* one.  With
    the tilts zeroed a row-mirror is exact to 6.7e-16 in qhat; it is broken only
    by the sub-degree tilts, linearly (see LAB_NOTEBOOK_TPS21A.md section 3a).
    """
    if mount not in MOUNTS:
        raise KeyError(f"unknown mount {mount!r}; have {sorted(MOUNTS)}")
    if tilt_convention not in TILT_CONVENTIONS:
        raise KeyError(f"unknown tilt convention {tilt_convention!r}; "
                       f"have {sorted(TILT_CONVENTIONS)}")
    if row_parity not in (+1, -1):
        raise ValueError(f"row_parity must be +1 or -1, got {row_parity!r}")

    nr_px_x = cal.n_second        # columns, fast axis
    nr_px_y = cal.n_first         # rows
    dx = dy = cal.px_size_m

    px_poni, py_poni = poni_pixel(cal)
    P0 = (0.5 * (nr_px_x - 1) - px_poni) * dx
    P1 = (0.5 * (nr_px_y - 1) - py_poni) * dy * row_parity
    P2 = cal.distance_mm * 1e-3

    R_mat = np.array(MOUNTS[mount], dtype=float)
    angles = (cal.roll_deg, cal.pitch_deg, cal.yaw_deg)
    for angle_deg, (axis_idx, sign) in zip(angles, TILT_CONVENTIONS[tilt_convention]):
        if angle_deg == 0.0:
            continue
        axis = np.zeros(3)
        axis[axis_idx] = 1.0
        # tilts act on the detector frame, so post-multiply
        R_mat = R_mat @ rodrigues_to_matrix(axis * np.radians(sign * angle_deg))

    R_vec = matrix_to_rodrigues(R_mat)
    return LaueGeometry(
        P=(float(P0), float(P1), float(P2)),
        R=(float(R_vec[0]), float(R_vec[1]), float(R_vec[2])),
        nr_px_x=nr_px_x, nr_px_y=nr_px_y, px_x=dx, px_y=dy,
        mount=mount, tilt_convention=tilt_convention, row_parity=row_parity,
        source=cal.provenance or cal.name,
    )


def enumerate_candidates(cal: XmasCalibration) -> Iterator[LaueGeometry]:
    """The physically distinct candidates, to be discriminated by the indexer
    on real data.  Never pick one by argument.

    4 mountings x 2 row parities x 4 tilt readings = 32.  Only the four mounts
    with the panel normal along +lab Y are enumerated: the four with -Y are
    those same four turned 180 deg about the beam, which is an exact gauge
    (verified to 2.0e-12 px), so including them would just duplicate every row.
    """
    for mount in [m for m in MOUNTS if not m.startswith("flip")]:
        for parity in (+1, -1):
            for tilt in TILT_CONVENTIONS:
                yield convert(cal, mount, tilt, parity)


# ---------------------------------------------------------------------------
# WORKED EXAMPLES, not defaults.  These are the two TPS 21A (NSRRC) calibrations
# transcribed from the ``Condition.txt`` files shipped with that data, kept here
# because the tests pin the conversion against them and because a real instance
# is the fastest way to see what an ``XmasCalibration`` looks like filled in.
#
# Each scan carries its OWN calibration.  Geometry from another run is not
# transferable -- these two differ by 0.46 mm in distance and 0.03 deg in tilt
# despite being the same station.
# ---------------------------------------------------------------------------
TPS21A_A5 = XmasCalibration(
    name="A5_fine",
    distance_mm=522.115, xcent=1298.02, ycent=1204.76,
    roll_deg=0.0, pitch_deg=-0.388, yaw_deg=0.308,
    provenance="TPS 21A XMAS calibration, A5_fine_Condition.txt, beamtime 2024-10-24",
)

TPS21A_NB1_3 = XmasCalibration(
    name="Nb1_3",
    distance_mm=521.652, xcent=1298.27, ycent=1212.62,
    roll_deg=0.0, pitch_deg=-0.35551, yaw_deg=0.280510,
    provenance="TPS 21A XMAS calibration, Nb1_3_Condition.txt, beamtime 2023-06-06",
)


if __name__ == "__main__":  # pragma: no cover - a worked example, not a CLI
    for cal in (TPS21A_A5, TPS21A_NB1_3):
        g = convert(cal)
        px, py = poni_pixel(cal)
        print(f"{cal.name:8s} PONI (col,row) = ({px:.2f}, {py:.2f})  "
              f"P = {g.P[0]:.7f} {g.P[1]:.7f} {g.P[2]:.6f}  "
              f"R = {g.R[0]:.6f} {g.R[1]:.6f} {g.R[2]:.6f}")
