"""Tests for scripts/laue_indexfile.py.

Run:
    cd ~/opt/LaueMatching && python -m pytest scripts/tests/test_indexfile.py -v
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import laue_indexfile as lif  # noqa: E402


# ---------------------------------------------------------------------------
# Euler<->Matrix round-trip (must match the C convention byte-for-byte)
# ---------------------------------------------------------------------------

def _euler_to_matrix_c(psi_rad: float, phi_rad: float, theta_rad: float) -> np.ndarray:
    """Port of Euler2OrientMat for use in tests."""
    cps, cph, cth = math.cos(psi_rad), math.cos(phi_rad), math.cos(theta_rad)
    sps, sph, sth = math.sin(psi_rad), math.sin(phi_rad), math.sin(theta_rad)
    return np.array([
        [cth * cps - sth * cph * sps, -cth * cph * sps - sth * cps, sph * sps],
        [cth * sps + sth * cph * cps,  cth * cph * cps - sth * sps, -sph * cps],
        [sth * sph,                    cth * sph,                   cph],
    ])


def test_euler_round_trip_identity():
    m = np.eye(3)
    psi, phi, theta = lif.orient_matrix_to_euler_deg(m)
    # Identity should be all zeros
    assert abs(phi) < 1e-7
    # psi + theta is arbitrary for identity; both should sum to 0
    assert abs(psi + theta) < 1e-7


@pytest.mark.parametrize("psi,phi,theta", [
    (30.0, 45.0, 60.0),
    (-10.0, 90.0, 20.0),
    (178.0, 30.0, -178.0),
    (15.3, 127.1, 86.4),   # close to the user's example
])
def test_euler_round_trip_random(psi: float, phi: float, theta: float):
    m = _euler_to_matrix_c(math.radians(psi), math.radians(phi), math.radians(theta))
    psi2, phi2, theta2 = lif.orient_matrix_to_euler_deg(m)
    # Reconstruct from reported angles; should match original matrix
    m_reco = _euler_to_matrix_c(math.radians(psi2), math.radians(phi2), math.radians(theta2))
    np.testing.assert_allclose(m_reco, m, atol=1e-9)


# ---------------------------------------------------------------------------
# Energy formula (matches the C line-609 formula)
# ---------------------------------------------------------------------------

def test_energy_matches_c_formula():
    # Arbitrary recip lattice (identity * 17.8307 1/nm ≈ a=0.35238 nm cubic)
    recip = np.eye(3) * 17.8307091979
    # hkl=(0,0,2) gives Q in +z direction → sinTheta = -1 (backscatter geometry
    # is invalid under the indexer's convention).  Use (1,1,1) instead.
    # For a -z component we need a specific orientation.
    # Use a simple case: Q = (0.1, 0, -0.5) (sinTheta=0.5, |Q|=sqrt(0.26))
    recip = np.array([[0.1, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, -0.5]])
    hkl = [1, 0, 1]
    e = lif.energy_from_recip_and_hkl(recip, hkl)
    qlen = math.sqrt(0.01 + 0.25)
    sin_theta = 0.5 / qlen  # -qhat_z = 0.5/qlen
    expected = 1.2398419739 * qlen / (4 * math.pi * sin_theta)
    assert abs(e - expected) < 1e-9


def test_energy_nan_for_back_half():
    # sinTheta <= 0 => NaN (physically invalid)
    recip = np.eye(3)
    e = lif.energy_from_recip_and_hkl(recip, [0, 0, 1])  # Q = (0,0,1), sinTheta = -1
    assert math.isnan(e)


# ---------------------------------------------------------------------------
# Pixel ↔ Qhat round trip
# ---------------------------------------------------------------------------

def test_pixel_to_qhat_center_points_to_neg_z():
    """With zero rotation vector and P=(0,0,0.5), the detector is a plane
    at z=0.5 facing +Z; the center pixel should give kf=(0,0,1) exactly,
    so Q=kf-ki=(0,0,0) → undefined. Shift to (nrPx/2, nrPx/4) so the
    kf has a y component and gives a valid Q."""
    center_x = 2048 / 2.0 - 0.5  # well inside detector
    center_y = 2048 / 4.0
    qhat = lif.pixel_to_qhat(
        center_x, center_y,
        r_array=[0.0, 0.0, 0.0],
        p_array=[0.0, 0.0, 0.5],
        px_x=0.0002, px_y=0.0002,
        nr_px_x=2048, nr_px_y=2048,
    )
    # Must be a unit vector
    np.testing.assert_allclose(np.linalg.norm(qhat), 1.0, atol=1e-12)
    # By geometry the kf is in the y<0 half; Q = kf - ki points into -y and -z
    assert qhat[1] < 0
    assert qhat[2] < 0


def test_angular_error_simple():
    assert abs(lif.angular_error_deg([1, 0, 0], [1, 0, 0])) < 1e-9
    assert abs(lif.angular_error_deg([1, 0, 0], [0, 1, 0]) - 90.0) < 1e-9


# ---------------------------------------------------------------------------
# Peak-index nearest lookup
# ---------------------------------------------------------------------------

def test_nearest_peak_index_within_threshold():
    centers = np.array([
        [1, 100.0, 200.0, 5.0],
        [2, 300.0, 400.0, 8.0],
        [3, 500.0, 600.0, 10.0],
    ])
    # Exactly at (300, 400) → label 2, 0-indexed peak_index = 1
    assert lif.nearest_peak_index(300.5, 399.5, centers) == 1
    # Way off → -1
    assert lif.nearest_peak_index(1000, 1000, centers, max_distance_px=5.0) == -1


def test_nearest_peak_index_empty():
    assert lif.nearest_peak_index(10, 10, None) == -1
    assert lif.nearest_peak_index(10, 10, np.empty((0, 4))) == -1


# ---------------------------------------------------------------------------
# Writer: schema + structural sanity
# ---------------------------------------------------------------------------

def _sample_header_and_patterns():
    hdr = lif.IndexFileHeader(
        peak_file="test_peaks.txt",
        input_image="test_img.h5",
        kev_max_calc=17.2,
        kev_max_test=35.0,
        angle_tolerance_deg=0.1,
        n_patterns_found=1,
        n_indexed=3,
        n_input_data=4,
        execution_time_sec=0.1,
        structure_desc="Ni",
        space_group=225,
        lattice_params_nm_deg=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        x_dim=2048, y_dim=2048, x_dim_det=2048, y_dim_det=2048,
        end_x=2047, end_y=2047,
    )
    rot = _euler_to_matrix_c(math.radians(15.0), math.radians(45.0), math.radians(30.0))
    recip = np.eye(3) * 17.83
    spots = [
        lif.IndexFileSpot(g_hat=(0.1, 0.2, -0.9), hkl=(0, 0, 2),
                          intensity=1.0, energy_kev=5.83, err_deg=0.003, peak_index=0),
        lif.IndexFileSpot(g_hat=(0.2, 0.3, -0.8), hkl=(1, 1, 5),
                          intensity=0.034, energy_kev=12.22, err_deg=0.002, peak_index=9),
        lif.IndexFileSpot(g_hat=(0.3, 0.4, -0.7), hkl=(2, 0, 6),
                          intensity=0.01, energy_kev=18.65, err_deg=0.008, peak_index=13),
    ]
    pat = lif.IndexFilePattern(
        euler_deg=(15.0, 45.0, 30.0),
        goodness=42.5,
        rms_error_deg=0.005,
        rotation_matrix=rot,
        recip_lattice=recip,
        spots=spots,
    )
    return hdr, [pat]


def test_write_indexfile_has_required_keywords(tmp_path: Path):
    hdr, patterns = _sample_header_and_patterns()
    out = lif.write_indexfile(tmp_path / "t.indexing.txt", hdr, patterns)
    txt = out.read_text()
    # Filetype line first
    assert txt.splitlines()[0] == "$filetype\tIndexFile"
    # Required header keywords
    for k in ("$peakFile", "$keVmaxCalc", "$angleTolerance", "$keVmaxTest",
              "$NpatternsFound", "$Nindexed", "$NiData", "$executionTime",
              "$SpaceGroup", "$latticeParameters", "$lengthUnit",
              "$xdim", "$ydim", "$xDimDet", "$yDimDet",
              "$pattern0", "$EulerAngles0", "$goodness0", "$rms_error0",
              "$rotation_matrix0", "$recip_lattice0", "$array0"):
        assert k in txt, f"missing keyword {k}"


def test_write_indexfile_pattern_block_layout(tmp_path: Path):
    hdr, patterns = _sample_header_and_patterns()
    out = lif.write_indexfile(tmp_path / "t.indexing.txt", hdr, patterns)
    txt = out.read_text()
    # Sample row format: "    [  N]   ( gx gy gz)     (h k l)    intens,  E,  err  PkIndex"
    m = re.search(r"^\s*\[\s*0\]\s+\(\s*[-.\d]+\s+[-.\d]+\s+[-.\d]+\)\s+"
                  r"\(\s*\d+\s+\d+\s+\d+\)", txt, re.MULTILINE)
    assert m is not None, "spot row 0 does not match expected Tischler format"
    # Column-vector commentary lines present (loose whitespace match)
    assert re.search(r"//\s+column vectors", txt)
    assert re.search(r"//\s+rotation matrix", txt)
    assert re.search(r"//\s+reciprocal matrix", txt)


def test_commentary_lines_show_columns_not_rows(tmp_path: Path):
    """The Tischler ``// column vectors`` commentary is the TRANSPOSE of
    the ``{{row}{row}{row}}`` form: line *i* must print the *i*-th column.
    This regression-locks that behaviour against the shape the user's
    sample file (``Ni_index150L0_...txt``) uses.
    """
    hdr, patterns = _sample_header_and_patterns()
    # Deliberately distinctive matrix so col vs row can't match by accident.
    patterns[0].rotation_matrix = np.array([
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [7.7, 8.8, 9.9],
    ])
    out = lif.write_indexfile(tmp_path / "t.indexing.txt", hdr, patterns)
    txt = out.read_text()
    # The ``rotation matrix`` line must contain the first *column*: 1.1, 4.4, 7.7.
    m = re.search(r"//\s+rotation matrix\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", txt)
    assert m is not None, "rotation matrix commentary line not found"
    a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
    assert a == pytest.approx(1.1) and b == pytest.approx(4.4) and c == pytest.approx(7.7), (
        f"commentary line shows {(a, b, c)} but should be first column (1.1, 4.4, 7.7)"
    )


def test_write_preserves_AtomDesctiption_typo(tmp_path: Path):
    hdr, patterns = _sample_header_and_patterns()
    hdr.atom_descriptions = ["Ni001  0 0 0 1"]
    out = lif.write_indexfile(tmp_path / "t.indexing.txt", hdr, patterns)
    txt = out.read_text()
    # Preserve original typo for compat with Tischler parsers
    assert "$AtomDesctiption1" in txt


# ---------------------------------------------------------------------------
# HDF5 build path (smoke-test)
# ---------------------------------------------------------------------------

def test_build_from_h5_empty_file(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "empty.h5"
    with h5py.File(p, "w") as hf:
        hf.create_group("/entry/results")
    header, patterns = lif.build_from_h5(p, cfg={"ehi": 20.0, "elo": 5.0})
    assert header.n_patterns_found == 0
    assert patterns == []


def test_build_from_h5_synthetic_round_trip(tmp_path: Path):
    """Construct a minimal HDF5 matching the stream schema and verify the
    builder pulls out the right Euler angles and spot counts."""
    h5py = pytest.importorskip("h5py")

    # Create a rotation matrix for a known Euler angle
    psi, phi, theta = 30.0, 45.0, 60.0
    rot = _euler_to_matrix_c(math.radians(psi), math.radians(phi), math.radians(theta))
    recip = np.eye(3) * 17.83

    # Stream orientation row: 35 cols (with ImageNr). Fill:
    #   [0]  ImageNr        = 1
    #   [1]  GrainNr        = 0
    #   [2]  NumberOfSolutions
    #   [3]  Intensity
    #   [4]  NMatches*Intensity
    #   [5]  NMatches*sqrt(Intensity) = 100.0 (goodness)
    #   [6]  NMatches
    #   [7]  NSpotsCalc
    #   [8..16]  Recip (3x3)
    #   [17..22] Lattice params
    #   [23..31] Orient matrix (3x3)
    #   [32] CoarseScore
    #   [33] Misorientation (deg)
    #   [34] orientationRowNr
    orient_row = np.zeros(35, dtype=np.float64)
    orient_row[0] = 1        # image_nr
    orient_row[1] = 0        # grain_nr
    orient_row[5] = 100.0    # quality
    orient_row[8:17] = recip.flatten()
    orient_row[17:23] = [0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0]
    orient_row[23:32] = rot.flatten()
    orient_row[33] = 0.01    # misorient

    # Stream spot row: 12 cols with ImageNr.
    # Layout in _slice_spot_row assumes (has_image_nr=True):
    #   off=1; col off+0 = 1 → grain_nr
    # That means the dataset columns are:
    #   [0] ImageNr, [1] GrainNr, [2] SpotNr, [3..5] h k l, [6] X, [7] Y,
    #   [8..10] Qhat, [11] Intensity
    spot_row = np.array([1, 0, 0, 0, 0, 2, 100.0, 200.0, 0.1, 0.2, -0.9, 1000.0])

    # No need for filtered_spots to have any more rows
    p = tmp_path / "synth.output.h5"
    with h5py.File(p, "w") as hf:
        results = hf.require_group("/entry/results")
        results.create_dataset("filtered_orientations", data=orient_row.reshape(1, -1))
        results.create_dataset("filtered_spots", data=spot_row.reshape(1, -1))

    header, patterns = lif.build_from_h5(
        p,
        cfg={
            "ehi": 30.0, "elo": 5.0,
            "space_group": 225,
            "lattice_parameter": "0.35238 0.35238 0.35238 90 90 90",
            "nr_px_x": 2048, "nr_px_y": 2048,
            "r_array": "-1.2 -1.2 -1.2",
            "p_array": "0.028745 0.002788 0.513115",
            "px_x": 0.0002, "px_y": 0.0002,
            "maxAngle": 0.1,
        },
        mapping_entry={"file": "img.h5", "frame": 0},
    )
    assert header.n_patterns_found == 1
    assert header.n_indexed == 1
    assert header.space_group == 225
    assert header.lattice_params_nm_deg[0] == pytest.approx(0.35238)
    pat = patterns[0]
    assert pat.goodness == 100.0
    # Euler round-trip
    assert abs(pat.euler_deg[1] - phi) < 1e-6
    # Spot unpacked
    assert len(pat.spots) == 1
    s = pat.spots[0]
    assert s.hkl == (0, 0, 2)
    assert s.intensity == 1000.0
