"""Integration test: exercise laue_postprocess.process_single_image with
synthetic stream-format inputs and confirm (a) the HDF5 is written,
(b) the /entry/provenance group is present, (c) the .indexing.txt is
produced by default, and (d) --no-indexfile suppresses it.

Run:
    cd ~/opt/LaueMatching && python -m pytest scripts/tests/test_postprocess_integration.py -v
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _euler_to_matrix_c(psi: float, phi: float, theta: float) -> np.ndarray:
    cps, cph, cth = math.cos(psi), math.cos(phi), math.cos(theta)
    sps, sph, sth = math.sin(psi), math.sin(phi), math.sin(theta)
    return np.array([
        [cth*cps - sth*cph*sps, -cth*cph*sps - sth*cps, sph*sps],
        [cth*sps + sth*cph*cps,  cth*cph*cps - sth*sps, -sph*cps],
        [sth*sph,                cth*sph,               cph],
    ])


def _build_inputs():
    """Build one orientation + two spots in the stream-format layout."""
    rot = _euler_to_matrix_c(0.3, 0.8, 1.0)
    recip = np.eye(3) * 17.83

    orient = np.zeros((1, 35), dtype=np.float64)
    orient[0, 0] = 1        # ImageNr
    orient[0, 1] = 0        # GrainNr
    orient[0, 5] = 120.0    # quality
    orient[0, 7] = 2        # NSpotsCalc
    orient[0, 8:17] = recip.flatten()
    orient[0, 17:23] = [0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0]
    orient[0, 23:32] = rot.flatten()

    # Spot row: [ImageNr, GrainNr, SpotNr, h, k, l, X, Y, Q0, Q1, Q2, Intensity]
    spots = np.array([
        [1, 0, 0, 0, 0, 2, 100.0, 200.0, 0.0, 0.1, -0.995, 1500.0],
        [1, 0, 1, 1, 1, 5, 110.0, 210.0, 0.05, 0.12, -0.99, 800.0],
    ], dtype=np.float64)

    return orient, spots


@pytest.fixture
def synthetic_cfg():
    """Minimal cfg dict matching what parse_config returns."""
    return {
        "space_group": 225,
        "symmetry": "F",
        "lattice_parameter": "0.35238 0.35238 0.35238 90 90 90",
        "nr_px_x": 2048, "nr_px_y": 2048,
        "px_x": 0.0002, "px_y": 0.0002,
        "r_array": "-1.2 -1.2 -1.2",
        "p_array": "0.028745 0.002788 0.513115",
        "elo": 5.0, "ehi": 30.0,
        "maxAngle": 0.1,
    }


def test_process_single_image_writes_h5_provenance_and_indexfile(tmp_path: Path, synthetic_cfg):
    """Default path: expect both H5 + indexing.txt."""
    h5py = pytest.importorskip("h5py")

    import laue_postprocess as pp
    import laue_provenance as lp

    orient, spots = _build_inputs()
    result = pp.process_single_image(
        image_nr=1,
        orientations=orient,
        spots=spots,
        cfg=synthetic_cfg,
        output_dir=str(tmp_path),
        min_unique=1,
        mapping_info={"file": "test_image.h5", "frame": 0},
        labels=None,
        folder="",
    )

    assert result["n_filtered"] >= 0

    h5_path = tmp_path / "image_00001.output.h5"
    idx_path = tmp_path / "image_00001.indexing.txt"
    assert h5_path.exists(), "output HDF5 not written"
    assert idx_path.exists(), "indexing.txt not written by default"

    # HDF5 has the provenance group
    with h5py.File(h5_path, "r") as hf:
        assert "/entry/provenance" in hf
        prov = lp.read_from_h5(hf, group="/entry/provenance")
        assert "timestamp_utc" in prov
        # Git commit should be the real repo commit (or "unknown" if unavailable)
        assert prov["git"]["commit"], "git commit not populated"

    # IndexFile has the expected opening keyword
    assert idx_path.read_text().splitlines()[0] == "$filetype\tIndexFile"


def test_process_single_image_no_indexfile_flag(tmp_path: Path, synthetic_cfg):
    """With write_indexfile=False, the .indexing.txt must NOT be created."""
    import laue_postprocess as pp

    orient, spots = _build_inputs()
    pp.process_single_image(
        image_nr=2,
        orientations=orient,
        spots=spots,
        cfg=synthetic_cfg,
        output_dir=str(tmp_path),
        min_unique=1,
        mapping_info={"file": "test2.h5", "frame": 0},
        labels=None,
        folder="",
        write_indexfile=False,
    )

    h5_path = tmp_path / "image_00002.output.h5"
    idx_path = tmp_path / "image_00002.indexing.txt"
    assert h5_path.exists()
    assert not idx_path.exists(), "indexing.txt should be suppressed with write_indexfile=False"


def test_process_single_image_indexfile_dir_redirects(tmp_path: Path, synthetic_cfg):
    """--indexfile-out should redirect the .indexing.txt to a different dir."""
    import laue_postprocess as pp

    orient, spots = _build_inputs()
    idx_dir = tmp_path / "text_outputs"

    pp.process_single_image(
        image_nr=3,
        orientations=orient,
        spots=spots,
        cfg=synthetic_cfg,
        output_dir=str(tmp_path),
        min_unique=1,
        mapping_info={"file": "test3.h5", "frame": 0},
        folder="",
        indexfile_dir=str(idx_dir),
    )

    # HDF5 still in tmp_path, indexing.txt in redirected dir
    assert (tmp_path / "image_00003.output.h5").exists()
    assert (idx_dir / "image_00003.indexing.txt").exists()
    # And NOT in the default location
    assert not (tmp_path / "image_00003.indexing.txt").exists()
