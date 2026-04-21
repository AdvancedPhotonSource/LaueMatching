"""Golden-file test against a curated subset of the Tischler sample
``Ni_index150L0_Ni_wire_pos1_2_10.txt`` the user attached when requesting
IndexFile output. The fixture lives at
``scripts/tests/golden/ni_sample.indexing.txt``.

We don't try to byte-match (whitespace varies); instead we verify that
our writer, given the same data, produces a file whose **structural
fingerprint** matches: same keywords present, same pattern block shape,
and — critically — that the ``// rotation matrix`` commentary shows the
*first column* of the matrix, not the first row.

Run:
    cd ~/opt/LaueMatching && python -m pytest scripts/tests/test_indexfile_golden.py -v
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

GOLDEN = Path(__file__).parent / "golden" / "ni_sample.indexing.txt"


def _parse_matrix_from_text(text: str, tag: str) -> np.ndarray:
    """Extract a ``{{a,b,c}{d,e,f}{g,h,i}}`` matrix for a given tag
    (e.g. ``$rotation_matrix0``) from an IndexFile text."""
    m = re.search(rf"\${tag}\d*\s+(\{{\{{.+?\}}\}})", text)
    assert m is not None, f"tag ${tag} not found"
    raw = re.findall(r"-?\d+\.\d+", m.group(1))
    return np.array([float(v) for v in raw]).reshape(3, 3)


def test_golden_fixture_loadable():
    """The fixture should itself be parseable (this anchors our regex)."""
    txt = GOLDEN.read_text()
    assert "$filetype\tIndexFile" in txt
    rot = _parse_matrix_from_text(txt, "rotation_matrix")
    assert rot.shape == (3, 3)
    # First row matches the sample
    np.testing.assert_allclose(rot[0], [0.9970272, 0.0037123, -0.0769611], atol=1e-7)


def test_rotation_matrix_commentary_matches_columns():
    """The fixture's ``// rotation matrix`` line must equal the *first
    column* of the stated matrix — this is the non-obvious convention
    we have to preserve in our writer."""
    txt = GOLDEN.read_text()
    rot = _parse_matrix_from_text(txt, "rotation_matrix")
    m = re.search(r"//\s+rotation matrix\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", txt)
    assert m is not None
    a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
    np.testing.assert_allclose([a, b, c], rot[:, 0], atol=1e-5)


def test_our_writer_matches_golden_shape(tmp_path: Path):
    """Reconstruct the Ni pattern 0 from the fixture and run it through
    our writer. Verify every required keyword and the per-spot row
    pattern from the original are present in our output."""

    # Pattern 0 data taken from the user's attached file
    rot = np.array([
        [ 0.9970272,  0.0037123, -0.0769611],
        [-0.0590224, -0.6052710, -0.7938283],
        [-0.0495293,  0.7960108, -0.6032526],
    ])
    recip = np.array([
        [17.7777017,   0.0661931,  -1.3722717],
        [-1.0524119, -10.7924121, -14.1545209],
        [-0.8831422,  14.1934369, -10.7564213],
    ])
    # Three of the 16 spots in the user's file (enough to verify structure)
    spots = [
        lif.IndexFileSpot((-0.0495221, 0.7960039, -0.6032622), (0, 0, 2),
                          intensity=1.0, energy_kev=5.8324, err_deg=0.00320, peak_index=0),
        lif.IndexFileSpot(( 0.1328918, 0.6501967, -0.7480535), (1, 1, 5),
                          intensity=0.0337, energy_kev=12.2201, err_deg=0.00206, peak_index=9),
        lif.IndexFileSpot(( 0.2683093, 0.7563513, -0.5966094), (2, 0, 6),
                          intensity=0.0098, energy_kev=18.6494, err_deg=0.00819, peak_index=13),
    ]
    pat = lif.IndexFilePattern(
        euler_deg=(-95.53748287, 127.10320335, 86.43953933),
        goodness=279.758,
        rms_error_deg=0.00498,
        rotation_matrix=rot,
        recip_lattice=recip,
        spots=spots,
    )
    hdr = lif.IndexFileHeader(
        peak_file="Ni_wire_pos1_2_10.txt",
        input_image="temp.h5",
        kev_max_calc=17.2,
        kev_max_test=35.0,
        angle_tolerance_deg=0.1,
        n_patterns_found=1,
        n_indexed=3,
        n_input_data=38,
        execution_time_sec=0.10,
        structure_desc="Ni",
        space_group=225,
        lattice_params_nm_deg=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        atom_descriptions=["Ni001  0 0 0 1"],
        x_dim=2048, y_dim=2048, x_dim_det=2048, y_dim_det=2048,
        end_x=2047, end_y=2047,
        peak_search_params={"threshold": 150},
    )
    out = lif.write_indexfile(tmp_path / "ours.indexing.txt", hdr, [pat])
    our_text = out.read_text()
    golden_text = GOLDEN.read_text()

    # Every top-level keyword in the golden must appear in our output
    keywords = set(re.findall(r"(\$[A-Za-z]\w*)", golden_text))
    # Strip the trailing integer suffix for pattern/array/etc.
    keywords = {re.sub(r"\d+$", "", kw) for kw in keywords}
    for kw in keywords:
        # Our output may use numbered form (e.g. $pattern0), so match the prefix.
        assert re.search(re.escape(kw) + r"\d*", our_text), f"missing keyword {kw}"

    # Our rotation commentary line == first column of rot
    m = re.search(r"//\s+rotation matrix\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", our_text)
    assert m is not None
    np.testing.assert_allclose(
        [float(m.group(i)) for i in (1, 2, 3)], rot[:, 0], atol=1e-4,
    )

    # Per-spot row layout matches
    row = re.search(
        r"\[\s*0\]\s+\(\s*[-.\d]+\s+[-.\d]+\s+[-.\d]+\)\s+"
        r"\(\s*\d+\s+\d+\s+\d+\)",
        our_text,
    )
    assert row is not None

    # Preserve the Tischler typo so downstream parsers keep working
    assert "$AtomDesctiption1" in our_text
