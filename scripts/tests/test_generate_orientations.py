"""Tests for scripts/GenerateOrientations.py and annotate_orientation_db.py.

Run:
    cd ~/opt/LaueMatching && python -m pytest scripts/tests/test_generate_orientations.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def tiny_orient_file(tmp_path: Path) -> Path:
    """Build a 1000-orientation binary by hand (identity + 999 random)."""
    rng = np.random.default_rng(42)
    # Random 3x3 with det +1 via QR
    mats = np.empty((1000, 3, 3), dtype=np.float64)
    mats[0] = np.eye(3)
    for i in range(1, 1000):
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        mats[i] = q
    p = tmp_path / "tiny.bin"
    p.write_bytes(mats.tobytes())
    return p


def test_annotate_produces_sidecar(tiny_orient_file: Path):
    from annotate_orientation_db import annotate, RECORD_BYTES
    sidecar = annotate(tiny_orient_file)
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    # Sanity: fingerprint covers the binary
    assert meta["inputs"][0]["basename"] == tiny_orient_file.name
    assert meta["inputs"][0]["size"] == 1000 * RECORD_BYTES
    # Retroactive metadata tag is present
    assert meta["extra"]["crystal_system"] == "cubic"
    assert "full SO(3)" in meta["extra"]["covers"]
    assert meta["extra"]["actual_n_orientations"] == 1000


def test_annotate_rejects_missing(tmp_path: Path):
    from annotate_orientation_db import annotate
    with pytest.raises(FileNotFoundError):
        annotate(tmp_path / "nope.bin")


def test_annotate_warns_on_bad_size(tmp_path: Path, capsys):
    from annotate_orientation_db import annotate
    p = tmp_path / "odd.bin"
    p.write_bytes(b"\x00" * (9 * 8 + 3))  # one-and-a-bit records
    annotate(p)
    err = capsys.readouterr().err
    assert "not a multiple" in err


@pytest.mark.skipif(
    pytest.importorskip("orix", reason="orix not installed") is None,
    reason="orix required",
)
def test_generate_produces_full_SO3(tmp_path: Path):
    """Smoke-test: generated rotations span more than the fundamental zone.

    Uses a coarse 20-degree spacing so the test runs in seconds. We verify
    that the generator emits the FULL SO(3), not a fundamental-zone-reduced
    set: the sample size should be ~24x the cubic fundamental count.
    """
    from GenerateOrientations import generate, RECORD_BYTES

    out = tmp_path / "small.bin"
    info = generate(
        spacing_deg=20.0,
        crystal_system="cubic",
        output_path=out,
        sampling="haar",
    )
    n = info["n_orientations"]
    assert n > 0
    # Shipped DB is full SO(3), 0.4 deg spacing ≈ 100M. At 20 deg we expect
    # thousands, definitely more than the fundamental-zone cubic count.
    from orix.sampling import get_sample_fundamental
    from orix.quaternion.symmetry import Oh
    fz = get_sample_fundamental(resolution=20, point_group=Oh)
    assert n > fz.size, (
        f"generated {n} rotations but fundamental-zone sample has {fz.size}; "
        f"generator may be reducing to the fundamental zone"
    )
    # Ratio should be at least ~10x (Oh has 24 operators; discretization
    # drags the ratio down to ~22 in practice).
    assert n / fz.size > 10, (
        f"ratio full/fundamental = {n / fz.size:.2f}; expected ~20-24"
    )

    # File has the right byte count
    assert out.stat().st_size == n * RECORD_BYTES

    # Sidecar exists and names the covers correctly
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    meta = json.loads(sidecar.read_text())
    assert meta["config"]["covers"].startswith("full SO(3)")
    assert meta["config"]["spacing_deg"] == 20.0
    assert meta["extra"]["n_orientations"] == n


@pytest.mark.skipif(
    pytest.importorskip("orix", reason="orix not installed") is None,
    reason="orix required",
)
def test_generate_orthogonal_matrices(tmp_path: Path):
    """Every emitted record should be a proper rotation (det +1, orthogonal)."""
    from GenerateOrientations import generate, RECORD_BYTES

    out = tmp_path / "rots.bin"
    generate(spacing_deg=30.0, crystal_system="cubic", output_path=out)
    raw = np.fromfile(out, dtype=np.float64)
    assert raw.size % 9 == 0
    mats = raw.reshape(-1, 3, 3)
    # Sample a handful for speed
    idx = np.linspace(0, len(mats) - 1, 50, dtype=int)
    for m in mats[idx]:
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-10)
