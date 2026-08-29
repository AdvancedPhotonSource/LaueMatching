"""The indexer must survive the run that WRITES the forward cache.

This is the regression test for a SIGSEGV that lived for two months without a
single test noticing, because every test that ran the binary handed it a cache
that already existed.

The crash was in the exit path: an ``if (orientsMapped) munmap(...) / else
free(orients)`` pair had a second ``if`` inserted between the two halves, so
the ``else`` re-parented onto it. Writing the cache leaves ``outArr`` NULL, the
``else`` fires, and ``free()`` is called on memory ``munmap``'d one line above.
Reading an existing cache takes the other branch, which is exactly why the
whole suite stayed green.

Two conditions have to coincide, so the test arranges both:
  * the orientation database is mmap'd -- which the C only does for a path
    under ``/dev/shm`` (LaueMatchingCPU.c, `strncmp(orientFN, "/dev/shm", 8)`);
  * the forward cache does not exist yet.

Everything is tiny (a few thousand orientations, a 256-px detector), so the
whole thing runs in seconds rather than the ~16 minutes a real cold run costs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from laue_index import indexer
from laue_index.pipeline import run_module

_SHM = Path("/dev/shm")
_N_ORIENTS = 3000
_N_PX = 256
_MAX_SPOTS = 10

_REASONS = []
if not _SHM.is_dir() or not os.access(_SHM, os.W_OK):
    _REASONS.append("needs /dev/shm (the C only mmaps the database from there)")
_BIN = Path(os.environ["LAUE_BIN"]) if os.environ.get("LAUE_BIN") else None
if _BIN is None and not indexer.available():
    _REASONS.append("LaueMatchingCPU not built")

pytestmark = pytest.mark.skipif(bool(_REASONS), reason="; ".join(_REASONS))

_CONFIG = """\
LatticeParameter 0.35238 0.35238 0.35238 90 90 90
SpaceGroup 225
Symmetry F
P_Array 0.0 0.0 0.05
R_Array -1.2013 -1.2140 -1.2188
PxX 0.0002
PxY 0.0002
NrPxX {npx}
NrPxY {npx}
Elo 5
Ehi 15
MinNrSpots 3
MinGoodSpots 3
MinIntensity 1
MaxAngle 5
MinArea 1
Threshold 0
WatershedImage 0
ResultDir {results}
HKLFile {hkls}
OrientationFile {db}
MaxNrLaueSpots {maxspots}
ForwardFile {fwd}
OrientationSpacing 5.0
DoFwd 0
"""


def _random_orientation_db(path: Path, n: int) -> None:
    """n orientation matrices, 9 float64 each -- the database's whole format."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    oms = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1)
    oms.astype(np.float64).tofile(path)
    assert path.stat().st_size == n * 9 * 8


def _hkls(tmp_path: Path) -> Path:
    out = tmp_path / "hkls.csv"
    run_module("GenerateHKLs", [
        "-resultFileName", str(out), "-sgnum", "225", "-sym", "F",
        "-latticeParameter", "0.35238", "0.35238", "0.35238", "90", "90", "90",
        "-RArray", "-1.2013", "-1.2140", "-1.2188",
        "-PArray", "0.0", "0.0", "0.05",
        "-NumPxX", str(_N_PX), "-NumPxY", str(_N_PX),
        "-dx", "0.0002", "-dy", "0.0002", "-Elo", "5", "-Ehi", "15"])
    assert out.is_file() and out.stat().st_size > 0
    return out


def _image(path: Path) -> None:
    """A frame with real intensity, so the run reaches the end rather than
    bailing out early on an empty image."""
    rng = np.random.default_rng(1)
    img = np.zeros((_N_PX, _N_PX), dtype=np.float64)
    ys = rng.integers(4, _N_PX - 4, 400)
    xs = rng.integers(4, _N_PX - 4, 400)
    for y, x in zip(ys, xs):
        img[y - 1:y + 2, x - 1:x + 2] += 500.0
    img.tofile(path)


def _run(tmp_path: Path, db: Path, fwd: Path) -> subprocess.CompletedProcess:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    image = tmp_path / "image.bin"
    if not image.exists():
        _image(image)
    hkls = tmp_path / "hkls.csv"
    if not hkls.exists():
        hkls = _hkls(tmp_path)
    cfg = tmp_path / "params.txt"
    cfg.write_text(_CONFIG.format(npx=_N_PX, results=results, hkls=hkls, db=db,
                                  maxspots=_MAX_SPOTS, fwd=fwd))
    exe = _BIN or indexer.binary_path()
    return subprocess.run(
        [str(exe), str(cfg), str(db), str(hkls), str(image), "2"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=600)


@pytest.fixture
def shm_db(tmp_path):
    """The database must be under /dev/shm or the C will not mmap it, and the
    bug this guards only exists on the mmap'd path."""
    db = _SHM / f"laue_test_orients_{os.getpid()}.bin"
    _random_orientation_db(db, _N_ORIENTS)
    yield db
    db.unlink(missing_ok=True)


def test_cold_forward_cache_run_exits_cleanly(tmp_path, shm_db):
    """The first run on any machine: no cache yet, so it writes one.

    Before the fix this exited -11 AFTER writing complete, correct output, and
    the pipeline reported the image as failed.
    """
    fwd = tmp_path / "fwd_cold.bin"
    assert not fwd.exists()
    proc = _run(tmp_path, shm_db, fwd)
    assert proc.returncode == 0, (
        f"cold-cache run exited {proc.returncode} "
        f"({'SIGSEGV' if proc.returncode == -11 else 'see output'})\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-1000:]}")
    assert fwd.is_file() and fwd.stat().st_size > 0, "it should have written the cache"
    assert "simulation mode" in proc.stdout, (
        "this test is only meaningful if the run actually wrote the cache")


def test_warm_forward_cache_run_still_exits_cleanly(tmp_path, shm_db):
    """The other branch of the same cleanup: cache present, so it is mmap'd."""
    fwd = tmp_path / "fwd_warm.bin"
    first = _run(tmp_path, shm_db, fwd)
    assert first.returncode == 0, first.stdout[-2000:]
    second = _run(tmp_path, shm_db, fwd)
    assert second.returncode == 0, (
        f"warm-cache run exited {second.returncode}\n{second.stdout[-2000:]}")
    assert "will not do forward simulation" in second.stdout.lower(), (
        "the second run should have read the cache the first one wrote")
