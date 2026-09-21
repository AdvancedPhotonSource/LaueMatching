"""Analysis scripts fail loudly instead of reporting success on nothing.

Pinned (synthetic inputs, each script run as the subprocess it is in production):

* parentbeta_validate.py: an unset LAUE_SCAN_DATA used to make every frame fail
  inside ``except: return None``, print "VALIDATED 0" and write an empty npz. It
  now exits before reading, and if frames were attempted but none validated it
  exits WITHOUT writing and names the first read error.
* batch_peel_driver.py: a failing orchestrator used to be hidden by
  ``capture_output`` and the empty pass logged as "converged". It now exits
  non-zero, keeps the orchestrator's output in a per-pass log, leaves the base
  file's detection thresholds alone, and writes GaussSigmaMax exactly once.
* collect_scan_metrics.py: used to glob the literal string "$LAUE_WORK", print
  ``{}`` and exit 0.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_hit_counting import _write_frames, _write_params  # noqa: E402


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _run(args, env, cwd):
    """Plain interpreter, default start method (spawn on macOS): the scripts'
    __main__ guards are what make this work."""
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture
def scan(tmp_path):
    """Two raw frames with stage coordinates, and an indexing run whose results
    carry random orientations (which will not validate)."""
    h5py = pytest.importorskip("h5py")
    data, run = tmp_path / "data", tmp_path / "run"
    _write_frames(data, run, n=2)
    for i, f in enumerate(sorted(data.glob("*.h5"))):
        with h5py.File(f, "a") as h:
            h["entry1/sample/sampleX"] = np.array([float(i)])
            h["entry1/sample/sampleZ"] = np.array([0.0])
    (run / "results").mkdir()
    rng = np.random.default_rng(0)
    for i in (1, 2):
        fo = np.zeros((3, 35))
        for r in range(3):
            q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
            fo[r, 23:32] = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]).ravel()
        with h5py.File(run / "results" / f"image_{i:05d}.output.h5", "w") as h:
            h.create_dataset("entry/results/filtered_orientations", data=fo)
    work = tmp_path / "work"
    (work / "peel_map").mkdir(parents=True)
    return dict(data=data, run=run, work=work, params=_write_params(tmp_path, "zn"))


def test_parentbeta_validate_exits_on_unset_data(scan):
    env = _env(LAUE_WORK=scan["work"], LAUE_SCAN_ZN=scan["run"], LAUE_PARAMS_ZN=scan["params"],
               LAUE_OUT_PREFIX="t")
    r = _run(["parentbeta_validate.py", "zn", "1", "env"], env, _analysis_dir())
    assert r.returncode != 0 and "LAUE_SCAN_DATA" in (r.stderr + r.stdout)
    assert not list((scan["work"] / "peel_map").glob("*.npz"))


def test_parentbeta_validate_zero_validated_exits_without_writing(scan):
    env = _env(LAUE_WORK=scan["work"], LAUE_SCAN_DATA=scan["data"], LAUE_SCAN_ZN=scan["run"],
               LAUE_PARAMS_ZN=scan["params"], LAUE_OUT_PREFIX="t", LAUE_GATE_STAT="nhit_distinct")
    r = _run(["parentbeta_validate.py", "zn", "1", "env"], env, _analysis_dir())
    out = r.stderr + r.stdout
    assert r.returncode != 0, out[-2000:]
    assert "0 instances validated out of 2 frames attempted" in out
    # the analytic gate says it is on nhit even when asked for the distinct count
    assert "analytic Poisson gate is on nhit" in out
    assert not list((scan["work"] / "peel_map").glob("*.npz")), "an empty npz was written"


def test_parentbeta_validate_reports_first_read_error(scan):
    """Frames missing from the data folder: every frame raises; the message says which."""
    for f in scan["data"].glob("*.h5"):
        f.unlink()
    env = _env(LAUE_WORK=scan["work"], LAUE_SCAN_DATA=scan["data"], LAUE_SCAN_ZN=scan["run"],
               LAUE_PARAMS_ZN=scan["params"], LAUE_OUT_PREFIX="t")
    r = _run(["parentbeta_validate.py", "zn", "1", "env"], env, _analysis_dir())
    out = r.stderr + r.stdout
    assert r.returncode != 0 and "First error:" in out and "frame_00001.h5" in out


def test_batch_peel_driver_orchestrator_failure_is_loud(tmp_path, scan):
    """A fake checkout whose orchestrator exits 3: the driver must exit non-zero,
    keep the orchestrator's message, never log 'converged', and its generated
    config must carry the base thresholds and ONE GaussSigmaMax."""
    fake = tmp_path / "fake_lm"
    (fake / "scripts").mkdir(parents=True)
    (fake / "scripts" / "laue_orchestrator.py").write_text(
        "import sys\nprint('ORCHESTRATOR BOOM', file=sys.stderr)\nsys.exit(3)\n")
    base = Path(scan["params"]).read_text() + (
        "ThresholdPercentile 99.5\nMinNrSpots 6\nMinIntensity 20\n"
        "GaussSigmaMax 2.5\nGaussSigmaMax 9.9\n")
    bp = tmp_path / "params_base.txt"
    bp.write_text(base)
    env = _env(LAUE_WORK=scan["work"], LAUE_SCAN_DATA=scan["data"], LAUE_PHASE="zn",
               LAUE_PARAMS_ZN=bp, LAUE_LM=fake)
    r = subprocess.run([sys.executable, "batch_peel_driver.py"], cwd=_analysis_dir(), env=env,
                       capture_output=True, text=True, timeout=300)
    out = r.stderr + r.stdout
    assert r.returncode != 0, out[-2000:]
    assert "ORCHESTRATOR BOOM" in out
    status = (scan["work"] / "batch_peel_status.txt").read_text()
    assert "converged" not in status and "FAILED" in status
    cfg = (scan["work"] / "params" / "params_batchpeel_p1.txt").read_text().splitlines()
    keys = [l.split()[0] for l in cfg if l.split()]
    assert keys.count("GaussSigmaMax") == 1 and "GaussSigmaMax 2.5" in cfg
    assert "ThresholdPercentile 99.5" in cfg and "MinNrSpots 6" in cfg and "MinIntensity 20" in cfg


def test_collect_scan_metrics_exits_on_nothing_to_collect(tmp_path):
    env = _env(LAUE_WORK=tmp_path, LAUE_MOUNT_DEG=45)
    r = subprocess.run([sys.executable, "collect_scan_metrics.py"], cwd=_analysis_dir(), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "no scan directories" in r.stderr
    r = subprocess.run([sys.executable, "collect_scan_metrics.py"], cwd=_analysis_dir(),
                       env=_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "LAUE_WORK is not set" in r.stderr


@pytest.mark.parametrize("args,needle", [
    (["scan_map.py", "zn"], "LAUE_WORK is not set"),
    (["texture_null.py"], "usage: texture_null.py"),
    (["cluster_orientations.py"], "usage: cluster_orientations.py"),
])
def test_missing_input_gives_usage_not_traceback(args, needle):
    r = subprocess.run([sys.executable, *args], cwd=_analysis_dir(), env=_env(),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert needle in r.stderr and "Traceback" not in r.stderr
