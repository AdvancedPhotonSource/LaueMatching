"""Characterization: full RunImage end-to-end (raw frame -> filtered_orientations).

The whole-pipeline behaviour anchor (REFACTOR_PLAN §6 step 0 / §6.5).  Runs the
real ``RunImage process`` on a real Ni frame through the C indexer and pins a
stable summary of ``/entry/results/filtered_orientations``.  This is the guard
that makes the §6.5 RunImage split (and the §6.2/§6.4 inline edits) provably
behaviour-preserving across the whole stack, not just the unit functions.

Heavy and machine-specific, so it is double-gated:
  * env ``LAUE_E2E=1`` must be set, and
  * the binary, orientation DB, HKL file, frame, and a *prebuilt* forward-sim
    cache must all be present.
The cache (~12 GB) is built once with a DoFwd=1 run; this test then runs DoFwd=0
against it (fast) so it can be re-run cheaply during the refactor.

Paths resolve from env with local defaults:
  LAUE_BIN        -> <repo>/bin/LaueMatchingCPU
  LAUE_ORIENT_DB  -> ~/opt/LaueMatching/100MilOrients.bin
  LAUE_FWD_CACHE  -> <SEED_DIR>/e2e/fwdcache_NiIndent.bin
  frame + hkls    -> SEED_DIR (LAUE_FIXTURES)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from _golden import check_golden, local_fixture, SEED_DIR

_REPO = Path(__file__).resolve().parents[1]
_FRAME = "Indent_KB_2D_scan2_1040.h5"
_HKLS = "valid_hkls_Ni.csv"

_BIN = Path(os.environ.get("LAUE_BIN", _REPO / "bin" / "LaueMatchingCPU"))
_DB = Path(os.environ.get("LAUE_ORIENT_DB",
                          Path.home() / "opt/LaueMatching/100MilOrients.bin"))
_CACHE = Path(os.environ.get("LAUE_FWD_CACHE", SEED_DIR / "e2e" / "fwdcache_NiIndent.bin"))

_REASONS = []
if os.environ.get("LAUE_E2E") != "1":
    _REASONS.append("set LAUE_E2E=1 to run")
for label, p in [("binary", _BIN), ("orient DB", _DB), ("fwd cache", _CACHE),
                 ("frame", local_fixture(_FRAME)), ("hkls", local_fixture(_HKLS))]:
    if p is None or not Path(p).exists():
        _REASONS.append(f"missing {label}")

pytestmark = pytest.mark.skipif(bool(_REASONS), reason="; ".join(_REASONS))

_CONFIG = """\
SpaceGroup 225
Symmetry F
LatticeParameter 0.352380 0.352380 0.352380 90.000000 90.000000 90.000000
P_Array 0.028828 0.002715 0.512993
R_Array -1.20161887 -1.21404493 -1.21852276
PxX 0.000200
PxY 0.000200
NrPxX 2048
NrPxY 2048
Elo 5.0
Ehi 30.0
MinNrSpots 5
MinGoodSpots 3
MinIntensity 50
MaxAngle 5
MinArea 5
Threshold 0
ThresholdMethod adaptive
WatershedImage 0
NMeadianPasses 0
MaxNrLaueSpots 30
OrientationFile {db}
ForwardFile {cache}
OrientationSpacing 0.4
DoFwd 0
HKLFile {hkls}
ResultDir {results}
"""


def _summarize(filtered):
    """Order-independent, refinement-jitter-robust summary of solutions."""
    from laue_index import parse_solutions
    sols = parse_solutions(np.asarray(filtered), fmt="runimage")
    sols = sorted(sols, key=lambda s: s.orientation_row_nr)
    return {
        "n_kept": len(sols),
        "row_nrs": [s.orientation_row_nr for s in sols],
        "grain_nrs": sorted(s.grain_nr for s in sols),
        "n_matches": [s.n_matches for s in sols],
        # quality coarsened to absorb any nlopt refinement jitter across runs
        "quality_rounded": [round(s.quality, 1) for s in sols],
    }


def test_char_e2e_runimage_1040():
    import h5py
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        results = d / "results"
        results.mkdir()
        cfg = d / "params_e2e.txt"
        cfg.write_text(_CONFIG.format(
            db=_DB, cache=_CACHE, hkls=local_fixture(_HKLS), results=results))

        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
        proc = subprocess.run(
            [sys.executable, str(_REPO / "scripts" / "RunImage.py"), "process",
             "-c", str(cfg), "-i", str(local_fixture(_FRAME)), "-n", "8",
             "--no-viz", "--no-sim", "--no-indexfile"],
            cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=1200)
        assert proc.returncode == 0, f"RunImage failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}"

        out_h5 = results / f"{Path(_FRAME).stem}.output.h5"
        assert out_h5.exists(), f"no output h5 at {out_h5}\n{proc.stdout[-1500:]}"
        with h5py.File(out_h5, "r") as hf:
            filtered = hf["/entry/results/filtered_orientations"][()]

    check_golden("e2e_runimage_1040", _summarize(filtered))


if __name__ == "__main__":
    if _REASONS:
        print("SKIP e2e:", "; ".join(_REASONS))
    else:
        test_char_e2e_runimage_1040()
        print("PASS  test_char_e2e_runimage_1040")
