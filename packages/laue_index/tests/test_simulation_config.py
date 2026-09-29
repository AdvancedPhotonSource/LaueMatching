"""GenerateSimulation parses the templates' values and lays the image out as
the indexer reads it; GenerateHKLs refuses arguments it would ignore.

* SimulationSmoothingWidth was parsed with int(); both shipped templates say
  2.0, so every RunImage run from a template failed its simulation step
  (EnableSimulation defaults on) with ValueError -> sys.exit(1).
* The image was allocated (nPxX, nPxY) but indexed [y, x]: transposed, and on a
  non-square panel spots with y >= nPxX were clipped away.
* The Symmetry check accepted any 1-char value and some multi-char ones
  (`sym not in 'FICAR' and len(sym) != 1`).
* GenerateHKLs used parse_known_args, so RunImage's -Elo (and any typo) was
  dropped without a word.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from laue_index.pipeline import add_to_path

add_to_path()
import GenerateSimulation  # noqa: E402

BASE = {"SpaceGroup": "225", "Symmetry": "F", "LatticeParameter": "0.36 0.36 0.36 90 90 90",
        "R_Array": "0 0 0", "P_Array": "0 0 60", "PxX": "0.2", "PxY": "0.2",
        "Elo": "5", "Ehi": "30", "SimulationSmoothingWidth": "2.0",
        "NrPxX": "64", "NrPxY": "32"}


def _params(tmp_path, **over):
    kv = dict(BASE, **over)
    p = tmp_path / "p.txt"
    p.write_text("".join(f"{k} {v}\n" for k, v in kv.items()))
    return p


def test_smoothing_width_accepts_the_template_value(tmp_path):
    cfg = GenerateSimulation.ConfigParser(str(_params(tmp_path)))
    assert cfg.params["gaussWidth"] == 2.0


@pytest.mark.parametrize("sym", ["P", "B", "F", "R"])
def test_valid_symmetry_letters_accepted(tmp_path, sym):
    assert GenerateSimulation.ConfigParser(str(_params(tmp_path, Symmetry=sym))).params["sym"] == sym


@pytest.mark.parametrize("sym", ["X", "f", "FI"])
def test_invalid_symmetry_refused(tmp_path, sym):
    with pytest.raises((ValueError, SystemExit)):
        GenerateSimulation.ConfigParser(str(_params(tmp_path, Symmetry=sym)))


def test_non_square_image_is_rows_by_columns(tmp_path):
    cfg = GenerateSimulation.ConfigParser(str(_params(tmp_path)))
    sim = GenerateSimulation.DiffractionSimulator(cfg.params, None, 0)
    assert sim.img.shape == (32, 64)
    sim.splat_spot(30, 60, 1.0)
    assert np.unravel_index(sim.img.argmax(), sim.img.shape) == (30, 60)


def test_generate_hkls_refuses_unknown_arguments(tmp_path):
    r = subprocess.run([sys.executable, "-c",
                        "import sys; from laue_index.pipeline import add_to_path; add_to_path();"
                        "import GenerateHKLs; sys.argv=['GenerateHKLs','-resultFileName','x.csv',"
                        "'-latticeParameter','0.36','0.36','0.36','90','90','90','-RArray','0','0','0',"
                        "'-PArray','0','0','0.5','-bogusFlag','1']; GenerateHKLs.main()"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode != 0 and "bogusFlag" in (r.stderr + r.stdout)


def test_generate_hkls_accepts_elo():
    # RunImage passes -Elo; it must be a declared argument (it does not change
    # the list: the C filters by Elo when predicting spots).
    src = open(GenerateSimulation.__file__.replace("GenerateSimulation.py", "GenerateHKLs.py")).read()
    assert "'-Elo'" in src and ".parse_known_args(" not in src


def test_zero_r_array_is_the_identity_rotation(tmp_path):
    # R_Array 0 0 0 is a legitimate unrotated detector (the C has taken it as
    # the identity since 0.7.2); the simulator divided the rotation vector by
    # its zero length and built a NaN rotation, so every spot was NaN.
    cfg = GenerateSimulation.ConfigParser(str(_params(tmp_path, R_Array="0 0 0")))
    sim = GenerateSimulation.DiffractionSimulator(cfg.params, None, 0)
    assert np.allclose(sim.rot, np.eye(3))
