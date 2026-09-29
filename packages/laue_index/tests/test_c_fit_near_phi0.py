"""FitOrientation reaches a correction whatever the seed's Euler angles.

Before 0.8.0 the fit bounded the ZXZ Euler angles by +-3 deg about the seed.
Near Phi = 0 (or 180) psi and theta both turn about z, so a correction about
the in-plane axis perpendicular to the line of nodes was unreachable unless
Phi_seed > ~delta / sin(3 deg) (about 0.3% of random orientations for a
0.3 deg grid error). The fit now refines a rotation vector about the seed.
The fixture lights the pixels the header predicts for a truth 0.3 deg (one
database grid step) from the seed and blurs them as the coarse stage does; the
fit must land on the truth. Control: a generic seed (Phi = 40 deg), which the
old parametrisation also handled.
"""
import os
import subprocess

import pytest

from _cbuild import FIXTURES, HEADERS, build


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    d = tmp_path_factory.mktemp("fit0")
    exe = str(d / "fit_near_phi0")
    err = build(exe, [os.path.join(FIXTURES, "fit_near_phi0.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err

    def _run(phi, axis, delta=0.3):
        out = subprocess.run([exe, str(phi), *map(str, axis), str(delta)],
                             capture_output=True, text=True, check=True).stdout.split()
        return float(out[1]), float(out[3]), int(out[5])
    return _run


@pytest.mark.parametrize("phi", [0.5, 40.0])
@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0.6, -0.8, 0), (0, 0, 1)])
def test_fit_recovers_the_truth(fit, phi, axis):
    miso, seed_miso, lit = fit(phi, axis)
    assert lit >= 20, f"only {lit} truth spots on the panel: geometry broken"
    assert seed_miso > 0.25
    assert miso < 0.03, f"Phi_seed={phi} axis={axis}: fit ended {miso:.3f} deg from truth"
