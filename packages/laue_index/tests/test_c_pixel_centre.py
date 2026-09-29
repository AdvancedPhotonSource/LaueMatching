"""The indexer reads the pixel a predicted spot actually lies in.

Pixel k is centred at detector coordinate k (the projection puts the panel
centre at (N-1)/2; GenerateSimulation, laue_torch and laue_index.calibrate use
the same grid). Before 0.8.0 every lookup truncated: a spot predicted at
k + 0.7 was scored on pixel k, a mean bias of 0.5 px toward -x and -y, and
`(int)` also admitted -1 < px < 0 as pixel 0. The three binaries' coarse
forward loops truncated the same way, and the forward cache stored those
truncated pixels, so the cache format was bumped (ForwardFile.meta).

The fixture solves the geometry by hand (see pixel_centre.c): with the (111)
reflection and this pose, fx = (100 - P0)/0.2 + 1023.5 and
fy = (-100 - P1)/0.2 + 1023.5, so P0, P1 place the spot at any fractional
pixel. One pixel is lit; a count of 1 says which pixel was read.
"""
import os
import re
import subprocess

import numpy as np
import pytest

from _cbuild import C_DIR, FIXTURES, HEADERS, build


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    d = tmp_path_factory.mktemp("pixc")
    exe = str(d / "pixel_centre")
    err = build(exe, [os.path.join(FIXTURES, "pixel_centre.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err

    def _run(fx, fy, lit):
        p0 = 100.0 - (fx - 1023.5) * 0.2
        p1 = -100.0 - (fy - 1023.5) * 0.2
        out = subprocess.run([exe, repr(p0), repr(p1), str(lit[0]), str(lit[1])],
                             capture_output=True, text=True, check=True, cwd=d).stdout
        c, f, w, X, Y = (int(v) for v in out.split()[1:])
        return c, f, w, (X, Y)
    return _run


@pytest.mark.parametrize("fx,fy", [(1023.7, 523.7), (1023.3, 523.3), (1023.5 + 0.02, 523.49),
                                   (0.3, 1500.8), (2047.2, 10.6)])
def test_the_pixel_read_is_the_nearest_centre(probe, fx, fy):
    want = (int(np.floor(fx + 0.5)), int(np.floor(fy + 0.5)))
    c, f, w, xy = probe(fx, fy, want)
    assert (c, f, w) == (1, 1, 1), f"spot at ({fx}, {fy}) not read from pixel {want}"
    assert xy == want, f"spots.txt reports {xy} for a spot in pixel {want}"


@pytest.mark.parametrize("fx,fy", [(1023.7, 523.7), (2047.2, 10.6)])
def test_the_truncated_pixel_is_no_longer_read(probe, fx, fy):
    trunc = (int(fx), int(fy))
    assert probe(fx, fy, trunc)[:3] == (0, 0, 0)


@pytest.mark.parametrize("fx,fy", [(-0.7, 523.2), (1023.2, -0.6), (2047.6, 523.2)])
def test_spots_off_the_panel_are_not_read(probe, fx, fy):
    # -0.7 truncates to 0 and used to be scored on the edge column.
    edge = (min(max(int(round(fx)), 0), 2047), min(max(int(round(fy)), 0), 2047))
    assert probe(fx, fy, edge)[:3] == (0, 0, 0)


def test_every_binary_uses_the_header_rounding():
    """The three mains each carry a copy of the coarse projection loop; they
    must all call pixelIndex() and none may truncate by hand."""
    if not os.path.isdir(C_DIR):
        pytest.skip("c_src not present")
    for name in ("LaueMatchingCPU.c", "LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu"):
        src = open(os.path.join(C_DIR, name)).read()
        assert "pixelIndex(" in src, f"{name} does not use pixelIndex()"
        assert not re.search(r"\(int\)\(\(\s*[xy]p\s*/\s*px[XY]\s*\)", src), (
            f"{name} still truncates a detector coordinate by hand")
    hdr = open(HEADERS).read()
    assert hdr.count("px = pixelIndex(") >= 4 and hdr.count("py = pixelIndex(") >= 4, (
        "a LaueMatchingHeaders.h lookup does not go through pixelIndex()")
    assert not re.search(r"if \(p[xy] < 0 \|\| p[xy] > \(nrPx[XY] - 1\)\)", hdr), (
        "LaueMatchingHeaders.h still range-checks a coordinate that is then truncated")
