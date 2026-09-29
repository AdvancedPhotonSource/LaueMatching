"""spots.txt holds spots only for grains that solutions.txt holds, and the
streaming column is chosen by a sentinel, not by imageNr > 0.

Before 0.8.0 writeCalcOverlap wrote a grain's spot lines unconditionally and
the MinNrSpots gate ran afterwards, so spots.txt carried GrainNrs that never
appear in solutions.txt. And the leading ImageNr column was written only when
imageNr > 0, so a streaming frame numbered 0 shifted every column. The
streaming daemon also gated fits on MinGoodSpots where the CPU and GPU
binaries used MinNrSpots (user decision 2026-09-28: MinNrSpots everywhere).
"""
import os
import re
import subprocess

import pytest

from _cbuild import C_DIR, FIXTURES, HEADERS, build


@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    d = tmp_path_factory.mktemp("sg")
    exe = str(d / "spots_gate")
    err = build(exe, [os.path.join(FIXTURES, "spots_gate.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err

    def _run(min_write, image_nr):
        out = subprocess.run([exe, str(min_write), str(image_nr)], capture_output=True,
                             text=True, check=True).stdout.split()
        return int(out[1]), int(out[3]), int(out[5])
    return _run


def test_a_grain_below_the_gate_writes_no_spot_lines(gate):
    n, lines, _ = gate(3, -1)
    assert n == 2 and lines == 0


def test_a_grain_at_the_gate_writes_its_spot_lines(gate):
    n, lines, cols = gate(2, -1)
    assert (n, lines, cols) == (2, 2, 11)


@pytest.mark.parametrize("image_nr", [0, 1, 65535])
def test_streaming_frames_always_carry_the_image_column(gate, image_nr):
    assert gate(2, image_nr)[2] == 12


def test_binaries_use_the_sentinel_and_one_gate():
    if not os.path.isdir(C_DIR):
        pytest.skip("c_src not present")
    hdr = open(HEADERS).read()
    assert not re.search(r"image(Nr|Num) > 0", hdr), "layout still keyed on imageNr > 0"
    for name in ("LaueMatchingCPU.c", "LaueMatchingGPU.cu"):
        src = open(os.path.join(C_DIR, name)).read()
        call = src[src.index("fitAndWriteOrientations("):]
        call = call[:call.index(";")]
        assert "NOT_STREAMING" in call, name
        assert "minNrSpots" in call, name
    src = open(os.path.join(C_DIR, "LaueMatchingGPUStream.cu")).read()
    call = src[src.index("fitAndWriteOrientations("):]
    call = call[:call.index(";")]
    assert "minNrSpots" in call and "minGoodSpots" not in call, (
        "the streaming daemon gates fits on a different key than CPU/GPU")
