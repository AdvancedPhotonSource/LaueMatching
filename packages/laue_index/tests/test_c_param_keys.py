"""Parameter keys match on the whole token; the coarse-score column is named
for what it holds.

Before 0.8.0 every binary matched keys with strncmp(line, key, strlen(key)),
a PREFIX test: harmless for today's keys (none is a prefix of another) but a
future key such as "PxXY" or "MinIntensityFrac" would silently have been read
as "PxX" / "MinIntensity". And the solutions header called the coarse score
CoarseNMatches*sqrt(Intensity) while the value is Intensity*sqrt(NMatches)
(matchedArr = totInt * sqrt(nSpots)).
"""
import os
import re
import subprocess

import pytest

from _cbuild import C_DIR, FIXTURES, HEADERS, build

MAINS = ("LaueMatchingCPU.c", "LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu")


def test_whole_token_key_matching(tmp_path):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present")
    exe = str(tmp_path / "param_key")
    err = build(exe, [os.path.join(FIXTURES, "param_key.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    got = [int(l.split()[2]) for l in out.splitlines()]
    assert got == [1, 1, 1, 0, 0, 1, 0]


@pytest.mark.parametrize("name", MAINS)
def test_binaries_do_not_prefix_match_keys(name):
    if not os.path.isdir(C_DIR):
        pytest.skip("c_src not present")
    src = open(os.path.join(C_DIR, name)).read()
    assert "strncmp(aline, str, strlen(str))" not in src
    assert "paramKeyCmp(aline, str)" in src


@pytest.mark.parametrize("name", MAINS)
def test_coarse_score_header_names_its_value(name):
    if not os.path.isdir(C_DIR):
        pytest.skip("c_src not present")
    src = open(os.path.join(C_DIR, name)).read()
    assert "CoarseNMatches*sqrt(Intensity)" not in src
    assert "CoarseIntensity*sqrt(NMatches)" in src
