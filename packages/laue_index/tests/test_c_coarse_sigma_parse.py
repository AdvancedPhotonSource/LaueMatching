"""All three binaries read OrientationSpacing and CoarseFitSigma.

The coarse-fit blur width is set from the orientation-database grid spacing
(autoCoarseSigma) unless CoarseFitSigma overrides it. Before 0.8.0 only the
CPU binary parsed either key; the GPU and streaming binaries passed 0.0 and
the header fell back to a hard-coded 0.4 deg grid, so a params file with a
different spacing or an explicit CoarseFitSigma indexed differently depending
on which binary ran.
"""
import os
import re

import pytest

from _cbuild import C_DIR

MAINS = ("LaueMatchingCPU.c", "LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu")


@pytest.mark.parametrize("name", MAINS)
def test_binary_parses_both_keys(name):
    if not os.path.isdir(C_DIR):
        pytest.skip("c_src not present")
    src = open(os.path.join(C_DIR, name)).read()
    for key in ("OrientationSpacing", "CoarseFitSigma"):
        assert f'str = "{key}";' in src, f"{name} does not parse {key}"
    assert "autoCoarseSigma(pArr[2], pxX, orientSpacing)" in src, name


@pytest.mark.parametrize("name", MAINS)
def test_no_binary_passes_a_literal_sigma(name):
    if not os.path.isdir(C_DIR):
        pytest.skip("c_src not present")
    src = open(os.path.join(C_DIR, name)).read()
    for m in re.finditer(r"fitAndWriteOrientations\(", src):
        call = src[m.start():src.index(";", m.start())]
        assert not re.search(r"\b0\.0\s*/\*\s*auto", call), f"{name}: literal 0.0 sigma"
