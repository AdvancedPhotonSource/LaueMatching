"""`MinSpotIntensity`: the floor a pixel must exceed to count as a matched spot.

The match test used to be a bare ``image[px] > 0``, so a pixel carrying 4e-06
counted as evidence. On real data 35% (Zn) and 29% (Si) of matched spots sat
below intensity 1.0, and the objective is ``nrPos * sqrt(sum)`` -- it multiplies
by the COUNT -- so those pixels inflated the score without contributing signal.

These are SOURCE-LEVEL invariants, not behaviour tests, and that is deliberate.
The failure this guards against is the three binaries drifting apart, which has
already happened once on this exact test: a comment in LaueMatchingCPU.c records
that the quantized ``image_u8`` clamped faint pixels up to 1, inflating totInt,
flipping the minIntensity test, and making the CACHED path report more solutions
than the FRESH one. A behaviour test on one backend cannot see that; a test that
reads all four files can.

The default is 0.0 everywhere, which reduces to the historical ``> 0`` exactly,
so an existing parameter file produces byte-identical output.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_C_SRC = Path(__file__).resolve().parent.parent / "c_src"

CPU = "LaueMatchingCPU.c"
GPU = "LaueMatchingGPU.cu"
STREAM = "LaueMatchingGPUStream.cu"
HEADERS = "LaueMatchingHeaders.h"
MAINS = [CPU, GPU, STREAM]


def _read(name: str) -> str:
    p = _C_SRC / name
    if not p.is_file():
        pytest.skip(f"{name} not present (installed wheel, not a checkout)")
    return p.read_text()


def test_no_bare_pixel_match_test_survives_anywhere():
    """The regression itself: a bare `> 0` on the pixel under a predicted spot.

    Deliberately excludes the `image[pxNr] > 0` diagnostic that counts non-zero
    pixels for the "Pixels with intensity" printout -- that is not a match test.
    """
    bare = re.compile(r"\b(raw|thisInt|thisIntC)\s*>\s*0\s*\)")
    offenders = []
    for name in MAINS + [HEADERS]:
        for i, line in enumerate(_read(name).splitlines(), 1):
            if bare.search(line):
                offenders.append(f"{name}:{i}: {line.strip()}")
    assert not offenders, "bare pixel match test still present:\n  " + "\n  ".join(offenders)


def test_every_main_parses_the_parameter_and_defaults_to_zero():
    """A main that does not parse it silently ignores the setting; a main that
    defaults to anything but 0.0 silently changes existing results."""
    for name in MAINS:
        src = _read(name)
        assert '"MinSpotIntensity"' in src, f"{name} does not parse MinSpotIntensity"
        assert re.search(r"&minSpotIntensity\s*\)", src), f"{name} never stores it"
        assert re.search(r"double\s+minSpotIntensity\s*=\s*0\.0\s*;", src), (
            f"{name} does not default MinSpotIntensity to 0.0 -- the default MUST "
            f"reproduce the historical `> 0` exactly"
        )


def test_the_parse_is_not_shadowed_by_minintensity():
    """Both parsers use prefix matching (`strncmp`), and `MinIntensity` is a
    prefix-collision hazard. `MinSpotIntensity` must be tested FIRST, or a
    config line would fall into the wrong branch."""
    for name in MAINS:
        src = _read(name)
        spot = src.index('"MinSpotIntensity"')
        plain = src.index('"MinIntensity"')
        assert spot < plain, (
            f"{name}: MinIntensity is matched before MinSpotIntensity; with "
            f"strncmp prefix matching the more specific key must come first"
        )


def test_all_three_stage2_sites_use_the_floor():
    """calcOverlap and calcOverlapFiltered are the REFINEMENT objective and
    writeCalcOverlap is the reported NMatches. The objective multiplies by the
    match count, so if these disagree the optimised count and the reported count
    describe different things."""
    src = _read(HEADERS)
    n = len(re.findall(r"image\[\(size_t\)\(\(size_t\)py \* nrPxX \+ \(size_t\)px\)\]\s*>\s*\n?\s*minSpotIntensity",
                       src))
    assert n == 3, f"expected 3 stage-2 match sites using the floor, found {n}"


def test_cpu_fresh_and_cached_paths_agree():
    """The two CPU paths must apply the same floor. They diverged once already
    (see the image_u8 comment in LaueMatchingCPU.c) and the symptom was the
    cached path reporting MORE solutions than the fresh one."""
    src = _read(CPU)
    assert "thisInt > minSpotIntensity" in src, "CPU fresh path does not use the floor"
    assert "thisIntC > minSpotIntensity" in src, "CPU cached path does not use the floor"


def test_device_kernels_receive_the_floor_as_an_argument():
    """A __global__ kernel cannot see a host variable. If the floor is not a
    kernel parameter it silently stays whatever the device default is."""
    for name in (GPU, STREAM):
        src = _read(name)
        assert re.search(r"__global__ void compare\([^)]*float minSpotInt", src, re.S), (
            f"{name}: compare() does not take the floor as a kernel argument")
        assert "raw > minSpotInt" in src, f"{name}: kernel does not apply the floor"
        assert re.search(r"\(float\)minSpotIntensity", src), (
            f"{name}: the launch does not pass the host value to the kernel")
