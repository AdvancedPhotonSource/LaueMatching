"""The coarse-fit blur: exact, and parallel.

`gaussianBlurImage` in LaueMatchingHeaders.h was the entire streaming
pipeline's bottleneck. Measured 2026-09-09 on sentosa (H200, 64 cores, real
sampleH frames, 100M-orientation cache): the daemon spent a median 464 ms/image in
`fitAndWriteOrientations`, of which 449 ms was this one single-threaded blur
(96.8%), against 41 ms on the GPU -- all n=150 from one run. The cost was FLAT in
the number of orientations to fit (459 ms for 1, 509 ms for 93), which is what
identified it: the only work upstream of the parallel loop. Parallelising it took
the completion-bounded 150-frame time from ~72.8 s to ~10.1 s, about 7x.

An earlier write-up of this said "468 ms" and "6.4x". Both were wrong and a
`/verify` reproduction lens killed the claim on them: 468 ms is not the median of
any logged series (it is 4 of 150 individual values, and was spliced in from a
different run), and 6.4x came from a harness timer that stopped when the frame
sender exited, leaving 19.8 s of daemon backlog uncounted in the slow arm. The
direction of that error favoured the claim, which is exactly why it had to go.

NOTE ON THE EXACT-EQUALITY ASSERTION. The fixture compares with ``memcmp`` and
demands bit-identity. That is NOT the cross-backend platform assertion this
project has been bitten by before: both implementations live in ONE binary, are
built by one compiler, and call the same libm ``exp`` to build the same kernel,
so there is no glibc-vs-Apple-libm gap to absorb. The exactness is a property of
the transformation (same taps, same order), not a tolerance chosen to make a test
pass -- which is why it is asserted exactly rather than bounded. Verified to hold
under ``-O3``, ``-march=native`` (FMA available) and ``-Ofast -ffast-math``.

A speedup in the matcher's capture-radius blur is only acceptable if it changes
nothing, and the end-to-end solutions output CANNOT establish that: the
streaming daemon assigns GrainNr nondeterministically, so the same binary on the
same input disagrees with itself on 54 of 1661 solution lines (verified by
running it twice). Strip the GrainNr column and all three comparisons -- run A
vs run B, and serial vs parallel -- are byte-identical. So exactness is pinned
here, at the function, where nondeterminism cannot get in.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
C_SRC = os.path.join(HERE, "fixtures", "coarse_blur_exactness.c")
HEADERS = os.path.abspath(os.path.join(
    HERE, "..", "c_src", "LaueMatchingHeaders.h"))


def _read(path):
    with open(path) as f:
        return f.read()


def _cc():
    for cand in ("cc", "gcc", "clang"):
        p = shutil.which(cand)
        if p:
            return p
    return None


@pytest.mark.skipif(_cc() is None, reason="no C compiler available")
def test_parallel_blur_is_bit_identical_to_the_serial_one(tmp_path):
    """Compile both implementations and compare every output pixel.

    Exact because each output pixel still accumulates the same klen products in
    the same k order: parallelising over y splits disjoint output rows, and the
    clamp hoist only skips branches that could not have fired in the interior.
    Checked at autoCoarseSigma's clamp floor (4.0), its ceiling (12.0), and the
    7.77 px the 0.4-degree 100M database actually produces.
    """
    exe = str(tmp_path / "blurx")
    cc = _cc()
    # -fopenmp is what makes this a real test rather than two serial runs; if
    # the toolchain lacks it, fall back but say so, because a single-threaded
    # "parallel" path would pass trivially.
    cmd = [cc, "-O3", "-fopenmp", "-o", exe, C_SRC, "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    openmp = proc.returncode == 0
    if not openmp:
        proc = subprocess.run([cc, "-O3", "-o", exe, C_SRC, "-lm"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"cannot build the fixture: {proc.stderr[-400:]}")

    run = subprocess.run([exe, "512", "4"], capture_output=True, text=True,
                         timeout=600)
    sys.stdout.write(run.stdout)
    assert run.returncode == 0, (
        "the parallel coarse-fit blur is NOT bit-identical to the serial one:\n"
        + run.stdout + run.stderr)
    assert "PASS" in run.stdout
    assert run.stdout.count("BIT-IDENTICAL") == 3, (
        "expected all three sigmas to be checked")
    if not openmp:
        pytest.skip("compiler has no OpenMP: exactness checked, "
                    "but the parallel path ran single-threaded")


def test_blur_is_still_parallel_and_the_clamp_is_still_hoisted():
    """Guard the speedup itself.

    A revert to a serial blur is not a test failure anywhere else -- every
    result stays identical, the pipeline just gets ~7x slower again. The 449 ms
    was invisible for as long as nothing looked for it, so pin the two
    structural properties that removed it.
    """
    src = _read(HEADERS)
    start = src.index("static inline void gaussianBlurImage(")
    end = src.index("static inline void fitAndWriteOrientations(", start)
    body = src[start:end]

    assert body.count("#pragma omp parallel for") == 2, (
        "both separable passes of the coarse-fit blur must be parallel; "
        "a serial pass costs ~449 ms/image at sigma 7.77 on a 2048^2 frame")
    assert "acc += kern[k] * w[k]" in body, (
        "the horizontal interior loop has lost its hoisted form -- the edge "
        "clamp is back in the innermost loop, 2 branches per tap")
    assert "col[(size_t)k * nx]" in body, (
        "the vertical interior loop has lost its hoisted form")
    assert "nThreads" in body, (
        "gaussianBlurImage must take a thread count; it is called once per "
        "image from fitAndWriteOrientations, which has numProcs")


def test_the_blur_is_timed_so_a_regression_is_visible():
    """It took an end-to-end streaming run to find this cost because nothing
    reported it. The daemon already prints merge/setup/flush/fitting per image;
    the blur now prints too, so the next regression shows up in the log rather
    than needing a campaign."""
    src = _read(HEADERS)
    assert "coarse blur:" in src, (
        "the per-image blur timing print has been removed")
