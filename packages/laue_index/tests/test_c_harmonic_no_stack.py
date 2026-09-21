"""The fit-stage objective counts a harmonic pair ONCE -- NMatches cannot stack.

(111) and (222) of one orientation share a unit q-hat, so they fall on one
detector pixel at two energies. `calcOverlap`, `calcOverlapFiltered` and
`writeCalcOverlap` in LaueMatchingHeaders.h keep only the first reflection with
a given q-hat (|d| < 1e-6 per component), so the refinement objective and the
NMatches column count that pixel once. A handbook entry once said the opposite
("anything >~ 2 is harmonic stacking"); this pins what the code does.

Real-data anchor (from the 0.7.2 audit, LAUE_CHANGES_NEEDED.md section 5): on
one sampleH shard, 0 of 25,172 solutions put two matches on one pixel.

WHAT THE FIXTURE EXERCISES: the real header functions, compiled from c_src with
the system compiler -- nothing is copied into the fixture. WHAT IT DOES NOT: the
coarse forward-cache stage (which dedups by integer pixel, a different and
deliberate rule; see `pixelClaimed`), the CUDA kernels, the optimiser, or any
main(). See the fixture's header comment for the geometry and the controls.

Verified to be able to fail: with the q-hat test disabled in a scratch copy of
the header, the fixture reports 2 for both harmonic orderings and exits 2.

The integer counts are compared exactly: they are counts, not floating-point
results, and the fixture recovers them from a constant-1.0 image where the
score is N*sqrt(N).
"""
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.abspath(os.path.join(HERE, "..", "c_src"))
FIXTURE = os.path.join(HERE, "fixtures", "harmonic_no_stack.c")
HEADERS = os.path.join(C_DIR, "LaueMatchingHeaders.h")

# LaueMatchingHeaders.h includes <omp.h> unconditionally, so an OpenMP-capable
# toolchain is required even though this fixture runs single-threaded. Apple
# clang has no -fopenmp; it needs -Xpreprocessor plus Homebrew's libomp.
_LIBOMP_PREFIXES = ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")


def _compilers():
    seen = []
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        p = shutil.which(cand) if cand else None
        if p and p not in seen:
            seen.append(p)
    return seen


def _omp_flag_sets():
    yield ["-fopenmp"], []
    for pre in _LIBOMP_PREFIXES:
        if os.path.isdir(pre):
            yield (["-Xpreprocessor", "-fopenmp", f"-I{pre}/include"],
                   [f"-L{pre}/lib", "-lomp"])


def _build(exe, sources):
    """Compile ``sources`` against c_src with the first toolchain that works.
    Returns None on success, or the last compiler error."""
    err = "no C compiler found"
    for cc in _compilers():
        for cflags, ldflags in _omp_flag_sets():
            cmd = ([cc, "-O2", "-std=gnu99"] + cflags + [f"-I{C_DIR}", "-o", exe]
                   + sources + [os.path.join(C_DIR, "nelder_mead.c")]
                   + ldflags + ["-lm"])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return None
            err = proc.stderr[-600:]
    return err


def test_harmonic_pair_counts_once_in_every_fit_stage_function(tmp_path):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    if not _compilers():
        pytest.skip("no C compiler available")
    exe = str(tmp_path / "harmonic_no_stack")
    err = _build(exe, [FIXTURE])
    if err is not None:
        pytest.skip(f"cannot build the fixture (needs OpenMP headers): {err}")
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    sys.stdout.write(run.stdout)
    assert run.returncode == 0, (
        "a harmonic pair is counted more than once, or a control failed:\n"
        + run.stdout + run.stderr)
    assert "PASS" in run.stdout
    # All five cases ran: two singles, both orderings of the pair, one control.
    assert run.stdout.count(" OK") == 5, run.stdout


def test_every_fit_stage_function_keeps_its_qhat_dedup():
    """Source guard, so the dedup cannot be dropped from ONE of the three
    copies while the fixture's other two still pass for the wrong reason."""
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    with open(HEADERS) as f:
        src = f.read()
    for fn in ("calcOverlap", "calcOverlapFiltered", "writeCalcOverlap"):
        m = re.search(r"static inline (?:double|int)\s+" + fn + r"\(", src)
        assert m, f"{fn} definition not found in LaueMatchingHeaders.h"
        start = m.start()
        end = src.index("\n}\n", start)
        body = src[start:end]
        assert body.count("* 100000 < 0.1") == 3, (
            f"{fn} has lost its per-component q-hat dedup (|d| < 1e-6)")
        assert "badSpot = 1;" in body, f"{fn} no longer rejects a repeat q-hat"
