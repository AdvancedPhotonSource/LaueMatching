"""The crystal-fit tolerance guard: FRACTIONS, not percents.

`tol_LatC` and `tol_c_over_a` bound the lattice fit as ``value * (1 -/+ tol)``.
The usage text said "in %" until 2026-09-20, so ``1.0`` meaning "1%" was the
natural mistake -- it puts the lower bound at zero and the run still looks
plausible. `validateCrystalFitTolerances()` (LaueMatchingHeaders.h), called by
all three main()s, therefore:

* rejects anything outside [0, 1) (and NaN) with a FATAL naming FRACTION;
* WARNS above 0.1, where a percent-minded 0.5 ("0.5%") would silently be +-50%;
* validates the EFFECTIVE values: when tol_c_over_a is set it overrides
  tol_LatC, so tol_LatC is reported as ignored instead of aborting the run.

The behaviour tests run the REAL CPU main(), compiled from c_src into a temp
dir (a pre-built binary would test whatever source it was built from). The
parameter file holds only the tolerance lines, so a run that passes the guard
then stops at the NrPxX check; only the guard's own messages are asserted on.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent / "c_src"
CPU, GPU, STREAM, HEADERS = ("LaueMatchingCPU.c", "LaueMatchingGPU.cu",
                             "LaueMatchingGPUStream.cu", "LaueMatchingHeaders.h")
MAINS = [CPU, GPU, STREAM]

# LaueMatchingHeaders.h includes <omp.h> unconditionally. Apple clang has no
# -fopenmp; it needs -Xpreprocessor plus Homebrew's libomp.
_LIBOMP_PREFIXES = ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")
# Strings only a binary built from the current guard contains.
_GUARD_MARKERS = (b"is not a valid FRACTION", b"c/a may move by")


def _read(name):
    p = C_DIR / name
    if not p.is_file():
        pytest.skip(f"{name} not present (installed wheel, not a checkout)")
    return p.read_text()


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


def _build_cpu(exe):
    """Build LaueMatchingCPU the way CMakeLists.txt does (gnu99, -O3, OpenMP,
    libm, CPU.c + nelder_mead.c). Returns None on success, else the error."""
    err = "no C compiler found"
    for cc in _compilers():
        for cflags, ldflags in _omp_flag_sets():
            cmd = ([cc, "-std=gnu99", "-O3"] + cflags + [f"-I{C_DIR}", "-o", exe,
                   str(C_DIR / CPU), str(C_DIR / "nelder_mead.c")]
                   + ldflags + ["-lm"])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return None
            err = proc.stderr[-600:]
    return err


def _prebuilt_with_current_guard():
    try:
        from laue_index.indexer import binary_path
        p = binary_path("CPU")
    except Exception:
        return None
    if not (p.is_file() and os.access(p, os.X_OK)):
        return None
    blob = p.read_bytes()
    return str(p) if all(m in blob for m in _GUARD_MARKERS) else None


@pytest.fixture(scope="module")
def cpu_binary(tmp_path_factory):
    if not (C_DIR / CPU).is_file():
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    exe = str(tmp_path_factory.mktemp("guard") / "LaueMatchingCPU")
    err = _build_cpu(exe)
    if err is None:
        return exe
    pre = _prebuilt_with_current_guard()
    if pre:
        return pre
    pytest.skip(f"cannot build LaueMatchingCPU and no pre-built binary carries "
                f"the current guard: {err}")


def _run(exe, tmp_path, lines):
    params = tmp_path / "params.txt"
    params.write_text("".join(l + "\n" for l in lines))
    missing = str(tmp_path / "does_not_exist")
    return subprocess.run([exe, str(params), missing, missing, missing, "1"],
                          capture_output=True, text=True, timeout=60,
                          cwd=str(tmp_path))


# ── behaviour: the real CPU main() ───────────────────────────────────────

def test_tol_c_over_a_of_one_is_fatal_and_says_fraction(cpu_binary, tmp_path):
    r = _run(cpu_binary, tmp_path, ["tol_c_over_a 1.0"])
    assert r.returncode != 0
    assert "FRACTION" in r.stderr
    assert "FATAL: tol_c_over_a" in r.stderr
    # It stopped AT the guard, not at a later check.
    assert "NrPxX" not in r.stderr


def test_tol_c_over_a_of_one_percent_passes_the_guard(cpu_binary, tmp_path):
    r = _run(cpu_binary, tmp_path, ["tol_c_over_a 0.01"])
    assert "FATAL: tol_c_over_a" not in r.stderr
    assert "WARNING: tol_c_over_a" not in r.stderr
    # Past the guard: the next thing to fail is the (absent) detector size.
    assert "NrPxX" in r.stderr


def test_percent_minded_half_warns_but_runs_on(cpu_binary, tmp_path):
    r = _run(cpu_binary, tmp_path, ["tol_c_over_a 0.5"])
    assert "WARNING: tol_c_over_a = 0.5" in r.stderr
    assert "+-50%" in r.stderr, "the warning must state the percent it means"
    assert "FATAL: tol_c_over_a" not in r.stderr
    assert "NrPxX" in r.stderr


def test_tol_latc_is_validated_and_warned_per_entry(cpu_binary, tmp_path):
    r = _run(cpu_binary, tmp_path, ["tol_LatC 1.0 0 0 0 0 0"])
    assert r.returncode != 0
    assert "FATAL: tol_LatC[0]" in r.stderr and "FRACTION" in r.stderr
    r = _run(cpu_binary, tmp_path, ["tol_LatC 0.01 0.01 0.5 0 0 0"])
    assert "WARNING: tol_LatC[2] = 0.5" in r.stderr
    assert "tol_LatC[0]" not in r.stderr and "FATAL: tol_LatC" not in r.stderr


def test_overridden_tol_latc_is_ignored_not_fatal(cpu_binary, tmp_path):
    """EFFECTIVE values: tol_c_over_a zeroes tol_LatC, so a bad tol_LatC never
    forms a bound and must not abort the run -- but it is reported."""
    r = _run(cpu_binary, tmp_path,
             ["tol_LatC 1.0 1.0 1.0 0 0 0", "tol_c_over_a 0.01"])
    assert "FATAL: tol_LatC" not in r.stderr
    assert "overrides tol_LatC" in r.stderr and "ignored" in r.stderr
    assert "NrPxX" in r.stderr


def test_nan_tolerance_is_rejected(cpu_binary, tmp_path):
    r = _run(cpu_binary, tmp_path, ["tol_c_over_a nan"])
    assert r.returncode != 0
    assert "FATAL: tol_c_over_a" in r.stderr


# ── source: all three mains, and the usage text ──────────────────────────

def _main_body(src):
    return src[src.index("int main(int argc, char *argv[])"):]


def test_every_main_calls_the_guard_once_before_the_override():
    for name in MAINS:
        body = _main_body(_read(name))
        calls = re.findall(r"if \(validateCrystalFitTolerances\(\)\)\s*\n\s*return 1;",
                           body)
        assert len(calls) == 1, f"{name}: main() must call the guard exactly once"
        assert body.index("validateCrystalFitTolerances()") < body.index(
            "if (tol_c_over_a != 0) {"), (
            f"{name}: the guard must run before tol_LatC is zeroed, or its "
            f"'tol_LatC is ignored' NOTE can never fire")


def _usage_body(src, fn):
    start = src.index(f"static void {fn}(")
    return src[start:src.index("\n}\n", start)]


@pytest.mark.parametrize("name,fn", [(CPU, "usageCPU"), (GPU, "usageGPU")])
def test_puts_usage_has_no_doubled_percent(name, fn):
    """puts() is not a format function: `%%` prints as two percent signs. The
    first version of the FRACTION text made exactly this mistake."""
    body = _usage_body(_read(name), fn)
    assert "puts(" in body and "printf(" not in body
    assert "%%" not in body, f"{name}: {fn} prints with puts(); write % not %%"
    assert body.count("FRACTION") == 2


def test_stream_printf_usage_escapes_percent_and_lists_tolerances():
    """The streaming daemon's usage is printf(), the opposite rule: every
    literal percent must be `%%`, and the only conversions are %s and %d."""
    body = _usage_body(_read(STREAM), "usage")
    assert "printf(" in body
    assert "tol_latC" in body and "tol_c_over_a" in body
    assert body.count("FRACTION") == 2
    convs = re.findall(r"%[^%]", body.replace("%%", ""))
    assert convs == ["%s", "%d"], (
        f"stray printf conversion(s) in the stream usage text: {convs}")


def test_no_vestigial_optimizer_switch():
    """`useBobyqa` was written, never read, and initialised to 1 'default:
    BOBYQA' -- an algorithm that no longer exists. It must not come back."""
    for name in MAINS + [HEADERS]:
        src = _read(name)
        assert "useBobyqa" not in src.replace("former `useBobyqa`", ""), name
        assert "/*BOBYQA*/" not in src, name
