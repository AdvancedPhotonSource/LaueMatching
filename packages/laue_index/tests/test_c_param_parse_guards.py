"""Parameter parsing in the three mains: no aliased sscanf buffers, and the
stream daemon validates what it reads like the other two binaries.

SOURCE-LEVEL checks, and only the ones a grep can make honestly: there is no
CUDA here to run the .cu mains, and the defects are textual (one buffer passed
twice to sscanf; locals read before they are written; a missing check).

* `sscanf(aline, "%s %s", dummy, dummy)` writes two conversions into one
  buffer. CPU.c always used `dummy2`; GPU.cu and GPUStream.cu now do too.
* LaueMatchingGPUStream's main() declared nrPxX, nrPxY, pArr, rArr, pxX, pxY and
  LatticeParameter without initialisers and never checked them, so a missing
  key put stack garbage into allocations and divisions. It now initialises them
  and carries LaueMatchingGPU.cu's NrPxX, LatticeParameter and numProcs checks
  with the same messages.
"""
import re
from pathlib import Path

import pytest

C_DIR = Path(__file__).resolve().parent.parent / "c_src"
CPU, GPU, STREAM = ("LaueMatchingCPU.c", "LaueMatchingGPU.cu",
                    "LaueMatchingGPUStream.cu")

_CHECKS = (
    '"FATAL: Invalid detector dimensions (NrPxX=%d, NrPxY=%d). Check "',
    '"FATAL: LatticeParameter not set in parameter file.\\n"',
    '"FATAL: Invalid number of CPU cores (%d). Must be >= 1.\\n"',
)


def _main(name):
    p = C_DIR / name
    if not p.is_file():
        pytest.skip(f"{name} not present (installed wheel, not a checkout)")
    src = p.read_text()
    return src[src.index("int main(int argc, char *argv[])"):]


@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_no_sscanf_writes_two_fields_into_one_buffer(name):
    body = _main(name)
    for m in re.finditer(r"sscanf\(aline,\s*\"[^\"]*\",([^;]*)\);", body):
        args = [a.strip() for a in m.group(1).split(",")]
        assert len(args) == len(set(args)), (
            f"{name}: sscanf passes one buffer twice: {m.group(0)}")
    assert 'strncmp(dummy2, "BOBYQA", 6)' in body, name


def test_stream_main_initialises_what_it_later_checks():
    body = _main(STREAM)
    for pat in (r"\bnrPxX = 0, nrPxY = 0\b",
                r"double pArr\[3\] = \{0, 0, 0\}, rArr\[3\] = \{0, 0, 0\}, "
                r"pxX = 0, pxY = 0,",
                r"double LatticeParameter\[6\] = \{0, 0, 0, 0, 0, 0\};"):
        assert re.search(pat, body), f"stream main() no longer initialises: {pat}"


@pytest.mark.parametrize("name", [GPU, STREAM])
def test_both_cuda_mains_carry_the_same_parameter_checks(name):
    body = _main(name)
    for msg in _CHECKS:
        assert body.count(msg) == 1, f"{name}: missing or duplicated check {msg}"
    assert re.search(r"if \(nrPxX <= 0 \|\| nrPxY <= 0\)", body)
    assert re.search(r"if \(LatticeParameter\[0\] == 0\)", body)
    assert re.search(r"if \(numProcs < 1\)", body)
    # The detector check must precede the first use of the sizes.
    assert body.index("if (nrPxX <= 0 || nrPxY <= 0)") < body.index(
        "(size_t)nrPxX * nrPxY")


def test_cpu_main_keeps_its_checks():
    body = _main(CPU)
    assert re.search(r"if \(nrPxX <= 0 \|\| nrPxY <= 0\)", body)
    assert re.search(r"if \(LatticeParameter\[0\] == 0\.0\)", body)
    assert re.search(r"if \(numProcs < 1\)", body)


# ── outfn and size arithmetic ─────────────────────────────────────────────

@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_outfn_starts_empty(name):
    """A missing ForwardFile must reach forwardCacheUsable()'s blank-path check
    deterministically, not read stack garbage."""
    assert re.search(r'\boutfn\[1000\] = "";', _main(name)), name


def test_gpu_forward_slab_arithmetic_is_size_t():
    """GPU.cu computed the slab as int * int (overflow at ~35M orientations per
    thread) and indexed orients with an int orientNr (overflow at ~238M)."""
    body = _main(GPU)
    fwd = body[body.index("#pragma omp parallel num_threads(numProcs)"):]
    fwd = fwd[:fwd.index("finishForwardCacheOrDie(")]
    assert "size_t orientNr;" in fwd
    assert "size_t szArr = nrOrientsThread * slabStride;" in fwd
    assert "size_t slabStride = (size_t)(1 + 2 * maxNrSpots);" in fwd
    assert not re.search(r"\bint (nrOrientsThread|startOrientNr|endOrientNr|"
                         r"orientNr)\b", fwd), "an int slab counter is back"
    assert "szArr ? szArr : 1" in fwd, "calloc(0) may return NULL"


def test_stream_forward_offset_is_not_cast_through_int():
    body = _main(STREAM)
    assert "(int)ceil(" not in body


@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_no_int_by_int_image_or_orientation_allocation(name):
    body = _main(name)
    assert not re.search(r"(malloc|fread)\([^;]*[^)]\bnrPxX \* nrPxY", body), (
        f"{name}: image size computed as int * int; cast to size_t first")
    for alloc in ("orients = (double *)malloc(szFile);",):
        i = body.index(alloc)
        assert "if (orients == NULL)" in body[i:i + 200], (
            f"{name}: orientation buffer allocation unchecked")
    i = body.index("int *hkls = ")
    assert "if (hkls == NULL)" in body[i:i + 200], f"{name}: hkls unchecked"


def _build_cpu(exe):
    """LaueMatchingCPU as CMakeLists.txt builds it (gnu99, OpenMP, libm)."""
    import os
    import shutil
    import subprocess
    ccs = [shutil.which(c) for c in (os.environ.get("CC"), "cc", "gcc", "clang")
           if c and shutil.which(c)]
    flagsets = [(["-fopenmp"], [])] + [
        (["-Xpreprocessor", "-fopenmp", f"-I{p}/include"], [f"-L{p}/lib", "-lomp"])
        for p in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")
        if os.path.isdir(p)]
    for cc in dict.fromkeys(ccs):
        for cflags, ldflags in flagsets:
            cmd = ([cc, "-std=gnu99", "-O1"] + cflags + [f"-I{C_DIR}", "-o", exe,
                   str(C_DIR / CPU), str(C_DIR / "nelder_mead.c")]
                   + ldflags + ["-lm"])
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                return True
    return False


@pytest.fixture(scope="module")
def cpu_exe(tmp_path_factory):
    if not (C_DIR / CPU).is_file():
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    exe = str(tmp_path_factory.mktemp("cpu") / "LaueMatchingCPU")
    if not _build_cpu(exe):
        pytest.skip("cannot build LaueMatchingCPU (needs OpenMP)")
    return exe


_GOOD = {
    "LatticeParameter": "0.4 0.4 0.4 90 90 90",
    "SpaceGroup": "225", "NrPxX": "8", "NrPxY": "8", "PxX": "0.2", "PxY": "0.2",
    "P_Array": "0 0 50", "R_Array": "0 0 0.0001",
}


def _run_cpu(exe, tmp_path, overrides=None, extra="",
             orient=(1, 0, 0, 0, 1, 0, 0, 0, 1), npx=8):
    import struct
    import subprocess
    kv = dict(_GOOD)
    kv.update(overrides or {})
    (tmp_path / "params.txt").write_text(
        "".join(f"{k} {v}\n" for k, v in kv.items() if v is not None) + extra)
    (tmp_path / "orients.bin").write_bytes(struct.pack("9d", *orient))
    (tmp_path / "hkls.txt").write_text("1 1 1\n")
    (tmp_path / "image.bin").write_bytes(
        struct.pack(f"{npx * npx}d", *([0.0] * (npx * npx))))
    return subprocess.run([exe, "params.txt", "orients.bin", "hkls.txt",
                           "image.bin", "1"], capture_output=True, text=True,
                          timeout=120, cwd=str(tmp_path))


def test_missing_forwardfile_takes_the_blank_path_branch(cpu_exe, tmp_path):
    """Behaviour, on the real CPU main(): with no ForwardFile line, DoFwd 0
    prints forwardCacheUsable()'s blank-path message, and the fallback to
    simulation is REFUSED by name before any file is created -- the old code
    open()ed whatever bytes the uninitialised buffer held. The first message
    alone could also appear on the old code when the stack byte happened to be
    0; the refusal and the unchanged directory could not."""
    r = _run_cpu(cpu_exe, tmp_path, extra="DoFwd 0\n")
    listing = sorted(p.name for p in tmp_path.iterdir())
    assert "No ForwardFile specified" in r.stdout, r.stdout + r.stderr
    assert r.returncode != 0
    assert "FATAL: no ForwardFile specified; cannot write the forward cache" in r.stderr
    assert listing == sorted(["params.txt", "orients.bin", "hkls.txt",
                              "image.bin"]), f"stray files created: {listing}"


# ── unfilled / degenerate geometry (real CPU main) ────────────────────────

@pytest.mark.parametrize("key,value,want", [
    ("P_Array", "0 0 __SET_ME__", 3),
    ("R_Array", "__SET_ME__ 0 0", 3),
    ("LatticeParameter", "0.4 0.4 __SET_ME__ 90 90 90", 6),
    ("Elo", "__SET_ME__", 1),
    ("Ehi", "__SET_ME__", 1),
])
def test_unfilled_template_value_is_fatal_naming_the_key(cpu_exe, tmp_path,
                                                         key, value, want):
    """sscanf stops at the first unconvertible token; the remaining numbers
    used to stay 0 and the run went ahead."""
    r = _run_cpu(cpu_exe, tmp_path, {key: value})
    assert r.returncode != 0
    assert f"FATAL: {key} needs {want} number(s)" in r.stderr, r.stderr
    assert "__SET_ME__" in r.stderr, "the offending line must be shown"


@pytest.mark.parametrize("elo,ehi", [("30", "5"), ("0", "30"), ("5", "5"),
                                     ("-1", "30"), ("nan", "30")])
def test_impossible_energy_band_is_fatal_naming_both(cpu_exe, tmp_path, elo, ehi):
    """Elo/Ehi default to 5/30; a band that is empty, reversed, non-positive or
    NaN is refused after the file is read, naming both values."""
    r = _run_cpu(cpu_exe, tmp_path, {"Elo": elo, "Ehi": ehi})
    assert r.returncode != 0
    assert "FATAL: energy band Elo = " in r.stderr, r.stderr
    assert "0 < Elo < Ehi" in r.stderr


def test_valid_energy_band_passes(cpu_exe, tmp_path):
    r = _run_cpu(cpu_exe, tmp_path, {"Elo": "5", "Ehi": "30"},
                 extra=f"ForwardFile {tmp_path / 'fwd.bin'}\nDoFwd 1\n")
    assert "energy band" not in r.stderr
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("value", ["0 0 0", None])
def test_zero_or_missing_detector_distance_is_fatal(cpu_exe, tmp_path, value):
    r = _run_cpu(cpu_exe, tmp_path, {"P_Array": value})
    assert r.returncode != 0
    assert "FATAL: P_Array[2] (detector distance) is 0" in r.stderr, r.stderr


def test_zero_r_array_runs_as_the_identity_rotation(cpu_exe, tmp_path):
    """A detector exactly perpendicular with no rotation is physical, so a zero
    R_Array must RUN, as the identity -- checked on the predicted spot, not the
    exit code (the old NaN rotation also exited 0; it just predicted nothing).

    Geometry: a = 0.2 nm cubic, orientation = 180 deg about x, reflection (111):
    q-hat = (1,-1,-1)/sqrt(3), E = 9.30 keV (in the default 5-30 band), and the
    diffracted beam is (2,-2,1)/3. With R_Array 0 (detector normal along the
    beam) at distance 50 and P_Array (100, -100, 50), the spot lands exactly at
    the detector centre: pixel (31, 31) of 64x64 at 0.2 units/pixel. The forward
    cache row is [n_spots, ipx, ipy, ...]; it must be [1, 31, 31]. With the old
    code the 0/0 axis made every coordinate NaN and the row was empty."""
    import struct
    fwd = tmp_path / "fwd.bin"
    r = _run_cpu(cpu_exe, tmp_path,
                 {"R_Array": "0 0 0", "LatticeParameter": "0.2 0.2 0.2 90 90 90",
                  "NrPxX": "64", "NrPxY": "64", "P_Array": "100 -100 50"},
                 extra=f"MaxNrLaueSpots 5\nForwardFile {fwd}\nDoFwd 1\n",
                 orient=(1, 0, 0, 0, -1, 0, 0, 0, -1), npx=64)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FATAL" not in r.stderr
    row = struct.unpack("11H", fwd.read_bytes())
    assert row[:3] == (1, 31, 31), f"forward-cache row {row}"


def test_detector_geometry_helpers(tmp_path):
    """The real header helpers, via fixtures/detector_geometry_guards.c:
    zero R_Array -> exact identity (it was NaN: 0/0 axis through Rodrigues);
    paramLineComplete on complete / unfilled lines; validateDetectorDistance
    on 0, NaN and a real distance."""
    import subprocess
    fixture = Path(__file__).resolve().parent / "fixtures" / "detector_geometry_guards.c"
    exe = str(tmp_path / "dgg")
    if not (C_DIR / CPU).is_file():
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    import os
    import shutil
    built = False
    ccs = [shutil.which(c) for c in (os.environ.get("CC"), "cc", "gcc", "clang")
           if c and shutil.which(c)]
    flagsets = [(["-fopenmp"], [])] + [
        (["-Xpreprocessor", "-fopenmp", f"-I{p}/include"], [f"-L{p}/lib", "-lomp"])
        for p in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")
        if os.path.isdir(p)]
    for cc in dict.fromkeys(ccs):
        for cflags, ldflags in flagsets:
            cmd = ([cc, "-std=gnu99", "-O2"] + cflags + [f"-I{C_DIR}", "-o", exe,
                   str(fixture)] + ldflags + ["-lm"])
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                built = True
                break
        if built:
            break
    if not built:
        pytest.skip("cannot build the fixture (needs OpenMP headers)")
    r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_geometry_lines_are_count_checked_and_rotation_guarded(name):
    body = _main(name)
    for key, want in (("LatticeParameter", 7), ("P_Array", 4), ("R_Array", 4),
                      ("Elo", 2), ("Ehi", 2)):
        assert re.search(rf'{want},\s*"{key}", aline\)\)\s*\n\s*return 1;', body), (
            f"{name}: the {key} sscanf count is not checked")
    assert "if (validateDetectorDistance(pArr))" in body, name
    assert "if (validateEnergyBand(Elo, Ehi))" in body, name
    assert body.index("if (validateEnergyBand(Elo, Ehi))") > body.index(
        "fclose(fileParam)"), f"{name}: band checked before the file was read"
    assert "detectorRotationTranspose(rArr, rotTranspose);" in body, name
    assert "/ rotang" not in body, f"{name}: an unguarded r/|r| is back"
