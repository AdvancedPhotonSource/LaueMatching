"""Compile a C fixture against the shipped c_src headers.

Shared by the C fixture tests added after 0.7.3; the older tests carry their own
copy of the same helper. LaueMatchingHeaders.h includes <omp.h>
unconditionally, so an OpenMP-capable toolchain is required even for
single-threaded fixtures. Apple clang has no -fopenmp; it needs
-Xpreprocessor plus Homebrew's libomp.
"""
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.abspath(os.path.join(HERE, "..", "c_src"))
HEADERS = os.path.join(C_DIR, "LaueMatchingHeaders.h")
FIXTURES = os.path.join(HERE, "fixtures")

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


def build(exe, sources):
    """Compile ``sources`` against c_src with the first toolchain that works.

    Returns None on success, or the last compiler error. A toolchain that cannot
    build a trivial OpenMP program is reported as ``"no toolchain: ..."`` so the
    caller can skip; any other error is a real compile failure and must fail.
    """
    err = "no toolchain: no C compiler found"
    for cc in _compilers():
        for cflags, ldflags in _omp_flag_sets():
            cmd = ([cc, "-O2", "-std=gnu99"] + cflags + [f"-I{C_DIR}", "-o", exe]
                   + sources + [os.path.join(C_DIR, "nelder_mead.c")]
                   + ldflags + ["-lm"])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return None
            if _toolchain_works(cc, cflags, ldflags, os.path.dirname(exe)):
                return proc.stderr[-1500:]
            err = "no toolchain: " + proc.stderr[-300:]
    return err


def _toolchain_works(cc, cflags, ldflags, tmpdir):
    src = os.path.join(tmpdir, "_omp_probe.c")
    with open(src, "w") as f:
        f.write("#include <omp.h>\nint main(void){return omp_get_max_threads()>0?0:1;}\n")
    exe = os.path.join(tmpdir, "_omp_probe")
    proc = subprocess.run([cc] + cflags + ["-o", exe, src] + ldflags,
                          capture_output=True, text=True)
    return proc.returncode == 0
