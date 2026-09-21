"""The shared header and the HOST code of both .cu files compile as C++.

nvcc hands host code to g++, so a C-only idiom in LaueMatchingHeaders.h (an
implicit void* conversion, a designated initialiser, `restrict`, a POSIX
function a feature-test macro hides) passes every C fixture in this suite and
fails only the CI CUDA build. Nothing here has CUDA, so this compiles the .cu
files with every kernel launch `<<<...>>>` stripped against a declarations-only
stub cuda.h, with `-fsyntax-only`: it type-checks all host code (including each
launch's argument list against the kernel signature) and does NOT compile
device code. CI's nvcc build remains the real gate for that.

It also pins a pre-existing hazard: both .cu files wrap the header in
`extern "C"`, and GCC >= 14's <omp.h> declares templates, so including it
inside that wrapper is a hard error ("template with C linkage"). The fix is to
include <omp.h> and <math.h> BEFORE the wrapper; this test deliberately does
NOT pass `-include omp.h`, so it fails if that ordering regresses -- on any
GCC >= 14 found (g++-14/g++-15 are tried by name when present).

Skips when no C++ compiler with OpenMP headers is available.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

C_DIR = Path(__file__).resolve().parent.parent / "c_src"
CUS = ("LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu")

_STUB = """/* Host-side syntax stub: declarations only, nothing is linked. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#define __global__
#define __device__
typedef int cudaError_t;
enum { cudaSuccess = 0 };
typedef enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 } cudaMemcpyKind;
typedef struct CUst_st *cudaStream_t;
typedef struct CUev_st *cudaEvent_t;
struct dim3_ { unsigned x, y, z; };
extern dim3_ blockIdx, blockDim, threadIdx;
template <class T> T __ldg(const T *p) { return *p; }
int atomicAdd(int *, int);
const char *cudaGetErrorString(cudaError_t);
cudaError_t cudaFree(void *);
template <class T> cudaError_t cudaMalloc(T **, size_t);
template <class T> cudaError_t cudaMallocHost(T **, size_t);
cudaError_t cudaMallocHost(void **, size_t);
cudaError_t cudaFreeHost(void *);
cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
cudaError_t cudaGetLastError(void);
cudaError_t cudaDeviceSynchronize(void);
"""

_LIBOMP_PREFIXES = ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")


def _cxx_candidates():
    seen = []
    for c in (os.environ.get("CXX"), "g++", "c++", "clang++", "g++-14",
              "g++-15"):
        p = shutil.which(c) if c else None
        if p and os.path.realpath(p) not in [os.path.realpath(s) for s in seen]:
            seen.append(p)
    return seen


def _omp_flags(cxx, tmp):
    """First OpenMP flag set this compiler accepts for a C++ TU."""
    probe = tmp / "probe.cpp"
    probe.write_text("#include <omp.h>\nint main(){return omp_get_max_threads();}\n")
    sets = [["-fopenmp"]] + [["-Xpreprocessor", "-fopenmp", f"-I{p}/include"]
                             for p in _LIBOMP_PREFIXES if os.path.isdir(p)]
    for fl in sets:
        if subprocess.run([cxx, "-x", "c++", "-fsyntax-only", *fl, str(probe)],
                          capture_output=True).returncode == 0:
            return fl
    return None


@pytest.fixture(scope="module")
def toolchains(tmp_path_factory):
    if not (C_DIR / CUS[0]).is_file():
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    tmp = tmp_path_factory.mktemp("cxx")
    (tmp / "cuda.h").write_text(_STUB)
    found = []
    for cxx in _cxx_candidates():
        fl = _omp_flags(cxx, tmp)
        if fl is not None:
            found.append((cxx, fl))
    if not found:
        pytest.skip("no C++ compiler with OpenMP headers available")
    return tmp, found


def _compile(cxx, flags, src, stubdir):
    cmd = [cxx, "-x", "c++", "-std=gnu++17", "-fsyntax-only", *flags,
           f"-I{stubdir}", f"-I{C_DIR}", str(src)]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.parametrize("cu", CUS)
def test_cu_host_code_compiles_as_cxx(toolchains, cu):
    tmp, found = toolchains
    text = (C_DIR / cu).read_text()
    stripped = re.sub(r"<<<.*?>>>", "", text, flags=re.S)
    assert stripped != text, f"{cu}: no kernel launch found to strip"
    src = tmp / (cu + ".cpp")
    src.write_text(stripped)
    for cxx, fl in found:
        r = _compile(cxx, fl, src, tmp)
        errors = [l for l in r.stderr.splitlines() if "error" in l]
        assert r.returncode == 0, f"{cxx} ({cu}):\n" + "\n".join(errors[:20])


def test_header_alone_compiles_as_cxx(toolchains):
    tmp, found = toolchains
    src = tmp / "header_only.cpp"
    src.write_text('#include "LaueMatchingHeaders.h"\n')
    for cxx, fl in found:
        r = _compile(cxx, fl, src, tmp)
        assert r.returncode == 0, f"{cxx}:\n{r.stderr[-2000:]}"


@pytest.mark.parametrize("cu", CUS)
def test_cxx_sensitive_headers_precede_the_extern_c_wrapper(cu):
    """Source guard for the GCC >= 14 hazard, independent of which compiler
    this machine happens to have."""
    text = (C_DIR / cu).read_text()
    wrap = text.index('extern "C" {')
    for h in ("#include <omp.h>", "#include <math.h>"):
        assert h in text[:wrap], f"{cu}: {h} must come before extern \"C\""
