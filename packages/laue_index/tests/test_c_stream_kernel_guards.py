"""LaueMatchingGPUStream carries the same device and I/O guards as LaueMatchingGPU.

SOURCE-LEVEL, deliberately: there is no CUDA on the development Mac and the
failure being guarded is drift between two copies of one kernel, which a test
reading both files can see and a behaviour test on one backend cannot.

* The `compare` kernel reads `im[py * nrPxX + px]` for every (px, py) in a
  forward-cache row. The cache is size-checked (`forwardCacheUsable`) but not
  content-checked, so LaueMatchingGPU.cu clamps the spot count and bounds-checks
  (px, py) before the read. The streaming copy did neither.
* The streaming daemon's forward-simulation `pwrite` ignored its return value:
  a failed or short write left a zero-filled hole that can still have the size
  the cache check accepts.
"""
import re
from pathlib import Path

import pytest

C_DIR = Path(__file__).resolve().parent.parent / "c_src"
GPU, STREAM = "LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu"


def _read(name):
    p = C_DIR / name
    if not p.is_file():
        pytest.skip(f"{name} not present (installed wheel, not a checkout)")
    return p.read_text()


def _kernel(src):
    start = src.index("__global__ void compare(")
    return src[start:src.index("\n}\n", start)]


def _normalise(body):
    return re.sub(r"\s+", " ", body)


@pytest.mark.parametrize("name", [GPU, STREAM])
def test_compare_kernel_bounds_checks_every_read(name):
    k = _kernel(_read(name))
    assert re.search(r"__global__ void compare\(size_t nrPxX, size_t nrPxY,", k), (
        f"{name}: compare() must take nrPxY to bounds-check py")
    assert "if (nrSpots > nrMaxSpots)" in k, f"{name}: spot count not clamped"
    guard = k.index("if (px < nrPxX && py < nrPxY)")
    assert guard < k.index("__ldg(&im[py * nrPxX + px])"), (
        f"{name}: the image read is not inside the (px, py) bounds check")


def test_the_two_kernels_are_identical():
    """Same signature, same body. If one changes, both must."""
    assert _normalise(_kernel(_read(GPU))) == _normalise(_kernel(_read(STREAM)))


def test_stream_launch_passes_nrpxy():
    src = _read(STREAM)
    launch = src[src.index("compare<<<"):]
    launch = launch[:launch.index(");")]
    assert re.search(r">>>\(\s*nrPxX, nrPxY, thisChunk,", launch), launch


def test_stream_forward_cache_write_goes_through_the_checked_writer():
    """The daemon used to discard pwrite's return value. It now calls the
    shared writeForwardSlabOrDie(); test_c_forward_cache_write.py covers that
    function's behaviour."""
    src = _read(STREAM)
    assert "pwrite(" not in src, (
        "a direct pwrite in the stream daemon bypasses the checked writer")
    assert "writeForwardSlabOrDie(fwdFd," in src
