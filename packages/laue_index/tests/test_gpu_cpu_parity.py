"""CPU, GPU and streaming binaries index one frame identically.

Runs only where the three binaries exist: set LAUE_GPU_BINDIR to a directory
holding LaueMatchingCPU, LaueMatchingGPU and LaueMatchingGPUStream built from
THIS checkout (a GPU host; skipped everywhere else, including CI).

What it pins (0.8.0):
* the GPU coarse score is accumulated in double, as on the CPU, so the
  MinIntensity gate and the merge order cannot depend on which binary ran;
* the merge sorts candidates by (score desc, row asc), so the GPU's atomicAdd
  arrival order does not change the clusters;
* the streaming daemon gates fits on MinNrSpots and honours OrientationSpacing,
  as the CPU does;
* all three read the forward cache only through its .meta sidecar.

The frame: three cubic crystals at known orientations, each a row of a small
random database that also holds a chain of its near-duplicates (to exercise
the merge), with their predicted spots lit, plus unrelated noise spots.

Measured 2026-09-28 on alleppey (H100, nvcc 13.3): 3 solutions from each of
CPU, GPU (twice) and the streaming daemon (images 0 and 1), identical. This pins
parity; it does not by itself show the frame is one on which the old float32
score or the old MinGoodSpots stream gate would have differed.
"""
import os
import signal
import socket
import struct
import subprocess
import time

import numpy as np
import pytest

from laue_index import lattice

BINDIR = os.environ.get("LAUE_GPU_BINDIR")
NAMES = ("LaueMatchingCPU", "LaueMatchingGPU", "LaueMatchingGPUStream")
pytestmark = pytest.mark.skipif(
    not BINDIR or not all(os.path.isfile(os.path.join(BINDIR, n)) for n in NAMES),
    reason="set LAUE_GPU_BINDIR to a directory with all three binaries (GPU host)")

NPX, PX, LSD = 512, 0.2, 60.0
LAT = (0.36, 0.36, 0.36, 90.0, 90.0, 90.0)
ELO, EHI = 5.0, 30.0


def _rand_rot(rng):
    q = rng.normal(size=4)
    return lattice.quat_to_matrix(q / np.linalg.norm(q))


def _axis_angle(ax, deg):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _spots(U, hkls):
    """Pixels the C predicts (identity detector rotation, P = (0, 0, LSD))."""
    q = (U @ lattice.reciprocal_matrix(LAT, 225) @ hkls.T).T
    qh = q / np.linalg.norm(q, axis=1, keepdims=True)
    kf = np.array([0, 0, 1.0]) - 2 * qh[:, 2:3] * qh
    ok = kf[:, 2] > 0
    E = -(1.2398419739 / (4 * np.pi)) * (q ** 2).sum(1) / q[:, 2]
    ok &= (E >= ELO) & (E <= EHI)
    fx = kf[:, 0] * LSD / kf[:, 2] / PX + (NPX - 1) / 2
    fy = kf[:, 1] * LSD / kf[:, 2] / PX + (NPX - 1) / 2
    ok &= (fx >= 0.5) & (fx < NPX - 1.5) & (fy >= 0.5) & (fy < NPX - 1.5)
    return np.rint(fx[ok]).astype(int), np.rint(fy[ok]).astype(int)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    d = tmp_path_factory.mktemp("parity")
    rng = np.random.default_rng(20260928)
    hkls = np.array([(h, k, l) for h in range(-4, 5) for k in range(-4, 5)
                     for l in range(-4, 5) if (h, k, l) != (0, 0, 0)], float)
    truths = [_rand_rot(rng) for _ in range(3)]
    oms = [_rand_rot(rng) for _ in range(20000)]
    for t, (row, chain) in zip(truths, ((7777, 1000), (12345, 2000), (15000, 3000))):
        oms[row] = t
        for i, deg in enumerate((0.15, 0.3, 0.45)):   # near-duplicates: a merge chain
            oms[chain + i] = _axis_angle(rng.normal(size=3), deg) @ t
    np.asarray(oms, float).reshape(-1, 9).tofile(d / "orients.bin")
    np.savetxt(d / "hkls.txt", hkls, fmt="%d")
    img = np.zeros((NPX, NPX))
    kern = np.array([[.3, .5, .3], [.5, 1, .5], [.3, .5, .3]])
    for t in truths:
        xs, ys = _spots(t, hkls)
        assert len(xs) >= 20, "geometry puts too few truth spots on the panel"
        for x, y, a in zip(xs, ys, rng.uniform(80, 400, len(xs))):
            img[y - 1:y + 2, x - 1:x + 2] += a * kern
    for x, y in rng.integers(2, NPX - 2, size=(40, 2)):
        img[y, x] += rng.uniform(80, 400)
    img.astype(np.float64).tofile(d / "image.bin")
    base = (f"LatticeParameter {' '.join(map(str, LAT))}\nSpaceGroup 225\n"
            f"P_Array 0 0 {LSD}\nR_Array 0 0 0\nPxX {PX}\nPxY {PX}\n"
            f"NrPxX {NPX}\nNrPxY {NPX}\nElo {ELO}\nEhi {EHI}\nMinNrSpots 6\n"
            "MinGoodSpots 2\nMinIntensity 200\nMaxNrLaueSpots 200\nMaxAngle 2\n"
            "OrientationSpacing 0.4\nMinSpotIntensity 0\n"
            f"ForwardFile {d / 'fwd.bin'}\nResultDir {d / 'stream'}\n")
    (d / "params_build.txt").write_text(base + "DoFwd 1\n")
    (d / "params.txt").write_text(base + "DoFwd 0\n")
    exe = lambda n: os.path.join(BINDIR, n)
    r = subprocess.run([exe("LaueMatchingCPU"), "params_build.txt", "orients.bin",
                        "hkls.txt", "image.bin", "4"], cwd=d, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (d / "fwd.bin.meta.json").is_file(), "the forward cache was published without its record"
    return d, truths


def _rows(path, drop_first=False):
    rows = []
    for line in open(path):
        if line.startswith("%") or not line.strip():
            continue
        v = line.split()
        rows.append(v[1:] if drop_first else v)
    return rows


def _compare(a, b):
    assert len(a) >= 3, f"only {len(a)} solutions: the comparison would be thin"
    assert len(a) == len(b), f"{len(a)} vs {len(b)} solutions"
    for ra, rb in zip(sorted(a, key=lambda r: int(r[-1])), sorted(b, key=lambda r: int(r[-1]))):
        # ints exactly: GrainNr-independent columns NumberOfSolutions, NMatches,
        # NSpotsCalc, orientationRowNr; floats to 1e-6 relative.
        for i in (1, 5, 6, len(ra) - 1):
            assert ra[i] == rb[i], (i, ra, rb)
        fa, fb = np.array(ra[2:5] + ra[7:-1], float), np.array(rb[2:5] + rb[7:-1], float)
        assert np.allclose(fa, fb, rtol=1e-6, atol=1e-9), (ra, rb)


def _run(d, name):
    r = subprocess.run([os.path.join(BINDIR, name), "params.txt", "orients.bin",
                        "hkls.txt", "image.bin", "4"], cwd=d, capture_output=True,
                       text=True, timeout=600,
                       env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "rebuilding it" not in r.stdout, "the sidecar written by the build run was refused"
    return _rows(d / "image.bin.solutions.txt")


def test_cpu_finds_every_truth(case):
    d, truths = case
    rows = _run(d, "LaueMatchingCPU")
    for t in truths:
        best = min(np.degrees(np.arccos(np.clip(
            (np.trace(t.T @ np.array(r[22:31], float).reshape(3, 3)) - 1) / 2, -1, 1)))
            for r in rows)
        assert best < 0.5, f"closest solution {best:.2f} deg from a truth"


def test_gpu_equals_cpu(case):
    d, _ = case
    cpu = _run(d, "LaueMatchingCPU")
    (d / "image.bin.solutions.txt").rename(d / "cpu.solutions.txt")
    gpu = _run(d, "LaueMatchingGPU")
    _compare(cpu, gpu)
    # run the GPU twice: arrival order differs, output must not
    (d / "image.bin.solutions.txt").rename(d / "gpu1.solutions.txt")
    _compare(gpu, _run(d, "LaueMatchingGPU"))


def test_stream_equals_cpu(case):
    d, _ = case
    cpu = _run(d, "LaueMatchingCPU")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, LAUE_STREAM_PORT=str(port), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    log = open(d / "stream.log", "w")
    p = subprocess.Popen([os.path.join(BINDIR, "LaueMatchingGPUStream"), "params.txt",
                          "orients.bin", "hkls.txt", "4"], cwd=d, env=env, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 300
        while True:
            try:
                c = socket.create_connection(("127.0.0.1", port), timeout=2)
                break
            except OSError:
                assert p.poll() is None, open(d / "stream.log").read()[-3000:]
                assert time.time() < deadline, "daemon never opened its port"
                time.sleep(1)
        frame = np.fromfile(d / "image.bin").astype("<f4")
        with c:
            for num in (0, 1):   # image 0 must keep its ImageNr column
                c.sendall(struct.pack("<H", num) + frame.tobytes())
        sol = d / "stream" / "solutions.txt"
        deadline = time.time() + 300
        while time.time() < deadline:
            if sol.is_file():
                ids = [r[0] for r in _rows(sol)]
                if ids.count("0") == len(cpu) and ids.count("1") == len(cpu):
                    break
            time.sleep(1)
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
        log.close()
    rows = _rows(sol)
    for num in ("0", "1"):
        _compare(cpu, [r[1:] for r in rows if r[0] == num])
