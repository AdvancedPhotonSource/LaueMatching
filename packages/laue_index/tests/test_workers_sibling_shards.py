"""Worker sizing when several shards share a host.

The incident these pin: a 112-core host with ``ulimit -u`` 8192 ran two shards.
Each sized itself from the whole host (112 workers), each worker let OpenCV
start its own thread pool, the user ran out of threads, OpenCV failed with
``res = 11``, the GPU daemon reported a misleading "GPUassert: device busy", and
the dispatcher could not fork the next shard. So the chooser must divide by
the shards on the host, take the daemon's cores out, respect RLIMIT_NPROC, and
honour a fractional cgroup quota; and the pool must pin library threads to 1.
``LAUE_PREPROCESS_WORKERS`` must still win outright.
"""
import os
from pathlib import Path

import pytest

from laue_index import workers as W


@pytest.fixture
def host(monkeypatch):
    """A 112-CPU, 512 GB host with no process limit and a clean environment."""
    monkeypatch.setattr(W, "usable_cpu_count", lambda: 112)
    monkeypatch.setattr(W, "available_memory_bytes", lambda: 512 * 10**9)
    monkeypatch.setattr(W, "_rlimit_nproc", lambda: None)
    for k in ("LAUE_PREPROCESS_WORKERS", "LAUE_SHARDS_PER_HOST",
              "LAUE_DAEMON_NCPUS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _choose(**kw):
    return W.choose_preprocess_workers(2048, 2048, **kw)


def test_one_shard_uses_the_host(host):
    n, d = _choose()
    assert n == 112 and d["bound_by"] == "usable_cpus"
    assert d["shards_per_host"] == 1 and d["daemon_ncpus"] == 0


def test_shards_per_host_divides_the_cpus(host):
    host.setenv("LAUE_SHARDS_PER_HOST", "2")
    n, d = _choose()
    assert n == 56
    assert d["shards_per_host"] == 2


def test_shards_per_host_divides_the_memory_budget(host):
    """Memory binding on one shard must bind twice as hard on two: each shard's
    parent queue is its own, but the host's memory is shared."""
    host.setattr(W, "available_memory_bytes", lambda: 20 * 10**9)
    one, d1 = _choose(queue_depth=64)
    host.setenv("LAUE_SHARDS_PER_HOST", "2")
    two, d2 = _choose(queue_depth=64)
    assert d1["bound_by"] == d2["bound_by"] == "memory"
    assert d2["memory_budget_bytes"] == int(20 * 10**9 * 0.75) // 2 - d2["queue_bytes"]
    assert two < one


def test_daemon_ncpus_is_subtracted(host):
    host.setenv("LAUE_SHARDS_PER_HOST", "2")
    host.setenv("LAUE_DAEMON_NCPUS", "8")
    n, d = _choose()
    assert n == 56 - 8
    assert d["daemon_ncpus"] == 8


def test_explicit_kwargs_beat_environment(host):
    host.setenv("LAUE_SHARDS_PER_HOST", "4")
    host.setenv("LAUE_DAEMON_NCPUS", "8")
    n, _ = _choose(shards_per_host=1, daemon_ncpus=0)
    assert n == 112


def test_never_below_one_worker(host):
    host.setenv("LAUE_SHARDS_PER_HOST", "200")
    host.setenv("LAUE_DAEMON_NCPUS", "64")
    n, _ = _choose()
    assert n == 1


def test_rlimit_nproc_caps_the_pool(host):
    """ulimit -u 8192 with two shards: half the limit, 4 threads a worker, split
    two ways -> 512 per shard. With few cores that does not bind; with a low
    limit it must."""
    host.setattr(W, "_rlimit_nproc", lambda: 8192)
    host.setenv("LAUE_SHARDS_PER_HOST", "2")
    n, d = _choose()
    assert d["workers_by_nproc"] == 8192 // 2 // W.THREADS_PER_WORKER // 2
    assert n == 56                       # cores bind first here
    host.setattr(W, "_rlimit_nproc", lambda: 256)
    n, d = _choose()
    assert d["workers_by_nproc"] == 256 // 2 // W.THREADS_PER_WORKER // 2 == 16
    assert n == 16 and d["bound_by"] == "nproc"


def test_rlimit_nproc_reader_handles_unlimited(monkeypatch):
    resource = pytest.importorskip("resource")
    monkeypatch.setattr(resource, "getrlimit",
                        lambda _r: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    assert W._rlimit_nproc() is None
    monkeypatch.setattr(resource, "getrlimit", lambda _r: (8192, 8192))
    assert W._rlimit_nproc() == 8192


def test_override_wins_over_shards_daemon_and_nproc(host, caplog):
    """An operator who has measured the host is not argued with -- but is told
    when the number exceeds the thread budget."""
    host.setattr(W, "_rlimit_nproc", lambda: 256)
    host.setenv("LAUE_SHARDS_PER_HOST", "2")
    host.setenv("LAUE_DAEMON_NCPUS", "8")
    host.setenv("LAUE_PREPROCESS_WORKERS", "100")
    with caplog.at_level("WARNING", logger="LaueStream"):
        n, d = _choose(max_workers=4)
    assert n == 100
    assert d["bound_by"] == "LAUE_PREPROCESS_WORKERS"
    assert "thread budget" in caplog.text


def test_config_cap_still_only_lowers(host):
    host.setenv("LAUE_SHARDS_PER_HOST", "2")
    assert _choose(max_workers=10)[0] == 10
    assert _choose(max_workers=500)[0] == 56


def test_bad_shard_value_is_ignored_with_a_warning(host, caplog):
    host.setenv("LAUE_SHARDS_PER_HOST", "two")
    with caplog.at_level("WARNING", logger="LaueStream"):
        n, d = _choose()
    assert n == 112 and d["shards_per_host"] == 1
    assert "LAUE_SHARDS_PER_HOST" in caplog.text


# ---------------------------------------------------------------------------
# usable_cpu_count: fractional cgroup quota
# ---------------------------------------------------------------------------

def test_fractional_cgroup_quota_is_one_cpu(monkeypatch):
    """A 0.5-CPU container quota used to be ignored (`quota >= 1`), so the pool
    was sized to every CPU on the host -- inside half a CPU."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _p: set(range(64)),
                        raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(W, "_cgroup_cpu_quota", lambda: 0.5)
    assert W.usable_cpu_count() == 1


def test_whole_quota_unchanged(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _p: set(range(64)),
                        raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(W, "_cgroup_cpu_quota", lambda: 6.0)
    assert W.usable_cpu_count() == 6


# ---------------------------------------------------------------------------
# the pool initializer
# ---------------------------------------------------------------------------

def test_parent_pins_library_threads_and_initializer_does_not(monkeypatch):
    from laue_index.pipeline import laue_image_server as srv
    for k in srv._SINGLE_THREAD_ENV:
        monkeypatch.setenv(k, "64")
    calls = []

    class _FakeCv2:
        @staticmethod
        def setNumThreads(n):
            calls.append(n)

        @staticmethod
        def getNumThreads():
            return 1

    import sys
    import types
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2)
    # Stub the other two so the test does not pin THIS process's BLAS / diplib
    # pools for the rest of the session.
    tpc = types.ModuleType("threadpoolctl")
    tpc.threadpool_limits = lambda n: calls.append(("blas", n))
    monkeypatch.setitem(sys.modules, "threadpoolctl", tpc)
    dip = types.ModuleType("diplib")
    dip.SetNumberOfThreads = lambda n: calls.append(("dip", n))
    monkeypatch.setitem(sys.modules, "diplib", dip)
    srv._pin_library_threads()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[k] == "1"
    assert calls == [1, ("blas", 1), ("dip", 1)]
    # The worker initializer must make NO library calls: a forked child that
    # touches OpenCV / libgomp / BLAS can deadlock (measured on Linux).
    calls.clear()
    for k in srv._SINGLE_THREAD_ENV:
        monkeypatch.setenv(k, "64")
    srv._init_preprocess_worker("fork")
    assert calls == []
    # A spawned worker is a fresh interpreter: it pins its own pools (safe there).
    srv._init_preprocess_worker("spawn")
    assert calls == [1, ("blas", 1), ("dip", 1)]
    assert all(os.environ[k] == "1" for k in srv._SINGLE_THREAD_ENV)


_POOL_PROBE = r"""
import multiprocessing as mp, os, sys
import numpy as np
import cv2
if sys.platform.startswith("linux"):
    mp.set_start_method("fork")        # production's path; the one that can hang
# Warm OpenCV's own thread pool in the parent first, as a long-lived server has:
# forking a process whose library threads exist is the hazardous case.
cv2.setNumThreads(8)
for _ in range(3):
    cv2.GaussianBlur(np.random.rand(2048, 2048).astype(np.float32), (0, 0), 5)
os.environ["OMP_NUM_THREADS"] = "8"
from laue_index.pipeline import laue_image_server as srv

def probe():
    import cv2
    return os.environ.get("OMP_NUM_THREADS"), cv2.getNumThreads()

if __name__ == "__main__":
    with srv._make_pool(2) as pool:
        print(*pool.submit(probe).result(timeout=60))
"""


def test_server_pool_workers_run_single_threaded(tmp_path):
    """Behavioural, in a FRESH interpreter (pinning is process-wide and must not
    leak into this session): a worker from the server's real pool factory, forked
    on Linux from a parent whose OpenCV pool is already running, comes up with
    the thread environment at 1 and OpenCV at one thread (or 0, sequential, on
    backends that ignore 1) -- and it comes up at all. The timeout is the test:
    the design this replaced hung here indefinitely on Linux."""
    pytest.importorskip("cv2")
    import subprocess
    import sys
    script = tmp_path / "pool_probe.py"
    script.write_text(_POOL_PROBE)
    env = dict(os.environ)
    # The pipeline modules import each other flat (see conftest.py), so the
    # pipeline directory goes on the path along with the package root.
    from laue_index.pipeline import PIPELINE_DIR
    pkg_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_root, str(PIPELINE_DIR), env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, timeout=180, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    omp, cv_threads = r.stdout.split()[-2:]
    assert omp == "1"
    assert int(cv_threads) <= 1


def test_orchestrator_env_keeps_0_7_1_worker_count(host):
    """Behavioural, for the review finding: the orchestrator used to set
    LAUE_DAEMON_NCPUS=--ncpus, and run_laue.sh defaults NCPUS=32, so a 32-CPU
    host got 1 worker. With the env the orchestrator now produces, a 32-CPU host
    with default settings gets 0.7.1's count (min(CPUs, memory) = 32) reduced
    only by the RLIMIT_NPROC cap."""
    import laue_orchestrator as lo
    host.setattr(W, "usable_cpu_count", lambda: 32)
    env = lo._server_env()
    assert "LAUE_DAEMON_NCPUS" not in env
    for k, v in env.items():
        host.setenv(k, v)
    host.setattr(W, "_rlimit_nproc", lambda: 8192)          # 1024 by nproc
    n, d = _choose()
    assert n == 32 and d["bound_by"] == "usable_cpus"
    host.setattr(W, "_rlimit_nproc", lambda: 64)            # 8 by nproc
    n, d = _choose()
    assert n == 8 and d["bound_by"] == "nproc"


def test_orchestrator_passes_a_callers_daemon_ncpus_through(host):
    import laue_orchestrator as lo
    host.setenv("LAUE_DAEMON_NCPUS", "6")
    assert lo._server_env()["LAUE_DAEMON_NCPUS"] == "6"


def test_daemon_subtraction_never_below_half_the_share(host, caplog):
    """Rule: workers = max(share - daemon_ncpus, share // 2, 1), share = CPUs //
    shards. 32 CPUs with a 32-core daemon -> 16, with a warning."""
    host.setattr(W, "usable_cpu_count", lambda: 32)
    host.setenv("LAUE_DAEMON_NCPUS", "32")
    with caplog.at_level("WARNING", logger="LaueStream"):
        n, _ = _choose()
    assert n == 16
    assert "half the share" in caplog.text
    host.setenv("LAUE_DAEMON_NCPUS", "4")
    assert _choose()[0] == 28
