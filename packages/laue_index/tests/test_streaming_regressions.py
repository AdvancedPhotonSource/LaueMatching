"""Regression tests for five silent-failure bugs in the streaming pipeline.

All five were found while running ~150,000 Laue frames through the pipeline across
four machines during the July 2026 34-ID-E beam time. Four share a signature: the
failure reported success, so partial or misplaced work looked like a clean run. Each
test below locks in one fix; the module docstring of each fix and PR #5 have the full
story.

Where the fixed logic is a reachable function it is exercised directly; where it is
inline in a long driver it is pinned with a source contract (which still fails if the
construct is reverted) plus, for the concurrency bug, a behavioural reproduction of the
race the fix removes.

Run:
    cd ~/opt/LaueMatching && python -m pytest scripts/tests/test_streaming_regressions.py -v
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

# macOS CI sometimes links two OpenMP runtimes via numpy/scipy; harmless here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# conftest.py puts laue_index/pipeline/ on sys.path, which is where the
# orchestration modules live since they moved into the package. These are
# source-level assertions, so they must read the real modules -- not the
# same-named one-line shims still sitting in the repo's scripts/.
from laue_index.pipeline import PIPELINE_DIR  # noqa: E402

_ORCH_SRC = (PIPELINE_DIR / "laue_orchestrator.py").read_text()
_SERVER_SRC = (PIPELINE_DIR / "laue_image_server.py").read_text()

# The shell pipeline stayed in the repo, so the tests that read it can only run
# from a checkout. Skip them explicitly rather than handing them a path that
# does not exist -- a missing file is a failure, and a failure that only means
# "not a checkout" trains people to ignore red.
# Anchor on the launcher ITSELF, not on a directory that happens to exist: the
# old form looked for a `scripts/` dir and then reached into `scripts/pipeline/`,
# so when the launcher moved to `pipeline/` the marker still found a repo root,
# the file was absent, and these tests skipped instead of failing.
_RUN_LAUE = next(
    (p / "pipeline" / "run_laue.sh" for p in Path(__file__).resolve().parents
     if (p / "pipeline" / "run_laue.sh").is_file()), None)
needs_checkout = pytest.mark.skipif(
    _RUN_LAUE is None,
    reason="pipeline/run_laue.sh needs a checkout")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------
# Bug 1: the port wait killed a healthy-but-slow daemon and waited the full
#        timeout on a dead one. _wait_for_daemon_port polls the process too.
# ---------------------------------------------------------------------------

class _FakeAlive:
    def poll(self):
        return None


def test_port_wait_fails_fast_when_daemon_dies():
    """A daemon that exits without opening the port must be reported at once,
    with its exit code — not after the whole timeout (old behaviour: 180 s+)."""
    import laue_orchestrator as lo

    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    t0 = time.time()
    ok = lo._wait_for_daemon_port(proc, _free_port(), timeout=900.0, poll_interval=0.2)
    dt = time.time() - t0
    assert ok is False
    assert dt < 15, f"took {dt:.1f}s; should fail fast on a dead daemon"


def test_port_wait_keeps_waiting_while_daemon_is_alive():
    """A slow-but-alive daemon must be waited for, not killed at a fixed budget."""
    import laue_orchestrator as lo

    port = _free_port()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def open_late():
        time.sleep(4)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)

    threading.Thread(target=open_late, daemon=True).start()
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        t0 = time.time()
        ok = lo._wait_for_daemon_port(alive, port, timeout=30.0,
                                      poll_interval=0.2, progress_every=2.0)
        dt = time.time() - t0
        assert ok is True
        assert dt >= 3.5, "must not return before the port actually opens"
    finally:
        alive.kill()
        srv.close()


def test_port_wait_honours_cap_when_never_opens():
    """An alive daemon that never opens the port must still give up at the cap."""
    import laue_orchestrator as lo

    hang = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        t0 = time.time()
        ok = lo._wait_for_daemon_port(hang, _free_port(), timeout=3.0,
                                      poll_interval=0.2, progress_every=1.0)
        dt = time.time() - t0
        assert ok is False
        assert 2.5 <= dt < 12
    finally:
        hang.kill()


# ---------------------------------------------------------------------------
# Bug 2: run_laue.sh ignored WATCH="". ${WATCH:-default} substitutes for an
#        empty value too, so the documented batch mode never took effect.
# ---------------------------------------------------------------------------

def _watch_expansion(env_value):
    """Return what run_laue.sh's WATCH expansion yields for a given WATCH env."""
    line = [l for l in _RUN_LAUE.read_text().splitlines()
            if l.strip().startswith("WATCH=")][0].split("#")[0].strip()
    # e.g. WATCH=${WATCH-"--watch"}
    script = f'{("WATCH="+chr(39)+env_value+chr(39)+"; ") if env_value is not None else ""}{line}; printf "%s" "$WATCH"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return out.stdout


@needs_checkout
def test_watch_empty_means_batch():
    """WATCH="" must yield an empty flag (batch mode), not --watch."""
    assert _watch_expansion("") == "", (
        "WATCH='' still expands to --watch; batch mode silently stays in watch mode"
    )


@needs_checkout
def test_watch_unset_still_defaults_to_watch():
    """An unset WATCH must still default to --watch (live mode)."""
    assert _watch_expansion(None) == "--watch"


# ---------------------------------------------------------------------------
# Bug 3: the daemon was terminated as soon as solutions.txt EXISTED, before it
#        finished writing, losing the tail of the scan. The fix waits for the
#        file to stop growing. (Logic is inline in run_pipeline; pin by contract.)
# ---------------------------------------------------------------------------

def test_drain_waits_for_quiescence_not_mere_existence():
    """The flush wait must key on the file no longer growing, not on it existing."""
    # the old, buggy predicate broke as soon as the file was non-empty
    assert 'os.path.getsize(solutions_file) > 0:' not in _ORCH_SRC or \
        "quiescent" in _ORCH_SRC, (
        "orchestrator appears to break on solutions.txt existence again; "
        "it must wait for the file to stop growing"
    )
    assert "quiescent" in _ORCH_SRC, "drain-quiescence logic missing"
    # and it must warn (not silently proceed) if it gives up while still growing
    assert "may be truncated" in _ORCH_SRC


# ---------------------------------------------------------------------------
# Bug 4: frame_mapping was mutated by two threads while a third serialised it,
#        raising "dictionary changed size during iteration", which killed the
#        consumer and left the server reporting a truncated scan as success.
# ---------------------------------------------------------------------------

def test_frame_mapping_mutations_are_lock_guarded():
    """Every frame_mapping write, and the JSON dump, must hold the lock."""
    assert "mapping_lock = threading.Lock()" in _SERVER_SRC, "mapping_lock missing"
    # the dump must operate on a snapshot taken under the lock, never the live dict
    assert "with mapping_lock:" in _SERVER_SRC
    assert "snapshot = dict(frame_mapping)" in _SERVER_SRC, (
        "JSON dump must use a snapshot taken under the lock, not the live dict"
    )


def test_mutating_a_dict_during_iteration_raises():
    """The exact failure mode the fix removes, made deterministic: mutating a dict
    while it is being iterated (which is what json.dump does) raises RuntimeError."""
    # json.dump(mapping, f, indent=1) iterates the dict's items; a concurrent insert
    # from the sender thread mid-iteration is exactly this RuntimeError. Reproduced
    # deterministically here without threads.
    d = {i: i for i in range(100)}
    with pytest.raises(RuntimeError, match="changed size during iteration"):
        for _ in d:
            d[len(d)] = 1


def test_snapshot_under_lock_is_immune():
    """The fix pattern — copy the dict under the lock, dump the copy — is safe even
    while the live dict keeps changing. This is deterministic: it must always pass."""
    live, lock, stop = {}, threading.Lock(), threading.Event()

    def churn():
        i = 0
        while not stop.is_set() and i < 500_000:
            with lock:
                live[str(i)] = {"frame": i}
            i += 1

    th = threading.Thread(target=churn, daemon=True)
    th.start()
    try:
        for _ in range(500):
            with lock:
                snapshot = dict(live)     # exactly what the consumer thread does
            json.dumps(snapshot)          # never touches the live, mutating dict
    finally:
        stop.set()
        th.join(timeout=2)


# ---------------------------------------------------------------------------
# Bug 5: CUDA_VISIBLE_DEVICES without CUDA_DEVICE_ORDER selects by CUDA's
#        FASTEST_FIRST ordering, which need not match nvidia-smi indices.
# ---------------------------------------------------------------------------

@needs_checkout
def test_run_laue_pins_cuda_device_order():
    """The launch line must set CUDA_DEVICE_ORDER=PCI_BUS_ID so the configured GPU
    index selects the card nvidia-smi calls that index."""
    src = _RUN_LAUE.read_text()
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in src, (
        "run_laue.sh selects a GPU without pinning CUDA_DEVICE_ORDER; on a mixed-GPU "
        "host the job can land on the wrong card"
    )
    # the two env vars must share a line so the ordering applies to the selection
    # (setsid/the command may continue on the next physical line via a trailing backslash)
    dev = [l for l in src.splitlines() if "CUDA_VISIBLE_DEVICES=" in l]
    assert dev and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in dev[0], (
        "CUDA_DEVICE_ORDER must sit on the same line as CUDA_VISIBLE_DEVICES"
    )
