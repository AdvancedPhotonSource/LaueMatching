"""Tests for the preprocessing worker-count chooser.

The thing being replaced was `min(os.cpu_count() or 4, 8)`. Its two failures are
what these tests pin: the cap of 8 threw away measured throughput, and
`os.cpu_count()` reports the machine rather than this process's slice, so under
a cpuset or a container CPU quota the pool oversubscribed whatever we were
actually given. Both failure modes are asserted directly, not implied.
"""
import os
from unittest import mock

import pytest

from laue_index import workers as W


# ---------------------------------------------------------------------------
# usable_cpu_count
# ---------------------------------------------------------------------------

def test_affinity_beats_cpu_count():
    """A cpuset/taskset slice must win over the machine's core count."""
    with mock.patch.object(os, "sched_getaffinity", lambda _pid: set(range(4)), create=True), \
         mock.patch.object(os, "cpu_count", lambda: 96), \
         mock.patch.object(W, "_cgroup_cpu_quota", lambda: None):
        assert W.usable_cpu_count() == 4


def test_cgroup_quota_caps_affinity():
    """A container CPU quota is invisible to sched_getaffinity -- all 96 CPUs are
    'allowed', we just get 2.5 CPUs' worth of time. The quota must still bind."""
    with mock.patch.object(os, "sched_getaffinity", lambda _pid: set(range(96)), create=True), \
         mock.patch.object(os, "cpu_count", lambda: 96), \
         mock.patch.object(W, "_cgroup_cpu_quota", lambda: 2.5):
        assert W.usable_cpu_count() == 2


def test_usable_cpu_count_never_zero():
    with mock.patch.object(os, "sched_getaffinity", lambda _pid: set(), create=True), \
         mock.patch.object(os, "cpu_count", lambda: None), \
         mock.patch.object(W, "_cgroup_cpu_quota", lambda: None):
        assert W.usable_cpu_count() >= 1


# ---------------------------------------------------------------------------
# the fitted memory model
# ---------------------------------------------------------------------------

def test_worker_peak_matches_the_measurement():
    """Fitted on shannon at 1024^2, 2048^2 and 4096^2; residuals were <=0.2 MB.
    If someone edits the constants, these three must still be reproduced."""
    for n, measured_mb in ((1024, 68.0), (2048, 207.4), (4096, 763.1)):
        got_mb = W.worker_peak_bytes(n * n) / 1e6
        assert got_mb == pytest.approx(measured_mb, abs=1.0), (
            f"{n}x{n}: model says {got_mb:.1f} MB, measurement said {measured_mb} MB")


def test_worker_peak_scales_with_pixels_not_frames():
    """A 4096^2 detector costs ~4x a 2048^2 one. The old constant-8 cap was blind
    to frame size entirely."""
    small = W.worker_peak_bytes(2048 * 2048)
    big = W.worker_peak_bytes(4096 * 4096)
    assert 3.5 < big / small < 4.0


# ---------------------------------------------------------------------------
# choose_preprocess_workers
# ---------------------------------------------------------------------------

def _choose(**kw):
    defaults = dict(nr_px_x=2048, nr_px_y=2048)
    defaults.update(kw)
    return W.choose_preprocess_workers(**defaults)


def test_uses_more_than_eight_when_the_machine_allows():
    """The whole point. 40 cores and 256 GB must not yield 8."""
    with mock.patch.object(W, "usable_cpu_count", lambda: 40), \
         mock.patch.object(W, "available_memory_bytes", lambda: 256 * 10**9), \
         mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
        n, d = _choose(ceiling=None)
        assert n == 40
        assert d["bound_by"] == "usable_cpus"


def test_memory_binds_before_cpus():
    """96 cores but 8 GB: memory must bind, not the core count."""
    with mock.patch.object(W, "usable_cpu_count", lambda: 96), \
         mock.patch.object(W, "available_memory_bytes", lambda: 8 * 10**9):
        os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
        n, d = _choose(ceiling=None)
        assert d["bound_by"] == "memory"
        assert n < 96
        # 8 GB * 0.75 / 207 MB ~= 28
        assert n == d["workers_by_memory"]


def test_queue_is_charged_before_workers():
    """The 512-deep futures queue holds dense float32 frames -- 8.6 GB at
    2048^2. It must come out of the budget before workers are counted, which is
    exactly what the ~17 GB RSS on a 13k-frame shard was."""
    with mock.patch.object(W, "usable_cpu_count", lambda: 64), \
         mock.patch.object(W, "available_memory_bytes", lambda: 14 * 10**9):
        os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
        free, _ = _choose(queue_depth=0, ceiling=None)
        charged, d = _choose(queue_depth=512, ceiling=None)
        assert d["queue_bytes"] == 512 * 2048 * 2048 * 4
        assert charged < free, "queue depth did not reduce the worker count"


def test_impossible_budget_returns_one_not_zero():
    """A huge frame and a tiny budget must not produce a pool of 0 workers."""
    with mock.patch.object(W, "usable_cpu_count", lambda: 16), \
         mock.patch.object(W, "available_memory_bytes", lambda: 100 * 10**6):
        os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
        n, d = _choose(nr_px_x=4096, nr_px_y=4096, queue_depth=512)
        assert n == 1
        assert d["bound_by"] == "memory"


def test_unknown_memory_falls_back_to_cpus():
    """No procfs, no cgroup (macOS): must still return something sane."""
    with mock.patch.object(W, "usable_cpu_count", lambda: 10), \
         mock.patch.object(W, "available_memory_bytes", lambda: None):
        os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
        n, d = _choose(ceiling=None)
        assert n == 10
        assert d["available_memory_bytes"] is None


def test_no_arbitrary_default_ceiling():
    """There was a default ceiling of 32, justified by a curve that only looked
    flat because pool construction was inside the timed region. Warm, throughput
    was still climbing at the core count. A default cap below the CPU count
    would discard measured throughput, so there must not be one."""
    assert W.DEFAULT_WORKER_CEILING is None
    with mock.patch.object(W, "usable_cpu_count", lambda: 40), \
         mock.patch.object(W, "available_memory_bytes", lambda: 111 * 10**9):
        os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
        n, d = _choose(queue_depth=512)
        assert n == 40, "a default ceiling has come back"
        assert d["bound_by"] == "usable_cpus"


def test_ceiling_and_max_workers_only_lower():
    with mock.patch.object(W, "usable_cpu_count", lambda: 64), \
         mock.patch.object(W, "available_memory_bytes", lambda: 512 * 10**9):
        os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
        assert _choose(max_workers=6, ceiling=None)[0] == 6
        assert _choose(ceiling=12)[0] == 12
        # neither may raise the count above what memory/CPUs allow
        with mock.patch.object(W, "usable_cpu_count", lambda: 3):
            assert _choose(max_workers=99, ceiling=99)[0] == 3


def test_env_override_wins_over_memory():
    """An operator who knows their machine is not argued with -- including past
    the memory model, which is a fit and can be wrong on a new pipeline."""
    with mock.patch.object(W, "usable_cpu_count", lambda: 4), \
         mock.patch.object(W, "available_memory_bytes", lambda: 2 * 10**9), \
         mock.patch.dict(os.environ, {"LAUE_PREPROCESS_WORKERS": "24"}):
        n, d = _choose()
        assert n == 24
        assert d["bound_by"] == "LAUE_PREPROCESS_WORKERS"


def test_bad_env_override_is_ignored_not_fatal():
    with mock.patch.object(W, "usable_cpu_count", lambda: 8), \
         mock.patch.object(W, "available_memory_bytes", lambda: 64 * 10**9), \
         mock.patch.dict(os.environ, {"LAUE_PREPROCESS_WORKERS": "banana"}):
        n, d = _choose(ceiling=None)
        assert n == 8
        assert d["bound_by"] != "LAUE_PREPROCESS_WORKERS"


def test_decision_explains_itself():
    """A surprising worker count must be explainable from the returned dict
    without re-deriving anything."""
    os.environ.pop("LAUE_PREPROCESS_WORKERS", None)
    n, d = _choose()
    for key in ("pixels", "usable_cpus", "available_memory_bytes",
                "per_worker_bytes", "queue_bytes", "bound_by", "workers"):
        assert key in d, f"decision is missing {key}"
    assert d["workers"] == n


# ---------------------------------------------------------------------------
# the server actually uses it
# ---------------------------------------------------------------------------

def test_config_schema_and_dataclass_agree():
    """laue_config.py warns in its own docstring that a schema key missing from
    the dataclass is parsed and then SILENTLY DROPPED. Pin the new key."""
    from laue_index import config_schema
    from laue_index.pipeline.laue_config import ImageProcessingConfig
    names = {p.field for p in config_schema.SCHEMA
             if p.target == "image_processing"}
    assert "preprocess_workers" in names, "PreprocessWorkers missing from SCHEMA"
    assert hasattr(ImageProcessingConfig, "preprocess_workers"), (
        "preprocess_workers is in the schema but not the dataclass -- it would "
        "be parsed and silently dropped")


def test_server_no_longer_hardcodes_eight():
    import inspect
    from laue_index.pipeline import laue_image_server as srv
    src = inspect.getsource(srv.serve_images)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "min(os.cpu_count() or 4, 8)" not in code, (
        "the hard-coded cap of 8 is back")
    assert "choose_preprocess_workers" in code
