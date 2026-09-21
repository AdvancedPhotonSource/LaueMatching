"""How many preprocessing workers this machine can actually feed.

``laue_image_server`` used to hard-code::

    PREPROCESS_WORKERS = min(os.cpu_count() or 4, 8)

Two things are wrong with that. The cap of 8 is a throughput ceiling on any
modern node, and ``os.cpu_count()`` reports the *machine's* CPUs, not the ones
this process is allowed to use -- under ``taskset``, a cpuset cgroup, a SLURM
allocation or a container CPU quota it over-reports, and the pool then
oversubscribes whatever slice we were actually given.

MEASURED, 2026-09-09, shannon (40 cores, idle), sampleH p99.8, 80 real frames per
point, ``OMP/MKL/OPENBLAS=1``, sparse-centroid preprocessing, POOL WARMED BEFORE
TIMING:

    workers     1     2     4     8    16    24    32    40
    frames/s 3.12  6.21 10.81 17.76 25.37 28.25 30.42 34.70
    canonical                  12.00 17.85             20.42

Still climbing at 40, which is this host's core count -- 8 -> 16 is +43% and
16 -> 40 a further +37%. The shipped cap of 8 gave away most of the machine:
canonical-at-8 is 12.00 f/s against patched-at-40's 34.70, a factor of 2.89.

An earlier version of this curve showed it flattening past 16, and that was an
artefact of the measurement, not the machine: ProcessPoolExecutor construction
was inside the timed region, and forking N workers (each importing numpy, scipy
and cv2) costs more the larger N is, so it taxed precisely the high-worker end.
``benchmark_workers`` now warms the pool before starting the clock. The flat
curve is what justified an absolute worker ceiling here; with the artefact
removed there is no evidence of flattening below the core count, so there is no
default ceiling any more.

Scaling is verified only to 40 = one host's core count. On a 64- or 96-core node
it may flatten earlier; ``benchmark_workers`` measures the machine in front of
you rather than extrapolating this curve onto it.

THE MEMORY MODEL IS FITTED, NOT GUESSED. Peak RSS of a forked worker
preprocessing one frame, measured at three frame sizes on the same host:

    frame        pixels     peak MB    bytes/pixel
    1024x1024   1048576        68.0          64.8
    2048x2048   4194304       207.4          49.4
    4096x4096  16777216       763.1          45.5

    least squares:  peak_bytes = 44.19 * pixels + 2.18e7
    residuals:      0.2, -0.2, 0.0 MB   (over a 16x range in pixels)

44.2 bytes/pixel is about 5.5 live float64 frames per worker, which is what the
pipeline holds at peak (background-subtracted, enhanced float32, thresholded,
uint16 copy, int32 labels, the float64 blur input and its output). The constant
term is the interpreter plus numpy/scipy/cv2.

The other term is the parent's, and on the streaming path it dominates: the
futures queue is ``FUTURES_QUEUE_DEPTH`` deep and each completed result carries
a dense float32 frame, so the parent can hold depth x 4 bytes x pixels of
finished work. At the shipped 512 x 2048^2 that is 8.6 GB, which is the bulk of
the ~17 GB RSS seen on a 13k-frame shard -- the workers were never the problem.

SIBLING SHARDS AND THE PROCESS LIMIT. Everything above sizes ONE server against
the whole host. That is wrong as soon as several shards share a host, and it
failed on a 112-core node with ``ulimit -u`` 8192: two shards each sized
themselves to 112 workers, each worker let OpenCV start its own thread pool,
and the user ran out of threads. OpenCV reported ``res = 11`` (EAGAIN), the GPU
daemon reported a misleading ``GPUassert: device busy or unavailable``, and the
dispatcher could no longer fork to launch the remaining shards. So the chooser
now also takes:

  * ``LAUE_SHARDS_PER_HOST``  -- how many servers share this host. The CPU
    count, the memory budget and the thread budget are each divided by it.
  * ``LAUE_DAEMON_NCPUS``     -- the co-resident daemon's ``ncpus``, ONLY when
    the caller sets it (nothing sets it automatically, so an unset variable is
    0.7.1 behaviour). Subtracted from this shard's CPU share, but never below
    HALF of that share: ``max(share - daemon, share // 2, 1)``. Core contention
    was not the failure being fixed (the thread limit was), so it may trim the
    pool, never collapse it. A warning says when the half-share floor binds.
  * ``RLIMIT_NPROC``          -- on Linux it counts THREADS per user, not
    processes. Half of it is budgeted to preprocessing, at
    ``THREADS_PER_WORKER`` threads per worker. Both numbers are stated
    assumptions, not measurements: the pool initializer pins the BLAS/OpenMP/
    OpenCV pools to one thread, but a worker still has its main thread and
    whatever a library starts that the initializer cannot reach.

``LAUE_PREPROCESS_WORKERS`` still wins over all of it (it is how an operator who
has measured the host says so), with a warning if it exceeds the thread budget.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "usable_cpu_count", "available_memory_bytes", "worker_peak_bytes",
    "choose_preprocess_workers", "describe_machine", "benchmark_workers",
    "BYTES_PER_PIXEL_PER_WORKER", "WORKER_BASE_BYTES",
    "THREADS_PER_WORKER", "NPROC_BUDGET_FRACTION",
]

logger = logging.getLogger("LaueStream")

# Fitted above. Deliberately module constants so a future re-measurement is a
# one-line change with the fit that justifies it sitting in the docstring.
BYTES_PER_PIXEL_PER_WORKER = 44.19
WORKER_BASE_BYTES = 21_800_000

# Fraction of available memory left alone: the OS page cache, the parent, the
# GPU daemon if it is co-resident, and the headroom to not be the process the
# OOM killer picks.
DEFAULT_MEMORY_RESERVE = 0.25

# No default ceiling. There was one (32), justified by a curve that appeared to
# flatten past 16 -- and that flattening turned out to be pool-construction cost
# inside the timed region. Warm, throughput was still climbing at the core count,
# so an absolute cap here would discard measured throughput on every large node.
# Memory and the usable CPU count are real bounds and still apply; a site that
# has measured its own knee sets PreprocessWorkers or LAUE_PREPROCESS_WORKERS.
DEFAULT_WORKER_CEILING = None

# Thread budget under RLIMIT_NPROC (see the module docstring). Assumptions, not
# fits: a worker is charged 4 threads even after its pools are pinned to one,
# and preprocessing may use half the per-user limit -- the rest belongs to the
# daemon, the parent's threads, the shell, and anything else the user runs.
THREADS_PER_WORKER = 4
NPROC_BUDGET_FRACTION = 0.5


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota() -> Optional[float]:
    """CPU quota in whole CPUs, or None if unlimited/absent.

    cgroup v2 writes "<quota> <period>" (or "max <period>") to cpu.max; v1
    splits it across two files and uses -1 for unlimited.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:              # v2
            quota_s, period_s = f.read().split()[:2]
        if quota_s != "max":
            period = int(period_s)
            if period > 0:
                return int(quota_s) / period
        return None
    except (OSError, ValueError, IndexError):
        pass
    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")    # v1
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and period and quota > 0:
        return quota / period
    return None


def usable_cpu_count() -> int:
    """CPUs this process may actually run on.

    ``sched_getaffinity`` covers taskset and cpuset cgroups; the CFS quota
    covers container CPU limits, which affinity does not show. Both are
    honoured, then ``os.cpu_count()`` as the floor of last resort.
    """
    n = None
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:
            n = None
    if not n:
        n = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None and quota > 0:
        # A fractional quota (e.g. 0.5 CPU) is still a limit: it means ONE
        # worker, not "ignore the quota and use every CPU on the host", which is
        # what the old `quota >= 1` test did.
        n = min(n, max(1, int(quota)))
    return max(1, n)


def available_memory_bytes() -> Optional[int]:
    """Memory we could plausibly use, or None if it cannot be determined.

    A container limit is what binds inside a container, and it is invisible in
    /proc/meminfo -- that file describes the host. Check the cgroup first and
    take the smaller of the two.
    """
    candidates = []

    # cgroup v2: memory.max may be "max"; subtract what the cgroup already uses.
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        if raw != "max":
            limit = int(raw)
            current = _read_int("/sys/fs/cgroup/memory.current") or 0
            candidates.append(max(limit - current, 0))
    except (OSError, ValueError):
        pass

    # cgroup v1. The "no limit" sentinel is a huge number, not -1.
    v1 = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 is not None and v1 < (1 << 62):
        used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0
        candidates.append(max(v1 - used, 0))

    # MemAvailable is the kernel's own estimate of what can be handed out
    # without swapping -- better than MemFree, which ignores reclaimable cache.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass

    if not candidates:
        # macOS and anything else without procfs.
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and size > 0:
                candidates.append(pages * size)
        except (ValueError, OSError, AttributeError):
            pass

    return min(candidates) if candidates else None


def _rlimit_nproc() -> Optional[int]:
    """Soft RLIMIT_NPROC, or None if unlimited or not available (Windows)."""
    try:
        import resource
        soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except (ImportError, AttributeError, ValueError, OSError):
        return None
    if soft == resource.RLIM_INFINITY or soft <= 0:
        return None
    return int(soft)


def _env_positive_int(name: str) -> Optional[int]:
    """A positive integer from the environment, or None (warning if malformed)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        v = 0
    if v < 1:
        logger.warning("Ignoring %s=%r: not a positive integer", name, raw)
        return None
    return v


def _env_nonneg_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        v = -1
    if v < 0:
        logger.warning("Ignoring %s=%r: not a non-negative integer", name, raw)
        return None
    return v


def worker_peak_bytes(pixels: int) -> int:
    """Peak RSS of one preprocessing worker for a frame of *pixels* pixels."""
    return int(BYTES_PER_PIXEL_PER_WORKER * pixels + WORKER_BASE_BYTES)


def choose_preprocess_workers(
    nr_px_x: int,
    nr_px_y: int,
    *,
    queue_depth: int = 0,
    max_workers: Optional[int] = None,
    memory_reserve: float = DEFAULT_MEMORY_RESERVE,
    ceiling: Optional[int] = DEFAULT_WORKER_CEILING,
    env_var: str = "LAUE_PREPROCESS_WORKERS",
    shards_per_host: Optional[int] = None,
    daemon_ncpus: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Pick a preprocessing worker count for this machine and this frame size.

    Bounded by, in order: an explicit override (which wins outright), this
    shard's share of the CPUs we may use less the co-resident daemon's
    ``ncpus``, this shard's share of the memory left after its parent's queue is
    paid for, this shard's share of the RLIMIT_NPROC thread budget, an optional
    caller cap, and an optional absolute ceiling (``None`` by default -- see
    ``DEFAULT_WORKER_CEILING`` for why there is no longer a default one).

    Args:
        nr_px_x, nr_px_y: frame dimensions -- the memory cost is per pixel, so
            a 4096^2 detector costs 4x what a 2048^2 one does per worker.
        queue_depth: how many finished float32 frames the PARENT may hold. This
            is charged before the workers, because on the streaming path it is
            the larger term. 0 to ignore.
        max_workers: caller's own cap (e.g. from the config file).
        memory_reserve: fraction of available memory left for everything else.
        ceiling: absolute cap; None to disable.
        env_var: environment override. Wins over everything, including memory,
            so an operator who knows their machine is never argued with.
        shards_per_host: servers sharing this host; CPUs, memory budget and
            thread budget are divided by it. None reads LAUE_SHARDS_PER_HOST
            (default 1).
        daemon_ncpus: the co-resident daemon's ncpus, subtracted from this
            shard's CPU share but never below half of it. None reads
            LAUE_DAEMON_NCPUS (unset: 0, nothing subtracted -- 0.7.1).

    Returns:
        ``(workers, decision)`` -- decision records every bound that was
        considered and which one actually bound, so a surprising number can be
        explained without re-deriving it.
    """
    pixels = int(nr_px_x) * int(nr_px_y)
    cpus = usable_cpu_count()
    avail = available_memory_bytes()
    per_worker = worker_peak_bytes(pixels)
    queue_bytes = int(queue_depth) * pixels * 4      # float32 on the wire
    if shards_per_host is None:
        shards_per_host = _env_positive_int("LAUE_SHARDS_PER_HOST") or 1
    shards = max(1, int(shards_per_host))
    if daemon_ncpus is None:
        daemon_ncpus = _env_nonneg_int("LAUE_DAEMON_NCPUS") or 0
    daemon_ncpus = max(0, int(daemon_ncpus))
    nproc = _rlimit_nproc()
    by_nproc = None
    if nproc is not None:
        by_nproc = max(1, int(nproc * NPROC_BUDGET_FRACTION)
                       // THREADS_PER_WORKER // shards)

    decision: Dict[str, Any] = {
        "pixels": pixels,
        "usable_cpus": cpus,
        "shards_per_host": shards,
        "daemon_ncpus": daemon_ncpus,
        "available_memory_bytes": avail,
        "per_worker_bytes": per_worker,
        "queue_bytes": queue_bytes,
        "rlimit_nproc": nproc,
        "workers_by_nproc": by_nproc,
        "bound_by": None,
    }

    override = os.environ.get(env_var, "").strip()
    if override:
        try:
            n = int(override)
            if n >= 1:
                decision["bound_by"] = env_var
                decision["workers"] = n
                if by_nproc is not None and n > by_nproc:
                    logger.warning(
                        "%s=%d exceeds this shard's thread budget of %d workers "
                        "(RLIMIT_NPROC %d, %d shard(s) per host, %d threads per "
                        "worker). Honoured because it was set explicitly; if the "
                        "pool fails to start threads (OpenCV 'res = 11'), lower it.",
                        env_var, n, by_nproc, nproc, shards, THREADS_PER_WORKER)
                return n, decision
            raise ValueError(override)
        except ValueError:
            logger.warning("Ignoring %s=%r: not a positive integer",
                           env_var, override)

    # This shard's share of the CPUs, less the daemon's fit threads.
    share = max(1, cpus // shards)
    n = max(share - daemon_ncpus, share // 2, 1)
    if daemon_ncpus and share - daemon_ncpus < share // 2:
        logger.warning(
            "LAUE_DAEMON_NCPUS=%d would leave %d of this shard's %d CPUs; "
            "keeping half the share (%d) instead.",
            daemon_ncpus, max(share - daemon_ncpus, 0), share, n)
    decision["cpus_for_workers"] = n
    decision["bound_by"] = "usable_cpus"

    if avail is not None:
        budget = int(avail * (1.0 - memory_reserve)) // shards - queue_bytes
        by_mem = budget // per_worker
        decision["memory_budget_bytes"] = budget
        decision["workers_by_memory"] = int(by_mem)
        if by_mem < 1:
            # The queue alone can exhaust the budget on a big detector. One
            # worker and a warning beats a pool that gets OOM-killed mid-scan.
            logger.warning(
                "Memory budget leaves no room for a preprocessing worker "
                "(available %.1f GB over %d shard(s), queue %.1f GB, worker "
                "%.1f GB). Using 1; reduce the futures queue depth or the "
                "frame size.",
                avail / 1e9, shards, queue_bytes / 1e9, per_worker / 1e9)
            decision["bound_by"] = "memory"
            decision["workers"] = 1
            return 1, decision
        if by_mem < n:
            n = int(by_mem)
            decision["bound_by"] = "memory"

    if by_nproc is not None and by_nproc < n:
        n = by_nproc
        decision["bound_by"] = "nproc"

    if max_workers is not None and max_workers >= 1 and max_workers < n:
        n = int(max_workers)
        decision["bound_by"] = "max_workers"

    if ceiling is not None and ceiling >= 1 and ceiling < n:
        n = int(ceiling)
        decision["bound_by"] = "ceiling"

    n = max(1, n)
    decision["workers"] = n
    return n, decision


def describe_machine(nr_px_x: int = 2048, nr_px_y: int = 2048,
                     queue_depth: int = 0) -> str:
    """One-screen summary of what this machine can feed. Used by the CLI."""
    n, d = choose_preprocess_workers(nr_px_x, nr_px_y, queue_depth=queue_depth)
    avail = d["available_memory_bytes"]
    lines = [
        "Preprocessing worker check",
        "  frame                %d x %d  (%d pixels)" % (nr_px_x, nr_px_y, d["pixels"]),
        "  usable CPUs          %d" % d["usable_cpus"],
        "  os.cpu_count()       %s" % (os.cpu_count(),),
        "  shards per host      %d   (LAUE_SHARDS_PER_HOST)" % d["shards_per_host"],
        "  daemon ncpus         %d   (LAUE_DAEMON_NCPUS)" % d["daemon_ncpus"],
        "  RLIMIT_NPROC         %s" % (
            "unlimited" if d["rlimit_nproc"] is None else
            "%d  -> at most %d workers" % (d["rlimit_nproc"], d["workers_by_nproc"])),
        "  available memory     %s" % (
            "%.1f GB" % (avail / 1e9) if avail is not None else "unknown"),
        "  per-worker peak      %.0f MB   (fitted: %.2f B/px + %.1f MB)" % (
            d["per_worker_bytes"] / 1e6, BYTES_PER_PIXEL_PER_WORKER,
            WORKER_BASE_BYTES / 1e6),
        "  parent queue         %.1f GB   (%d frames x 4 B/px)" % (
            d["queue_bytes"] / 1e9, queue_depth),
    ]
    if "workers_by_memory" in d:
        lines.append("  memory allows        %d workers" % d["workers_by_memory"])
    lines.append("  --> workers          %d   (bound by %s)" % (n, d["bound_by"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Measuring this machine, rather than extrapolating one host's curve
# ---------------------------------------------------------------------------
#
# choose_preprocess_workers() answers from CPU count, memory and the thread
# limit alone, which is what a server start-up can afford. There is no default
# ceiling any more (DEFAULT_WORKER_CEILING is None), so on a big node the CPU
# count decides -- and whether throughput is still climbing there was measured
# only up to 40 cores. benchmark_workers() measures the machine actually in
# front of you instead of extrapolating that curve onto it.

def _synthetic_frame(nr_px_x: int, nr_px_y: int, n_spots: int = 150,
                     seed: int = 0):
    """A frame with Laue-like statistics: sparse Gaussian spots on a noise floor.

    Only for TIMING. The spot positions are random, so nothing measured on this
    frame is a scientific result -- but preprocessing cost is driven by frame
    size and spot count, both of which this reproduces.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 5.0, size=(nr_px_y, nr_px_x))
    yy, xx = np.mgrid[-12:13, -12:13]
    kern = np.exp(-(yy ** 2 + xx ** 2) / (2 * 2.0 ** 2))
    for _ in range(n_spots):
        cy = int(rng.integers(20, nr_px_y - 20))
        cx = int(rng.integers(20, nr_px_x - 20))
        img[cy - 12:cy + 13, cx - 12:cx + 13] += kern * rng.uniform(500, 5000)
    return np.maximum(img, 0.0)


_BENCH_STATE: Dict[str, Any] = {}


def _bench_init(nr_px_x: int, nr_px_y: int, cfg: Dict[str, Any]) -> None:
    _BENCH_STATE["frame"] = _synthetic_frame(nr_px_x, nr_px_y)
    _BENCH_STATE["cfg"] = cfg
    import numpy as np
    _BENCH_STATE["bg"] = np.zeros((nr_px_y, nr_px_x), dtype=float)


def _bench_one(_i: int) -> int:
    import numpy as np
    from laue_index.preprocess import preprocess_image
    out = preprocess_image(_BENCH_STATE["frame"], _BENCH_STATE["cfg"],
                           background=_BENCH_STATE["bg"])
    return len(np.ascontiguousarray(out[0], dtype=np.float32).tobytes())


def benchmark_workers(
    nr_px_x: int = 2048,
    nr_px_y: int = 2048,
    frames_per_point: int = 40,
    worker_counts: Optional[list] = None,
    efficiency_target: float = 0.90,
) -> Tuple[int, list]:
    """Measure preprocessing throughput on THIS machine and recommend a count.

    The recommendation rule is stated rather than tuned: the SMALLEST worker
    count that reaches ``efficiency_target`` of the best throughput observed.
    Past that point you are paying memory and latency for a few percent.

    Returns ``(recommended, [(workers, seconds, frames_per_second), ...])``.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor

    cpus = usable_cpu_count()
    if worker_counts is None:
        worker_counts, w = [], 1
        while w < cpus:
            worker_counts.append(w)
            w *= 2
        worker_counts.append(cpus)
        worker_counts = sorted(set(worker_counts))

    cfg = {
        "nr_px_x": nr_px_x, "nr_px_y": nr_px_y,
        "threshold_method": "percentile", "threshold_value": 0.0,
        "threshold_percentile": 99.8, "min_area": 4,
        "px_x": 0.0002, "distance": 0.5134, "orientation_spacing": 0.4,
        "gauss_sigma_max": 2.5, "background_file": "",
        "filter_radius": 101, "median_passes": 1,
    }

    rows = []
    for w in worker_counts:
        with ProcessPoolExecutor(
                max_workers=w, initializer=_bench_init,
                initargs=(nr_px_x, nr_px_y, cfg)) as ex:
            # Warm the pool OUTSIDE the timed region. Forking w processes, each
            # importing numpy/scipy/cv2 and building a synthetic frame, costs
            # more the more workers there are -- timing it would bias against
            # exactly the high-worker end this function exists to resolve.
            list(ex.map(_bench_one, range(w)))
            t0 = time.perf_counter()
            list(ex.map(_bench_one, range(frames_per_point)))
            dt = time.perf_counter() - t0
        rows.append((w, dt, frames_per_point / dt))

    best = max(r[2] for r in rows)
    recommended = next(r[0] for r in rows if r[2] >= efficiency_target * best)
    return recommended, rows


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nr-px-x", type=int, default=2048)
    ap.add_argument("--nr-px-y", type=int, default=2048)
    ap.add_argument("--queue-depth", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true",
                    help="measure this machine instead of extrapolating the shipped curve")
    ap.add_argument("--frames", type=int, default=40,
                    help="frames per worker-count point (default 40)")
    a = ap.parse_args()
    print(describe_machine(a.nr_px_x, a.nr_px_y, a.queue_depth))
    if a.benchmark:
        print()
        print("Measuring this machine (synthetic frames, timing only)...")
        rec, rows = benchmark_workers(a.nr_px_x, a.nr_px_y, a.frames)
        print()
        print("%9s %10s %12s %11s" % ("workers", "wall s", "frames/s", "vs best"))
        best = max(r[2] for r in rows)
        for w, dt, fps in rows:
            print("%9d %10.2f %12.2f %10.0f%%" % (w, dt, fps, 100 * fps / best))
        print()
        print("recommended: %d workers "
              "(smallest count reaching 90%% of the best observed throughput)"
              % rec)
        print("set it with:  LAUE_PREPROCESS_WORKERS=%d   "
              "or  PreprocessWorkers %d  in the params file" % (rec, rec))
