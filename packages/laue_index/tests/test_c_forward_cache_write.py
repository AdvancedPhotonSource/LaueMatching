"""Forward-cache writes: retried, fatal on failure, atomic, never half-published.

Each thread pwrites its own slab of the forward cache at its own offset. The
three binaries used to handle a failed write three different ways -- the CPU
loop spun forever on a 0-byte return, LaueMatchingGPU printed and carried on,
the stream daemon ignored the return value -- and in every case, as in any run
killed mid-write, ForwardFile could be left with a zero-filled hole while other
threads' slabs extended it to exactly the size `forwardCacheUsable()` accepts.
A later `DoFwd 0` run then read the hole as "no spots", silently.

All three now go through the shared helpers in LaueMatchingHeaders.h, with the
standard atomic pattern: write `<ForwardFile>.partial.<pid>` (same directory),
retry short writes, fsync + close, and only then `rename()` onto ForwardFile.
Any failure removes the partial file and exits non-zero; a prior ForwardFile
is never touched. A KILLED run leaves only a `.partial.<pid>` leftover, which
`forwardCacheUsable()` never opens.

The behaviour tests drive the real functions (fixture `forward_slab_write.c`),
each against a prior ForwardFile holding "OLD". A genuine short write is not
reproducible here without fault injection; the loop is covered by the source
check only.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent / "c_src"
FIXTURE = HERE / "fixtures" / "forward_slab_write.c"
CPU, GPU, STREAM, HEADERS = ("LaueMatchingCPU.c", "LaueMatchingGPU.cu",
                             "LaueMatchingGPUStream.cu", "LaueMatchingHeaders.h")

_LIBOMP_PREFIXES = ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")


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


@pytest.fixture(scope="module")
def fsw(tmp_path_factory):
    if not (C_DIR / HEADERS).is_file():
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    exe = str(tmp_path_factory.mktemp("fsw") / "forward_slab_write")
    err = "no C compiler found"
    for cc in _compilers():
        for cflags, ldflags in _omp_flag_sets():
            cmd = ([cc, "-std=gnu99", "-O2"] + cflags + [f"-I{C_DIR}", "-o", exe,
                   str(FIXTURE)] + ldflags + ["-lm"])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return exe
            err = proc.stderr[-600:]
    pytest.skip(f"cannot build the fixture (needs OpenMP headers): {err}")


def _fsw(fsw, mode, out):
    return subprocess.run([fsw, mode, str(out)], capture_output=True, text=True,
                          timeout=60)


def _partials(out):
    return sorted(out.parent.glob(out.name + ".partial.*"))


def _says_old(out):
    return out.is_file() and out.read_bytes() == b"OLD"


def test_publishes_atomically_only_after_finish(fsw, tmp_path):
    """The partial name exists during the write, the final name keeps its prior
    content until finish, then holds the new slab and the partial is gone."""
    out = tmp_path / "cache.bin"
    r = _fsw(fsw, "ok", out)
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr
    words = r.stdout.split("PARTIAL ", 1)[1].split()
    partial, pid = words[0], words[2]
    # Built independently of the printed name: realpath + host + the pid the
    # fixture reports for itself.
    want = f"{os.path.realpath(out)}.partial.{os.uname().nodename}.{pid}"
    assert partial == want, (partial, want)
    assert not _partials(out)


def test_failed_write_removes_the_partial_and_keeps_the_prior_file(fsw, tmp_path):
    out = tmp_path / "cache.bin"
    r = _fsw(fsw, "fail", out)
    assert r.returncode != 0, "a failed forward-cache write must not return"
    assert "returned without exiting" not in r.stdout
    assert "FATAL: thread 7 forward-cache pwrite failed" in r.stderr
    assert "Removed the partial forward cache" in r.stderr
    assert not _partials(out), "the partial file was left behind"
    assert _says_old(out), "abandon touched the prior ForwardFile"


def test_killed_run_leaves_only_a_partial_leftover(fsw, tmp_path):
    """The SIGKILL case: no handler runs, so nothing is cleaned up -- but the
    final name was never written, so no later run can accept a holed cache."""
    out = tmp_path / "cache.bin"
    r = _fsw(fsw, "killed", out)
    assert r.returncode != 0
    assert _says_old(out), "a killed run changed ForwardFile"
    assert len(_partials(out)) == 1, "expected exactly one .partial.<pid> leftover"


def test_empty_forwardfile_is_refused_before_anything_is_created(fsw, tmp_path):
    before = set(tmp_path.iterdir())
    r = subprocess.run([fsw, "empty", "unused"], capture_output=True, text=True,
                       timeout=60, cwd=str(tmp_path))
    assert r.returncode == 0 and "REFUSED" in r.stdout, r.stdout + r.stderr
    assert "no ForwardFile specified" in r.stderr
    assert set(tmp_path.iterdir()) == before, "a file was created for an empty name"


def test_the_writer_retries_and_treats_zero_as_failure():
    """Source check for what the fixture cannot provoke: the short-write retry
    and rc == 0 (the CPU loop used to spin forever on it)."""
    src = _read(HEADERS)
    start = src.index("static inline void writeForwardSlabOrDie(")
    body = src[start:src.index("\n}\n", start)]
    assert "while (done < nbytes)" in body
    assert "if (rc <= 0)" in body, "rc == 0 must be a failure, not a retry"
    assert "done += (size_t)rc;" in body
    assert "abandonForwardCache(partialfn);" in body
    assert src.count("static inline void writeForwardSlabOrDie(") == 1
    start = src.index("static inline void abandonForwardCache(")
    abandon = src[start:src.index("\n}\n", start)]
    assert "unlink(partialfn)" in abandon and "exit(EXIT_FAILURE)" in abandon
    start = src.index("static inline int beginForwardCacheWrite(")
    opener = src[start:src.index("\n}\n", start)]
    assert '"%s.partial.%s.%ld"' in opener and "getpid()" in opener
    assert "un.nodename" in opener, "the partial name must carry the host"
    assert "O_CREAT | O_EXCL | O_WRONLY" in opener and "O_TRUNC" not in opener, (
        "never truncate an existing partial: it may be a live writer's")


@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_every_binary_writes_the_cache_only_through_the_checked_writer(name):
    src = _read(name)
    assert "pwrite(" not in src, (
        f"{name}: a direct pwrite bypasses writeForwardSlabOrDie()")
    assert len(re.findall(r"writeForwardSlabOrDie\(fwdFd,", src)) == 1, name
    assert re.search(r"writeForwardSlabOrDie\([^;]*procNr, fc\.partial\);", src), (
        f"{name}: slabs must be written to the partial file")


# ── fsync / close: durability failures take the write-failure path ────────

def test_finish_publishes_the_partial(fsw, tmp_path):
    out = tmp_path / "cache.bin"
    r = _fsw(fsw, "finish", out)
    assert r.returncode == 0 and "FINISHED" in r.stdout, r.stdout + r.stderr
    assert "FATAL" not in r.stderr
    assert out.read_bytes() == b"x" and not _partials(out)


def test_failed_fsync_is_fatal_and_publishes_nothing(fsw, tmp_path):
    """fsync EBADF stands in for EIO/ENOSPC: any failure outside the
    'not supported' set must be treated like a failed write."""
    out = tmp_path / "cache.bin"
    r = _fsw(fsw, "finish-badfd", out)
    assert r.returncode != 0
    assert "returned without exiting" not in r.stdout
    assert "FATAL: fsync of the forward cache" in r.stderr
    assert not _partials(out), "the undurable partial was left behind"
    assert _says_old(out), "an undurable cache was published"


def test_fsync_unsupported_only_warns_and_still_publishes(fsw, tmp_path):
    """A pipe: Linux fsync(2) returns EINVAL, macOS ENOTSUP -- 'fsync is not
    this fd's contract'. That must WARN and carry on, not abort. (A platform
    whose pipe fsync simply succeeds prints nothing; also acceptable.)"""
    out = tmp_path / "cache.bin"
    r = _fsw(fsw, "finish-pipe", out)
    assert r.returncode == 0 and "FINISHED" in r.stdout, r.stdout + r.stderr
    assert "FATAL" not in r.stderr
    assert out.read_bytes() == b"x" and not _partials(out)
    if r.stderr:
        assert "WARNING: fsync is not supported" in r.stderr


def test_the_finisher_policy_in_source():
    src = _read(HEADERS)
    start = src.index("static inline void finishForwardCacheOrDie(")
    body = src[start:src.index("\n}\n", start)]
    assert "while (rc != 0 && errno == EINTR)" in body, "EINTR must be retried"
    for e in ("EINVAL", "EROFS", "ENOTSUP", "EOPNOTSUPP"):
        assert e in body, f"{e} missing from the 'fsync not supported' set"
    assert body.count("abandonForwardCache(partialfn);") == 3, (
        "a failed fsync, close and rename must each abandon the partial")
    assert body.index("rename(partialfn, outfn)") > body.index("close(fd)"), (
        "publish only after the partial is durable and closed")


@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_every_binary_finishes_the_cache_through_the_checked_finisher(name):
    src = _read(name)
    assert "fsync(" not in src and "close(fwdFd)" not in src, (
        f"{name}: a direct fsync/close of the cache bypasses "
        f"finishForwardCacheOrDie()")
    fin = "finishForwardCacheOrDie(fwdFd, fc.partial, fc.target);"
    assert src.count(fin) == 1, name
    # The lock is released right after publication, and only there.
    assert src.count("releaseForwardCacheLock(&fc);") == 1, name
    assert src.index("releaseForwardCacheLock(&fc);") > src.index(fin), name
    # The per-thread allocation failure, reached after the cache is opened,
    # must also abandon it rather than plain-exit.
    assert src.count("abandonForwardCache(fc.partial);") == 1, name
    # Nothing opens the FINAL name for writing any more.
    assert "O_CREAT" not in src, f"{name}: writes a file other than via the partial"
    assert src.count("beginForwardCacheWrite(outfn,") == 1, name
    assert "if (fwdFd == FWD_CACHE_REUSE)" in src, (
        f"{name}: a sibling's fresh publication must switch to the read path")


@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_nothing_reads_forwardfile_by_name_while_this_run_writes_it(name):
    """Between beginForwardCacheWrite() and the publishing rename, no code may
    open ForwardFile by name: it would see the prior (possibly stale) file or
    nothing. LaueMatchingGPUStream is the one binary that re-reads by name
    after simulating, so that read must come after the finish; CPU and GPU only
    read by name on the DoFwd 0 path. Checked for EVERY by-name read."""
    src = _read(name)
    begin = src.index("beginForwardCacheWrite(outfn,")
    fin = src.index("finishForwardCacheOrDie(fwdFd, fc.partial, fc.target);")
    reads = [m.start() for m in re.finditer(
        r"\b(open|fopen)\(outfn,", src)]
    assert reads, f"{name}: no by-name read found; the check would be vacuous"
    inside = [p for p in reads if begin < p < fin]
    # CPU.c's DoFwd 0 setup sits textually between begin and finish but is
    # guarded by `if (doFwd == 0)`, which beginForwardCacheWrite only enters
    # when it did NOT open a partial (FWD_CACHE_REUSE). Allow reads there only.
    for p in inside:
        guard = src.rfind("if (doFwd", begin, p)
        cond = src[guard:src.index("\n", guard)] if guard != -1 else ""
        assert "doFwd == 0" in cond, (
            f"{name}: ForwardFile read by name at offset {p} while writing; "
            f"nearest doFwd condition: {cond!r}")
    if name == STREAM:
        assert all(p > fin for p in reads if p > begin), (
            "the stream daemon reads the cache back before it is published")


@pytest.mark.parametrize("name", [CPU, GPU, STREAM])
def test_the_usability_check_runs_before_any_write(name):
    """An existing, possibly stale ForwardFile is judged by forwardCacheUsable()
    before the partial is opened, never while a write is in progress."""
    src = _read(name)
    assert src.index("forwardCacheUsable(outfn,") < src.index(
        "beginForwardCacheWrite(outfn,")


# ── one writer at a time: the sibling-shard case ─────────────────────────

def _sim(fsw, out, ms=0):
    return subprocess.Popen([fsw, "sim", str(out), str(ms)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def test_sibling_waits_for_the_lock_then_reads_instead_of_simulating(fsw, tmp_path):
    """Two processes cold-start on one ForwardFile. The first simulates while
    holding <out>.lock; the second waits, re-checks, finds the published cache
    and reuses it. Only ONE partial ever exists, so peak space is 1x."""
    out = tmp_path / "cache.bin"
    a = _sim(fsw, out, 1500)
    first = a.stdout.readline()
    assert first.startswith("SIMULATING") and "locked=1" in first, first
    b = _sim(fsw, out, 0)
    b_out, b_err = b.communicate(timeout=60)
    a_rest, a_err = a.communicate(timeout=60)
    assert a.returncode == 0 and "PUBLISHED" in a_rest, a_rest + a_err
    assert b.returncode == 0, b_out + b_err
    assert "waiting for" in b_out and "REUSED" in b_out, b_out + b_err
    assert "SIMULATING" not in b_out, "the sibling simulated anyway"
    assert out.stat().st_size == 40 and not _partials(out)


def test_explicit_resimulation_is_not_short_circuited(fsw, tmp_path):
    """DoFwd 1 over an existing right-sized cache must still simulate: the
    re-check only accepts a file that CHANGED while this run waited."""
    out = tmp_path / "cache.bin"
    out.write_bytes(b"\0" * 40)
    r = subprocess.run([fsw, "sim", str(out), "0"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0 and "SIMULATING" in r.stdout, r.stdout + r.stderr
    assert "REUSED" not in r.stdout
    assert out.read_bytes() != b"\0" * 40


def test_the_lock_holder_deletes_dead_writers_partials(fsw, tmp_path):
    out = tmp_path / "cache.bin"
    dead = tmp_path / "cache.bin.partial.deadhost.1"
    dead.write_bytes(b"\0" * 4096)
    r = subprocess.run([fsw, "sim", str(out), "0"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not dead.exists() and "left by a dead writer" in r.stdout


def test_symlinked_forwardfile_is_replaced_at_its_target(fsw, tmp_path):
    big = tmp_path / "bigvolume"
    big.mkdir()
    target = big / "fwd.bin"
    target.write_bytes(b"OLD")
    link = tmp_path / "fwd.bin"
    link.symlink_to(target)
    r = subprocess.run([fsw, "sim", str(link), "0"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert link.is_symlink(), "the symlink was replaced by a regular file"
    assert target.stat().st_size == 40
    assert not list(tmp_path.glob("fwd.bin.partial.*")), "partial in the link's dir"
    assert not list(big.glob("fwd.bin.partial.*"))


def test_published_cache_keeps_the_replaced_files_mode(fsw, tmp_path):
    out = tmp_path / "cache.bin"
    out.write_bytes(b"OLD")
    os.chmod(out, 0o640)
    r = subprocess.run([fsw, "sim", str(out), "0"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out.stat().st_mode & 0o777) == 0o640


def test_new_cache_is_group_and_world_readable_subject_to_umask(fsw, tmp_path):
    """The old open() created the cache 0600, locking other beamline accounts
    out of a shared cache; it is now 0644 before the umask."""
    out = tmp_path / "cache.bin"
    old = os.umask(0o022)
    try:
        r = subprocess.run([fsw, "sim", str(out), "0"], capture_output=True,
                           text=True, timeout=60)
    finally:
        os.umask(old)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out.stat().st_mode & 0o777) == 0o644


def test_abandon_is_single_entry():
    """Concurrent exit() from several failing writer threads is undefined in
    POSIX; abandonForwardCache admits one thread through a named critical."""
    src = _read(HEADERS)
    start = src.index("static inline void abandonForwardCache(")
    body = src[start:src.index("\n}\n", start)]
    assert "#pragma omp critical(laue_abandon_forward_cache)" in body
    crit = body.index("#pragma omp critical")
    assert crit < body.index("unlink(partialfn)") < body.index("exit(EXIT_FAILURE)")
