"""laue_provenance.py — Provenance tracking for LaueMatching artifacts.

Single source of truth for stamping every generated file (HDF5 outputs,
CSV/TXT artifacts, sidecar JSON) with the information needed to reproduce
or audit it: git commit, timestamp, config snapshot, and fingerprints of
the input files that went into its creation.

Public API:

    collect(config=None, input_files=(), extra=None, *, executable=None) -> dict
    write_to_h5(h5_obj, prov, group="provenance") -> None
    read_from_h5(h5_obj, group="provenance") -> dict
    header_lines(prov, comment="#") -> list[str]
    write_sidecar_json(path, prov) -> None
    file_fingerprint(path, strong=False) -> dict

File fingerprints default to a weak ``sha256_head`` (first 1 MiB + last
1 MiB + size) because full SHA-256 of the 7.2 GB (6.7 GiB) orientation database
takes ~40 s. Pass ``strong=True`` for a true SHA-256.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import socket
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from ._version import __version__ as _LAUE_VERSION  # package-relative
except ImportError:
    try:
        from _version import __version__ as _LAUE_VERSION  # script-relative
    except ImportError:
        _LAUE_VERSION = "unknown"

logger = logging.getLogger("LaueMatching")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEAK_HASH_WINDOW = 1 << 20  # 1 MiB head + tail
# Schema history:
#   "1"  git, laue_version (the stale pipeline/_version.py string), config,
#        inputs, extra.
#   "2"  adds ``build``: laue_index version, the CMake manifest and its
#        c_src_sha256, a full SHA-256 of each binary beside the default one, and
#        -- when the caller says which binary it ran -- ``build.executable``
#        (kind + path + SHA-256 of THAT file). Adds ``config_notes`` when the
#        config snapshot carries ``processing_type``, which is a config label
#        (RunImage's -g flag; "CPU" by default, including on a GPUStream run),
#        not a record of the binary.
_PROVENANCE_SCHEMA_VERSION = "2"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: Sequence[str], cwd: Path = _REPO_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _collect_git(cwd: Path = _REPO_ROOT) -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"], cwd)
    if not commit:
        return {
            "commit": "unknown",
            "dirty": False,
            "branch": "unknown",
            "remote": "unknown",
        }
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or "unknown"
    remote = _git(["remote", "get-url", "origin"], cwd) or "unknown"
    dirty = bool(_git(["status", "--porcelain"], cwd))
    return {
        "commit": commit,
        "commit_short": commit[:12],
        "dirty": dirty,
        "branch": branch,
        "remote": remote,
    }


# ---------------------------------------------------------------------------
# Build identity -- WHICH CODE RAN
# ---------------------------------------------------------------------------
#
# Added 2026-09-21. Before this, every provenance record on a pip-installed
# pipeline read `git.commit: "unknown"` and `laue_version: "2.2.0"`:
#
#   * `git rev-parse` runs in THIS package's directory, which is a git checkout
#     only for an editable/dev install. For `pip install laue-index` -- the
#     production case -- it is site-packages, so the commit is always unknown.
#   * `laue_version` is `pipeline/_version.py`, a hand-maintained string that
#     stayed at 2.2.0 across at least 84 commits and several laue-index releases.
#
# So two indexing runs of the same scan a month apart, which differed in ~3% of
# their solutions, could not be attributed to a build from their own metadata --
# the question "which code produced the July sampleH result" was unanswerable, and
# rebuilding it was not possible. The information existed the whole time: CMake
# writes `_build_info.json` (see laue_index.buildmeta) with the installed version
# and a SHA-256 of the C sources. It was just never read here.
#
# Both identities are recorded, because they answer different questions:
#
#   c_src_sha256    "was it built from the same SOURCE?"  Stable across rebuilds.
#   binary sha256   "is it the same EXECUTABLE?"          NOT stable across
#                   rebuilds of identical source: nvcc embeds a PID-derived
#                   tmpxft_<pid> filename, so two CUDA builds of unchanged code
#                   differ. Use c_src_sha256 to compare builds; use the binary
#                   hash only to prove two runs used the very same file.

_BINARIES = ("LaueMatchingCPU", "LaueMatchingGPU", "LaueMatchingGPUStream")


def _executable_record(executable: str | os.PathLike) -> dict[str, Any]:
    """Kind + full fingerprint of the binary a caller actually ran.

    The ``binaries`` block hashes whatever sits beside ``indexer.binary_path()``
    (the CPU binary's directory). The streaming orchestrator can run a daemon
    from ``<project_root>/build/`` instead, so without this the binary that ran
    and the binary recorded could differ.
    """
    p = Path(executable)
    rec = file_fingerprint(p, strong=True)
    rec["kind"] = p.name
    try:
        rec["realpath"] = str(p.resolve())
    except OSError:
        pass
    return rec


def _collect_build(executable: str | os.PathLike | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if executable:
        # First, so it is recorded even when laue_index itself is not importable.
        out["executable"] = _executable_record(executable)
    try:
        import laue_index  # noqa: WPS433 -- deliberate lazy import
        out["laue_index_version"] = getattr(laue_index, "__version__", "unknown")
    except Exception as exc:  # the pipeline can run outside the package
        out["laue_index_version"] = "unknown"
        out["error"] = f"laue_index not importable: {exc}"
        return out
    try:
        info = laue_index.build_info()
        out["manifest"] = info
        out["c_src_sha256"] = info.get("c_src_sha256", "unknown")
        out["manifest_version"] = info.get("version", "unknown")
    except Exception as exc:
        out["manifest"] = {"available": False, "reason": str(exc)}
        out["c_src_sha256"] = "unknown"
    bins: dict[str, Any] = {}
    try:
        from laue_index import indexer
        bindir = Path(indexer.binary_path()).resolve().parent
        for name in _BINARIES:
            p = bindir / name
            if p.is_file():
                # ~1 MB each, so a full hash costs nothing
                bins[name] = file_fingerprint(p, strong=True)
            else:
                bins[name] = {"path": str(p), "missing": True}
        out["bindir"] = str(bindir)
    except Exception as exc:
        bins["error"] = str(exc)
    out["binaries"] = bins
    return out


# ---------------------------------------------------------------------------
# File fingerprinting
# ---------------------------------------------------------------------------

def _weak_hash(path: Path, size: int) -> str:
    """SHA-256 over (head || tail || size). ``size`` is included so a
    zero-padded or truncated file produces a different digest even if the
    head and tail happen to collide.
    """
    h = hashlib.sha256()
    window = min(_WEAK_HASH_WINDOW, size)
    try:
        with open(path, "rb") as fh:
            head = fh.read(window)
            h.update(head)
            if size > window:
                fh.seek(-window, os.SEEK_END)
                tail = fh.read(window)
                h.update(tail)
    except OSError as exc:
        logger.warning("weak-hash read failed for %s: %s", path, exc)
        return "unreadable"
    h.update(struct.pack("<Q", size))
    return h.hexdigest()


def _strong_hash(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError as exc:
        logger.warning("strong-hash read failed for %s: %s", path, exc)
        return "unreadable"
    return h.hexdigest()


def file_fingerprint(path: str | os.PathLike, strong: bool = False) -> dict[str, Any]:
    """Return ``{path, basename, size, mtime_utc, sha256_head|sha256}``.

    Missing files yield ``{"path": ..., "missing": True}``.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "missing": True}
    st = p.stat()
    out: dict[str, Any] = {
        "path": str(p),
        "basename": p.name,
        "size": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }
    if strong:
        out["sha256"] = _strong_hash(p)
    else:
        out["sha256_head"] = _weak_hash(p, st.st_size)
    return out


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def _config_snapshot(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception as exc:
            logger.warning("config.to_dict() failed: %s", exc)
    # Fall back to dataclass __dict__ if available
    if hasattr(config, "__dict__"):
        return {k: v for k, v in vars(config).items() if not k.startswith("_")}
    return None


# ---------------------------------------------------------------------------
# Top-level collector
# ---------------------------------------------------------------------------

def collect(
    config: Any = None,
    input_files: Iterable[str | os.PathLike] = (),
    extra: Mapping[str, Any] | None = None,
    *,
    strong_hash: bool = False,
    executable: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Gather a provenance dict.

    Arguments:
        config: ``LaueConfig`` instance, dict, or any object with ``to_dict()``.
        input_files: iterable of paths that contributed to the artifact being
            produced (e.g. config file, orientation DB, HKL list, source
            image). Each is fingerprinted.
        extra: caller-specific fields (e.g. ``{"n_orientations": 100_000_000}``).
        strong_hash: use full SHA-256 instead of weak head+tail hash.
        executable: the indexer binary this run actually executes (daemon or
            single-image). Recorded under ``build.executable`` with its kind and
            full SHA-256. Pass it whenever it is known.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    prov: dict[str, Any] = {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "timestamp_utc": now,
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": sys.version.split()[0],
        "laue_version": _LAUE_VERSION,
        "script": {
            "argv0": sys.argv[0] if sys.argv else "",
            "argv": list(sys.argv),
        },
        "git": _collect_git(),
        # WHICH CODE RAN. See _collect_build: on a pip install `git` above is
        # always "unknown" and `laue_version` is a stale string -- this is the
        # field that actually identifies the build.
        "build": _collect_build(executable),
    }
    snapshot = _config_snapshot(config)
    if snapshot is not None:
        prov["config"] = _sanitize_for_json(snapshot)
        if "processing_type" in snapshot:
            prov["config_notes"] = {
                "processing_type": (
                    "config label only (RunImage -g sets it; the default is "
                    "'CPU' even on a GPUStream run). The binary that ran is "
                    "build.executable, when the caller recorded it."),
            }
    prov["inputs"] = [file_fingerprint(p, strong=strong_hash) for p in input_files]
    if extra:
        prov["extra"] = _sanitize_for_json(dict(extra))
    return prov


def _sanitize_for_json(obj: Any) -> Any:
    """Convert non-JSON-serializable values (Path, Enum, ndarray, etc.) to
    JSON-friendly forms. Keeps dicts/lists recursive.
    """
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Enum
    name = getattr(obj, "name", None)
    if name is not None and hasattr(obj, "value"):
        return name
    # Path-like
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    # numpy scalar / array — avoid hard import
    mod = type(obj).__module__
    if mod.startswith("numpy"):
        try:
            return obj.item() if hasattr(obj, "item") and getattr(obj, "shape", None) == () else obj.tolist()
        except Exception:
            return repr(obj)
    return repr(obj)


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_to_h5(h5_obj: Any, prov: Mapping[str, Any], group: str = "provenance") -> Any:
    """Write the provenance dict into ``h5_obj`` under ``group``.

    ``h5_obj`` may be an ``h5py.File`` or any ``h5py.Group``. The resulting
    subgroup stores scalar fields as string attributes and nested dicts/lists
    as JSON-encoded string datasets.

    Returns the created group (for chaining / further attribute writes).
    """
    import h5py  # local import so tests can skip if h5py missing

    if group in h5_obj:
        del h5_obj[group]
    g = h5_obj.create_group(group)

    for key, value in prov.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            g.attrs[key] = "null" if value is None else value
        else:
            # Nested dict/list → JSON string dataset
            payload = json.dumps(_sanitize_for_json(value), indent=2, sort_keys=True)
            g.create_dataset(key, data=payload)
    g.attrs["schema_version"] = prov.get("schema_version", _PROVENANCE_SCHEMA_VERSION)
    return g


def read_from_h5(h5_obj: Any, group: str = "provenance") -> dict[str, Any]:
    """Inverse of :func:`write_to_h5`."""
    if group not in h5_obj:
        return {}
    g = h5_obj[group]
    out: dict[str, Any] = {}
    for key, val in g.attrs.items():
        out[key] = val.decode() if isinstance(val, bytes) else val
    for key in g.keys():
        raw = g[key][()]
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            out[key] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            out[key] = raw
    return out


# ---------------------------------------------------------------------------
# Text/CSV header lines
# ---------------------------------------------------------------------------

def header_lines(prov: Mapping[str, Any], comment: str = "#") -> list[str]:
    """Return commented header lines suitable for prepending to CSV/TXT files.

    Keeps the important scalars human-readable on dedicated lines and drops
    the full JSON at the end so automated tools can parse it.
    """
    lines: list[str] = []
    git = prov.get("git", {}) or {}
    lines.append(f"{comment} LaueMatching provenance (schema {prov.get('schema_version', _PROVENANCE_SCHEMA_VERSION)})")
    lines.append(f"{comment}   generated_at_utc: {prov.get('timestamp_utc', 'unknown')}")
    build = prov.get("build", {}) or {}
    lines.append(f"{comment}   laue_index:       {build.get('laue_index_version', 'unknown')}")
    lines.append(f"{comment}   c_src_sha256:     {build.get('c_src_sha256', 'unknown')}")
    exe = build.get("executable") or {}
    if exe:
        lines.append(f"{comment}   executable:       {exe.get('kind', '?')} "
                     f"{exe.get('path', '?')}"
                     + ("  (MISSING)" if exe.get("missing") else ""))
        if exe.get("sha256"):
            lines.append(f"{comment}   executable sha256: {exe['sha256']}")
    for name, rec in sorted((build.get("binaries") or {}).items()):
        if not isinstance(rec, Mapping):
            continue
        if rec.get("sha256"):
            lines.append(f"{comment}   binary {name}: sha256={rec['sha256']}")
        elif rec.get("missing"):
            lines.append(f"{comment}   binary {name}: missing ({rec.get('path')})")
    lines.append(f"{comment}   laue_version:     {prov.get('laue_version', 'unknown')}  "
                 f"(stale script string -- use laue_index / c_src_sha256)")
    lines.append(f"{comment}   git_commit:       {git.get('commit', 'unknown')}{'  (dirty)' if git.get('dirty') else ''}")
    lines.append(f"{comment}   git_branch:       {git.get('branch', 'unknown')}")
    lines.append(f"{comment}   host:             {prov.get('host', 'unknown')}")
    lines.append(f"{comment}   user:             {prov.get('user', 'unknown')}")
    script = prov.get("script", {}) or {}
    if script.get("argv"):
        lines.append(f"{comment}   argv:             {' '.join(script['argv'])}")
    for inp in prov.get("inputs", []) or []:
        if inp.get("missing"):
            lines.append(f"{comment}   input (missing):  {inp.get('path')}")
        else:
            digest = inp.get("sha256") or inp.get("sha256_head", "")
            lines.append(
                f"{comment}   input:            {inp.get('basename')} "
                f"size={inp.get('size')} sha256_head={digest[:16]}"
            )
    lines.append(f"{comment} ---")
    lines.append(f"{comment} provenance_json: {json.dumps(_sanitize_for_json(dict(prov)), sort_keys=True)}")
    return lines


# ---------------------------------------------------------------------------
# Sidecar JSON
# ---------------------------------------------------------------------------

def write_sidecar_json(path: str | os.PathLike, prov: Mapping[str, Any]) -> Path:
    """Write ``prov`` as pretty-printed JSON to ``path``. Returns the Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(_sanitize_for_json(dict(prov)), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return p


def prepend_header_to_text_file(path: str | os.PathLike, lines: Sequence[str]) -> None:
    """Prepend ``lines`` (already comment-prefixed) to an existing text file.

    Reads the current content, writes header + content. Intended for files
    generated by ``np.savetxt``/``fprintf`` where the generator does not
    offer a header hook.
    """
    p = Path(path)
    existing = p.read_text() if p.exists() else ""
    with open(p, "w") as fh:
        for line in lines:
            fh.write(line.rstrip("\n") + "\n")
        fh.write(existing)
