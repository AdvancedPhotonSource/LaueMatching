"""laue_provenance.py — Provenance tracking for LaueMatching artifacts.

Single source of truth for stamping every generated file (HDF5 outputs,
CSV/TXT artifacts, sidecar JSON) with the information needed to reproduce
or audit it: git commit, timestamp, config snapshot, and fingerprints of
the input files that went into its creation.

Public API:

    collect(config=None, input_files=(), extra=None) -> dict
    write_to_h5(h5_obj, prov, group="provenance") -> None
    read_from_h5(h5_obj, group="provenance") -> dict
    header_lines(prov, comment="#") -> list[str]
    write_sidecar_json(path, prov) -> None
    file_fingerprint(path, strong=False) -> dict

File fingerprints default to a weak ``sha256_head`` (first 1 MiB + last
1 MiB + size) because full SHA-256 of the 6.7 GB orientation database
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
_PROVENANCE_SCHEMA_VERSION = "1"


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
) -> dict[str, Any]:
    """Gather a provenance dict.

    Arguments:
        config: ``LaueConfig`` instance, dict, or any object with ``to_dict()``.
        input_files: iterable of paths that contributed to the artifact being
            produced (e.g. config file, orientation DB, HKL list, source
            image). Each is fingerprinted.
        extra: caller-specific fields (e.g. ``{"n_orientations": 100_000_000}``).
        strong_hash: use full SHA-256 instead of weak head+tail hash.
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
    }
    snapshot = _config_snapshot(config)
    if snapshot is not None:
        prov["config"] = _sanitize_for_json(snapshot)
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
    lines.append(f"{comment}   laue_version:     {prov.get('laue_version', 'unknown')}")
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
