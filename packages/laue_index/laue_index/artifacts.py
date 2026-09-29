"""Provenance records for LaueMatching's data artifacts.

The orientation database, HKL lists, forward caches and backgrounds are bare
binary (or plain-text) files. Before 0.8.0 nothing said which code, which
inputs or which configuration produced one, so a right-sized file from another
campaign was accepted silently: a forward cache built for a different detector
(RUNBOOK open item 10), a background computed from whatever frame came first
(invariant 16), an HKL list generated for another lattice. Each artifact now
carries a JSON record beside it, ``<file>.meta.json``:

    {"schema": "laue-artifact/1", "kind": "orientation_db" | "hkl_list" |
                                          "forward_cache" | "background",
     "retroactive": false,
     "artifact": {"path", "basename", "size", "quick", "sha256", "layout"},
     "config":   {... the FULL configuration that generated it: every argument
                  or parameter-file key, e.g. the orientation spacing, the
                  HKL generator's lattice and detector, the forward cache's
                  sample/detector/energy settings ...},
     "config_file": {"path", "sha256", "text"}   (when a file was used),
     "inputs":   [{"role", "path", "size", "quick", "sha256", "record"}],
     "extra":    {...},
     "producer": {... laue_provenance.collect(): host, user, time, build ...}}

``quick`` is SHA-256 over the first and last MiB and the size (cheap on a 12 GB
cache); ``sha256`` is the full hash, computed when the record is written and
checked on request (``full=True``).

POLICY (2026-09-28): a MISSING record warns; a record that DISAGREES with the
file (size, hash, kind) or with the configuration (``expect``) refuses, by
raising :class:`ArtifactMismatch` from :func:`require`.

The forward cache's record is written by the C binaries themselves
(``LaueMatchingHeaders.h``, ``writeForwardCacheMeta``) in this same schema.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union

from .pipeline.laue_provenance import _strong_hash, _weak_hash, collect

__all__ = ["SCHEMA", "SIDECAR_SUFFIX", "KINDS", "ORIENT_DB_SHA256", "ArtifactMismatch",
           "Check", "sidecar_path", "write_record", "read_record", "check", "require",
           "summary", "read_params", "check_run_inputs", "HKL_CRYSTAL_KEYS",
           "HKL_GEOMETRY_KEYS"]

logger = logging.getLogger(__name__)

SCHEMA = "laue-artifact/1"
SIDECAR_SUFFIX = ".meta.json"
KINDS = frozenset({"orientation_db", "hkl_list", "forward_cache", "background"})

#: Full SHA-256 of the released 100MilOrients.bin (GitHub release v1.0-data,
#: reassembled from its four parts). Measured 2026-09-28, identical on two
#: independent copies: the Mac checkout and a beamline-host copy.
ORIENT_DB_SHA256 = "351dc8e0dec0db91aff67493bf749957663e9926dda55db145205a1e0915fd20"

#: Sidecars written before 0.8.0 (GenerateHKLs). Read so a user can see them,
#: never trusted to vouch for the bytes (they carry no artifact hash).
_LEGACY_SUFFIXES = (".provenance.json",)

PathLike = Union[str, os.PathLike]


class ArtifactMismatch(RuntimeError):
    """A record exists and disagrees with the file or with the configuration."""


@dataclass
class Check:
    status: str                      # "ok" | "missing" | "mismatch"
    reasons: list = field(default_factory=list)
    record: dict | None = None


def sidecar_path(path: PathLike) -> Path:
    return Path(str(path) + SIDECAR_SUFFIX)


def _fingerprint(path: Path, full: bool) -> dict[str, Any]:
    st = path.stat()
    out = {"path": str(path.resolve()), "basename": path.name, "size": st.st_size,
           "quick": _weak_hash(path, st.st_size)}
    if full:
        out["sha256"] = _strong_hash(path)
    return out


def _input_entry(role: str, path: PathLike) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"role": role, "path": str(p), "missing": True}
    entry = {"role": role, **_fingerprint(p, full=False)}
    rec = read_record(p)
    if rec is not None and not rec.get("legacy"):
        entry["record"] = True
        sha = rec.get("artifact", {}).get("sha256")
        if sha:
            entry["sha256"] = sha
    else:
        entry["record"] = False
    return entry


def write_record(path: PathLike, kind: str, *, layout: Mapping[str, Any] | None = None,
                 config: Mapping[str, Any] | None = None,
                 config_file: PathLike | None = None,
                 inputs: Iterable[Tuple[str, PathLike]] = (),
                 extra: Mapping[str, Any] | None = None, full_hash: bool = True,
                 retroactive: bool = False) -> dict[str, Any]:
    """Write ``<path>.meta.json`` for an existing artifact and return the record.

    ``config`` is the full generating configuration (every argument / key);
    ``config_file``, when the artifact was made from a parameter file, is
    recorded with its hash and full text, so the record stands alone even if
    the file is later edited or lost. ``inputs`` are ``(role, path)`` pairs; each is fingerprinted and, when it
    has a record of its own, carries that record's full hash (lineage).
    Written atomically (temp file + rename) so a reader never sees half a record.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {sorted(KINDS)}")
    p = Path(path)
    artifact = _fingerprint(p, full=full_hash)
    if layout:
        artifact["layout"] = dict(layout)
    prod = collect()
    prod.pop("inputs", None)
    record = {"schema": SCHEMA, "kind": kind, "retroactive": bool(retroactive),
              "artifact": artifact, "config": dict(config or {}),
              "inputs": [_input_entry(r, ip) for r, ip in inputs],
              "extra": dict(extra or {}), "producer": prod}
    if config_file is not None:
        cf = Path(config_file)
        try:
            text = cf.read_text(errors="replace")
            record["config_file"] = {"path": str(cf.resolve()), "sha256": _strong_hash(cf),
                                     "text": text}
        except OSError as exc:
            record["config_file"] = {"path": str(cf), "error": str(exc)}
    out = sidecar_path(p)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, out)
    return record


def read_record(path: PathLike) -> dict | None:
    """The record for ``path``, a legacy sidecar marked ``legacy: True``, or None."""
    sc = sidecar_path(path)
    if sc.is_file():
        try:
            return json.loads(sc.read_text())
        except (OSError, ValueError) as exc:
            return {"schema": "unreadable", "error": str(exc)}
    for suf in _LEGACY_SUFFIXES:
        lp = Path(str(path) + suf)
        if lp.is_file():
            try:
                rec = json.loads(lp.read_text())
            except (OSError, ValueError):
                rec = {}
            rec["legacy"] = True
            rec["legacy_path"] = str(lp)
            return rec
    return None


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        fa, fb = float(a), float(b)
        return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def check(path: PathLike, kind: str | None = None, *,
          expect: Mapping[str, Any] | None = None, full: bool = False) -> Check:
    """Compare ``path`` with its record (and the record with ``expect``)."""
    p = Path(path)
    if not p.exists():
        return Check("mismatch", [f"{p} does not exist"])
    rec = read_record(p)
    if rec is None:
        return Check("missing", [f"{p} has no provenance record ({sidecar_path(p).name})"])
    if rec.get("legacy"):
        return Check("missing", [f"{p} has only a pre-0.8.0 sidecar "
                                 f"({Path(rec['legacy_path']).name}), which cannot vouch for "
                                 f"its bytes"], rec)
    if rec.get("schema") != SCHEMA:
        return Check("mismatch", [f"{sidecar_path(p).name}: unreadable or unknown schema "
                                  f"{rec.get('schema')!r}"], rec)
    reasons = []
    if kind is not None and rec.get("kind") != kind:
        reasons.append(f"kind is {rec.get('kind')!r}, expected {kind!r}")
    art = rec.get("artifact", {})
    size = p.stat().st_size
    if art.get("size") != size:
        reasons.append(f"size {size} B, record says {art.get('size')} B")
    elif art.get("quick") and art["quick"] != _weak_hash(p, size):
        reasons.append("contents differ from the record (head/tail hash)")
    if full and not reasons and art.get("sha256"):
        if _strong_hash(p) != art["sha256"]:
            reasons.append("contents differ from the record (full SHA-256)")
    cfg = rec.get("config", {})
    for k, v in (expect or {}).items():
        if k in cfg and not _same(cfg[k], v):
            reasons.append(f"{k}: record {cfg[k]!r}, configuration {v!r}")
    return Check("mismatch" if reasons else "ok", reasons, rec)


def require(path: PathLike, kind: str | None = None, *,
            expect: Mapping[str, Any] | None = None, full: bool = False,
            log: logging.Logger | None = None) -> Check:
    """:func:`check`, with the policy: warn when missing, raise when mismatched."""
    c = check(path, kind, expect=expect, full=full)
    lg = log or logger
    if c.status == "missing":
        lg.warning("%s: %s (it is used unverified; `laue-index provenance stamp` records it)",
                   kind or "artifact", "; ".join(c.reasons))
    elif c.status == "mismatch":
        raise ArtifactMismatch(f"{path}: " + "; ".join(c.reasons))
    return c


def summary(path: PathLike) -> dict[str, Any]:
    """Compact identity of an artifact for a run's lineage block."""
    p = Path(path)
    out: dict[str, Any] = {"path": str(p)}
    if not p.exists():
        out["missing"] = True
        return out
    out["size"] = p.stat().st_size
    rec = read_record(p)
    if rec is None or rec.get("legacy"):
        out["record"] = False
        out["quick"] = _weak_hash(p, out["size"])
        return out
    art = rec.get("artifact", {})
    out.update(record=True, kind=rec.get("kind"), quick=art.get("quick"),
               sha256=art.get("sha256"))
    if rec.get("kind") == "forward_cache":
        out["key"] = rec.get("config", {}).get("key")
    return out


# ---------------------------------------------------------------------------
# What a run checks before it launches the indexer
# ---------------------------------------------------------------------------

#: An HKL list for another crystal is always wrong: refuse.
HKL_CRYSTAL_KEYS = ("SpaceGroup", "Symmetry", "LatticeParameter")
#: A list made for a slightly different detector or energy window differs only
#: in which reflections reach the panel edges (a recalibration does this):
#: warn, do not refuse.
HKL_GEOMETRY_KEYS = ("P_Array", "R_Array", "NrPxX", "NrPxY", "PxX", "PxY", "Elo", "Ehi")


def read_params(path: PathLike) -> dict[str, Any]:
    """A parameter file as {key: value}, keys matched on the whole first token
    (as the C does). Numbers are parsed; one token -> scalar, several -> list."""
    out: dict[str, Any] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        toks = line.split("#", 1)[0].split()
        if not toks:
            continue
        vals = []
        for t in toks[1:]:
            try:
                vals.append(int(t) if t.lstrip("-").isdigit() else float(t))
            except ValueError:
                vals.append(t)
        out[toks[0]] = vals[0] if len(vals) == 1 else vals
    return out


def check_run_inputs(params_file: PathLike, *, orient_db: PathLike | None = None,
                     hkl: PathLike | None = None, forward: PathLike | None = None,
                     background: PathLike | None = None,
                     log: logging.Logger | None = None) -> dict[str, Any]:
    """Check a run's data artifacts against their records and the params file.

    Refuses (ArtifactMismatch) when a record disagrees with its file, or when
    the HKL list was made for another crystal; warns when a record is missing
    or the HKL list was made for a different detector / energy window. Returns
    the lineage block (identity of every artifact used) for the run's
    provenance. The forward cache is checked by the C itself; here it is only
    summarised.
    """
    lg = log or logger
    pf = Path(params_file)
    params = read_params(pf)
    lineage: dict[str, Any] = {"params_file": {"path": str(pf.resolve()),
                                               "sha256": _strong_hash(pf)}}
    if hkl is not None:
        c = require(hkl, "hkl_list",
                    expect={k: params[k] for k in HKL_CRYSTAL_KEYS if k in params}, log=lg)
        if c.status == "ok":
            cfg = c.record.get("config", {})
            diff = [f"{k}: list {cfg[k]!r}, params {params[k]!r}" for k in HKL_GEOMETRY_KEYS
                    if k in cfg and k in params and cfg[k] is not None
                    and not _same(cfg[k], params[k])]
            if diff:
                lg.warning("HKL list %s was generated for a different detector or energy "
                           "window (%s); reflections near the panel edges may be missing or "
                           "extra. Regenerate it if that matters.", hkl, "; ".join(diff))
        lineage["hkl_list"] = summary(hkl)
    if orient_db is not None:
        require(orient_db, "orientation_db", log=lg)
        lineage["orientation_db"] = summary(orient_db)
    if background is not None and Path(background).exists():
        require(background, "background", log=lg)
        lineage["background"] = summary(background)
    if forward is not None and Path(forward).exists():
        lineage["forward_cache"] = summary(forward)
    return lineage
