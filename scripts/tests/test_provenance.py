"""Tests for scripts/laue_provenance.py.

Run:
    cd ~/opt/LaueMatching && python -m pytest scripts/tests/test_provenance.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable regardless of where pytest is invoked from.
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import laue_provenance as lp  # noqa: E402


def test_weak_fingerprint_changes_on_content_change(tmp_path: Path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"hello world" * 100)
    fp1 = lp.file_fingerprint(a)
    a.write_bytes(b"hello world" * 100 + b"!")
    fp2 = lp.file_fingerprint(a)
    assert fp1["sha256_head"] != fp2["sha256_head"]
    assert fp1["size"] != fp2["size"]


def test_weak_fingerprint_changes_on_tail_change(tmp_path: Path):
    # 3 MiB file so the tail-window actually differs from the head-window.
    path = tmp_path / "big.bin"
    path.write_bytes(b"A" * (3 << 20))
    fp1 = lp.file_fingerprint(path)

    # Overwrite last 1 KiB only — the weak hash covers head+tail so this must
    # flip the digest. ``truncate`` would change the size too, which is also
    # detected, so we overwrite in place to isolate the tail read.
    with open(path, "r+b") as fh:
        fh.seek(-1024, os.SEEK_END)
        fh.write(b"Z" * 1024)
    fp2 = lp.file_fingerprint(path)
    assert fp1["size"] == fp2["size"]
    assert fp1["sha256_head"] != fp2["sha256_head"]


def test_strong_hash_matches_shasum(tmp_path: Path):
    path = tmp_path / "c.bin"
    path.write_bytes(b"0123456789" * 2048)
    fp = lp.file_fingerprint(path, strong=True)
    # Reference via hashlib
    import hashlib
    ref = hashlib.sha256(path.read_bytes()).hexdigest()
    assert fp["sha256"] == ref
    assert "sha256_head" not in fp


def test_missing_file_yields_missing_marker(tmp_path: Path):
    fp = lp.file_fingerprint(tmp_path / "does_not_exist")
    assert fp["missing"] is True


def test_collect_contains_required_keys():
    prov = lp.collect(config={"foo": "bar"}, input_files=[], extra={"n_items": 42})
    for k in ("schema_version", "timestamp_utc", "host", "user", "git",
              "laue_version", "script", "config", "inputs", "extra"):
        assert k in prov, f"missing key {k}"
    assert prov["config"]["foo"] == "bar"
    assert prov["extra"]["n_items"] == 42


def test_collect_config_to_dict(monkeypatch):
    class FakeCfg:
        def to_dict(self):
            return {"space_group": 225, "symmetry": "F"}

    prov = lp.collect(config=FakeCfg())
    assert prov["config"] == {"space_group": 225, "symmetry": "F"}


def test_collect_git_matches_rev_parse():
    """Smoke-test: running inside this repo must return a real commit SHA."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not expected:
        pytest.skip("not inside a git repo")
    prov = lp.collect()
    assert prov["git"]["commit"] == expected


def test_header_lines_include_commit_and_inputs(tmp_path: Path):
    inp = tmp_path / "in.bin"
    inp.write_bytes(b"xx")
    prov = lp.collect(input_files=[inp])
    lines = lp.header_lines(prov, comment="#")
    joined = "\n".join(lines)
    assert "provenance" in joined.lower()
    assert "git_commit" in joined
    assert inp.name in joined
    # Must end with the machine-parseable JSON line
    assert lines[-1].startswith("# provenance_json:")
    payload = lines[-1].split("provenance_json:", 1)[1].strip()
    json.loads(payload)  # parseable


def test_sidecar_json_roundtrip(tmp_path: Path):
    prov = lp.collect(config={"k": "v"})
    out = lp.write_sidecar_json(tmp_path / "prov.json", prov)
    loaded = json.loads(out.read_text())
    assert loaded["config"] == {"k": "v"}
    assert "timestamp_utc" in loaded


def test_prepend_header_to_text_file(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("1 2 3\n4 5 6\n")
    lp.prepend_header_to_text_file(p, ["# hello", "# world"])
    content = p.read_text().splitlines()
    assert content[0] == "# hello"
    assert content[1] == "# world"
    assert content[2] == "1 2 3"
    assert content[3] == "4 5 6"


def test_h5_write_read_roundtrip(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    prov = lp.collect(
        config={"space_group": 225, "symmetry": "F"},
        extra={"scalar": 7, "nested": {"a": 1, "b": [2, 3]}},
    )
    with h5py.File(tmp_path / "t.h5", "w") as fh:
        lp.write_to_h5(fh, prov, group="provenance")
    with h5py.File(tmp_path / "t.h5", "r") as fh:
        loaded = lp.read_from_h5(fh, group="provenance")
    assert loaded["timestamp_utc"] == prov["timestamp_utc"]
    assert loaded["config"] == {"space_group": 225, "symmetry": "F"}
    assert loaded["extra"]["nested"]["b"] == [2, 3]


def test_h5_write_overwrites_existing(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "t.h5", "w") as fh:
        lp.write_to_h5(fh, lp.collect(config={"v": 1}))
        lp.write_to_h5(fh, lp.collect(config={"v": 2}))
    with h5py.File(tmp_path / "t.h5", "r") as fh:
        loaded = lp.read_from_h5(fh)
    assert loaded["config"]["v"] == 2


def test_sanitize_handles_enum_and_numpy():
    np = pytest.importorskip("numpy")
    from enum import Enum

    class Color(Enum):
        RED = 1

    obj = {"c": Color.RED, "arr": np.array([1, 2]), "scalar": np.float64(3.5)}
    out = lp._sanitize_for_json(obj)
    assert out["c"] == "RED"
    assert out["arr"] == [1, 2]
    assert out["scalar"] == 3.5
    json.dumps(out)  # fully JSON-safe


def test_git_helpers_tolerate_non_repo(tmp_path: Path):
    # Tell _git to run from a non-repo dir; should return empty strings.
    out = lp._collect_git(cwd=tmp_path)
    assert out["commit"] == "unknown"
    assert out["dirty"] is False
