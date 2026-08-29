"""§6.5 — Indexer stage (C-binary wrapper) unit checks.

The execution path itself is covered end-to-end by test_char_e2e (which calls
RunImage -> run_indexer -> the real binary).  Here we pin the cheap, pure bits:
executable selection, the search order, and the missing-binary guard.

NOTE on the missing-binary test. It used to pass a nonexistent repo_root and
assume that meant "no binary". That stopped being true the moment the C started
being compiled into the package at `pip install` time: on CI the binary IS
present in site-packages, so run_indexer found it, ran it on fake inputs, and
failed with exit code 1 instead of "not found". The absent condition now has to
be constructed, not assumed.
"""
import os

import pytest

from laue_index import indexer
from laue_index.indexer import (BINARY_ENV, available, binary_path,
                                resolve_executable, run_indexer)


# ── selection ───────────────────────────────────────────────────────────────

def test_resolve_executable_cpu_default():
    assert resolve_executable("/repo", "CPU", do_forward=False).endswith("LaueMatchingCPU")


def test_resolve_executable_gpu():
    assert resolve_executable("/repo", "GPU", do_forward=False).endswith("LaueMatchingGPU")


def test_resolve_executable_gpu_with_forward_falls_back_to_cpu():
    # DoFwd has no GPU path -> CPU binary.
    assert resolve_executable("/repo", "GPU", do_forward=True).endswith("LaueMatchingCPU")


def test_stream_names_the_daemon():
    """The streaming orchestrator asks for this by name; it is not CPU-able."""
    for alias in ("STREAM", "stream", "GPUStream"):
        assert indexer._executable_name(alias) == "LaueMatchingGPUStream"


def test_stream_never_falls_back_to_the_single_image_binary():
    """A caller that wants the daemon must be told it is missing, not handed
    a binary that cannot serve a socket."""
    assert indexer._executable_name("STREAM", do_forward=True) == "LaueMatchingGPUStream"


# ── search order ────────────────────────────────────────────────────────────

def test_env_var_wins_and_accepts_a_file_or_a_directory(tmp_path, monkeypatch):
    exe = tmp_path / "LaueMatchingCPU"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)

    monkeypatch.setenv(BINARY_ENV, str(exe))
    assert binary_path() == exe and available()

    monkeypatch.setenv(BINARY_ENV, str(tmp_path))      # a directory
    assert binary_path() == exe and available()


def test_repo_root_is_still_searched(tmp_path, monkeypatch):
    """The historical location must keep working for a source checkout."""
    monkeypatch.delenv(BINARY_ENV, raising=False)
    monkeypatch.setattr(indexer, "_candidates",
                        lambda name, repo_root=None: [tmp_path / "bin" / name])
    (tmp_path / "bin").mkdir()
    exe = tmp_path / "bin" / "LaueMatchingCPU"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    assert available(repo_root=str(tmp_path))


# ── the guard ───────────────────────────────────────────────────────────────

def _no_candidates_anywhere(monkeypatch, tmp_path):
    """Force the genuinely-absent case, whatever is installed on this machine."""
    missing = tmp_path / "definitely" / "not" / "here" / "LaueMatchingCPU"
    monkeypatch.setattr(indexer, "_candidates",
                        lambda name, repo_root=None: [missing])
    return missing


def test_available_is_false_when_the_binary_is_absent(tmp_path, monkeypatch):
    _no_candidates_anywhere(monkeypatch, tmp_path)
    assert available() is False


def test_binary_path_returns_a_diagnosable_path_when_absent(tmp_path, monkeypatch):
    """Even with nothing found, the caller gets a concrete path to report."""
    missing = _no_candidates_anywhere(monkeypatch, tmp_path)
    assert binary_path() == missing


def test_require_binary_raises_with_the_full_diagnosis(tmp_path, monkeypatch):
    """One diagnosis, shared by every caller that cannot proceed without it."""
    missing = _no_candidates_anywhere(monkeypatch, tmp_path)
    with pytest.raises(indexer.BinaryUnavailableError) as e:
        indexer.require_binary()
    msg = str(e.value)
    assert str(missing) in msg and BINARY_ENV in msg


def test_require_binary_returns_the_path_when_it_is_there(tmp_path, monkeypatch):
    exe = tmp_path / "LaueMatchingCPU"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    monkeypatch.setenv(BINARY_ENV, str(exe))
    assert indexer.require_binary() == exe


def test_a_missing_cuda_binary_says_how_to_get_one(tmp_path, monkeypatch):
    """The CUDA build is opt-in, so 'not found' is not the whole story."""
    monkeypatch.setattr(indexer, "_candidates",
                        lambda name, repo_root=None: [tmp_path / "nope" / name])
    with pytest.raises(indexer.BinaryUnavailableError) as e:
        indexer.require_binary("STREAM")
    assert "LAUEMATCHING_CUDA=1" in str(e.value)


def test_run_indexer_missing_binary_names_where_it_looked(tmp_path, monkeypatch):
    missing = _no_candidates_anywhere(monkeypatch, tmp_path)
    res = run_indexer(
        repo_root="/nonexistent_repo_xyz", config_file="c.txt",
        orient_db_file="o.bin", hkl_file="h.csv", image_bin="i.bin",
        ncpus=1, output_path=str(tmp_path / "_out"))
    assert res.success is False
    err = res.error or ""
    # The old message advised running `make` in a build directory that does not
    # exist for a pip user. The new one must name the paths tried and the escape
    # hatch, so the error is actionable rather than merely true.
    assert "not found" in err
    assert str(missing) in err
    assert BINARY_ENV in err
    assert res.returncode is None, "should not have tried to execute anything"
