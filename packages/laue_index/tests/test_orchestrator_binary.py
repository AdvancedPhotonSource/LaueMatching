"""The streaming orchestrator finds the daemon wherever the package put it.

`_find_daemon_binary` used to search build/, bin/ and the repo root on its own,
so a user who had just run `LAUEMATCHING_CUDA=1 pip install laue-index` -- and
therefore HAD LaueMatchingGPUStream, in site-packages -- was told to "build it
first". Discovery now goes through laue_index.indexer, which is the one place
that knows every location, including LAUEMATCHING_BIN.
"""
from __future__ import annotations

import pytest

from laue_index import indexer

laue_orchestrator = pytest.importorskip(
    "laue_orchestrator", reason="orchestrator deps (h5py/numpy) not installed")


def _fake_daemon(d):
    d.mkdir(parents=True, exist_ok=True)
    exe = d / "LaueMatchingGPUStream"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    return exe


def test_env_var_is_honoured(tmp_path, monkeypatch):
    exe = _fake_daemon(tmp_path / "elsewhere")
    monkeypatch.setenv(indexer.BINARY_ENV, str(exe))
    assert laue_orchestrator._find_daemon_binary() == str(exe)


def test_the_installed_package_is_found(tmp_path, monkeypatch):
    """What `LAUEMATCHING_CUDA=1 pip install` produces: site-packages/laue_index/bin."""
    exe = _fake_daemon(tmp_path / "site-packages" / "laue_index" / "bin")
    monkeypatch.delenv(indexer.BINARY_ENV, raising=False)
    monkeypatch.setattr(indexer, "_candidates",
                        lambda name, repo_root=None: [exe.parent / name])
    assert laue_orchestrator._find_daemon_binary() == str(exe)


def test_the_error_names_every_path_and_the_escape_hatch(tmp_path, monkeypatch):
    missing = tmp_path / "nowhere" / "LaueMatchingGPUStream"
    monkeypatch.delenv(indexer.BINARY_ENV, raising=False)
    monkeypatch.setattr(indexer, "_candidates",
                        lambda name, repo_root=None: [missing])
    with pytest.raises(FileNotFoundError) as e:
        laue_orchestrator._find_daemon_binary()
    msg = str(e.value)
    assert str(missing) in msg, "must say where it looked"
    assert indexer.BINARY_ENV in msg, "must say how to override"
    assert "LAUEMATCHING_CUDA=1" in msg, "must say the CUDA build is opt-in"
    assert "cmake --build build/" not in msg, (
        "the old advice was useless to a pip user, who has no build directory")
