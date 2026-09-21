"""Provenance records WHICH CODE RAN (schema 2).

Schema 1 recorded a git commit that is always "unknown" on a pip install and a
``laue_version`` string that never moved. The ``build`` block (added this
release) records the laue_index version, the CMake manifest's
``c_src_sha256``, a full SHA-256 per binary beside the default one, and --
because the streaming orchestrator can run a daemon from
``<project_root>/build/`` instead -- the binary the caller actually executed.
``processing_type`` in the config snapshot is a config label ("CPU" by default
even on a GPUStream run), so it is annotated rather than trusted.
"""
import hashlib
import json
from pathlib import Path

import pytest

import laue_index
import laue_provenance as lp
from laue_index import indexer

_MANIFEST = {"version": "9.9.9", "c_src_sha256": "ab" * 32, "schema": 1}


@pytest.fixture
def fake_install(tmp_path, monkeypatch):
    """A bin/ with the CPU and GPUStream binaries present and GPU missing, and a
    monkeypatched build manifest."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "LaueMatchingCPU").write_bytes(b"cpu-binary")
    (bindir / "LaueMatchingGPUStream").write_bytes(b"stream-binary")
    monkeypatch.setattr(indexer, "binary_path",
                        lambda *a, **k: bindir / "LaueMatchingCPU")
    monkeypatch.setattr(laue_index, "build_info", lambda: dict(_MANIFEST))
    return bindir


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_schema_version_is_two():
    assert lp.collect()["schema_version"] == "2"


def test_build_block_from_manifest(fake_install):
    b = lp.collect()["build"]
    assert b["laue_index_version"] == laue_index.__version__
    assert b["c_src_sha256"] == "ab" * 32
    assert b["manifest_version"] == "9.9.9"
    assert set(b["binaries"]) == set(lp._BINARIES)
    assert b["binaries"]["LaueMatchingCPU"]["sha256"] == \
        _sha(fake_install / "LaueMatchingCPU")
    assert b["binaries"]["LaueMatchingGPUStream"]["sha256"] == \
        _sha(fake_install / "LaueMatchingGPUStream")
    assert b["binaries"]["LaueMatchingGPU"]["missing"] is True


def test_executable_is_recorded_with_its_own_hash(fake_install, tmp_path):
    """A daemon outside the default bin/ (e.g. <project_root>/build/) must be
    recorded as what ran, with its own hash -- not a neighbour of the CPU
    binary."""
    elsewhere = tmp_path / "build"
    elsewhere.mkdir()
    exe = elsewhere / "LaueMatchingGPUStream"
    exe.write_bytes(b"a-different-stream-build")
    b = lp.collect(executable=exe)["build"]
    assert b["executable"]["kind"] == "LaueMatchingGPUStream"
    assert b["executable"]["sha256"] == _sha(exe)
    assert b["executable"]["sha256"] != b["binaries"]["LaueMatchingGPUStream"]["sha256"]


def test_header_lines_print_the_build(fake_install, tmp_path):
    exe = fake_install / "LaueMatchingGPUStream"
    lines = "\n".join(lp.header_lines(lp.collect(executable=exe)))
    assert "c_src_sha256:     " + "ab" * 32 in lines
    assert f"laue_index:       {laue_index.__version__}" in lines
    assert "executable:       LaueMatchingGPUStream" in lines
    assert _sha(exe) in lines
    for name in lp._BINARIES:
        assert f"binary {name}:" in lines
    assert "binary LaueMatchingGPU: missing" in lines


def test_processing_type_is_flagged_as_a_config_label():
    prov = lp.collect(config={"processing_type": "CPU", "space_group": 225})
    assert prov["config"]["processing_type"] == "CPU"      # recorded as given
    assert "config label" in prov["config_notes"]["processing_type"]
    assert "config_notes" not in lp.collect(config={"space_group": 225})


def test_orchestrator_records_the_daemon_it_launches(fake_install, tmp_path,
                                                      monkeypatch):
    """run_pipeline resolves the daemon BEFORE stamping provenance and records
    it; launching is stubbed to stop right after the stamp."""
    import laue_orchestrator as lo
    daemon = tmp_path / "build" / "LaueMatchingGPUStream"
    daemon.parent.mkdir()
    daemon.write_bytes(b"project-root-build-daemon")
    monkeypatch.setattr(lo, "_find_daemon_binary", lambda: str(daemon))

    class _Stop(Exception):
        pass

    real_popen = lo.subprocess.Popen

    def _no_launch(cmd, *a, **k):
        # Only the daemon launch is stopped; provenance's own `git` calls run.
        if cmd and cmd[0] == str(daemon):
            raise _Stop()
        return real_popen(cmd, *a, **k)

    monkeypatch.setattr(lo.subprocess, "Popen", _no_launch)
    params = tmp_path / "params.txt"
    params.write_text("SpaceGroup 225\nResultDir results_stream\n")
    (tmp_path / "frames").mkdir()
    out = tmp_path / "run"
    with pytest.raises(_Stop):
        lo.run_pipeline(config_file=str(params), folder=str(tmp_path / "frames"),
                        orient_file=str(tmp_path / "o.bin"),
                        hkl_file=str(tmp_path / "h.bin"), output_dir=str(out))
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["extra"]["daemon_bin"] == str(daemon)
    assert prov["build"]["executable"]["sha256"] == _sha(daemon)
    assert prov["schema_version"] == "2"
