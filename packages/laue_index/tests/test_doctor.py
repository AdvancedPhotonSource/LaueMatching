"""Tests for the build manifest and `laue-index doctor`.

These pin the contract that makes attempting CUDA by default safe: the build
records what it did, and doctor can tell the two silent states apart -- no GPU
binary on a CUDA host, and a GPU binary that cannot launch on the card in front
of it.
"""
from __future__ import annotations

import json

import pytest

from laue_index import buildmeta as bi
from laue_index import doctor


# ── architecture parsing ────────────────────────────────────────────────────
@pytest.mark.parametrize("spec,cubin,ptx", [
    ("75-real;80-real;120-virtual", {75, 80}, {120}),
    ("80-real", {80}, set()),
    ("90", {90}, {90}),                 # CMake shorthand: real AND virtual
    ("", set(), set()),
    ("all-major", set(), set()),        # not parseable, and must not pretend
])
def test_parse_architectures(spec, cubin, ptx):
    assert doctor.parse_architectures(spec) == (cubin, ptx)


def test_a_cubin_for_the_exact_arch_can_launch():
    ok, why = doctor.arch_supported("8.6", "75-real;86-real;120-virtual")
    assert ok and "cubin for sm_86" in why


def test_ptx_below_the_device_can_jit_forward():
    """PTX JIT is forward-only: sm_80 PTX runs on an sm_90 card."""
    ok, why = doctor.arch_supported("9.0", "70-real;80-real;80-virtual")
    assert ok and "JIT from PTX sm_80" in why


def test_ptx_above_the_device_cannot_jit_backward():
    """The failure that prints 'Unique Orientations: 0' and exits 0. PTX for a
    NEWER arch than the card does not help; only a cubin or lower PTX does."""
    ok, why = doctor.arch_supported("8.6", "75-real;80-real;120-virtual")
    assert not ok
    assert "cannot launch" in why and "exits 0" in why.replace("exit 0", "exits 0")


def test_no_cubin_and_no_ptx_at_all_cannot_launch():
    ok, _ = doctor.arch_supported("9.0", "80-real")
    assert not ok


def test_an_unparseable_arch_list_is_reported_as_unknown_not_broken():
    """`all-major` emits no PTX but we cannot enumerate it here -- say so rather
    than raise a false alarm. A probe that reports a false FAIL costs the same
    investigation as a real one."""
    ok, why = doctor.arch_supported("8.6", "all-major")
    assert ok and "cannot check" in why


# ── the manifest ────────────────────────────────────────────────────────────
def test_a_missing_manifest_is_reported_not_faked(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "manifest_path", lambda: tmp_path / "nope.json")
    info = bi.build_info()
    assert info["available"] is False
    assert "predates" in info["reason"] or "no _build_info.json" in info["reason"]


def test_an_unreadable_manifest_does_not_raise(monkeypatch, tmp_path):
    bad = tmp_path / "_build_info.json"
    bad.write_text("{not json")
    monkeypatch.setattr(bi, "manifest_path", lambda: bad)
    assert bi.build_info()["available"] is False


def test_a_real_manifest_round_trips(monkeypatch, tmp_path):
    good = tmp_path / "_build_info.json"
    good.write_text(json.dumps({
        "schema": 1, "version": "0.6.0", "c_src_sha256": "abc123",
        "cuda": {"built": True, "architectures": "80-real;90-virtual"}}))
    monkeypatch.setattr(bi, "manifest_path", lambda: good)
    assert bi.build_info()["version"] == "0.6.0"
    assert bi.c_src_sha256() == "abc123"


# ── diagnose ────────────────────────────────────────────────────────────────
def _stub(monkeypatch, tmp_path, *, devices, binaries, cuda):
    bindir = tmp_path / "bin"; bindir.mkdir(exist_ok=True)
    for b in binaries:
        (bindir / b).write_text("#!/bin/sh\n")
    monkeypatch.setattr(doctor.indexer, "binary_path",
                        lambda *a, **k: bindir / "LaueMatchingCPU")
    monkeypatch.setattr(doctor, "detect_devices", lambda: devices)
    monkeypatch.setattr(doctor, "build_info",
                        lambda: {"available": True, "version": "0.6.0",
                                 "cuda": cuda})


def test_cuda_device_but_no_gpu_binary_is_a_problem(monkeypatch, tmp_path):
    """The exact state two beamline environments were left in by a routine
    upgrade on 2026-09-06, and which nothing reported at the time."""
    _stub(monkeypatch, tmp_path, devices=[{"name": "A6000", "compute_cap": "8.6"}],
          binaries=["LaueMatchingCPU"],
          cuda={"built": False, "reason": "skipped: no CUDA compiler (nvcc) found"})
    d = doctor.diagnose()
    assert not d["healthy"]
    assert any("no GPU binary" in p for p in d["problems"])
    assert any("nvcc" in p for p in d["problems"]), "must say WHY it is missing"


def test_no_device_and_no_gpu_binary_is_consistent(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path, devices=[], binaries=["LaueMatchingCPU"],
          cuda={"built": False, "reason": "skipped: no CUDA compiler (nvcc) found"})
    d = doctor.diagnose()
    assert d["healthy"] and any("consistent" in n for n in d["notes"])


def test_a_binary_that_cannot_launch_on_this_card_is_a_problem(monkeypatch, tmp_path):
    """Built on a NEWER toolkit than the card it is handed to -- /home/beams is
    shared, so binaries travel. JIT is forward-only, so PTX sm_120 does not
    rescue an sm_86 card. The reverse direction is fine and is tested above."""
    _stub(monkeypatch, tmp_path,
          devices=[{"name": "A6000", "compute_cap": "8.6"}],
          binaries=["LaueMatchingCPU", "LaueMatchingGPU", "LaueMatchingGPUStream"],
          cuda={"built": True, "architectures": "100-real;120-real;120-virtual"})
    d = doctor.diagnose()
    assert not d["healthy"]
    assert any("cannot launch" in p for p in d["problems"])


def test_a_healthy_gpu_install_reports_healthy(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path,
          devices=[{"name": "A6000", "compute_cap": "8.6"}],
          binaries=["LaueMatchingCPU", "LaueMatchingGPU", "LaueMatchingGPUStream"],
          cuda={"built": True, "architectures": "80-real;86-real;90-virtual"})
    d = doctor.diagnose()
    assert d["healthy"], d["problems"]


def test_no_cpu_binary_is_always_a_problem(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path, devices=[], binaries=[], cuda={"built": False})
    d = doctor.diagnose()
    assert not d["healthy"]
    assert any("Nothing can index" in p for p in d["problems"])


def test_doctor_json_is_machine_readable(monkeypatch, tmp_path, capsys):
    _stub(monkeypatch, tmp_path, devices=[], binaries=["LaueMatchingCPU"],
          cuda={"built": False, "reason": "skipped: no CUDA compiler (nvcc) found"})
    rc = doctor.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload["healthy"] is True and "binaries" in payload


def test_cli_exposes_doctor():
    from laue_index import cli
    with pytest.raises(SystemExit):
        cli.main(["doctor", "--help"])
