"""The binaries refuse a params file missing a key whose default used to differ
between RunImage, streaming and the C (user decision D5, 2026-09-28).

The C defaults (MaxNrLaueSpots 500, MinIntensity 1000, R_Array 0 0 0, PxX 0)
disagreed with both Python parsers (7 / 400, 50 / 0, ...), so one params file
indexed differently depending on which path ran. Python refuses these keys when
absent (config_schema.REQUIRED_KEYS); so does the C, with one message naming
every missing key. R_Array needs this in particular: 0 0 0 is a legitimate
identity, so its absence cannot be caught from the value.
"""
import os
import re
import subprocess

import pytest

from _cbuild import C_DIR, HEADERS, build

C_REQUIRED = ("LatticeParameter", "P_Array", "R_Array", "PxX", "PxY", "NrPxX",
              "NrPxY", "MaxNrLaueSpots", "MinIntensity")
FULL = {"LatticeParameter": "0.4 0.4 0.4 90 90 90", "SpaceGroup": "225",
        "NrPxX": "8", "NrPxY": "8", "PxX": "0.2", "PxY": "0.2", "P_Array": "0 0 50",
        "R_Array": "0 0 0", "MaxNrLaueSpots": "5", "MinIntensity": "50"}


@pytest.fixture(scope="module")
def cpu(tmp_path_factory):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present")
    exe = str(tmp_path_factory.mktemp("req") / "LaueMatchingCPU")
    err = build(exe, [os.path.join(C_DIR, "LaueMatchingCPU.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err
    return exe


def _run(cpu, tmp_path, drop):
    import struct
    kv = {k: v for k, v in FULL.items() if k not in drop}
    (tmp_path / "p.txt").write_text("".join(f"{k} {v}\n" for k, v in kv.items()))
    (tmp_path / "o.bin").write_bytes(struct.pack("9d", 1, 0, 0, 0, 1, 0, 0, 0, 1))
    (tmp_path / "h.txt").write_text("1 1 1\n")
    (tmp_path / "i.bin").write_bytes(struct.pack("64d", *([0.0] * 64)))
    return subprocess.run([cpu, "p.txt", "o.bin", "h.txt", "i.bin", "1"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("key", ["R_Array", "MaxNrLaueSpots", "MinIntensity", "PxX", "PxY"])
def test_missing_key_is_fatal_and_named(cpu, tmp_path, key):
    r = _run(cpu, tmp_path, {key})
    assert r.returncode != 0
    assert "FATAL" in r.stderr and key in r.stderr, r.stderr


def test_every_missing_key_is_named_at_once(cpu, tmp_path):
    r = _run(cpu, tmp_path, {"R_Array", "MinIntensity"})
    assert "R_Array" in r.stderr and "MinIntensity" in r.stderr


def test_all_binaries_check_the_same_keys():
    hdr = open(HEADERS).read()
    body = hdr[hdr.index("static inline int requireParamKeys("):]
    body = body[:body.index("\n}\n")]
    for k in C_REQUIRED:
        assert f'"{k}"' in body, k
    for name in ("LaueMatchingCPU.c", "LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu"):
        assert "requireParamKeys(argv[1])" in open(os.path.join(C_DIR, name)).read(), name


def test_c_and_python_agree_on_the_required_geometry_keys():
    from laue_index.config_schema import REQUIRED_KEYS
    assert set(C_REQUIRED) <= set(REQUIRED_KEYS)


def test_stream_flushes_its_no_match_report():
    # the orchestrator's drain reads these lines; unflushed, a run ending in
    # no-match frames waits out the stall timeout
    src = open(os.path.join(C_DIR, "LaueMatchingGPUStream.cu")).read()
    i = src.index("No matches, skipping fitting")
    assert "fflush(stdout)" in src[i:i + 200]
