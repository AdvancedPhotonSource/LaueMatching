"""The forward cache carries a provenance record and is only reused if it was
built for THIS configuration.

Before 0.8.0 the file size was the whole contract, so a right-sized cache built
for another detector pose, lattice, energy window or HKL list was reused and
silently wrong (RUNBOOK open item 10), and the 0.8.0 change of pixel rounding
would have been invisible to every existing cache. A sidecar
``<ForwardFile>.meta`` now carries the cache format (2 = rounded pixel centres)
and a key over everything the cached pixels depend on. Missing, old-format or
mismatched sidecar -> the cache is rebuilt.

0.8.0 (data-artifact provenance, HS 2026-09-28): the sidecar is the JSON
artifact record `<ForwardFile>.meta.json` (laue_index.artifacts schema) with the
full generating configuration (every key input under its params-file name and
the params file's text), the orientation-DB and HKL inputs and the producer.
Policy: no record -> warn and rebuild; a record for a DIFFERENT configuration
-> FATAL (never silently overwrite a cache another configuration may use),
unless the user asked for DoFwd 1.
"""
import json
import os
import re
import subprocess

import pytest

from _cbuild import C_DIR, FIXTURES, HEADERS, build


def test_key_and_sidecar_behaviour(tmp_path):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    exe = str(tmp_path / "fcm")
    err = build(exe, [os.path.join(FIXTURES, "forward_cache_meta.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err
    params = tmp_path / "params.txt"
    params.write_text('SpaceGroup 194\nLatticeParameter 0.2921 0.2921 0.4665 90 90 120\n'
                      '# a "quoted" comment \\ with a backslash\tand a tab\n')
    proc = subprocess.run([exe, str(tmp_path), str(params)], capture_output=True, text=True)
    failed = [l for l in proc.stdout.splitlines() if l.startswith("CHECK") and l.endswith(" 0")]
    assert proc.returncode == 0 and not failed, proc.stdout + proc.stderr
    # the record the C wrote is valid JSON in the laue_index.artifacts schema
    from laue_index import artifacts as A
    rec = json.loads((tmp_path / "forward.bin.meta.json").read_text())
    assert rec["schema"] == A.SCHEMA and rec["kind"] == "forward_cache"
    cfg = rec["config"]
    assert cfg["SpaceGroup"] == 194 and cfg["LatticeParameter"] == [0.2921, 0.2921, 0.4665, 90, 90, 120]
    assert cfg["MaxNrLaueSpots"] == 500 and cfg["n_orientations"] == 1000 and cfg["format"] == 2
    assert len(cfg["key"]) == 16
    assert rec["config_file"]["text"] == params.read_text()      # escaping round-trips
    roles = {i["role"]: i for i in rec["inputs"]}
    assert roles["orientation_db"]["path"].endswith("db.bin")
    assert roles["hkl_list"]["n_hkls"] == 2
    assert rec["producer"]["program"] == "LaueMatchingCPU" and rec["producer"]["threads"] == 8
    assert A.check(tmp_path / "forward.bin", "forward_cache").status == "ok"


def test_every_binary_checks_and_writes_the_sidecar():
    if not os.path.isdir(C_DIR):
        pytest.skip("c_src not present")
    for name in ("LaueMatchingCPU.c", "LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu"):
        src = open(os.path.join(C_DIR, name)).read()
        assert "forwardCacheKey(" in src, name
        # the read path checks the status (refusing a foreign record), the
        # sibling-reuse path checks the key
        assert "forwardCacheMetaStatus(" in src and "forwardCacheMetaMatches(" in src, name
        assert "makeFwdCacheInfo(" in src, name
        # published caches get their sidecar before the lock is released
        fin = src.index("finishForwardCacheOrDie(fwdFd")
        rel = src.index("releaseForwardCacheLock(&fc)", fin)
        assert "writeForwardCacheMeta(" in src[fin:rel], name


def test_real_cpu_refuses_a_foreign_cache_and_rebuilds_on_request(tmp_path):
    """End to end on the real CPU main: a cache built for lattice A, then a run
    for lattice B. DoFwd 0 must refuse (the file may be lattice A's runs'
    cache) and leave it untouched; DoFwd 1 rebuilds it for B and rewrites the
    record."""
    import test_c_param_parse_guards as G
    exe = str(tmp_path / "LaueMatchingCPU")
    if not G._build_cpu(exe):
        pytest.skip("cannot build LaueMatchingCPU")
    fwd = tmp_path / "fwd.bin"
    fwd_meta = tmp_path / "fwd.bin.meta.json"
    extra = f"MaxNrLaueSpots 5\nForwardFile {fwd}\n"
    a = {"LatticeParameter": "0.2 0.2 0.2 90 90 90"}
    b = {"LatticeParameter": "0.21 0.21 0.21 90 90 90"}
    r = G._run_cpu(exe, tmp_path, a, extra=extra + "DoFwd 1\n", npx=64)
    assert r.returncode == 0, r.stdout + r.stderr
    rec_a = json.loads(fwd_meta.read_text())
    assert rec_a["config"]["LatticeParameter"][0] == 0.2
    assert "LatticeParameter 0.2 0.2 0.2" in rec_a["config_file"]["text"]
    bytes_a = fwd.read_bytes()

    r = G._run_cpu(exe, tmp_path, b, extra=extra + "DoFwd 0\n", npx=64)
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr, r.stdout + r.stderr
    assert fwd.read_bytes() == bytes_a and json.loads(fwd_meta.read_text()) == rec_a

    r = G._run_cpu(exe, tmp_path, b, extra=extra + "DoFwd 1\n", npx=64)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(fwd_meta.read_text())["config"]["LatticeParameter"][0] == 0.21
