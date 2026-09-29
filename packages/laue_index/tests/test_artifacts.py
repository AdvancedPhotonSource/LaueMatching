"""laue_index.artifacts: one provenance record per data artifact.

The orientation database, HKL lists, forward caches and backgrounds were bare
binaries: nothing said which code, which inputs or which configuration made
them, so a right-sized file from another campaign was accepted silently
(RUNBOOK open item 10, invariant 16). Each now carries `<file>.meta.json`.

Policy (HS, 2026-09-28): a MISSING record warns, a record that DISAGREES with
the file or with the configuration refuses.
"""
import json
import logging
import os

import pytest

from laue_index import artifacts as A


def _blob(tmp_path, name="x.bin", data=b"\x01\x02" * 5000):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_write_then_check_is_ok(tmp_path):
    p = _blob(tmp_path)
    rec = A.write_record(p, "background", layout={"shape": [100, 100], "dtype": "uint8"},
                         config={"FilterRadius": 101})
    assert A.sidecar_path(p).is_file()
    assert rec["schema"] == A.SCHEMA and rec["kind"] == "background"
    assert rec["artifact"]["size"] == p.stat().st_size
    assert len(rec["artifact"]["sha256"]) == 64
    c = A.check(p, "background", full=True)
    assert c.status == "ok", c.reasons


def test_missing_record_is_missing_and_require_warns(tmp_path, caplog):
    p = _blob(tmp_path)
    assert A.check(p, "background").status == "missing"
    with caplog.at_level(logging.WARNING):
        c = A.require(p, "background")
    assert c.status == "missing"
    assert "no provenance record" in caplog.text


@pytest.mark.parametrize("damage", ["flip_head", "flip_middle", "truncate", "append"])
def test_changed_bytes_are_a_mismatch_and_require_refuses(tmp_path, damage):
    data = bytes(range(256)) * 20000          # 5 MB: middle is outside the quick hash
    p = _blob(tmp_path, data=data)
    A.write_record(p, "background")
    b = bytearray(p.read_bytes())
    if damage == "flip_head":
        b[10] ^= 0xFF
    elif damage == "flip_middle":
        b[len(b) // 2] ^= 0xFF
    elif damage == "truncate":
        b = b[:-8]
    else:
        b += b"\x00" * 8
    p.write_bytes(bytes(b))
    # the middle byte is only visible to the full hash; everything else to both
    c = A.check(p, "background", full=True)
    assert c.status == "mismatch", damage
    with pytest.raises(A.ArtifactMismatch):
        A.require(p, "background", full=True)
    if damage != "flip_middle":
        assert A.check(p, "background").status == "mismatch"


def test_wrong_kind_is_a_mismatch(tmp_path):
    p = _blob(tmp_path)
    A.write_record(p, "background")
    c = A.check(p, "hkl_list")
    assert c.status == "mismatch" and any("kind" in r for r in c.reasons)


def test_params_that_disagree_with_the_configuration_refuse(tmp_path):
    p = _blob(tmp_path)
    A.write_record(p, "hkl_list", config={"SpaceGroup": 194,
                                          "LatticeParameter": [0.2665, 0.2665, 0.4947, 90, 90, 120]})
    ok = A.check(p, "hkl_list", expect={"SpaceGroup": 194,
                                        "LatticeParameter": [0.2665, 0.2665, 0.4947, 90, 90, 120.0]})
    assert ok.status == "ok", ok.reasons
    bad = A.check(p, "hkl_list", expect={"LatticeParameter": [0.2921, 0.2921, 0.4665, 90, 90, 120]})
    assert bad.status == "mismatch" and any("LatticeParameter" in r for r in bad.reasons)
    # a parameter the record does not carry is not a mismatch (older record)
    assert A.check(p, "hkl_list", expect={"Ehi": 30}).status == "ok"


def test_inputs_carry_the_parents_recorded_hash(tmp_path):
    parent = _blob(tmp_path, "db.bin")
    A.write_record(parent, "orientation_db")
    child = _blob(tmp_path, "fwd.bin", data=b"z" * 100)
    rec = A.write_record(child, "forward_cache", inputs=[("orientation_db", parent)])
    (inp,) = rec["inputs"]
    assert inp["role"] == "orientation_db"
    assert inp["sha256"] == A.read_record(parent)["artifact"]["sha256"]
    assert inp["record"] is True


def test_record_is_written_atomically_and_is_json(tmp_path):
    p = _blob(tmp_path)
    A.write_record(p, "background")
    json.loads(A.sidecar_path(p).read_text())
    assert not any(n.name.endswith(".tmp") for n in tmp_path.iterdir())


def test_retroactive_stamp_keeps_unknowns(tmp_path):
    p = _blob(tmp_path)
    rec = A.write_record(p, "orientation_db", retroactive=True,
                         extra={"generator": "unknown -- pre-repo-history"})
    assert rec["retroactive"] is True
    assert rec["extra"]["generator"].startswith("unknown")


def test_legacy_hkl_provenance_sidecar_is_read(tmp_path):
    p = _blob(tmp_path, "valid_hkls.csv", data=b"1 0 0\n")
    legacy = tmp_path / "valid_hkls.csv.provenance.json"
    legacy.write_text(json.dumps({"schema_version": "2", "extra": {"sgNum": 225}}))
    rec = A.read_record(p)
    assert rec is not None and rec.get("legacy") is True
    # a legacy record cannot vouch for the bytes: treated as missing (warn)
    assert A.check(p, "hkl_list").status == "missing"


def test_known_orientation_db_hash():
    # Full SHA-256 of the released 100MilOrients.bin, identical on two
    # independent copies (Mac checkout, reassembled from the release parts,
    # and a beamline-host copy), 2026-09-28.
    assert A.ORIENT_DB_SHA256 == "351dc8e0dec0db91aff67493bf749957663e9926dda55db145205a1e0915fd20"


def test_record_carries_the_full_generating_configuration(tmp_path):
    """HS 2026-09-28: the configuration that generated a file travels with it,
    e.g. the orientation spacing, or the sample/detector params of a forward
    simulation, with the parameter file's full text and hash."""
    p = _blob(tmp_path)
    cfg = tmp_path / "params.txt"
    cfg.write_text("SpaceGroup 194\nLatticeParameter 0.2665 0.2665 0.4947 90 90 120\n"
                   "P_Array 0.0288 0.0027 0.5134\nOrientationSpacing 0.4\n")
    rec = A.write_record(p, "forward_cache", config={"OrientationSpacing": 0.4},
                         config_file=cfg)
    assert rec["config"]["OrientationSpacing"] == 0.4
    assert rec["config_file"]["text"] == cfg.read_text()
    assert len(rec["config_file"]["sha256"]) == 64
    cfg.write_text("edited later\n")               # the record still stands alone
    assert "P_Array 0.0288" in A.read_record(p)["config_file"]["text"]


# ---- check_run_inputs: what a run checks before it launches the indexer -----

PARAMS = ("SpaceGroup 194\nSymmetry P\nLatticeParameter 0.26649 0.26649 0.49468 90 90 120\n"
          "P_Array 0.0288 0.0027 0.513\nR_Array -1.2 -1.21 -1.22\nNrPxX 256\nNrPxY 256\n"
          "PxX 0.0016\nPxY 0.0016\nElo 5\nEhi 20\n")


def _hkl(tmp_path, **cfg_over):
    h = _blob(tmp_path, "hkls.csv", data=b"1 0 0 1\n")
    cfg = {"SpaceGroup": 194, "Symmetry": "P", "LatticeParameter": [0.26649, 0.26649, 0.49468, 90, 90, 120],
           "P_Array": [0.0288, 0.0027, 0.513], "R_Array": [-1.2, -1.21, -1.22], "NrPxX": 256,
           "NrPxY": 256, "PxX": 0.0016, "PxY": 0.0016, "Elo": 5, "Ehi": 20}
    cfg.update(cfg_over)
    A.write_record(h, "hkl_list", config=cfg)
    return h


def test_run_inputs_ok_returns_lineage(tmp_path):
    params = tmp_path / "p.txt"; params.write_text(PARAMS)
    db = _blob(tmp_path, "db.bin"); A.write_record(db, "orientation_db")
    lin = A.check_run_inputs(params, orient_db=db, hkl=_hkl(tmp_path))
    assert lin["hkl_list"]["record"] and lin["orientation_db"]["record"]
    assert lin["params_file"]["sha256"]


def test_run_inputs_refuse_an_hkl_list_for_another_crystal(tmp_path):
    params = tmp_path / "p.txt"; params.write_text(PARAMS)
    h = _hkl(tmp_path, LatticeParameter=[0.2921, 0.2921, 0.4665, 90, 90, 120])
    with pytest.raises(A.ArtifactMismatch, match="LatticeParameter"):
        A.check_run_inputs(params, hkl=h)


def test_run_inputs_only_warn_on_a_different_detector(tmp_path, caplog):
    # a recalibrated pose changes which reflections reach the panel at the
    # edges; that is worth a warning, not a refusal
    params = tmp_path / "p.txt"; params.write_text(PARAMS)
    h = _hkl(tmp_path, P_Array=[0.0290, 0.0027, 0.513], Ehi=30)
    with caplog.at_level(logging.WARNING):
        A.check_run_inputs(params, hkl=h)
    assert "P_Array" in caplog.text and "Ehi" in caplog.text


def test_run_inputs_warn_on_unrecorded_files(tmp_path, caplog):
    params = tmp_path / "p.txt"; params.write_text(PARAMS)
    h = _blob(tmp_path, "hkls.csv", data=b"1 0 0 1\n")
    db = _blob(tmp_path, "db.bin")
    with caplog.at_level(logging.WARNING):
        lin = A.check_run_inputs(params, orient_db=db, hkl=h)
    assert lin["hkl_list"]["record"] is False and caplog.text.count("no provenance record") >= 2


def test_run_inputs_check_and_record_the_background(tmp_path):
    params = tmp_path / "p.txt"; params.write_text(PARAMS)
    bg = _blob(tmp_path, "median.bin"); A.write_record(bg, "background")
    lin = A.check_run_inputs(params, background=bg)
    assert lin["background"]["record"] is True
    bg.write_bytes(b"changed" * 100)
    with pytest.raises(A.ArtifactMismatch):
        A.check_run_inputs(params, background=bg)
