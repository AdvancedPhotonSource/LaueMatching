"""The orientation database carries an artifact record, and the released one is
recognised by its full SHA-256.

fetch-db used to check only the size; copies of the database on the hosts were
anonymous 7.2 GB files. Now: fetch-db verifies a release-sized download against
artifacts.ORIENT_DB_SHA256 and refuses a corrupt one; every route that writes
or stamps a database writes <db>.meta.json (kind orientation_db) with its full
generating configuration, or says the generator is unknown.
"""
import hashlib
import json

import numpy as np
import pytest

from laue_index import artifacts as A
from laue_index import cli
from test_fetch_db import _serve, _whole_orientations


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fetch_db_verifies_the_release_hash_and_writes_the_record(tmp_path, monkeypatch):
    parts = _whole_orientations(10)
    blob = b"".join(parts[n] for n in cli.ORIENT_DB_PARTS)
    monkeypatch.setattr(cli, "ORIENT_DB_BYTES", len(blob))
    monkeypatch.setattr(A, "ORIENT_DB_SHA256", hashlib.sha256(blob).hexdigest())
    _serve(parts, monkeypatch)
    assert cli.main(["fetch-db", "--dest", str(tmp_path)]) == 0
    db = tmp_path / "100MilOrients.bin"
    rec = A.read_record(db)
    assert rec["kind"] == "orientation_db" and rec["extra"]["verified_against_release"] is True
    assert rec["config"]["source"].endswith("v1.0-data")
    assert A.check(db, "orientation_db", full=True).status == "ok"


def test_fetch_db_refuses_a_release_sized_download_with_the_wrong_hash(tmp_path, monkeypatch, capsys):
    parts = _whole_orientations(10)
    monkeypatch.setattr(cli, "ORIENT_DB_BYTES", 10 * 72)
    monkeypatch.setattr(A, "ORIENT_DB_SHA256", "0" * 64)
    _serve(parts, monkeypatch)
    assert cli.main(["fetch-db", "--dest", str(tmp_path)]) == 1
    assert "SHA-256" in capsys.readouterr().err


def test_annotate_recognises_the_released_database(tmp_path, monkeypatch):
    from annotate_orientation_db import annotate
    db = tmp_path / "100MilOrients.bin"
    db.write_bytes(np.eye(3).tobytes() * 5)
    monkeypatch.setattr(A, "ORIENT_DB_SHA256", _sha(db))
    annotate(db)
    rec = A.read_record(db)
    assert rec["kind"] == "orientation_db" and rec["retroactive"] is True
    assert rec["config"]["spacing_deg"] == 0.4 and rec["extra"]["is_release"] is True


def test_annotate_does_not_invent_a_generator(tmp_path):
    from annotate_orientation_db import annotate
    db = tmp_path / "mine.bin"
    db.write_bytes(np.eye(3).tobytes() * 5)
    annotate(db)
    rec = A.read_record(db)
    assert rec["extra"]["is_release"] is False
    assert "spacing_deg" not in rec["config"]
    assert rec["extra"]["generator"].startswith("unknown")


def test_generate_orientations_records_its_configuration(tmp_path):
    pytest.importorskip("orix")
    from GenerateOrientations import generate
    out = tmp_path / "small.bin"
    generate(spacing_deg=20.0, crystal_system="cubic", output_path=out, sampling="haar")
    rec = A.read_record(out)
    assert rec["schema"] == A.SCHEMA and rec["kind"] == "orientation_db"
    assert rec["config"]["spacing_deg"] == 20.0 and rec["config"]["sampling_method"] == "haar"
    assert rec["artifact"]["sha256"] == _sha(out)


def test_cli_provenance_show_verify_stamp(tmp_path, capsys):
    f = tmp_path / "bg.bin"
    f.write_bytes(b"\x00" * 800)
    assert cli.main(["provenance", "verify", str(f)]) == 1            # no record
    assert cli.main(["provenance", "stamp", "--kind", "background", str(f)]) == 0
    assert A.read_record(f)["retroactive"] is True
    assert cli.main(["provenance", "verify", "--full", str(f)]) == 0
    assert cli.main(["provenance", "show", str(f)]) == 0
    assert '"kind": "background"' in capsys.readouterr().out
    f.write_bytes(b"\x01" * 800)
    assert cli.main(["provenance", "verify", str(f)]) == 2            # mismatch


def test_cli_stamp_needs_a_kind(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"\x00" * 8)
    assert cli.main(["provenance", "stamp", str(f)]) == 2
    assert A.read_record(f) is None
