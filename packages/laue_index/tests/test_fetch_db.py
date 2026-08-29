"""`laue-index fetch-db` — the orientation database, without build.sh.

A pip user has no build.sh, so this is the only route to the 6.7 GB database
the indexer matches against. The network is faked: what is being tested is the
reassembly, the size arithmetic, and the refusal to hand back a corrupt file --
not GitHub.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from laue_index import cli


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _serve(parts: dict[str, bytes], monkeypatch, calls: list | None = None):
    def fake_urlopen(url, *a, **kw):
        name = url.rsplit("/", 1)[-1]
        if calls is not None:
            calls.append(name)
        if name not in parts:
            raise IOError(f"404 {name}")
        return _FakeResponse(parts[name])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def _whole_orientations(n: int) -> dict[str, bytes]:
    """Four parts that reassemble to n orientations (9 doubles = 72 B each)."""
    blob = bytes(n * 72)
    q = len(blob) // 4
    chunks = [blob[i * q:(i + 1) * q] for i in range(3)] + [blob[3 * q:]]
    return dict(zip(cli.ORIENT_DB_PARTS, chunks))


def test_fetch_db_reassembles_the_parts(tmp_path, monkeypatch, capsys):
    _serve(_whole_orientations(10), monkeypatch)
    rc = cli.main(["fetch-db", "--dest", str(tmp_path)])
    cap = capsys.readouterr()
    assert rc == 0
    db = tmp_path / "100MilOrients.bin"
    assert db.is_file() and db.stat().st_size == 10 * 72
    assert "10 orientations" in cap.out
    # 10 orientations is not the shipped database. Say so, rather than hand
    # back something that looks fine and indexes nothing.
    assert "warning" in cap.err.lower()
    assert cli.ORIENT_DB_ENV in cap.out, "must say how to point runs at it"


def test_fetch_db_cleans_up_the_parts_unless_asked(tmp_path, monkeypatch):
    _serve(_whole_orientations(4), monkeypatch)
    cli.main(["fetch-db", "--dest", str(tmp_path)])
    assert not list(tmp_path.glob("*.part.*"))

    _serve(_whole_orientations(4), monkeypatch)
    cli.main(["fetch-db", "--dest", str(tmp_path), "--force", "--keep-parts"])
    assert len(list(tmp_path.glob("*.part.*"))) == len(cli.ORIENT_DB_PARTS)


def test_fetch_db_refuses_a_size_that_is_not_whole_orientations(tmp_path, monkeypatch, capsys):
    """A truncated download must fail loudly, not leave a plausible file."""
    parts = _whole_orientations(4)
    parts[cli.ORIENT_DB_PARTS[-1]] += b"\x00" * 5      # 5 bytes, not 72
    _serve(parts, monkeypatch)
    rc = cli.main(["fetch-db", "--dest", str(tmp_path)])
    assert rc == 1
    assert "corrupt" in capsys.readouterr().err.lower()


def test_fetch_db_keeps_an_existing_database_unless_forced(tmp_path, monkeypatch, capsys):
    db = tmp_path / "100MilOrients.bin"
    db.write_bytes(bytes(72))
    called = []
    _serve(_whole_orientations(4), monkeypatch, calls=called)
    cli.main(["fetch-db", "--dest", str(db)])
    assert called == [], "must not re-download over an existing database"
    assert "already exists" in capsys.readouterr().out


def test_fetch_db_reports_a_failed_part_and_keeps_what_it_has(tmp_path, monkeypatch, capsys):
    parts = _whole_orientations(4)
    del parts[cli.ORIENT_DB_PARTS[2]]                  # third part 404s
    _serve(parts, monkeypatch)
    rc = cli.main(["fetch-db", "--dest", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "resume" in err
    assert not (tmp_path / "100MilOrients.bin").exists()
    assert (tmp_path / cli.ORIENT_DB_PARTS[0]).is_file(), "downloaded parts should survive"


def test_the_expected_size_is_arithmetic_not_a_magic_number():
    assert cli.ORIENT_DB_BYTES == 100_000_000 * 9 * 8
    assert cli.ORIENT_DB_BYTES % 72 == 0
