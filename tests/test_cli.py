"""§6.6 — laue_index.cli smoke tests."""
import numpy as np

from laue_index.cli import main
from _golden import fixture


def test_cli_parse(capsys):
    rc = main(["parse", str(fixture("sample.solutions.txt")), "--fmt", "runimage"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "solutions (runimage format)" in out


def test_cli_filter_frame_via_files(tmp_path, capsys):
    # write the committed real C output to files, run the filter subcommand
    rc = main([
        "filter",
        "--solutions", str(fixture("sample.solutions.txt")),
        "--spots", str(fixture("sample.spots.txt")),
        "--legacy", "--min-unique", "1",
        "--out", str(tmp_path / "filtered.txt"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "kept" in out and "orientations" in out
    assert (tmp_path / "filtered.txt").exists()


def test_cli_version(capsys):
    import pytest
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


if __name__ == "__main__":
    import types
    cap = types.SimpleNamespace(readouterr=lambda: types.SimpleNamespace(out=""))
    print("run via pytest")
