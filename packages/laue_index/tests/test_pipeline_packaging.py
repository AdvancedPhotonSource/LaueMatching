"""The orchestration layer ships with the package, and can be driven from it.

`pip install laue-index` gives the library and the C binaries; before the
orchestration modules moved into the package it gave nothing that could run an
image through them, and `laue-index` had no `run`. These tests pin that
arrangement: the modules are importable from the installed package, the shims
in scripts/ point at the same files, and the CLI hands off verbatim.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from laue_index import cli
from laue_index.pipeline import PIPELINE_DIR, SCRIPTS, add_to_path, run_module

_REPO = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "scripts").is_dir() and (p / "CMakeLists.txt").is_file()), None)


# ── the modules ship ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SCRIPTS)
def test_every_declared_entry_point_is_present(name):
    """SCRIPTS is what the shims and `laue-index run` promise exists."""
    assert (PIPELINE_DIR / f"{name}.py").is_file()


def test_the_whole_orchestration_layer_moved_not_just_the_entry_points():
    """RunImage shells out to siblings and imports others flat; all must ship."""
    for helper in ("laue_config", "laue_stream_utils", "laue_visualization",
                   "laue_simulation", "laue_provenance", "laue_indexfile",
                   "_version"):
        assert (PIPELINE_DIR / f"{helper}.py").is_file(), (
            f"{helper}.py is missing from the package: RunImage imports it, so "
            f"an installed pipeline would fail at import time")


def test_add_to_path_makes_the_modules_import_flat():
    """They import each other as top-level modules; that is what needs to work."""
    add_to_path()
    assert sys.path[0] == str(PIPELINE_DIR)
    add_to_path()                       # idempotent, not duplicated
    assert sys.path.count(str(PIPELINE_DIR)) == 1


def test_run_module_rejects_an_unknown_name():
    with pytest.raises(FileNotFoundError) as e:
        run_module("NotAScript")
    assert "NotAScript" in str(e.value)


# ── the shims ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(_REPO is None, reason="not running from a checkout")
@pytest.mark.parametrize("name", SCRIPTS)
def test_a_shim_exists_for_every_entry_point(name):
    """Every documented `python scripts/X.py` invocation must still work."""
    shim = _REPO / "scripts" / f"{name}.py"
    assert shim.is_file(), f"scripts/{name}.py disappeared; documented calls break"
    text = shim.read_text()
    assert "run_module" in text and name in text


@pytest.mark.skipif(_REPO is None, reason="not running from a checkout")
def test_no_orphaned_implementation_left_in_scripts():
    """A copy left behind in scripts/ would shadow the package and drift."""
    leftovers = [p.name for p in (_REPO / "scripts").glob("*.py")
                 if p.stat().st_size > 2000]
    assert not leftovers, f"scripts/ should hold only thin shims, found: {leftovers}"


@pytest.mark.skipif(_REPO is None, reason="not running from a checkout")
def test_the_shim_actually_runs_the_packaged_module():
    """End to end through the shim: it must reach RunImage's own parser."""
    proc = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "RunImage.py"), "--help"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RunImage.py" in proc.stdout and "process" in proc.stdout


# ── the CLI ─────────────────────────────────────────────────────────────────

def test_cli_run_passes_everything_through_verbatim(monkeypatch):
    """--help and RunImage's subcommands must not be eaten by our parser."""
    seen = {}
    monkeypatch.setattr("laue_index.pipeline.run_module",
                        lambda name, argv=None: seen.update(name=name, argv=argv))
    assert cli.main(["run", "process", "-c", "p.txt", "--help"]) == 0
    assert seen["name"] == "RunImage"
    assert seen["argv"] == ["process", "-c", "p.txt", "--help"]


def test_cli_run_reports_a_missing_dependency_as_an_install_hint(monkeypatch):
    def boom(name, argv=None):
        raise ImportError("No module named 'plotly'")
    monkeypatch.setattr("laue_index.pipeline.run_module", boom)
    assert cli.main(["run", "process"]) == 1


def test_cli_run_forwards_the_exit_code(monkeypatch):
    def bye(name, argv=None):
        raise SystemExit(3)
    monkeypatch.setattr("laue_index.pipeline.run_module", bye)
    assert cli.main(["run", "process"]) == 3
