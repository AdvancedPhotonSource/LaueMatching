"""Regression: LaueConfig.write_indexfile must be a real field so that
ConfigurationManager.set('write_indexfile', False) actually takes effect.

Previously this silently no-op'd because the flag wasn't defined on
LaueConfig — a bug that would have left --no-indexfile inert on
RunImage.py. This test locks it in.

Run:
    cd ~/opt/LaueMatching && python -m pytest scripts/tests/test_config_flags.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_write_indexfile_field_exists():
    """LaueConfig must expose write_indexfile so ConfigurationManager.set works."""
    from laue_config import LaueConfig
    cfg = LaueConfig()
    assert hasattr(cfg, "write_indexfile"), "LaueConfig missing write_indexfile field"
    assert cfg.write_indexfile is True, "default must be True (on by default)"


def test_set_write_indexfile_roundtrips(tmp_path: Path):
    """ConfigurationManager.set('write_indexfile', False) must take effect."""
    from laue_config import ConfigurationManager

    # Minimal valid params file (ConfigurationManager requires one to init)
    p = tmp_path / "params.txt"
    p.write_text(
        "SpaceGroup 225\n"
        "Symmetry F\n"
        "LatticeParameter 0.35 0.35 0.35 90 90 90\n"
        "R_Array -1.2 -1.2 -1.2\n"
        "P_Array 0.02 0.002 0.513\n"
    )
    mgr = ConfigurationManager(str(p))

    # Default: True
    assert mgr.get("write_indexfile") is True

    # Flip it via set()
    mgr.set("write_indexfile", False)
    assert mgr.get("write_indexfile") is False

    # And set back
    mgr.set("write_indexfile", True)
    assert mgr.get("write_indexfile") is True


def test_optional_indexfile_metadata_fields():
    """XtalFile/StructureDesc config keys round-trip through the parser."""
    from laue_config import LaueConfig
    cfg = LaueConfig()
    assert cfg.xtal_file == ""
    assert cfg.structure_desc == ""
    assert cfg.atom_description == ""


def test_parser_accepts_indexfile_metadata(tmp_path: Path):
    """Parsing StructureDesc / XtalFile / AtomDesctiption must populate
    the LaueConfig fields without warning about unknown keys."""
    from laue_config import ConfigurationManager

    p = tmp_path / "params.txt"
    p.write_text(
        "SpaceGroup 225\n"
        "Symmetry F\n"
        "LatticeParameter 0.35 0.35 0.35 90 90 90\n"
        "R_Array -1.2 -1.2 -1.2\n"
        "P_Array 0.02 0.002 0.513\n"
        "StructureDesc Ni\n"
        "XtalFile /path/to/Ni.xml\n"
        "AtomDesctiption Ni001  0 0 0 1\n"
    )
    mgr = ConfigurationManager(str(p))
    assert mgr.config.structure_desc == "Ni"
    assert mgr.config.xtal_file == "/path/to/Ni.xml"
    # AtomDesctiption retains everything after the key
    assert "Ni001" in mgr.config.atom_description
