"""Characterization: lock the two current config paths.

The codebase has (pain point #4) two parallel config readers:
  * ``laue_config.ConfigurationManager`` -> ``LaueConfig`` dataclass
    (with a hand-written ``_parse_classic_config_line`` elif chain and a
    parallel ``_write_to_text`` block), and
  * ``laue_stream_utils.parse_config`` -> flat dict.

Both are pinned here on a real param file (``params_NiIndent.txt``).  When the
declarative ``SCHEMA`` replaces them (REFACTOR_PLAN §5 / §6.4), the parsed
values and the round-tripped text must reproduce these goldens.
"""
import shutil
import tempfile
from pathlib import Path

import laue_config as lc
import laue_stream_utils as lsu
from _golden import check_golden, fixture


def _noncomment_lines(text: str):
    """Stripped, non-blank, non-comment lines (drops the nondeterministic
    'Generated on: <timestamp>' header so the snapshot is stable)."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def test_char_configmanager_parse():
    cm = lc.ConfigurationManager(str(fixture("params_NiIndent.txt")))
    check_golden("config_laueconfig_todict", cm.config.to_dict())


def test_char_configmanager_roundtrip_text():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "params.txt"
        shutil.copy(fixture("params_NiIndent.txt"), src)
        cm = lc.ConfigurationManager(str(src))
        cm.write_config()                     # overwrites src with canonical text
        written = src.read_text()
        # 1. The canonical written form (sans comments) is pinned.
        check_golden("config_written_text", _noncomment_lines(written))
        # 2. Re-parsing the written file is idempotent at the value level.
        cm2 = lc.ConfigurationManager(str(src))
        assert cm2.config.to_dict() == cm.config.to_dict(), \
            "ConfigurationManager text round-trip is not idempotent"


def test_char_lsu_parse_config():
    cfg = lsu.parse_config(str(fixture("params_NiIndent.txt")))
    check_golden("config_lsu_dict", cfg)


if __name__ == "__main__":
    for fn in [test_char_configmanager_parse,
               test_char_configmanager_roundtrip_text,
               test_char_lsu_parse_config]:
        fn(); print(f"PASS  {fn.__name__}")
