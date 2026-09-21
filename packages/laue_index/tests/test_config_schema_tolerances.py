"""tol_LatC, tol_c_over_a and MinSpotIntensity are schema keys, validated as
the C validates them.

The C reads them straight from the params file, so the run always saw them; the
Python config, with no schema rows, logged "Ignoring unknown configuration key"
for each and never checked them. They are FRACTIONS: the C forms the bounds as
``value * (1 -/+ tol)``, so 1.0 (a percent written where a fraction was meant)
puts the lower bound at zero. Python now mirrors
``validateCrystalFitTolerances`` in LaueMatchingHeaders.h: reject >= 1 (and
negative / NaN), warn above 0.1.
"""
import logging

import pytest

from laue_index import config_schema as S
from laue_index.pipeline.laue_config import ConfigurationManager, LaueConfig


def _parse(line):
    cfg = LaueConfig()
    assert S.parse_line(cfg, line) is True
    return cfg


def test_keys_are_in_the_schema():
    for key in ("tol_LatC", "tol_c_over_a", "MinSpotIntensity"):
        assert key in S.SCHEMA_BY_KEY, key
        assert hasattr(LaueConfig(), S.SCHEMA_BY_KEY[key].field), (
            f"{key} is in the schema but not the dataclass -- parsed then dropped")


def test_valid_fractions_are_stored():
    assert _parse("tol_c_over_a 0.01").tol_c_over_a == pytest.approx(0.01)
    assert _parse("tol_LatC 0.01 0.01 0.02 0 0 0").tol_lat_c == "0.01 0.01 0.02 0 0 0"
    assert _parse("MinSpotIntensity 12.5").min_spot_intensity == 12.5


@pytest.mark.parametrize("line", [
    "tol_c_over_a 1.0",           # "1 percent" written as a fraction: lower bound 0
    "tol_c_over_a 5",
    "tol_c_over_a -0.01",
    "tol_c_over_a nan",
    "tol_LatC 0.01 0.01 0.01",    # too few values
])
def test_invalid_tolerances_are_rejected(line):
    with pytest.raises(ValueError):
        S.parse_line(LaueConfig(), line)


def test_wide_fraction_warns_but_is_accepted(caplog):
    with caplog.at_level(logging.WARNING, logger="LaueMatching"):
        cfg = _parse("tol_c_over_a 0.5")
    assert cfg.tol_c_over_a == 0.5
    assert "FRACTION" in caplog.text and "50%" in caplog.text


def test_small_fraction_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="LaueMatching"):
        _parse("tol_LatC 0.01 0.01 0.01 0 0 0")
    assert caplog.text == ""


def test_not_logged_as_unknown_and_kept_on_rewrite(tmp_path, caplog):
    """ConfigurationManager used to warn 'Ignoring unknown configuration key'.
    And a rewrite of the params file (render_text) must keep the keys, since
    the C reads the same file."""
    p = tmp_path / "params.txt"
    p.write_text("SpaceGroup 194\ntol_c_over_a 0.01\n"
                 "tol_LatC 0 0 0.02 0 0 0\nMinSpotIntensity 3\n")
    with caplog.at_level(logging.WARNING, logger="LaueMatching"):
        cm = ConfigurationManager(str(p))
    assert "unknown configuration key" not in caplog.text
    cm.write_config()
    text = p.read_text()
    assert any(ln.split()[:2] == ["tol_c_over_a", "0.01"] for ln in text.splitlines())
    assert any(ln.split()[:7] == ["tol_LatC", "0", "0", "0.02", "0", "0", "0"]
               for ln in text.splitlines())
    cm2 = ConfigurationManager(str(p))
    assert cm2.config.tol_c_over_a == 0.01
    assert cm2.config.tol_lat_c == "0 0 0.02 0 0 0"
    assert cm2.config.min_spot_intensity == 3.0


def _load(tmp_path, text):
    p = tmp_path / "params.txt"
    p.write_text(text)
    return ConfigurationManager(str(p))


def test_bad_tol_latc_is_rejected_when_it_is_used(tmp_path, caplog):
    """tol_c_over_a 0: the C validates tol_LatC, so Python does too (after the
    whole file is read); the bad value is logged and not kept."""
    with caplog.at_level(logging.ERROR, logger="LaueMatching"):
        cm = _load(tmp_path, "tol_LatC 0.01 0.01 1 0 0 0\n")
    assert "FRACTION" in caplog.text
    assert cm.config.tol_lat_c == "0 0 0 0 0 0"


@pytest.mark.parametrize("order", ["latc_first", "ca_first"])
def test_tol_latc_not_validated_when_c_over_a_overrides_it(tmp_path, caplog, order):
    """The C ignores tol_LatC when tol_c_over_a != 0 and does not validate it.
    Python must do the same whichever line comes first, log a NOTE rather
    than an error, and keep the value as written."""
    lines = ["tol_LatC 1 1 1 0 0 0", "tol_c_over_a 0.01"]
    if order == "ca_first":
        lines.reverse()
    with caplog.at_level(logging.INFO, logger="LaueMatching"):
        cm = _load(tmp_path, "\n".join(lines) + "\n")
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, [r.getMessage() for r in errors]
    assert "overrides tol_LatC" in caplog.text
    assert cm.config.tol_lat_c == "1 1 1 0 0 0"
    assert cm.config.tol_c_over_a == 0.01


def test_extra_tol_tokens_warn_and_use_the_first(caplog):
    with caplog.at_level(logging.WARNING, logger="LaueMatching"):
        cfg = _parse("tol_LatC 0.01 0.01 0.02 0 0 0 0.5")
    assert cfg.tol_lat_c == "0.01 0.01 0.02 0 0 0"
    assert "first 6" in caplog.text


def test_bad_tolerance_does_not_reach_the_config(tmp_path, caplog):
    """A rejected line is logged as an error and not stored (the manager logs
    per-line errors rather than aborting, as for every other key)."""
    p = tmp_path / "params.txt"
    p.write_text("tol_c_over_a 1.0\n")
    with caplog.at_level(logging.ERROR, logger="LaueMatching"):
        cm = ConfigurationManager(str(p))
    assert cm.config.tol_c_over_a == 0.0
    assert "FRACTION" in caplog.text
