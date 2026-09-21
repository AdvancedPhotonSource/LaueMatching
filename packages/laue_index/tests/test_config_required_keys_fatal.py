"""A malformed key with no safe default stops the run, in both config parsers.

Both parsers used to log a malformed SpaceGroup / Symmetry / LatticeParameter /
P_Array / R_Array (and, since the templates carry them as placeholders, Elo /
Ehi) and carry on with the built-in default -- SpaceGroup 225, the
Ni lattice, a 0.513 m detector -- so a params template with its literal
``__SET_ME__`` placeholders indexed as nickel on another beamline's geometry and
looked like it worked. Other keys keep the old warn-and-continue behaviour.
Also: GaussSigmaMax, which the streaming parser reads, is a schema key now.
"""
import logging

import pytest

import laue_stream_utils as lsu
from laue_index import config_schema as S
from laue_index.pipeline.laue_config import ConfigurationManager

_GOOD = {
    "SpaceGroup": "194",
    "Symmetry": "P",
    "LatticeParameter": "0.2665 0.2665 0.4947 90 90 120",
    "P_Array": "0.028 0.002 0.513",
    "R_Array": "-1.2 -1.2 -1.2",
    "Elo": "7.5",
    "Ehi": "28",
}
_BAD = [
    ("SpaceGroup", "__SET_ME__"),
    ("SpaceGroup", "0"),
    ("SpaceGroup", "231"),
    ("Symmetry", "__SET_ME__"),
    ("Symmetry", "Q"),
    ("LatticeParameter", " ".join(["__SET_ME__"] * 6)),
    ("LatticeParameter", "0.2665 0.2665 0.4947"),
    ("P_Array", "__SET_ME__ __SET_ME__ __SET_ME__"),
    ("P_Array", "0.028 0.002"),
    ("R_Array", "__SET_ME__ __SET_ME__ __SET_ME__"),
    ("Elo", "__SET_ME__"),
    ("Ehi", "__SET_ME__"),
]


def _params(tmp_path, **override):
    vals = dict(_GOOD)
    vals.update(override)
    p = tmp_path / "params.txt"
    p.write_text("".join(f"{k} {v}\n" for k, v in vals.items()))
    return p


def test_good_file_parses_in_both(tmp_path):
    p = _params(tmp_path)
    assert ConfigurationManager(str(p)).config.space_group == 194
    cfg = lsu.parse_config(str(p))
    assert cfg["space_group"] == 194 and cfg["distance"] == 0.513


@pytest.mark.parametrize("key,value", _BAD)
def test_configuration_manager_exits(tmp_path, key, value, caplog):
    p = _params(tmp_path, **{key: value})
    with caplog.at_level(logging.ERROR, logger="LaueMatching"), \
            pytest.raises(SystemExit) as exc:
        ConfigurationManager(str(p))
    assert exc.value.code not in (0, None)
    assert key in caplog.text


@pytest.mark.parametrize("key,value", _BAD)
def test_streaming_parser_raises(tmp_path, key, value):
    p = _params(tmp_path, **{key: value})
    with pytest.raises(ValueError) as exc:
        lsu.parse_config(str(p))
    assert key in str(exc.value)
    assert value.split()[0] in str(exc.value)


# Both Python parsers follow the C's token-count rule (sscanf reads the first N:
# too few is fatal, extra trailing tokens warn and are ignored), accept an
# integral float for SpaceGroup (C: %d), and refuse a lowercase Symmetry (the C
# does not read it; GenerateHKLs, which does, is case-sensitive).
_ACCEPTED = [
    ("P_Array", "0.028 0.002 0.513 0.0", "p_array", "0.028 0.002 0.513"),
    ("R_Array", "-1.2 -1.2 -1.2 7", "r_array", "-1.2 -1.2 -1.2"),
    ("LatticeParameter", "0.2665 0.2665 0.4947 90 90 120 1",
     "lattice_parameter", "0.2665 0.2665 0.4947 90 90 120"),
    ("SpaceGroup", "194.0", "space_group", 194),
]


@pytest.mark.parametrize("key,value,field,expected", _ACCEPTED)
def test_c_permissive_inputs_are_accepted_on_both_paths(tmp_path, caplog, key,
                                                        value, field, expected):
    p = _params(tmp_path, **{key: value})
    with caplog.at_level(logging.WARNING):
        cm = ConfigurationManager(str(p))
        cfg = lsu.parse_config(str(p))
    assert getattr(cm.config, field) == expected
    assert cfg[field] == expected
    if key != "SpaceGroup":
        assert caplog.text.count("using the first") == 2   # one per parser


@pytest.mark.parametrize("key,value", [("SpaceGroup", "194.5"), ("Symmetry", "p")])
def test_both_paths_reject(tmp_path, key, value, caplog):
    p = _params(tmp_path, **{key: value})
    with caplog.at_level(logging.ERROR, logger="LaueMatching"), \
            pytest.raises(SystemExit):
        ConfigurationManager(str(p))
    with pytest.raises(ValueError) as exc:
        lsu.parse_config(str(p))
    if key == "Symmetry":
        assert "case-sensitive" in caplog.text
        assert "case-sensitive" in str(exc.value)


def test_schema_raises_the_fatal_subclass():
    from laue_index.pipeline.laue_config import LaueConfig
    with pytest.raises(S.FatalConfigError, match="SpaceGroup"):
        S.parse_line(LaueConfig(), "SpaceGroup __SET_ME__   # comment")
    # a non-fatal key still raises the plain ValueError the manager logs
    with pytest.raises(ValueError) as exc:
        S.parse_line(LaueConfig(), "MinArea many")
    assert not isinstance(exc.value, S.FatalConfigError)


def test_other_malformed_keys_still_warn_and_continue(tmp_path, caplog):
    p = _params(tmp_path)
    p.write_text(p.read_text() + "MinArea many\nFilterRadius wide\n")
    with caplog.at_level(logging.WARNING):
        cm = ConfigurationManager(str(p))
        cfg = lsu.parse_config(str(p))
    assert cm.config.image_processing.min_area == 10
    assert cfg["filter_radius"] == 101


def test_gauss_sigma_max_is_a_schema_key(tmp_path, caplog):
    p = _params(tmp_path)
    p.write_text(p.read_text() + "GaussSigmaMax 2.5\n")
    with caplog.at_level(logging.WARNING, logger="LaueMatching"):
        cm = ConfigurationManager(str(p))
    assert "unknown configuration key" not in caplog.text
    assert cm.config.image_processing.gauss_sigma_max == 2.5
    assert lsu.parse_config(str(p))["gauss_sigma_max"] == 2.5


def test_energy_band_valid_and_absent(tmp_path):
    """Elo/Ehi: a valid number is taken on both paths; an absent key keeps
    today's default (5 / 30 keV) on both."""
    p = _params(tmp_path)
    assert ConfigurationManager(str(p)).config.elo == 7.5
    assert lsu.parse_config(str(p))["ehi"] == 28.0
    p.write_text("".join(f"{k} {v}\n" for k, v in _GOOD.items()
                         if k not in ("Elo", "Ehi")))
    cm = ConfigurationManager(str(p))
    cfg = lsu.parse_config(str(p))
    assert (cm.config.elo, cm.config.ehi) == (5.0, 30.0)
    assert (cfg["elo"], cfg["ehi"]) == (5.0, 30.0)
