"""P3 (code read 2026-09-28): one source of configuration defaults.

The two Python parsers (``laue_config.ConfigurationManager`` for RunImage and
``laue_stream_utils.parse_config`` for streaming) each carried their own
defaults, and those disagreed with each other and with the C:

    key              RunImage   streaming       C (CPU/GPU/Stream)
    MaxNrLaueSpots   7          400             500
    MinIntensity     50         0               1000
    NMeadianPasses   5          1               -
    MinGoodSpots     5          2               5 (Stream final gate)
    RobustFilter     1          absent=legacy   -
    PxX / PxY        0.2        0.2             0 (and 0.2 is not metres)
    P_Array          .02 .002 .513  0 0 .513    0 0 0
    R_Array          -1.2 x3    -1.2 x3         0 0 0
    NrPxX / NrPxY    2048       2048            FATAL
    LatticeParameter Ni         Ni              FATAL
    BackgroundFile   median.bin ""              -

Decision D5: each of these is now REQUIRED (fatal if absent) in both Python
parsers; every other key takes its default from config_schema.SCHEMA.
"""
from __future__ import annotations

import logging

import pytest

import laue_stream_utils as lsu
from laue_index import config_schema as S
from laue_index.pipeline.laue_config import ConfigurationManager, LaueConfig

NEW_REQUIRED = ("MaxNrLaueSpots", "MinIntensity", "NMeadianPasses", "PxX", "PxY",
                "P_Array", "R_Array", "NrPxX", "NrPxY", "LatticeParameter",
                "MinGoodSpots", "RobustFilter", "BackgroundFile")

FULL = {
    "SpaceGroup": "194", "Symmetry": "P",
    "LatticeParameter": "0.2665 0.2665 0.4947 90 90 120",
    "P_Array": "0.028 0.002 0.513", "R_Array": "-1.2 -1.2 -1.2",
    "Elo": "7.5", "Ehi": "28", "NrPxX": "1024", "NrPxY": "512",
    "PxX": "0.0002", "PxY": "0.0002", "MaxNrLaueSpots": "30",
    "MinIntensity": "50", "NMeadianPasses": "1", "MinGoodSpots": "4",
    "RobustFilter": "1", "BackgroundFile": "/data/bg.bin",
}


def _write(tmp_path, d):
    p = tmp_path / "params.txt"
    p.write_text("".join(f"{k} {v}\n" for k, v in d.items()))
    return p


def test_required_keys_are_declared_in_the_schema():
    assert set(NEW_REQUIRED) <= set(S.REQUIRED_KEYS)
    assert set(S.REQUIRED_KEYS) <= set(S.SCHEMA_BY_KEY)


@pytest.mark.parametrize("key", NEW_REQUIRED)
def test_absent_required_key_stops_configuration_manager(tmp_path, key, caplog):
    p = _write(tmp_path, {k: v for k, v in FULL.items() if k != key})
    with caplog.at_level(logging.ERROR, logger="LaueMatching"), \
            pytest.raises(SystemExit) as exc:
        ConfigurationManager(str(p))
    assert exc.value.code not in (0, None)
    assert key in caplog.text


@pytest.mark.parametrize("key", NEW_REQUIRED)
def test_absent_required_key_stops_streaming_parser(tmp_path, key):
    p = _write(tmp_path, {k: v for k, v in FULL.items() if k != key})
    with pytest.raises(ValueError) as exc:
        lsu.parse_config(str(p))
    assert key in str(exc.value)


def test_all_missing_keys_are_named_at_once(tmp_path):
    p = _write(tmp_path, {"SpaceGroup": "225"})
    with pytest.raises(ValueError) as exc:
        lsu.parse_config(str(p))
    for key in NEW_REQUIRED:
        assert key in str(exc.value)


@pytest.mark.parametrize("key,bad", [("PxX", "wide"), ("MaxNrLaueSpots", "many"),
                                     ("RobustFilter", "yes"), ("NMeadianPasses", "x")])
def test_malformed_required_key_is_fatal_too(tmp_path, key, bad):
    p = _write(tmp_path, {**FULL, key: bad})
    with pytest.raises(SystemExit):
        ConfigurationManager(str(p))
    with pytest.raises(ValueError):
        lsu.parse_config(str(p))


def test_full_file_parses_the_same_in_both(tmp_path):
    p = _write(tmp_path, FULL)
    c = ConfigurationManager(str(p)).config
    d = lsu.parse_config(str(p))
    assert (c.max_laue_spots, d["max_laue_spots"]) == (30, 30)
    assert (c.min_intensity, d["min_intensity"]) == (50.0, 50.0)
    assert (c.image_processing.median_passes, d["median_passes"]) == (1, 1)
    assert (c.min_good_spots, d["min_good_spots"]) == (4, 4)
    assert c.robust_filter is True and d["robust_filter"] is True
    assert (c.px_x, d["px_x"]) == (0.0002, 0.0002)
    assert (c.nr_px_x, d["nr_px_x"], c.nr_px_y, d["nr_px_y"]) == (1024, 1024, 512, 512)
    assert c.background_file == d["background_file"] == "/data/bg.bin"


# Streaming dict name -> schema field, where they differ.
_STREAM_TO_FIELD = {"max_angle": "maxAngle"}


def _defaults_of(obj_for_field):
    out = {}
    for p in S.SCHEMA:
        v = obj_for_field(p)
        if v is not _MISSING:
            out[p.key] = v
    return out


_MISSING = object()


def test_every_default_comes_from_the_schema():
    """Absent, non-required keys: both parsers' defaults equal SCHEMA's."""
    cfg = LaueConfig()
    field_to_stream = {v: k for k, v in _STREAM_TO_FIELD.items()}
    bad = []
    for p in S.SCHEMA:
        tgt = cfg if p.target == "config" else getattr(cfg, p.target)
        if getattr(tgt, p.field) != p.default:
            bad.append(f"RunImage {p.key}: {getattr(tgt, p.field)!r} != schema {p.default!r}")
        sk = field_to_stream.get(p.field, p.field)
        if sk in lsu.DEFAULT_CONFIG and lsu.DEFAULT_CONFIG[sk] != p.default:
            bad.append(f"streaming {p.key}: {lsu.DEFAULT_CONFIG[sk]!r} != schema {p.default!r}")
    assert not bad, "\n".join(bad)


def test_parsers_agree_on_every_shared_key(tmp_path):
    """Every schema key the streaming parser also reads: a non-default value in
    the file comes out the same from both parsers."""
    probe = {
        "NrPxX": "1000", "NrPxY": "999", "PxX": "0.00015", "PxY": "0.00016",
        "OrientationSpacing": "0.3", "Elo": "6", "Ehi": "25", "MinNrSpots": "7",
        "MaxNrLaueSpots": "33", "MaxAngle": "1.5", "MinIntensity": "12",
        "MinGoodSpots": "3", "RobustFilter": "0", "ThresholdMethod": "otsu",
        "Threshold": "7", "ThresholdPercentile": "95", "MinArea": "3",
        "GaussSigmaMax": "2", "PreprocessWorkers": "3", "ExcludeSpotsDir": "d",
        "ExcludeSpotsFile": "f.txt", "GaussianFactor": "0.3", "FilterRadius": "51",
        "NMeadianPasses": "2", "WatershedImage": "0", "EnhanceContrast": "1",
        "DenoiseImage": "1", "DenoiseStrength": "2", "EdgeEnhancement": "1",
        "ResultDir": "r", "OrientationFile": "o.bin", "HKLFile": "h.bin",
        "BackgroundFile": "b.bin",
    }
    p = _write(tmp_path, {**FULL, **probe})
    c = ConfigurationManager(str(p)).config
    d = lsu.parse_config(str(p))
    bad = []
    for key in probe:
        prm = S.SCHEMA_BY_KEY[key]
        tgt = c if prm.target == "config" else getattr(c, prm.target)
        sk = {"maxAngle": "max_angle"}.get(prm.field, prm.field)
        if getattr(tgt, prm.field) != d[sk]:
            bad.append(f"{key}: RunImage {getattr(tgt, prm.field)!r} vs streaming {d[sk]!r}")
    assert not bad, "\n".join(bad)


def test_enable_simulation_and_visualization_default_to_the_schema():
    """RunImage's dataclass said True for both; the schema (and the docs) say 0."""
    cfg = LaueConfig()
    assert cfg.simulation.enable_simulation is False
    assert cfg.visualization.enable_visualization is False
