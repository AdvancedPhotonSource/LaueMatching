"""The streaming path reads and honours the same filter/worker keys as RunImage.

Before 0.7.2:
  * ``lsu.parse_config`` (the streaming parser) never parsed ``PreprocessWorkers``
    or ``RobustFilter``, so both were silently ignored on the production path;
  * ``laue_postprocess.process_single_image`` was hard-wired to the LEGACY
    filter (which deletes real Sigma3 twins), never applied ``MinNrSpots``, and
    took its exclusive-spot floor from ``--min-unique`` (default 2) instead of
    ``MinGoodSpots``.
So one frame could keep different orientations depending on which pipeline ran
it. These pin that the streaming path now honours every EXPLICIT key exactly as
RunImage does, while an ABSENT key keeps 0.7.1 streaming behaviour (RobustFilter
absent -> legacy, MinGoodSpots absent -> 2), so re-running an existing config
reproduces its result.
"""
import math

import numpy as np
import pytest

import laue_stream_utils as lsu
import laue_postprocess as lp


# ---------------------------------------------------------------------------
# parse_config round trip
# ---------------------------------------------------------------------------

def test_absent_keys_keep_0_7_1_streaming_defaults():
    from laue_index.pipeline.laue_config import ImageProcessingConfig
    assert lsu.DEFAULT_CONFIG["robust_filter"] is None      # absent, not False
    assert lsu.DEFAULT_CONFIG["min_good_spots"] == 2        # 0.7.1 --min-unique
    assert lsu.DEFAULT_CONFIG["preprocess_workers"] == \
        ImageProcessingConfig().preprocess_workers == 0


@pytest.mark.parametrize("workers,robust,good", [(12, 0, 4), (0, 1, 7)])
def test_parse_config_round_trips_the_three_keys(tmp_path, workers, robust, good):
    p = tmp_path / "params.txt"
    p.write_text(
        "SpaceGroup 225\n"
        f"PreprocessWorkers {workers}\n"
        f"RobustFilter {robust}   # inline comment\n"
        f"MinGoodSpots {good}\n")
    cfg = lsu.parse_config(str(p))
    assert cfg["preprocess_workers"] == workers
    assert cfg["robust_filter"] is bool(robust)
    assert cfg["min_good_spots"] == good


# ---------------------------------------------------------------------------
# process_single_image honours the config
# ---------------------------------------------------------------------------

def _R(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


_MAT = np.array([0.5425461, 0.7753705, 0.3231785,
                 0.8400185, -0.4991547, -0.2126346,
                 -0.0035545, 0.3868400, -0.9221400]).reshape(3, 3)
_NPX = 256


def _stream_frame():
    """A parent (grain 1) with 10 spots in 10 labels of its own, and its exact
    Sigma3 twin (grain 2) with 8 spots on unlabelled pixels: 8 own pixels,
    0 own labels. Legacy drops the twin (0 exclusive labels < 2); robust keeps
    it (CSL-related, 8 own pixels >= MinNrSpots 5).

    Stream layouts: solutions ImageNr col0, GrainNr col1, quality col5,
    NMatches col6, OM cols 23:32 (35 cols); spots ImageNr col0, GrainNr col1,
    X col6, Y col7 (12 cols)."""
    labels = np.zeros((_NPX, _NPX), dtype=np.int32)
    sols, spots = [], []
    for g, q, nm, om in ((1, 1000.0, 10, _MAT),
                         (2, 600.0, 8, _MAT @ _R([1, 1, 1], 60))):
        r = np.zeros(35)
        r[0], r[1], r[5], r[6] = 1, g, q, nm
        r[23:32] = om.reshape(-1)
        sols.append(r)
    for i in range(10):
        x, y = 10 + 10 * i, 30
        labels[y, x] = 1 + i
        s = np.zeros(12)
        s[0], s[1], s[6], s[7] = 1, 1, x, y
        spots.append(s)
    for i in range(8):
        s = np.zeros(12)
        s[0], s[1], s[6], s[7] = 1, 2, 10 + 10 * i, 150
        spots.append(s)
    return np.array(sols), np.array(spots), labels


def _cfg(**over):
    cfg = dict(lsu.DEFAULT_CONFIG)
    cfg.update(nr_px_x=_NPX, nr_px_y=_NPX, space_group=225)
    cfg.update(over)
    return cfg


def _kept(tmp_path, cfg, **kw):
    sols, spots, labels = _stream_frame()
    res = lp.process_single_image(
        image_nr=1, orientations=sols, spots=spots, cfg=cfg,
        output_dir=str(tmp_path), labels=labels, write_indexfile=False, **kw)
    return {int(r[1]) for r in res["filtered_orientations"]}


def test_robust_filter_on_keeps_the_twin(tmp_path):
    assert _kept(tmp_path, _cfg(robust_filter=True)) == {1, 2}


def test_robust_filter_off_is_legacy(tmp_path):
    assert _kept(tmp_path, _cfg(robust_filter=False)) == {1}


def test_absent_robust_filter_is_legacy_as_in_0_7_1(tmp_path):
    assert _kept(tmp_path, _cfg()) == {1}


def test_production_config_without_robustfilter(tmp_path):
    """The shape of the production configs: MinNrSpots 8, MinGoodSpots 4,
    MaxAngle 3, no RobustFilter line. 0.7.2 streaming: legacy filter (as
    0.7.1), exclusive-label floor 4 (0.7.1: 2, from --min-unique). MinNrSpots
    and MaxAngle are not read by the legacy filter on either version."""
    p = tmp_path / "params.txt"
    p.write_text(f"NrPxX {_NPX}\nNrPxY {_NPX}\nSpaceGroup 225\n"
                 "MinNrSpots 8\nMinGoodSpots 4\nMaxAngle 3\n")
    cfg = lsu.parse_config(str(p))
    assert cfg["robust_filter"] is None and cfg["min_good_spots"] == 4
    assert lp._robust_in_force(cfg) is False
    # parent: 10 exclusive labels >= 4 -> kept; twin: 0 -> dropped (legacy)
    assert _kept(tmp_path, cfg) == {1}
    # the floor really is 4 now: a floor of 11 drops the parent too
    assert _kept(tmp_path, dict(cfg, min_good_spots=11)) == set()


def test_absent_robust_filter_is_announced_once(tmp_path, caplog):
    import logging
    import h5py  # noqa: F401  (postprocess writes HDF5)
    sols, spots, _ = _stream_frame()
    sol_f, spot_f = tmp_path / "solutions.txt", tmp_path / "spots.txt"
    import numpy as np
    np.savetxt(sol_f, sols, header="hdr", comments="")
    np.savetxt(spot_f, spots, header="hdr", comments="")
    p = tmp_path / "params.txt"
    p.write_text(f"NrPxX {_NPX}\nNrPxY {_NPX}\nSpaceGroup 225\n")
    with caplog.at_level(logging.WARNING, logger="laue_postprocess"):
        lp.postprocess(str(sol_f), str(spot_f), str(p), str(tmp_path / "out"),
                       mapping_file=str(tmp_path / "none.json"),
                       write_indexfile=False)
    lines = [r for r in caplog.records if "RobustFilter is not set" in r.getMessage()]
    assert len(lines) == 1
    assert "RobustFilter 0" in lines[0].getMessage()
    assert "RunImage" in lines[0].getMessage()
    p.write_text(f"NrPxX {_NPX}\nNrPxY {_NPX}\nSpaceGroup 225\nRobustFilter 1\n")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="laue_postprocess"):
        lp.postprocess(str(sol_f), str(spot_f), str(p), str(tmp_path / "out2"),
                       mapping_file=str(tmp_path / "none.json"),
                       write_indexfile=False)
    assert "RobustFilter is not set" not in caplog.text


def test_min_nr_spots_is_applied_in_python(tmp_path):
    """The twin has 8 own pixels; MinNrSpots 9 must drop it on the robust path."""
    assert _kept(tmp_path, _cfg(robust_filter=True, min_nr_spots=9)) == {1}


def test_min_good_spots_is_the_default_floor(tmp_path):
    """min_unique=None reads MinGoodSpots; an explicit min_unique still wins."""
    cfg = _cfg(robust_filter=False, min_good_spots=11)
    assert _kept(tmp_path, cfg) == set()          # parent has 10 exclusive labels
    assert _kept(tmp_path, cfg, min_unique=2) == {1}


def test_params_file_drives_the_filter_end_to_end(tmp_path):
    """From a params file through parse_config into process_single_image."""
    p = tmp_path / "params.txt"
    p.write_text(f"NrPxX {_NPX}\nNrPxY {_NPX}\nSpaceGroup 225\nRobustFilter 0\n")
    assert _kept(tmp_path, lsu.parse_config(str(p))) == {1}
    p.write_text(f"NrPxX {_NPX}\nNrPxY {_NPX}\nSpaceGroup 225\nRobustFilter 1\n")
    assert _kept(tmp_path, lsu.parse_config(str(p))) == {1, 2}


def test_cli_min_unique_defaults_to_config():
    """--min-unique no longer defaults to 2 on either CLI; unset means
    MinGoodSpots from the config."""
    import inspect
    import laue_orchestrator as lo
    assert inspect.signature(lp.postprocess).parameters["min_unique"].default is None
    assert inspect.signature(lo.run_pipeline).parameters["min_unique"].default is None
