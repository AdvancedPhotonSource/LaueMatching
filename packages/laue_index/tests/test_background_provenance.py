"""Backgrounds carry a provenance record (kind "background").

A median background was a bare float64 file, and a missing BackgroundFile was
silently recomputed from whichever frame came first (invariant 16), so nothing
recorded which frame or filter made it. preprocess.save_background now writes
the record (FilterRadius, NMeadianPasses, the source frame's file and index,
shape) and preprocess.load_background checks it: no record warns, bytes that
disagree with the record refuse.
"""
import logging
import os
import shutil

import h5py
import numpy as np
import pytest

from laue_index import artifacts as A
from laue_index import preprocess as P


def _bg(n=16):
    return np.arange(n * n, dtype=np.float64).reshape(n, n)


def test_save_background_writes_the_record(tmp_path):
    src = tmp_path / "frame.h5"
    src.write_bytes(b"frame")
    p = tmp_path / "median.bin"
    P.save_background(_bg(), str(p), source=str(src), frame_index=3,
                      filter_radius=101, median_passes=1)
    rec = A.read_record(p)
    assert rec["kind"] == "background"
    assert rec["config"] == {"FilterRadius": 101, "NMeadianPasses": 1}
    assert rec["artifact"]["layout"] == {"shape": [16, 16], "dtype": "float64"}
    (inp,) = rec["inputs"]
    assert inp["role"] == "source_frame" and inp["path"].endswith("frame.h5")
    assert rec["extra"]["frame_index"] == 3


def test_load_background_checks_the_record(tmp_path, caplog):
    p = tmp_path / "median.bin"
    P.save_background(_bg(), str(p), filter_radius=101, median_passes=1)
    assert np.array_equal(P.load_background(str(p), 16, 16), _bg())
    raw = bytearray(p.read_bytes()); raw[0] ^= 0xFF; p.write_bytes(bytes(raw))
    with pytest.raises(A.ArtifactMismatch):
        P.load_background(str(p), 16, 16)


def test_unrecorded_background_warns_and_loads(tmp_path, caplog):
    p = tmp_path / "median.bin"
    _bg().tofile(p)
    with caplog.at_level(logging.WARNING):
        out = P.load_background(str(p), 16, 16)
    assert np.array_equal(out, _bg()) and "no provenance record" in caplog.text


def test_runimage_records_the_frame_it_computed_the_background_from(tmp_path, monkeypatch):
    import test_run_artifact_checks as T
    res, called, out_h5 = T._runimage(tmp_path, monkeypatch, [0.35238, 0.35238, 0.35238, 90, 90, 90])
    assert res["success"], res
    bg = tmp_path / "results" / "median.bin"
    rec = A.read_record(bg)
    assert rec is not None and rec["kind"] == "background"
    assert rec["inputs"][0]["path"].endswith("frame.h5")
    assert "FilterRadius" in rec["config"]


def test_the_image_server_uses_the_recording_writer():
    import laue_image_server
    src = open(laue_image_server.__file__).read()
    assert ".tofile(" not in src, "a background is written without its record"
    assert src.count("save_background(") >= 2
