"""P1 (code read 2026-09-28): streaming post-processing fixes.

Each test failed before its fix:

* ``frame_idx=`` was passed to ``load_h5_image`` (whose keyword is
  ``frame_index``), the TypeError was swallowed by a warning, and no streaming
  output h5 ever carried ``/entry/data``.
* the re-preprocessing in post-processing recomputed a per-frame background and
  applied no exclusion, so ``/entry/data`` did not show what was indexed.
* frames the daemon never reports on (skipped by the server, or no solution)
  got no ``image_*.output.h5``, so ``wait_static.sh`` could never reach
  COMPLETE on a scan with any empty frame.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import laue_postprocess as pp  # noqa: E402
import laue_stream_utils as lsu  # noqa: E402

NPX = 64

# Every key the Python parsers require (config_schema.REQUIRED_KEYS), so the
# fixture keeps parsing after P3.
PARAMS = f"""SpaceGroup 225
Symmetry F
LatticeParameter 0.35238 0.35238 0.35238 90 90 90
P_Array 0.0 0.0 0.05
R_Array 0.0 0.0 0.0
Elo 5
Ehi 30
NrPxX {NPX}
NrPxY {NPX}
PxX 0.0002
PxY 0.0002
MaxNrLaueSpots 30
MinIntensity 0
NMeadianPasses 1
MinGoodSpots 1
RobustFilter 1
BackgroundFile bg_from_frame1.bin
FilterRadius 5
ThresholdMethod fixed
Threshold 5
MinArea 1
"""


def _frames():
    """Two frames that differ: one spot each, at different places."""
    f = np.zeros((2, NPX, NPX), dtype=np.float32)
    f[0, 10:13, 10:13] = 500.0
    f[1, 40:43, 20:23] = 900.0
    return f


def _write_scan(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / "scan.h5", "w") as hf:
        hf.create_dataset("/entry/data/data", data=_frames())


def _stream_inputs(image_nr):
    orient = np.zeros((1, 35))
    orient[0, 0] = image_nr
    orient[0, 5] = 10.0
    orient[0, 6] = 2
    orient[0, 23:32] = np.eye(3).ravel()
    spots = np.array([[image_nr, 0, 0, 1, 1, 1, 21.0, 41.0, 0, 0, 1, 900.0],
                      [image_nr, 0, 1, 2, 0, 0, 30.0, 30.0, 0, 0, 1, 10.0]])
    return orient, spots


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "params.txt"
    p.write_text(PARAMS)
    return lsu.parse_config(str(p))


def test_output_h5_embeds_the_right_frame(tmp_path, cfg):
    """2-frame file, image mapped to frame 1: /entry/data/raw_data is frame 1."""
    folder = tmp_path / "frames"
    _write_scan(folder)
    out = tmp_path / "results"
    out.mkdir()
    orient, spots = _stream_inputs(2)
    pp.process_single_image(image_nr=2, orientations=orient, spots=spots, cfg=cfg,
                            output_dir=str(out), min_unique=1,
                            mapping_info={"file": "scan.h5", "frame": 1},
                            folder=str(folder), write_indexfile=False)
    with h5py.File(out / "image_00002.output.h5", "r") as hf:
        assert "/entry/data/raw_data" in hf, "no /entry/data written"
        np.testing.assert_array_equal(hf["/entry/data/raw_data"][()], _frames()[1])


# --------------------------------------------------------------------------- #
# The re-preprocessing uses what the server used                              #
# --------------------------------------------------------------------------- #

def _fake_daemon():
    """A TCP listener that accepts one client and reads until it closes."""
    import socket
    import threading
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    got = {"bytes": 0}

    def run():
        c, _ = srv.accept()
        while True:
            b = c.recv(1 << 16)
            if not b:
                break
            got["bytes"] += len(b)
        c.close()
        srv.close()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return srv.getsockname()[1], th, got


def test_server_saves_its_background_and_records_it(tmp_path):
    import laue_image_server as srv
    folder = tmp_path / "frames"
    _write_scan(folder)
    run = tmp_path / "run"
    run.mkdir()
    (run / "excl.txt").write_text("21 41 4\n")
    params = tmp_path / "params.txt"
    params.write_text(PARAMS + "ExcludeSpotsFile excl.txt\n")
    port, th, _ = _fake_daemon()
    cwd = os.getcwd()
    os.chdir(run)                   # the orchestrator starts the server in the output dir
    try:
        srv.serve_images(str(params), str(folder), mapping_file=str(run / "frame_mapping.json"),
                         labels_file=str(run / "labels.h5"), port=port)
    finally:
        os.chdir(cwd)
    th.join(timeout=10)
    rec = lsu.read_preprocess_record(str(run / lsu.PREPROCESS_RECORD))
    assert rec is not None, "server wrote no preprocess record"
    assert os.path.isabs(rec["background_file"]) and os.path.isfile(rec["background_file"])
    assert rec["exclude_spots_file"] == str(run / "excl.txt")
    cfg = lsu.parse_config(str(params))
    expect = lsu.compute_background(_frames()[0].astype(np.float64),
                                    filter_radius=cfg["filter_radius"],
                                    median_passes=cfg["median_passes"])
    got = lsu.load_background(rec["background_file"], NPX, NPX)
    np.testing.assert_array_equal(got, expect)


def test_postprocess_uses_the_recorded_background_and_exclusion(tmp_path, monkeypatch):
    """Run from another cwd: the embedded frame is preprocessed with the server's
    background (a constant 400 here, not a per-frame median) and the server's
    exclusion list (a relative path the server resolved in the output dir)."""
    folder = tmp_path / "frames"
    _write_scan(folder)
    run = tmp_path / "run"
    run.mkdir()
    params = tmp_path / "params.txt"
    params.write_text(PARAMS)
    np.full((NPX, NPX), 400.0).tofile(run / "bg.bin")
    (run / "excl.txt").write_text("11 11 4\n")        # the spot of frame 0
    (run / lsu.PREPROCESS_RECORD).write_text(json.dumps({
        "background_file": str(run / "bg.bin"),
        "exclude_spots_file": str(run / "excl.txt"), "exclude_spots_dir": "",
        "nr_px_x": NPX, "nr_px_y": NPX}))
    mapping = {"1": {"file": "scan.h5", "frame": 0, "skipped": False, "n_spots": 1},
               "2": {"file": "scan.h5", "frame": 1, "skipped": False, "n_spots": 1}}
    (run / "frame_mapping.json").write_text(json.dumps(mapping))
    o1, s1 = _stream_inputs(1)
    o2, s2 = _stream_inputs(2)
    sol = run / "solutions.txt"
    spt = run / "spots.txt"
    np.savetxt(sol, np.vstack([o1, o2]), header="hdr", comments="")
    np.savetxt(spt, np.vstack([s1, s2]), header="hdr", comments="")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    pp.postprocess(str(sol), str(spt), str(params), str(run / "results"),
                   mapping_file=str(run / "frame_mapping.json"), folder=str(folder),
                   write_indexfile=False)
    with h5py.File(run / "results" / "image_00002.output.h5", "r") as hf:
        # 900 - 400: the recorded background, not a 5x5 median of this frame (0)
        assert hf["/entry/data/cleaned_data_threshold"][()].max() == 500
    with h5py.File(run / "results" / "image_00001.output.h5", "r") as hf:
        assert hf["/entry/data/component_centers"].shape[0] == 0, \
            "the recorded exclusion was not applied"


# --------------------------------------------------------------------------- #
# Every mapped frame gets an output h5, so wait_static.sh can reach COMPLETE  #
# --------------------------------------------------------------------------- #

def _pp_run(tmp_path, mapping, rows_by_image):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    params = tmp_path / "params.txt"
    params.write_text(PARAMS)
    (run / "frame_mapping.json").write_text(json.dumps(mapping))
    sol = run / "solutions.txt"
    spt = run / "spots.txt"
    o = [_stream_inputs(i)[0] for i in rows_by_image]
    s = [_stream_inputs(i)[1] for i in rows_by_image]
    sol.write_text("hdr\n" + "".join(" ".join(map(str, r)) + "\n" for a in o for r in a))
    spt.write_text("hdr\n" + "".join(" ".join(map(str, r)) + "\n" for a in s for r in a))
    pp.postprocess(str(sol), str(spt), str(params), str(run / "results"),
                   mapping_file=str(run / "frame_mapping.json"), write_indexfile=False)
    return run / "results"


def test_skipped_and_no_solution_frames_get_a_stub(tmp_path):
    mapping = {"1": {"file": "a.h5", "frame": 0, "skipped": False, "n_spots": 3},
               "2": {"file": "a.h5", "frame": 1, "skipped": True, "reason": "no_spots"},
               "3": {"file": "a.h5", "frame": 2, "skipped": False, "n_spots": 2}}
    res = _pp_run(tmp_path, mapping, rows_by_image=[1])
    for n in (1, 2, 3):
        assert (res / f"image_{n:05d}.output.h5").is_file(), f"no output for image {n}"
    with h5py.File(res / "image_00002.output.h5", "r") as hf:
        g = hf["/entry/results"]
        assert g.attrs["n_filtered"] == 0
        assert g.attrs["skip_reason"] == "no_spots"
        assert g.attrs["source_frame"] == 1
        assert g["filtered_orientations"].shape[0] == 0
    with h5py.File(res / "image_00003.output.h5", "r") as hf:
        assert hf["/entry/results"].attrs["skip_reason"] == "no_solution"
    summary = (res / "summary.txt").read_text()
    assert len([ln for ln in summary.splitlines()[2:] if ln.strip()]) == 3


def test_no_solutions_at_all_still_writes_stubs(tmp_path):
    mapping = {"1": {"file": "a.h5", "frame": 0, "skipped": False, "n_spots": 3},
               "2": {"file": "a.h5", "frame": 1, "skipped": True, "reason": "no_spots"}}
    res = _pp_run(tmp_path, mapping, rows_by_image=[])
    assert sorted(p.name for p in res.glob("image_*.output.h5")) == \
        ["image_00001.output.h5", "image_00002.output.h5"]


def test_wait_static_counts_what_postprocess_writes():
    """wait_static.sh counts image_*.output.h5 against the frames SENT to the
    run; with a stub per skipped / empty frame the two now agree, and its
    header says so."""
    from conftest import _REPO_ROOT
    if _REPO_ROOT is None:
        pytest.skip("pipeline/ needs a checkout")
    src = (_REPO_ROOT / "pipeline" / "dispatch" / "wait_static.sh").read_text()
    assert "image_*.output.h5" in src
    assert "stub" in src
    assert "legitimately produce no output" not in src
