"""Indexer output columns and image numbers, read the same way everywhere.

* The orientation matrix and the other solution columns are located by the table's
  COLUMN COUNT through laue_index.records.SOLUTION_FORMATS (frame_peaks.solution_format):
  34 columns = RunImage (matrix 22..30), 35 = stream (matrix 23..31), anything else
  exits. Seven scripts hard-coded the stream ``23:32``; on a RunImage file
  distinct_peak_gate.py joined on the wrong column and still wrote a wrong npz.
* ``image_<N>.output.h5`` numbers are parsed from the full digit run; the old
  ``[:5]`` truncated from image 100000 on.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


@pytest.fixture
def fp():
    d = str(_analysis_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    import frame_peaks
    return frame_peaks


def _table(n_cols, oms, grains, image=7):
    """A solution table of the given layout with known OMs, grain ids, NMatches."""
    from laue_index.records import SOLUTION_FORMATS
    fmt = [f for f in SOLUTION_FORMATS.values() if f.n_cols == n_cols][0]
    t = np.zeros((len(oms), n_cols))
    for i, (om, g) in enumerate(zip(oms, grains)):
        t[i, fmt.om_start:fmt.om_start + 9] = om.ravel()
        t[i, fmt.grain] = g
        t[i, fmt.n_matches] = 20 + i
        t[i, fmt.quality] = 1.0
        t[i, fmt.intensity] = 100.0
        if fmt.image_nr >= 0:
            t[i, fmt.image_nr] = image
    return t


def _rot(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


@pytest.mark.parametrize("n_cols,name", [(34, "runimage"), (35, "stream")])
def test_orientation_block_by_column_count(fp, n_cols, name):
    oms = [_rot(1), _rot(2)]
    t = _table(n_cols, oms, [3, 4])
    assert fp.solution_format(n_cols).name == name
    assert np.array_equal(fp.orientation_block(t).reshape(-1, 3, 3), np.array(oms))


def test_unknown_layout_exits(fp):
    with pytest.raises(SystemExit, match="36-column"):
        fp.orientation_block(np.zeros((2, 36)))


def test_spot_columns_match_the_stream_positions_scripts_used(fp):
    """Stream: grain 1, h,k,l 3,4,5, x,y 6,7, intensity 11 (the old literals);
    RunImage: the same, one lower."""
    st = fp.spot_columns(fp.solution_format(35))
    assert (st["grain"], st["h"], st["k"], st["l"], st["x"], st["y"], st["intensity"]) == \
        (1, 3, 4, 5, 6, 7, 11)
    ri = fp.spot_columns(fp.solution_format(34))
    assert all(ri[k] == st[k] - 1 for k in st)


@pytest.mark.parametrize("name,n", [("image_00042.output.h5", 42),
                                    ("/a/b/image_100000.output.h5", 100000),
                                    ("image_1234567.output.h5", 1234567)])
def test_image_number_full_digit_run(fp, name, n):
    assert fp.image_number(name) == n
    if n >= 100000:        # what the old parse returned
        assert int(os.path.basename(name).split("image_")[1][:5]) != n


def test_no_script_hard_codes_the_stream_layout_or_truncates_image_numbers():
    bad = []
    for p in sorted(_analysis_dir().glob("*.py")):
        code = "\n".join(l.split("#")[0] for l in p.read_text().splitlines())
        code = re.sub(r'""".*?"""', "", code, flags=re.S)       # prose may cite the old literals
        if re.search(r"\b23\s*:\s*32\b|slice\(\s*23\s*,\s*32\s*\)", code):
            bad.append(f"{p.name}: 23:32")
        if re.search(r"split\(\s*[\"']image_[\"']\s*\)\s*\[\s*1\s*\]\s*\[\s*:\s*5\s*\]", code):
            bad.append(f"{p.name}: image number truncated to 5 digits")
    assert not bad, bad


@pytest.mark.parametrize("n_cols", [34, 35])
def test_distinct_peak_gate_on_both_layouts(tmp_path, n_cols):
    """End to end: the joined wta_unique counts and the written matrices are right
    for a RunImage file as well as a stream file."""
    h5py = pytest.importorskip("h5py")
    res = tmp_path / "results"
    res.mkdir()
    oms = [_rot(5), _rot(6)]
    t = _table(n_cols, oms, [3, 4], image=100001)
    with h5py.File(res / "image_100001.output.h5", "w") as h:
        h.create_dataset("entry/results/filtered_orientations", data=t)
        h.create_dataset("entry/results/unique_spots_per_orientation",
                         data=np.array([[3, 15], [4, 9]]))
    out = tmp_path / "gate.npz"
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    r = subprocess.run([sys.executable, "distinct_peak_gate.py", str(res), str(out), "--nw=1"],
                       cwd=_analysis_dir(), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    z = np.load(out, allow_pickle=True)
    assert np.allclose(z["oms"], np.array(oms))
    assert list(z["distinct"]) == [15, 9] and list(z["matched"]) == [20, 21]
    assert list(z["image"]) == [100001, 100001]


def test_distinct_peak_gate_refuses_unknown_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    res = tmp_path / "results"
    res.mkdir()
    with h5py.File(res / "image_00001.output.h5", "w") as h:
        h.create_dataset("entry/results/filtered_orientations", data=np.zeros((2, 33)))
    r = subprocess.run([sys.executable, "distinct_peak_gate.py", str(res), str(tmp_path / "o.npz"),
                        "--nw=1"], cwd=_analysis_dir(), capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "33-column" in (r.stderr + r.stdout)
    assert not (tmp_path / "o.npz").exists()


def test_scan_map_requires_laue_work():
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    env["LAUE_PHASES"] = "zn"
    r = subprocess.run([sys.executable, "scan_map.py"], cwd=_analysis_dir(), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "LAUE_WORK is not set" in r.stderr


def test_fullped_creates_output_dir_before_reading_frames(tmp_path):
    work, data = tmp_path / "work", tmp_path / "data"
    work.mkdir(); data.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    env.update(LAUE_WORK=str(work), LAUE_SCAN_DATA=str(data), LAUE_NROWS="2", LAUE_NR="2")
    r = subprocess.run([sys.executable, "fullped.py"], cwd=_analysis_dir(), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "no frames matched" in r.stderr     # stops at the read pass
    assert (work / "analysis_out").is_dir()                          # ...after making its output dir
