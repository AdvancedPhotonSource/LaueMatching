"""IndexFile header honesty: NiData is set, and err(deg) is labelled for what it is.

``build_from_h5`` never set ``n_input_data``, so every IndexFile said
``$NiData 0`` ("total number of data spots"). And ``err(deg)`` compares each
PREDICTED Qhat with the Qhat of its own integer-truncated predicted pixel (the
only pixel spots.txt carries), so it is pixel quantisation, not a
(measured - predicted) residual; the file used to call it the latter.
"""
import math

import numpy as np
import pytest

import laue_indexfile as lif

h5py = pytest.importorskip("h5py")

_GEOM = {
    "ehi": 30.0, "elo": 5.0, "space_group": 225,
    "lattice_parameter": "0.35238 0.35238 0.35238 90 90 90",
    "nr_px_x": 2048, "nr_px_y": 2048,
    "r_array": "-1.2 -1.2 -1.2", "p_array": "0.028745 0.002788 0.513115",
    "px_x": 0.0002, "px_y": 0.0002, "maxAngle": 0.1,
}


def _write(path, n_centers=None, with_results=True):
    orient = np.zeros(35)
    orient[0], orient[1], orient[5] = 1, 0, 100.0
    orient[8:17] = (np.eye(3) * 17.83).reshape(-1)
    orient[17:23] = [0.35238, 0.35238, 0.35238, 90, 90, 90]
    orient[23:32] = np.eye(3).reshape(-1)
    spot = np.array([1, 0, 0, 0, 0, 2, 100.0, 200.0, 0.1, 0.2, -0.9, 1000.0])
    with h5py.File(path, "w") as hf:
        g = hf.require_group("/entry/results")
        if with_results:
            g.create_dataset("filtered_orientations", data=orient.reshape(1, -1))
            g.create_dataset("filtered_spots", data=spot.reshape(1, -1))
        if n_centers is not None:
            cc = np.column_stack([np.arange(1, n_centers + 1),
                                  np.linspace(50, 1500, n_centers),
                                  np.linspace(60, 1600, n_centers),
                                  np.full(n_centers, 9.0)])
            hf.require_group("/entry/data").create_dataset(
                "component_centers", data=cc)


def test_nidata_from_component_centers(tmp_path):
    p = tmp_path / "a.output.h5"
    _write(p, n_centers=37)
    header, patterns = lif.build_from_h5(p, _GEOM, {"file": "x.h5", "n_spots": 99})
    assert header.n_input_data == 37          # the segmented peaks win
    assert header.n_indexed == 1


def test_nidata_from_frame_mapping_when_no_centers(tmp_path):
    """The streaming server records n_spots per frame in frame_mapping.json."""
    p = tmp_path / "b.output.h5"
    _write(p, n_centers=None)
    header, _ = lif.build_from_h5(p, _GEOM, {"file": "x.h5", "n_spots": 42})
    assert header.n_input_data == 42


def test_nidata_set_even_when_nothing_indexed(tmp_path):
    p = tmp_path / "c.output.h5"
    _write(p, n_centers=12, with_results=False)
    header, patterns = lif.build_from_h5(p, _GEOM)
    assert patterns == [] and header.n_input_data == 12


def test_nidata_reaches_the_file(tmp_path):
    p = tmp_path / "d.output.h5"
    _write(p, n_centers=21)
    out = lif.write_from_h5(p, tmp_path / "d.indexing.txt", _GEOM, {"file": "x.h5"})
    txt = out.read_text()
    assert "$NiData\t\t21" in txt
    assert "out of 21 spots" in txt


def test_err_is_labelled_as_pixel_quantisation(tmp_path):
    p = tmp_path / "e.output.h5"
    _write(p, n_centers=3)
    txt = lif.write_from_h5(p, tmp_path / "e.indexing.txt", _GEOM).read_text()
    assert "quantisation" in txt
    assert "rms error of (measured-predicted)" not in txt


def test_err_is_bounded_by_one_pixel():
    """What err(deg) actually measures: a predicted Qhat against the Qhat of the
    same prediction truncated to an integer pixel. Bounded by about one pixel's
    angular size (0.2 mm at ~0.5 m ~ 0.02 deg in 2theta, less in Qhat), and it
    can be far from 0 for a perfect fit. Bound 0.05 deg is geometry, not a
    platform tolerance."""
    kw = dict(r_array=(-1.2, -1.2, -1.2), p_array=(0.028745, 0.002788, 0.513115),
              px_x=0.0002, px_y=0.0002, nr_px_x=2048, nr_px_y=2048)
    errs = []
    for px, py in ((100.7, 200.3), (1024.99, 1023.5), (1800.5, 300.9)):
        exact = lif.pixel_to_qhat(px, py, **kw)
        trunc = lif.pixel_to_qhat(int(px), int(py), **kw)
        errs.append(lif.angular_error_deg(exact, trunc))
    assert max(errs) < 0.05
    assert max(errs) > 1e-4          # not ~0: it is quantisation, not a residual
