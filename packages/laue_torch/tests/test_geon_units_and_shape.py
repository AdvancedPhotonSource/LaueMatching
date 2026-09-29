"""34-ID-E geoN XML -> LaueParams: units, pixel count and binning.

* ``unit`` attributes are checked (size / P in mm, R in radian) instead of
  assumed; a wrong unit raises.
* ``make_lauematching_params(..., frame_shape=(rows, cols))`` checks the frame
  against ``Npixels``: an integer binning rescales the pixel count and size
  (and is recorded), anything else raises.
* End to end: a reflection whose diffracted ray points at the detector point
  P (+ k pixels along detector x) lands on the centre pixel (+ k) of the
  forward model built from the parsed file.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch import LaueForwardModel
from laue_torch.geometry import reciprocal_matrix, rodrigues_to_matrix
from laue_torch.realdata import Crystal, make_lauematching_params, parse_geon_xml

DT = torch.float64

XML = """<?xml version="1.0"?>
<geoN xmlns="http://sector34.xray.aps.anl.gov/34ide/geoN">
  <Sample><Origin unit="micron">0 0 0</Origin><R unit="radian">0 0 0</R></Sample>
  <Detectors Ndetectors="1">
    <Detector N="0">
      <Npixels>2048 1024</Npixels>
      <size unit="mm">409.6 204.8</size>
      <R unit="radian">{R}</R>
      <P unit="{PU}">25.0 -2.0 510.0</P>
      <ID>PE1621 test</ID>
    </Detector>
  </Detectors>
</geoN>
"""
R_TXT = "-1.20131258 -1.21399082 -1.21881158"
CRYSTAL = Crystal(sg_num=225, lattice_nm=(0.35238,) * 3 + (90.0,) * 3)


def _write(tmp_path, pu="mm", R=R_TXT):
    f = tmp_path / "geoN.xml"
    f.write_text(XML.format(PU=pu, R=R))
    return f


def test_units_are_checked(tmp_path):
    g = parse_geon_xml(_write(tmp_path))
    assert g.P_mm == (25.0, -2.0, 510.0) and g.px_size_mm_x == pytest.approx(0.2)
    with pytest.raises(ValueError, match="unit"):
        parse_geon_xml(_write(tmp_path, pu="micron"))


def test_frame_shape_and_binning(tmp_path):
    g = parse_geon_xml(_write(tmp_path))
    p = make_lauematching_params(g, CRYSTAL, E_lo=5.0, E_hi=30.0, frame_shape=(1024, 2048))
    assert (p.n_pix_x, p.n_pix_y) == (2048, 1024)
    b = make_lauematching_params(g, CRYSTAL, E_lo=5.0, E_hi=30.0, frame_shape=(512, 1024))
    assert (b.n_pix_x, b.n_pix_y) == (1024, 512)
    assert b.px_x == pytest.approx(2 * p.px_x) and b.px_y == pytest.approx(2 * p.px_y)
    assert b.extras["binning"] == (2, 2)
    with pytest.raises(ValueError, match="Npixels"):
        make_lauematching_params(g, CRYSTAL, E_lo=5.0, E_hi=30.0, frame_shape=(2048, 1024))
    with pytest.raises(ValueError, match="Npixels"):
        make_lauematching_params(g, CRYSTAL, E_lo=5.0, E_hi=30.0, frame_shape=(1000, 2048))


def _rotation_taking(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    v = torch.linalg.cross(a, b)
    s, c = v.norm(), torch.dot(a, b)
    return rodrigues_to_matrix(v / s * torch.atan2(s, c))


@pytest.mark.parametrize("k_px", [0.0, 10.0])
def test_ray_to_detector_point_lands_on_that_pixel(tmp_path, k_px):
    g = parse_geon_xml(_write(tmp_path))
    p = make_lauematching_params(g, CRYSTAL, E_lo=1.0, E_hi=100.0)
    t = p.to_tensors()
    Rm = rodrigues_to_matrix(t["R"])
    target_det = t["P"] + torch.tensor([k_px * p.px_x, 0.0, 0.0], dtype=DT)
    kf = Rm @ target_det
    kf = kf / kf.norm()
    ki = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    qhat = (kf - ki) / (kf - ki).norm()
    h = torch.tensor([1.0, 1.0, 1.0], dtype=DT)
    g111 = reciprocal_matrix(t["lattice"]) @ h
    U = _rotation_taking(g111 / g111.norm(), qhat)
    m = LaueForwardModel(hkls=torch.tensor([[1, 1, 1]]), n_pix=t["n_pix"],
                         px_size=t["px_size"], psf_sigma=1.0, hard=True)
    _, aux = m(U.unsqueeze(0), t["lattice"], t["P"], t["R"], E_range=(1.0, 100.0),
               return_aux=True)
    assert float(aux.px[0]) == pytest.approx(0.5 * (p.n_pix_x - 1) + k_px, abs=1e-6)
    assert float(aux.py[0]) == pytest.approx(0.5 * (p.n_pix_y - 1), abs=1e-6)
