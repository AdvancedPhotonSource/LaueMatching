"""laue-torch CLI ``-axisOrder {XY,YX}``.

Default XY (model layout, unchanged, with a warning that the indexer reads
YX). YX writes the transpose of every image array, the marker b"YX", and a
TIFF in detector layout, so the file can go straight to the indexer.
"""
from __future__ import annotations

import logging

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.cli import main
from laue_torch.io import generate_hkls, parse_params

NX, NY = 96, 64


def _setup(tmp_path):
    cfg = tmp_path / "params.txt"
    cfg.write_text(
        "LatticeParameter 0.35238 0.35238 0.35238 90 90 90\nSpaceGroup 225\nSymmetry F\n"
        "P_Array 0.028745 0.002788 0.513115\nR_Array -1.20131258 -1.21399082 -1.21881158\n"
        f"PxX 0.006\nPxY 0.006\nNrPxX {NX}\nNrPxY {NY}\n"
        "Elo 5\nEhi 15\nSimulationSmoothingWidth 1\n")
    U = quat_to_orient_mat(torch.tensor(
        [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
        dtype=torch.float64)).reshape(3, 3)
    ori = tmp_path / "orient.txt"
    np.savetxt(ori, U.numpy().reshape(1, 9))
    return cfg, ori, U


def test_yx_writes_detector_layout(tmp_path):
    cfg, ori, U = _setup(tmp_path)
    out = tmp_path / "sim_yx.h5"
    assert main(["-configFile", str(cfg), "-orientationFile", str(ori),
                 "-outputFile", str(out), "-axisOrder", "YX", "-energyImage",
                 "-dtype", "float64"]) == 0
    p = parse_params(cfg)
    t = p.to_tensors()
    m = LaueForwardModel(hkls=generate_hkls(p.sg_num, p.lattice, p.E_hi),
                         n_pix=(NX, NY), px_size=t["px_size"], psf_sigma=1.0, hard=True)
    with torch.no_grad():
        _, aux = m(U.unsqueeze(0), t["lattice"], t["P"], t["R"], E_range=(5.0, 15.0),
                   return_aux=True)
    inside = ((aux.mask > 0.5) & (aux.px > 2) & (aux.px < NX - 3)
              & (aux.py > 2) & (aux.py < NY - 3)).nonzero().reshape(-1)
    assert inside.numel() > 0
    h = int(inside[0])
    px, py = int(round(float(aux.px[h]))), int(round(float(aux.py[h])))
    with h5py.File(out, "r") as hf:
        assert hf["/entry1/axis_order"][()] == b"YX"
        data = hf["/entry1/data/data"][()]
        assert data.shape == (NY, NX)
        assert hf["/entry1/energy_image"].shape == (NY, NX)
        assert hf["/entry1/average_energy"].shape == (NY, NX)
    patch = data[py - 1:py + 2, px - 1:px + 2]
    assert data[py, px] == patch.max() and data[py, px] > 0
    tif = np.asarray(Image.open(str(out) + ".tif"))
    assert tif.shape == (NY, NX)


def test_xy_default_unchanged_and_warns(tmp_path, caplog):
    cfg, ori, _ = _setup(tmp_path)
    out = tmp_path / "sim_xy.h5"
    with caplog.at_level(logging.WARNING, logger="laue_torch.cli"):
        assert main(["-configFile", str(cfg), "-orientationFile", str(ori),
                     "-outputFile", str(out)]) == 0
    assert any("YX" in r.getMessage() for r in caplog.records)
    with h5py.File(out, "r") as hf:
        assert hf["/entry1/axis_order"][()] == b"XY"
        assert hf["/entry1/data/data"].shape == (NX, NY)


def test_bad_axis_order_rejected(tmp_path):
    cfg, ori, _ = _setup(tmp_path)
    with pytest.raises(SystemExit):
        main(["-configFile", str(cfg), "-orientationFile", str(ori),
              "-outputFile", str(tmp_path / "x.h5"), "-axisOrder", "RC"])
