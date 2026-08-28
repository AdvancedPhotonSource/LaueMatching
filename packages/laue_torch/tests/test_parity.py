"""Numerical parity vs the existing NumPy/C forward simulator.

Compares the set of integer detector pixels each pipeline lights up for the
golden case (simulation/params_sim.txt + simulation/fourOrientations.csv).
We test the *geometry* path, not the rasterizer — the original code does
``int(pixel)`` (floor) splat then a Gaussian blur on uint16 intensities; we
exercise the differentiable forward and use ``aux.px/py`` to recover the
floor-pixel set. This isolates the math.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from laue_torch import LaueForwardModel, parse_params

HC_KEV_NM = 1.2398419739


def _ref_pixels(params: dict, hkl_data: np.ndarray, U_all: np.ndarray):
    rotang = np.linalg.norm(params["R"])
    rv = params["R"] / rotang
    c = math.cos(rotang); s = math.sin(rotang); C = 1 - c
    x, y, z = rv
    rot = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    rot_T = rot.T
    out = []
    for gi, U_g in enumerate(U_all):
        recip = U_g * params["astar"]
        hkls = hkl_data[:, :3]
        qvecs = recip.dot(hkls.T).T
        qlens = np.linalg.norm(qvecs, axis=1)
        good = qlens > 0
        qvecs = qvecs[good]; qlens = qlens[good]
        qhats = qvecs / qlens[:, None]
        dots = qhats[:, 2]
        ki = np.array([0.0, 0.0, 1.0])
        kfs = ki - 2 * dots[:, None] * qhats
        xyz = rot_T.dot(kfs.T).T
        gz = xyz[:, 2] > 0
        qhats = qhats[gz]; xyz = xyz[gz]; qlens = qlens[gz]
        xyz = xyz * params["P"][2] / xyz[:, 2:3]
        xp = xyz[:, 0] - params["P"][0]
        yp = xyz[:, 1] - params["P"][1]
        pxf = xp / params["dx"] + 0.5 * (params["nPxX"] - 1)
        pyf = yp / params["dy"] + 0.5 * (params["nPxY"] - 1)
        within = ((pxf >= 0) & (pxf < params["nPxX"] - 1)
                  & (pyf >= 0) & (pyf < params["nPxY"] - 1))
        pxf = pxf[within]; pyf = pyf[within]
        qlens = qlens[within]; qhats = qhats[within]
        sin_t = -qhats[:, 2]
        E = HC_KEV_NM * qlens / sin_t / (4 * math.pi)
        e_ok = (E > params["Elo"]) & (E < params["Ehi"])
        pxf = pxf[e_ok]; pyf = pyf[e_ok]
        out.extend((gi, int(x_), int(y_)) for x_, y_ in zip(pxf, pyf))
    return out


def test_pixel_set_matches_numpy_reference(params_path, hkl_csv, four_orients):
    import GenerateSimulation as gs

    cp = gs.ConfigParser(str(params_path))
    params = cp.get_params()
    hkl_data = np.genfromtxt(hkl_csv)
    orients_flat = np.genfromtxt(four_orients)

    p = parse_params(str(params_path))
    t = p.to_tensors()
    hkls = torch.tensor(hkl_data[:, :3].astype(np.int64))
    model = LaueForwardModel(
        hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
        psf_sigma=1.0, render_window=3,
        strain_mode="none", rotation="matrix", detector_rotation="rodrigues",
        hard=True,
    )
    U = torch.tensor(orients_flat.reshape(-1, 3, 3), dtype=torch.float64)
    _, aux = model(U, t["lattice"], t["P"], t["R"], return_aux=True)

    keep = aux.mask > 0.5
    gi = aux.grain_idx[keep].tolist()
    px = aux.px[keep].tolist()
    py = aux.py[keep].tolist()
    torch_set = {(g, int(x_), int(y_)) for g, x_, y_ in zip(gi, px, py)}
    ref_set = set(_ref_pixels(params, hkl_data, orients_flat.reshape(-1, 3, 3)))

    missing = ref_set - torch_set
    extra = torch_set - ref_set
    assert not missing, f"{len(missing)} reference pixels missing from torch output: {list(missing)[:5]}"
    assert not extra, f"{len(extra)} torch pixels not in reference: {list(extra)[:5]}"


def test_render_produces_image(params_path, hkl_csv, four_orients):
    import GenerateSimulation as gs

    p = parse_params(str(params_path))
    t = p.to_tensors()
    cp = gs.ConfigParser(str(params_path))
    params = cp.get_params()
    # Use the hkl_csv FIXTURE, not a path built from params_path. valid_hkls.csv
    # is generated, not committed, so it is absent from a fresh clone and from
    # an installed sdist; the fixture skips, a hand-built path raises
    # FileNotFoundError. The other three parity tests already use it -- this one
    # did not, and was the only parity test that failed on CI.
    hkl_data = np.genfromtxt(hkl_csv)
    orients = np.genfromtxt(four_orients).reshape(-1, 3, 3)

    hkls = torch.tensor(hkl_data[:, :3].astype(np.int64))
    model = LaueForwardModel(
        hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
        psf_sigma=2.0, render_window=9,
        strain_mode="none", rotation="matrix", detector_rotation="rodrigues",
        hard=True,
    )
    U = torch.tensor(orients, dtype=torch.float64)
    img = model(U, t["lattice"], t["P"], t["R"])
    assert img.shape == t["n_pix"]
    assert img.sum() > 0
    assert torch.isfinite(img).all()


def test_stack_mode_returns_per_grain(params_path, hkl_csv, four_orients):
    p = parse_params(str(params_path))
    t = p.to_tensors()
    hkl_data = np.genfromtxt(hkl_csv)
    orients = np.genfromtxt(four_orients).reshape(-1, 3, 3)

    hkls = torch.tensor(hkl_data[:, :3].astype(np.int64))
    model_sum = LaueForwardModel(
        hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
        psf_sigma=2.0, render_window=9,
        strain_mode="none", rotation="matrix", detector_rotation="rodrigues",
        hard=True, reduce="sum",
    )
    model_stk = LaueForwardModel(
        hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
        psf_sigma=2.0, render_window=9,
        strain_mode="none", rotation="matrix", detector_rotation="rodrigues",
        hard=True, reduce="stack",
    )
    U = torch.tensor(orients, dtype=torch.float64)
    img_sum = model_sum(U, t["lattice"], t["P"], t["R"])
    img_stk = model_stk(U, t["lattice"], t["P"], t["R"])
    G = orients.shape[0]
    assert img_stk.shape == (G,) + tuple(t["n_pix"])
    assert torch.allclose(img_stk.sum(0), img_sum, atol=1e-9)


def test_energy_image_is_intensity_weighted(params_path, hkl_csv, four_orients):
    p = parse_params(str(params_path))
    t = p.to_tensors()
    hkl_data = np.genfromtxt(hkl_csv)
    orients = np.genfromtxt(four_orients).reshape(-1, 3, 3)

    hkls = torch.tensor(hkl_data[:, :3].astype(np.int64))
    model = LaueForwardModel(
        hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
        psf_sigma=2.0, render_window=9,
        strain_mode="none", rotation="matrix", detector_rotation="rodrigues",
        hard=True, energy_image=True,
    )
    U = torch.tensor(orients, dtype=torch.float64)
    img, aux = model(U, t["lattice"], t["P"], t["R"], return_aux=True)
    assert aux.energy_image is not None
    assert aux.energy_image.shape == t["n_pix"]
    # Wherever intensity is positive, energy_image / intensity gives a
    # weighted-average energy; should fall inside [Elo, Ehi].
    nonzero = img > 1e-3
    avg_E = aux.energy_image[nonzero] / img[nonzero]
    Elo, Ehi = t["E_range"]
    margin = 0.5
    assert (avg_E >= Elo - margin).all()
    assert (avg_E <= Ehi + margin).all()
