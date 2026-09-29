"""Forward model: energy image uses the same PSF as the intensity image, and
the render window is wide enough for a pseudo-Voigt PSF.

* The aux-path energy image was splatted with the Gaussian PSF even when
  psf_eta > 0, so energy_image / image was not the spot energy.
* The default window was 2*ceil(3 sigma)+1 whatever eta was; a Lorentzian
  tail truncated at 3 sigma biases the spot centroid by ~0.06 px.
"""
from __future__ import annotations

import torch

from laue_torch import LaueForwardModel
from laue_torch.rasterize import pseudo_voigt_splat

DT = torch.float64
P = torch.tensor([0.028745, 0.002788, 0.513115], dtype=DT)
R = torch.tensor([-1.20131258, -1.21399082, -1.21881158], dtype=DT)
LAT = torch.tensor([0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0], dtype=DT)


def test_energy_image_over_image_is_the_spot_energy_with_eta():
    from midas_stress.orientation import quat_to_orient_mat
    U = quat_to_orient_mat(torch.tensor(
        [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
        dtype=DT)).reshape(1, 3, 3)
    # Pick one reflection that lands well inside a 96x64 detector.
    from laue_torch.io import generate_hkls
    hkls = generate_hkls(225, tuple(LAT.tolist()), 15.0)
    probe = LaueForwardModel(hkls=hkls, n_pix=(96, 64), px_size=(0.006, 0.006),
                             psf_sigma=1.5, hard=True)
    _, a = probe(U, LAT, P, R, E_range=(5.0, 15.0), return_aux=True)
    inside = ((a.mask > 0.5) & (a.px > 15) & (a.px < 80)
              & (a.py > 15) & (a.py < 48)).nonzero().reshape(-1)
    assert inside.numel() > 0, "fixture: no reflection inside the detector"
    m = LaueForwardModel(hkls=hkls[int(inside[0])].unsqueeze(0), n_pix=(96, 64),
                         px_size=(0.006, 0.006), psf_sigma=1.5, psf_eta=0.6,
                         energy_image=True, hard=True)
    img, aux = m(U, LAT, P, R, E_range=(5.0, 15.0), return_aux=True)
    sel = img > 1e-6 * img.max()
    ratio = aux.energy_image[sel] / img[sel]
    assert torch.allclose(ratio, aux.energy[0].expand_as(ratio), rtol=1e-10)


def _centroid(img):
    Nx, Ny = img.shape
    x = torch.arange(Nx, dtype=DT)[:, None]
    y = torch.arange(Ny, dtype=DT)[None, :]
    s = img.sum()
    return float((img * x).sum() / s), float((img * y).sum() / s)


def test_default_window_keeps_pseudo_voigt_centroid_unbiased():
    sigma, eta = 1.0, 0.5
    m = LaueForwardModel(hkls=torch.tensor([[1, 1, 1]]), n_pix=(64, 64),
                         px_size=(1e-4, 1e-4), psf_sigma=sigma, psf_eta=eta)
    worst = 0.0
    for dx in (0.1, 0.25, 0.4, 0.49):
        px = torch.tensor([32.0 + dx], dtype=DT)
        py = torch.tensor([31.0 - 0.3 * dx], dtype=DT)
        img = pseudo_voigt_splat(px, py, torch.ones(1, dtype=DT), n_pix=(64, 64),
                                 sigma=sigma, window=m.render_window, eta=eta)
        cx, cy = _centroid(img)
        worst = max(worst, abs(cx - float(px)), abs(cy - float(py)))
    assert worst < 1e-3, f"centroid bias {worst:.4f} px with window {m.render_window}"


def test_gaussian_default_window_unchanged():
    m = LaueForwardModel(hkls=torch.tensor([[1, 1, 1]]), n_pix=(64, 64),
                         px_size=(1e-4, 1e-4), psf_sigma=2.0)
    assert m.render_window == 13
