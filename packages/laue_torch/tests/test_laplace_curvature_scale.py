"""The Laplace sigma is on the right scale: it matches repeated-noise scatter.

``laplace_posterior(loss, theta, noise_variance=s2)`` returns ``pinv(H(loss) /
s2)``. Fed the per-point MSE (its documented usage) the Hessian is N/2 too
small and every sigma sqrt(N/2) too large -- a factor 10 at N = 200. The
real-data refiners now go through ``laplace_posterior_from_residuals``
(``0.5 * SSR`` with plug-in ``s2 = SSR / N``); this pins that helper against
the empirical standard deviation of the estimate over repeated noise draws of
a synthetic with known noise.

Tolerance: 10% on sigma (linear, 2000 draws), 15% (nonlinear, 300 draws).
Sources of spread, all smaller: the empirical sd over the draws (relative SE 1/sqrt(2*n_draws) = 1.6% at 2000 draws, 4% at 300),
the plug-in noise variance (biased low by p/N = 1.5% here), and for the
nonlinear model the local-linearisation error at SNR ~ 40.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from laue_torch.uncertainty import laplace_posterior, laplace_posterior_from_residuals

DT = torch.float64
N = 200
SIGMA = 0.05


def test_linear_model_sigma_matches_empirical_scatter():
    x = torch.linspace(-1.0, 1.0, N, dtype=DT)
    X = torch.stack([torch.ones_like(x), x], dim=1)
    truth = torch.tensor([0.3, -1.2], dtype=DT)
    g = torch.Generator().manual_seed(7)
    n_draws = 2000
    Y = (X @ truth).unsqueeze(0) + SIGMA * torch.randn(n_draws, N, dtype=DT, generator=g)
    est = torch.linalg.lstsq(X, Y.T).solution.T                 # (n_draws, 2)
    emp = est.std(dim=0)

    lap = []
    for i in range(0, 50):                                       # a few draws suffice
        y = Y[i]
        post = laplace_posterior_from_residuals(lambda th: X @ th - y, est[i])
        lap.append(post.sigma)
    lap = torch.stack(lap).median(dim=0).values
    assert torch.allclose(lap, emp, rtol=0.10), (lap, emp)

    # The per-point-MSE usage is off by sqrt(N/2), which is what this guards.
    y = Y[0]
    mse = float(((X @ est[0] - y) ** 2).mean())
    wrong = laplace_posterior(lambda th: ((X @ th - y) ** 2).mean(), est[0],
                              noise_variance=mse).sigma
    assert torch.allclose(wrong / lap, torch.full((2,), math.sqrt(N / 2), dtype=DT),
                          rtol=0.10)


def test_gaussian_peak_sigma_matches_empirical_scatter():
    """Nonlinear: a 1-D Gaussian peak (amplitude, centre, log-width), the shape
    the forward model splats. Fitted by Gauss-Newton from the truth per draw."""
    x = torch.linspace(-6.0, 6.0, N, dtype=DT)
    truth = torch.tensor([2.0, 0.3, math.log(1.1)], dtype=DT)

    def model(th):
        return th[0] * torch.exp(-0.5 * ((x - th[1]) / th[2].exp()) ** 2)

    rng = torch.Generator().manual_seed(11)
    n_draws = 300
    ests, laps = [], []
    for i in range(n_draws):
        y = model(truth) + SIGMA * torch.randn(N, dtype=DT, generator=rng)
        th = truth.clone()
        for _ in range(8):                                        # Gauss-Newton
            J = torch.autograd.functional.jacobian(model, th)
            r = model(th) - y
            th = th - torch.linalg.lstsq(J, r.unsqueeze(1)).solution.squeeze(1)
        ests.append(th)
        if i < 30:
            laps.append(laplace_posterior_from_residuals(lambda t: model(t) - y, th).sigma)
    emp = torch.stack(ests).std(dim=0)
    lap = torch.stack(laps).median(dim=0).values
    assert np.all(np.isfinite(lap.numpy()))
    assert torch.allclose(lap, emp, rtol=0.15), (lap, emp)


# ── VoxelODFRefiner wiring ─────────────────────────────────────────────────

def _driver_case():
    from midas_stress.orientation import quat_to_orient_mat
    from laue_torch.io import LaueParams
    from laue_torch.realdata import VoxelMeasurement, VoxelODFRefiner
    p = LaueParams(
        sg_num=225, symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.006, px_y=0.006, n_pix_x=96, n_pix_y=64,
        E_lo=5.0, E_hi=12.0, psf_sigma=1.0)
    U = quat_to_orient_mat(torch.tensor(
        [0.56153266089081, -0.1069242896544219, -0.7939419137346801,
         0.2071340258144413], dtype=DT)).reshape(3, 3)
    ref = VoxelODFRefiner(p, sigma_init_deg=0.05, n_steps=2, M_render=4)
    t = ref.tensors
    with torch.no_grad():
        img = ref.model(U.unsqueeze(0), t["lattice"], t["P"], t["R"],
                        strain=torch.zeros(1, 6, dtype=DT), E_range=ref.E_range)
    g = torch.Generator().manual_seed(3)
    img = img + 0.01 * torch.randn(img.shape, dtype=DT, generator=g)
    return ref, VoxelMeasurement(0, img.T.contiguous(), U.unsqueeze(0), {},
                                 axis_order="YX")


def test_voxel_refiner_returns_posterior_with_diagnostics():
    ref, m = _driver_case()
    res = ref.refine(m)
    post = res.posterior
    assert post is not None and post.theta.shape == (6,)
    assert post.eigvals.shape == (6,) and isinstance(post.rank_eff, int)
    if post.is_positive_definite:
        assert math.isfinite(res.posterior_sigma_U_deg)
    elif torch.isnan(post.sigma[:3]).any():
        assert math.isnan(res.posterior_sigma_U_deg)


def test_voxel_refiner_posterior_does_not_swallow_errors(monkeypatch):
    import laue_torch.realdata.driver as drv
    ref, m = _driver_case()

    def boom(*a, **k):
        raise RuntimeError("not a numerical failure")
    monkeypatch.setattr(drv, "laplace_posterior_from_residuals", boom)
    with pytest.raises(RuntimeError, match="not a numerical failure"):
        ref.refine(m)

    def linalg(*a, **k):
        raise torch.linalg.LinAlgError("eigvalsh failed")
    monkeypatch.setattr(drv, "laplace_posterior_from_residuals", linalg)
    res = ref.refine(m)
    assert math.isnan(res.posterior_sigma_U_deg) and res.posterior is None
    assert "LinAlgError" in res.metadata["posterior_error"]
