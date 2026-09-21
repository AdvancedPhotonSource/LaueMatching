"""The Laplace posterior is the curvature of the objective the fit minimised.

Both refiners rebuild the fit's render inside a standalone ``residual_fn`` for
the Hessian (flat parameter vector, fixed-seed MC draws replayed by hand). If
that replay drifts from the fit -- a different seed, draw order, energy band,
per-sample intensity or strain draw -- the posterior silently describes some
other objective. This pins it: ``residual_fn(theta_map)`` must equal the fit's
own render at the final parameters, minus the observation, EXACTLY (same
float64 operations on the same draws; max abs diff 0).

Mechanism: the renderer class methods are wrapped to remember the fitted
distribution, the posterior helper is wrapped to capture ``residual_fn``, and
after ``refine`` the fitted distribution is re-rendered with a fresh generator
on the fit's seed.
"""
from __future__ import annotations

import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

import laue_torch.realdata.driver as drv
import laue_torch.realdata.multi_grain as mg
from laue_torch.distributions import (IndependentVoxelDistribution,
                                      MixtureOfTangentGaussianSO3,
                                      MixtureOfVoxelDistributions)
from laue_torch.io import LaueParams
from laue_torch.realdata import (MultiGrainVoxelRefiner, VoxelMeasurement,
                                 VoxelODFRefiner)

DT = torch.float64


def _params() -> LaueParams:
    return LaueParams(
        sg_num=225, symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.006, px_y=0.006, n_pix_x=96, n_pix_y=64,
        E_lo=5.0, E_hi=12.0, psf_sigma=1.0)


def _U(q):
    q = torch.tensor(q, dtype=DT)
    return quat_to_orient_mat(q / q.norm()).reshape(3, 3)


US = torch.stack([_U([0.56153266089081, -0.1069242896544219,
                      -0.7939419137346801, 0.2071340258144413]),
                  _U([0.9, 0.1, 0.3, 0.2])])


def _noisy_obs(ref, Us):
    t = ref.tensors
    with torch.no_grad():
        img = ref.model(Us, t["lattice"], t["P"], t["R"],
                        strain=torch.zeros(Us.shape[0], 6, dtype=DT),
                        E_range=ref.E_range)
    g = torch.Generator().manual_seed(5)
    return img + 0.01 * torch.randn(img.shape, dtype=DT, generator=g)


def _capture(monkeypatch, module, render_cls):
    seen = {}
    orig_render = render_cls.render

    def render(self, *a, **k):
        seen["dist"], seen["args"], seen["kwargs"] = self, a, k
        return orig_render(self, *a, **k)
    monkeypatch.setattr(render_cls, "render", render)

    orig_post = module.laplace_posterior_from_residuals

    def post(residual_fn, theta, **k):
        seen["residual_fn"], seen["theta"] = residual_fn, theta.detach().clone()
        return orig_post(residual_fn, theta, **k)
    monkeypatch.setattr(module, "laplace_posterior_from_residuals", post)
    return seen, orig_render


def _closure(fn) -> dict:
    return dict(zip(fn.__code__.co_freevars,
                    (c.cell_contents for c in fn.__closure__)))


@pytest.mark.parametrize("mode", ["orient_only", "strain_voigt", "strain_deviatoric"])
@pytest.mark.parametrize("refine_means", [False, True])
def test_multi_grain_posterior_residual_is_the_fit_residual(monkeypatch, mode, refine_means):
    cls = (MixtureOfTangentGaussianSO3 if mode == "orient_only"
           else MixtureOfVoxelDistributions)
    seen, orig_render = _capture(monkeypatch, mg, cls)
    ref = MultiGrainVoxelRefiner(_params(), sigma_init_deg=0.05, n_steps=2,
                                 M_render=5, mode=mode, refine_means=refine_means,
                                 compute_posterior=True)
    ref.refine(_noisy_obs(ref, US).T.contiguous(), US, axis_order="YX")

    k = dict(seen["kwargs"])
    k["generator"] = torch.Generator().manual_seed(drv.FIXED_PRED_SEED)
    with torch.no_grad():
        I_fit = orig_render(seen["dist"], *seen["args"], **k)
        r = seen["residual_fn"](seen["theta"])
    c = _closure(seen["residual_fn"])
    expect = (I_fit - c["I_obs"])[c["patch_sel"]]
    assert r.shape == expect.shape
    assert float((r - expect).abs().max()) == 0.0


def test_driver_posterior_residual_is_the_fit_residual(monkeypatch):
    seen, orig_render = _capture(monkeypatch, drv, IndependentVoxelDistribution)
    ref = VoxelODFRefiner(_params(), sigma_init_deg=0.05, n_steps=2, M_render=5)
    obs = _noisy_obs(ref, US[:1])
    res = ref.refine(VoxelMeasurement(0, obs.T.contiguous(), US[:1], {},
                                      axis_order="YX"))
    assert res.metadata["posterior_conditional_on_fixed_means"] is True

    k = dict(seen["kwargs"])
    k["generator"] = torch.Generator().manual_seed(drv.FIXED_PRED_SEED)
    with torch.no_grad():
        I_fit = orig_render(seen["dist"], *seen["args"], **k)
        r = seen["residual_fn"](seen["theta"])
    expect = I_fit - _closure(seen["residual_fn"])["I_obs"]
    assert float((r - expect).abs().max()) == 0.0
