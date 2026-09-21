"""MultiGrainVoxelRefiner: deviatoric strain, Laplace posterior, fallback target.

* ``mode="strain_deviatoric"`` was a no-op alias of ``strain_voigt``: it fitted
  6 strain components including the hydrostatic one, which a white-beam Laue
  pattern cannot see (a pure dilatation moves no spot). It now has exactly 5
  free strain parameters, and a pure hydrostatic strain is not in their span.
* ``compute_posterior`` was accepted and ignored. It now returns the Laplace
  posterior with its eigen-diagnostics and parameter names.
* The fallback per-spot target gave every harmonic the full observed patch SUM
  (the splat is peak-normalised, and harmonics share a pixel) and summed all
  grains into one vector. It is now the patch PEAK, one reflection per
  predicted pixel, per mode.
"""
from __future__ import annotations

import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.io import LaueParams
from laue_torch.realdata import MultiGrainVoxelRefiner
from laue_torch.realdata.multi_grain import (
    _DeviatoricGaussianStrain,
    _dev5_to_voigt6,
    _per_sample_intensity,
)

DT = torch.float64
NX, NY = 96, 64


def _params() -> LaueParams:
    return LaueParams(
        sg_num=225, symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.006, px_y=0.006, n_pix_x=NX, n_pix_y=NY,
        E_lo=5.0, E_hi=30.0, psf_sigma=1.0,
    )


def _U(q) -> torch.Tensor:
    q = torch.tensor(q, dtype=DT)
    return quat_to_orient_mat(q / q.norm()).reshape(3, 3)


# U_A has no co-located harmonics on this detector; U_B has a triple at one pixel.
U_A = _U([0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413])
U_B = _U([0.9, 0.1, 0.3, 0.2])


def _observe(ref: MultiGrainVoxelRefiner, Us: torch.Tensor) -> torch.Tensor:
    """Sum of single-orientation renders (every reflection intensity 1), [X, Y]."""
    t = ref.tensors
    with torch.no_grad():
        return ref.model(Us, t["lattice"], t["P"], t["R"],
                         strain=torch.zeros(Us.shape[0], 6, dtype=DT))


# ── deviatoric strain ──────────────────────────────────────────────────────

def test_deviatoric_strain_has_five_free_components_and_no_hydrostatic():
    st = _DeviatoricGaussianStrain(sigma_init=1e-6)
    free = [p for p in st.parameters() if p is not st.cov.log_diag
            and p is not st.cov.off_diag]
    assert [p.numel() for p in free] == [5]
    # The span of the 5 -> Voigt-6 map is exactly the trace-free subspace:
    # rank 5, and orthogonal to a pure hydrostatic strain (e, e, e, 0, 0, 0).
    J = torch.stack([_dev5_to_voigt6(e) for e in torch.eye(5, dtype=DT)], dim=1)
    assert torch.linalg.matrix_rank(J) == 5
    hydro = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=DT)
    assert torch.allclose(hydro @ J, torch.zeros(5, dtype=DT))
    with torch.no_grad():
        st.mean5.copy_(torch.tensor([1e-3, -2e-3, 3e-4, -4e-4, 5e-4], dtype=DT))
        assert abs(float(st.mean[:3].sum())) < 1e-15
        assert abs(float(st.sample(7)[:, :3].sum(-1).abs().max())) < 1e-15


def test_dev5_matches_forward_models_own_deviatoric_mode():
    """Same layout as geometry.deviatoric5_to_symmetric: the Voigt model on the
    mapped 6-vector renders what strain_mode='deviatoric' renders on the 5-vector
    (bounded, not bit-equal: two float64 code paths)."""
    p = _params()
    ref = MultiGrainVoxelRefiner(p, n_steps=1, M_render=2)
    t = ref.tensors
    dev = LaueForwardModel(hkls=ref.hkls, n_pix=t["n_pix"], px_size=t["px_size"],
                           psf_sigma=1.0, rotation="matrix",
                           detector_rotation="rodrigues", strain_mode="deviatoric",
                           hard=False, reduce="sum")
    e5 = torch.tensor([[2e-3, -1e-3, 5e-4, -7e-4, 1e-3]], dtype=DT)
    U = U_A.unsqueeze(0)
    with torch.no_grad():
        a = ref.model(U, t["lattice"], t["P"], t["R"], strain=_dev5_to_voigt6(e5))
        b = dev(U, t["lattice"], t["P"], t["R"], strain=e5)
    assert torch.allclose(a, b, rtol=0, atol=1e-10)


def test_strain_deviatoric_refine_returns_trace_free_strain_and_posterior():
    p = _params()
    ref = MultiGrainVoxelRefiner(p, sigma_init_deg=0.05, n_steps=2, M_render=4,
                                 mode="strain_deviatoric", compute_posterior=True)
    res = ref.refine(_observe(ref, U_A.unsqueeze(0)).T.contiguous(), U_A.unsqueeze(0),
                     axis_order="YX")
    assert res.eps_means.shape == (1, 6)
    assert abs(float(res.eps_means[0, :3].sum())) < 1e-15
    names = res.posterior_param_names
    strain_names = [n for n in names if ".strain_mean." in n]
    assert len(strain_names) == 5 and not any(n.endswith("e33") for n in strain_names)
    assert res.posterior.theta.numel() == len(names) == 6 + 5


# ── posterior ──────────────────────────────────────────────────────────────

def test_compute_posterior_returns_laplace_diagnostics():
    p = _params()
    ref = MultiGrainVoxelRefiner(p, sigma_init_deg=0.05, n_steps=2, M_render=4,
                                 compute_posterior=True)
    res = ref.refine(_observe(ref, U_A.unsqueeze(0)).T.contiguous(), U_A.unsqueeze(0),
                     axis_order="YX")
    post = res.posterior
    assert post is not None
    n = len(res.posterior_param_names)
    assert n == 6                                        # 3 log-diag + 3 off-diag
    assert post.theta.shape == (n,) and post.eigvals.shape == (n,)
    assert post.hessian.shape == (n, n) and post.sigma.shape == (n,)
    assert isinstance(post.rank_eff, int) and 0 <= post.rank_eff <= n
    assert isinstance(post.cond_number, float)
    # Not converged after 2 steps: the Hessian can be indefinite, and then the
    # affected sigma entries must be NaN, not a confident-looking number.
    if post.is_positive_definite is False:
        assert torch.isnan(post.sigma).any()
    assert res.metadata["posterior_conditional_on_fixed_means"] is True


def test_posterior_off_by_default_and_gauge_free_with_two_modes():
    p = _params()
    Us = torch.stack([U_A, U_B])
    ref = MultiGrainVoxelRefiner(p, sigma_init_deg=0.05, n_steps=1, M_render=4)
    assert ref.refine(_observe(ref, Us).T.contiguous(), Us, axis_order="YX").posterior is None
    ref = MultiGrainVoxelRefiner(p, sigma_init_deg=0.05, n_steps=1, M_render=4,
                                 refine_means=True, compute_posterior=True)
    res = ref.refine(_observe(ref, Us).T.contiguous(), Us, axis_order="YX")
    names = res.posterior_param_names
    # 2 x (6 spread + 3 tangent rotation) + 1 relative logit (not 2 absolute).
    assert len(names) == 2 * 9 + 1
    assert sum("logit" in n for n in names) == 1
    assert res.metadata["posterior_conditional_on_fixed_means"] is False


# ── fallback per-spot target ───────────────────────────────────────────────

def test_fallback_target_is_peak_per_mode_one_reflection_per_pixel():
    p = _params()
    Us = torch.stack([U_A, U_B])
    ref = MultiGrainVoxelRefiner(p, sigma_init_deg=0.05, n_steps=1, M_render=4)
    I_xy = _observe(ref, Us)
    target, patch = ref._compute_per_spot_target(None, Us, I_xy)
    assert target.shape == (2, ref.hkls.shape[0])
    t = ref.tensors
    for k in range(2):
        with torch.no_grad():
            _, aux = ref.model(Us[k:k + 1], t["lattice"], t["P"], t["R"],
                               strain=torch.zeros(1, 6, dtype=DT), return_aux=True)
        nz = (target[k] > 0).nonzero().reshape(-1)
        pix = {(int(aux.px[h].round()), int(aux.py[h].round())) for h in nz.tolist()}
        assert len(pix) == len(nz), "two reflections of one mode share a predicted pixel"
    # U_B's co-located harmonics: exactly one of them carries the target.
    with torch.no_grad():
        _, aux = ref.model(U_B.unsqueeze(0), t["lattice"], t["P"], t["R"],
                           strain=torch.zeros(1, 6, dtype=DT), return_aux=True)
    on = (aux.mask > 0.5).nonzero().reshape(-1)
    keys = [(int(aux.px[h].round()), int(aux.py[h].round())) for h in on.tolist()]
    shared = [h for h, key in zip(on.tolist(), keys) if keys.count(key) > 1]
    assert len(shared) >= 2, "fixture no longer has co-located harmonics"
    assert int((target[1, shared] > 0).sum()) == len(set(
        keys[i] for i, h in enumerate(on.tolist()) if h in shared))

    # Rendered at the seeds with the target, peak amplitude matches observation
    # (the old window-sum target predicted ~2*pi*sigma^2 times too bright).
    psi = _per_sample_intensity(target, 2)
    with torch.no_grad():
        pred = ref.model(Us, t["lattice"], t["P"], t["R"],
                         strain=torch.zeros(2, 6, dtype=DT), per_spot_intensity=psi)
    ratio = float(pred.max() / I_xy.max())
    assert 0.8 < ratio < 1.25, ratio
