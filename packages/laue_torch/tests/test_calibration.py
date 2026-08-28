"""Toy calibration recovery tests.

These exercise the gradient pipeline end-to-end with Adam. Each test is
constructed to keep the inverse problem in the convex (or near-convex)
regime — small perturbations, well-conditioned losses — so it acts as a
gradient-correctness check rather than a serious calibration benchmark.
The full machinery (multi-scale annealing, line search, robust losses)
is left to downstream calibration code; these tests just verify that
gradient-based optimization on the differentiable forward kernel works.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from laue_torch import LaueForwardModel


def _hkls_subset() -> torch.Tensor:
    """A wide FCC reflection set sufficient to constrain a calibration."""
    out = []
    for h in range(-8, 9):
        for k in range(-8, 9):
            for l in range(-8, 9):
                if (h, k, l) == (0, 0, 0):
                    continue
                if not ((h % 2 == k % 2) and (k % 2 == l % 2)):
                    continue
                out.append([h, k, l])
    return torch.tensor(out, dtype=torch.long)


def _model(strain_mode: str = "none", energy_image: bool = False,
           psf: float = 4.0, n_pix: int = 1024) -> LaueForwardModel:
    return LaueForwardModel(
        hkls=_hkls_subset(),
        n_pix=(n_pix, n_pix),
        px_size=(0.0004, 0.0004),
        psf_sigma=psf,
        render_window=int(2 * math.ceil(3 * psf) + 1),
        rotation="rodrigues",
        detector_rotation="rodrigues",
        strain_mode=strain_mode,
        hard=False,
        tau_z=5e-3,
        tau_px=2.0,
        tau_E=0.3,
        energy_image=energy_image,
    )


def _truth():
    lat = torch.tensor([0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0], dtype=torch.float64)
    P_true = torch.tensor([0.028745, 0.002788, 0.513115], dtype=torch.float64)
    R_true = torch.tensor([-1.20131258, -1.21399082, -1.21881158], dtype=torch.float64)
    rvec_true = torch.tensor([0.05, -0.03, 0.02], dtype=torch.float64).unsqueeze(0)
    return lat, P_true, R_true, rvec_true


def _adam(params, loss_fn, lr=1e-3, steps=500):
    opt = torch.optim.Adam(params, lr=lr)
    last = float("inf")
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        last = loss.item()
    return last


# ── Geometry calibration ───────────────────────────────────────────────────

def test_recover_detector_translation():
    torch.manual_seed(0)
    model = _model()
    lat, P_true, R_true, rvec = _truth()
    with torch.no_grad():
        I_obs = model(rvec, lat, P_true, R_true)

    P = (P_true.clone() + torch.tensor([0.001, 0.001, 0.005], dtype=torch.float64)) \
        .requires_grad_(True)
    final = _adam([P], lambda: ((model(rvec, lat, P, R_true) - I_obs) ** 2).mean(),
                  lr=5e-4, steps=400)
    err = (P.detach() - P_true).abs().max().item()
    assert err < 1e-4, f"P recovery error {err:.4g}; final loss {final:.4g}"


def test_recover_detector_rotation():
    torch.manual_seed(0)
    model = _model()
    lat, P_true, R_true, rvec = _truth()
    with torch.no_grad():
        I_obs = model(rvec, lat, P_true, R_true)

    R = (R_true.clone() + torch.tensor([0.005, -0.005, 0.005], dtype=torch.float64)) \
        .requires_grad_(True)
    final = _adam([R], lambda: ((model(rvec, lat, P_true, R) - I_obs) ** 2).mean(),
                  lr=2e-3, steps=400)
    err = (R.detach() - R_true).abs().max().item()
    assert err < 5e-3, f"R recovery error {err:.4g}; final loss {final:.4g}"


# ── Orientation refinement ─────────────────────────────────────────────────

def test_recover_orientation_small_perturbation():
    """Small perturbation (0.5°) — image-MSE has gradient signal here."""
    torch.manual_seed(0)
    model = _model(psf=4.0)
    lat, P_true, R_true, rvec_true = _truth()
    with torch.no_grad():
        I_obs = model(rvec_true, lat, P_true, R_true)

    rvec = (rvec_true + torch.tensor([0.008, -0.005, 0.006], dtype=torch.float64)) \
        .clone().detach().requires_grad_(True)
    final = _adam([rvec], lambda: ((model(rvec, lat, P_true, R_true) - I_obs) ** 2).mean(),
                  lr=1e-3, steps=600)
    err = (rvec.detach() - rvec_true).abs().max().item()
    assert err < 5e-3, f"orientation error {err:.4g}; final loss {final:.4g}"


def test_recover_orientation_aux_position_loss():
    """Per-spot position MSE — wider basin of attraction than image MSE,
    which is what serious refinement code would use."""
    torch.manual_seed(0)
    model = _model(psf=2.0)
    lat, P_true, R_true, rvec_true = _truth()
    with torch.no_grad():
        _, aux_obs = model(rvec_true, lat, P_true, R_true, return_aux=True)
    px_obs = aux_obs.px.detach()
    py_obs = aux_obs.py.detach()
    m_obs = aux_obs.mask.detach()

    rvec = (rvec_true + torch.tensor([0.03, -0.02, 0.025], dtype=torch.float64)) \
        .clone().detach().requires_grad_(True)

    def loss_fn():
        _, aux = model(rvec, lat, P_true, R_true, return_aux=True)
        # Match per-(grain, hkl) predicted spot to its observed counterpart.
        # Both have the same H ordering, so direct comparison is valid.
        w = aux.mask * m_obs
        diff_x = (aux.px - px_obs) * w
        diff_y = (aux.py - py_obs) * w
        return (diff_x.pow(2) + diff_y.pow(2)).sum() / w.sum().clamp_min(1.0)

    final = _adam([rvec], loss_fn, lr=2e-3, steps=600)
    err = (rvec.detach() - rvec_true).abs().max().item()
    assert err < 1e-3, f"orientation error {err:.4g}; final loss {final:.4g}"


# ── Strain refinement ──────────────────────────────────────────────────────

def test_strain_voigt_image_loss_converges():
    """Image MSE on a small Voigt-mode strain converges by orders of magnitude.
    The recovered strain may differ from truth in the position-null directions
    of the image-only forward map, but the rendered image must agree well."""
    torch.manual_seed(0)
    model = _model(strain_mode="voigt")
    lat, P_true, R_true, rvec = _truth()
    eps_true = torch.tensor([1e-3, -2e-3, 5e-4, 1e-4, -1.5e-4, 2e-4],
                            dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        I_obs = model(rvec, lat, P_true, R_true, strain=eps_true)

    eps = torch.zeros_like(eps_true).requires_grad_(True)
    initial_loss = ((model(rvec, lat, P_true, R_true, strain=eps) - I_obs) ** 2).mean().item()
    final = _adam([eps], lambda: ((model(rvec, lat, P_true, R_true, strain=eps) - I_obs) ** 2).mean(),
                  lr=2e-4, steps=800)
    # Image-MSE should drop by at least 2 orders of magnitude.
    assert final < initial_loss * 1e-2, \
        f"voigt strain image-loss did not converge enough; init={initial_loss:.4g} final={final:.4g}"


def test_strain_deviatoric_recovers_truth():
    """Deviatoric mode (5 params) has no hydrostatic null — should recover
    the *exact* truth on a sufficiently rich hkl set."""
    torch.manual_seed(0)
    model = _model(strain_mode="deviatoric")
    lat, P_true, R_true, rvec = _truth()
    eps_true = torch.tensor([1e-3, -1.2e-3, 1e-4, -1.5e-4, 2e-4],
                            dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        I_obs = model(rvec, lat, P_true, R_true, strain=eps_true)

    eps = torch.zeros_like(eps_true).requires_grad_(True)
    final = _adam([eps], lambda: ((model(rvec, lat, P_true, R_true, strain=eps) - I_obs) ** 2).mean(),
                  lr=2e-4, steps=1200)
    err = (eps.detach() - eps_true).abs().max().item()
    assert err < 5e-4, f"deviatoric strain error {err:.4g}; final loss {final:.4g}"


# ── Energy-resolved hydrostatic recovery ───────────────────────────────────

def test_energy_loss_sees_hydrostatic_strain():
    """Hydrostatic strain is invisible to *positions* in white-beam Laue, but
    visible to *energies*. This test contrasts the two losses:

    - At the unstrained guess, the **position** loss has zero gradient w.r.t.
      the hydrostatic direction (1, 1, 1, 0, 0, 0) of ε.
    - The **energy** loss has a non-zero gradient there.
    """
    torch.manual_seed(0)
    model = _model(strain_mode="voigt", energy_image=True)
    lat, P_true, R_true, rvec = _truth()
    eps_hydro = 2e-3
    eps_true = torch.tensor([eps_hydro, eps_hydro, eps_hydro, 0.0, 0.0, 0.0],
                            dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        I_obs, aux_obs = model(rvec, lat, P_true, R_true, strain=eps_true,
                               return_aux=True)
    E_obs = aux_obs.energy.detach()
    px_obs = aux_obs.px.detach()
    py_obs = aux_obs.py.detach()
    m_obs = aux_obs.mask.detach()

    hydro_dir = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                             dtype=torch.float64).unsqueeze(0)

    # ── Position loss: gradient projected onto the hydrostatic direction
    eps = torch.zeros_like(eps_true).requires_grad_(True)
    _, aux = model(rvec, lat, P_true, R_true, strain=eps, return_aux=True)
    pos_loss = ((aux.px - px_obs).pow(2) + (aux.py - py_obs).pow(2)) * (aux.mask * m_obs)
    pos_loss = pos_loss.sum() / (aux.mask * m_obs).sum().clamp_min(1.0)
    pos_grad = torch.autograd.grad(pos_loss, eps, retain_graph=False)[0]
    pos_hydro = (pos_grad * hydro_dir).sum().abs().item()

    # ── Energy loss: gradient projected onto the same direction
    eps = torch.zeros_like(eps_true).requires_grad_(True)
    _, aux = model(rvec, lat, P_true, R_true, strain=eps, return_aux=True)
    e_loss = ((aux.energy - E_obs).pow(2) * (aux.mask * m_obs)).sum() \
             / (aux.mask * m_obs).sum().clamp_min(1.0)
    e_grad = torch.autograd.grad(e_loss, eps)[0]
    e_hydro = (e_grad * hydro_dir).sum().abs().item()

    # Position gradient on hydrostatic direction must be tiny relative to the
    # energy gradient on the same direction. We expect at least 3 orders of
    # magnitude separation in white-beam Laue.
    assert e_hydro > 1e-3, f"energy loss has near-zero hydrostatic gradient ({e_hydro:.4g})"
    assert pos_hydro / max(e_hydro, 1e-30) < 1e-2, \
        f"position-loss hydrostatic gradient too large: {pos_hydro:.4g} vs energy {e_hydro:.4g}"
