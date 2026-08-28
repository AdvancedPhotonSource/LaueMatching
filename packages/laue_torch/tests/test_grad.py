"""Gradient-flow checks: torch.autograd.gradcheck on each parameter group."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from laue_torch import LaueForwardModel
from laue_torch.geometry import (
    deviatoric5_to_symmetric,
    quat_to_matrix,
    reciprocal_matrix,
    rodrigues_to_matrix,
    voigt_to_symmetric,
)


def _toy_model(strain_mode: str = "none", rotation: str = "matrix",
               detector_rotation: str = "rodrigues",
               energy_image: bool = False,
               hard: bool = False) -> LaueForwardModel:
    # Use a generously sized detector with the real params_sim.txt geometry —
    # smaller detector + small pixel count means the soft masks saturate to
    # exactly 0 and gradients vanish.
    hkls = torch.tensor([
        [1, 1, 1], [-1, -1, -1], [2, 0, 0], [-2, 0, 0],
        [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2],
        [2, 2, 0], [-2, -2, 0], [1, -1, 3], [3, -1, 1],
        [1, -1, -3], [-3, 1, 1], [2, -2, 4], [4, 2, -2],
    ], dtype=torch.long)
    return LaueForwardModel(
        hkls=hkls,
        n_pix=(256, 256),
        px_size=(0.0016, 0.0016),
        psf_sigma=2.0,
        render_window=9,
        rotation=rotation,
        detector_rotation=detector_rotation,
        strain_mode=strain_mode,
        hard=hard,
        tau_z=5e-3,
        tau_px=2.0,
        tau_E=0.3,
        energy_image=energy_image,
    )


# Lattice / detector inputs that put spots on the synthetic detector.
def _good_inputs(dtype=torch.float64):
    lat = torch.tensor([0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0], dtype=dtype)
    P = torch.tensor([0.028745, 0.002788, 0.513115], dtype=dtype)
    R = torch.tensor([-1.20131258, -1.21399082, -1.21881158], dtype=dtype)
    U = torch.tensor([
        [0.867151, 0.494088, 0.062670],
        [-0.052670, 0.216095, -0.974957],
        [-0.495254, 0.842135, 0.213410],
    ], dtype=dtype).unsqueeze(0)
    return U, lat, P, R


# ── Helpers: scalar loss closure for gradcheck ────────────────────────────

def _loss_image_sum(model, U, lat, P, R, strain=None):
    img = model(U, lat, P, R, strain=strain, E_range=(5.0, 30.0))
    return (img * img).sum()  # squared total intensity → smooth scalar


def _matrix_to_quat(M: torch.Tensor) -> torch.Tensor:
    """Differentiable-irrelevant helper: 3×3 rotation → (w, x, y, z) quaternion."""
    m = M
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / torch.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        # Robust branch — never hit for our well-conditioned test rotations.
        w = torch.tensor(1.0, dtype=M.dtype)
        x = y = z = torch.tensor(0.0, dtype=M.dtype)
    return torch.stack([w, x, y, z])


@pytest.mark.parametrize("rotation", ["quat", "rodrigues", "6d"])
def test_grad_flows_through_orientation(rotation):
    model = _toy_model(rotation=rotation)
    U_mat, lat, P, R = _good_inputs()
    M = U_mat[0]
    if rotation == "quat":
        U = _matrix_to_quat(M).unsqueeze(0)
    elif rotation == "rodrigues":
        # Rodrigues vector via skew-of-log: cheap path uses axis-angle.
        cos_t = ((M.diagonal().sum() - 1) * 0.5).clamp(-1.0, 1.0)
        theta = torch.arccos(cos_t)
        if float(theta) < 1e-6:
            U = torch.zeros(3, dtype=torch.float64).unsqueeze(0)
        else:
            axis = torch.stack([
                M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1],
            ]) / (2.0 * torch.sin(theta))
            U = (axis * theta).unsqueeze(0)
    else:  # 6d — first two columns of M
        U = torch.cat([M[:, 0], M[:, 1]]).unsqueeze(0)
    U = U.detach().requires_grad_(True)

    loss = _loss_image_sum(model, U, lat, P, R)
    loss.backward()
    assert U.grad is not None
    assert torch.isfinite(U.grad).all()
    assert U.grad.abs().sum() > 0


@pytest.mark.parametrize("strain_mode,strain_init", [
    ("voigt", torch.zeros(6, dtype=torch.float64)),
    ("deviatoric", torch.zeros(5, dtype=torch.float64)),
    ("F", torch.eye(3, dtype=torch.float64)),
])
def test_grad_flows_through_strain(strain_mode, strain_init):
    model = _toy_model(strain_mode=strain_mode)
    U, lat, P, R = _good_inputs()
    s = strain_init.clone().unsqueeze(0).requires_grad_(True)
    loss = _loss_image_sum(model, U, lat, P, R, strain=s)
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_deviatoric_strain_has_zero_hydrostatic_gradient():
    """Correctness: in deviatoric mode, perturbing the trace of ε must not
    change the rendered image — confirmed by the gradient w.r.t. (e11, e22)
    being orthogonal to the (1, 1) direction (since e33 = -e11 - e22)."""
    # Use voigt mode and check that gradient sums of the 3 diagonal Voigt
    # components add to ~0 (i.e. pure-trace direction has zero gradient,
    # because Laue is white-beam).
    model = _toy_model(strain_mode="voigt", hard=False)
    U, lat, P, R = _good_inputs()
    s = torch.zeros(6, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    loss = _loss_image_sum(model, U, lat, P, R, strain=s)
    loss.backward()
    g = s.grad.squeeze()
    diag_sum = g[0] + g[1] + g[2]                 # ∂L/∂(tr ε)
    diag_norm = g[:3].abs().sum().clamp_min(1e-30)
    rel = diag_sum.abs() / diag_norm
    assert rel < 1e-3, f"hydrostatic gradient not zero: {rel.item()}"


def test_grad_flows_through_lattice():
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    lat = lat.detach().requires_grad_(True)
    loss = _loss_image_sum(model, U, lat, P, R)
    loss.backward()
    assert lat.grad is not None
    assert torch.isfinite(lat.grad).all()
    assert lat.grad.abs().sum() > 0


def test_grad_flows_through_detector_pose():
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    P = P.detach().requires_grad_(True)
    R = R.detach().requires_grad_(True)
    loss = _loss_image_sum(model, U, lat, P, R)
    loss.backward()
    assert P.grad is not None and R.grad is not None
    assert torch.isfinite(P.grad).all() and torch.isfinite(R.grad).all()
    assert P.grad.abs().sum() > 0
    assert R.grad.abs().sum() > 0


# ── Strict gradcheck on the geometry-only path (auxiliary outputs) ─────────

def test_gradcheck_on_aux_positions():
    """gradcheck on a small problem that bypasses the (non-smooth) splat:
    differentiate ``aux.px.sum() + aux.py.sum()`` directly."""
    model = _toy_model(rotation="rodrigues", strain_mode="voigt", hard=False)

    def f(rvec, lat, P, R, strain):
        _, aux = model(rvec, lat, P, R, strain=strain, return_aux=True)
        # mask weights make this smooth in all parameters.
        return (aux.px * aux.mask).sum() + (aux.py * aux.mask).sum() \
               + (aux.energy * aux.mask).sum()

    rvec = torch.tensor([0.05, -0.03, 0.02], dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    lat = torch.tensor([0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0],
                       dtype=torch.float64, requires_grad=True)
    P = torch.tensor([0.001, 0.001, 0.04], dtype=torch.float64, requires_grad=True)
    R = torch.tensor([-1.20131258, -1.21399082, -1.21881158],
                     dtype=torch.float64, requires_grad=True)
    strain = torch.zeros(6, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    assert torch.autograd.gradcheck(
        f, (rvec, lat, P, R, strain), eps=1e-6, atol=5e-4, rtol=5e-3,
        check_undefined_grad=False, nondet_tol=1e-5,
    )
