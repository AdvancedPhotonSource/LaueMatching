"""Unit tests for MixtureOfTangentGaussianSO3 and laue_torch.nye."""

from __future__ import annotations

import math

import torch

from laue_torch import (
    LaueForwardModel,
    MixtureOfTangentGaussianSO3,
    TangentGaussianSO3,
)
from laue_torch.geometry import rodrigues_to_matrix
from laue_torch.nye import (
    matrix_to_rotvec,
    nye_tensor,
    synthetic_linear_gradient_field,
)


# ── Mixture distribution ──────────────────────────────────────────────────

def test_mixture_weights_sum_to_one():
    torch.manual_seed(0)
    U_inits = torch.stack([torch.eye(3, dtype=torch.float64)] * 3, dim=0)
    mix = MixtureOfTangentGaussianSO3(U_inits, sigma_init=1e-2)
    w = mix.weights()
    assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_mixture_sample_weights_sum_to_one():
    torch.manual_seed(0)
    U_inits = torch.stack([torch.eye(3, dtype=torch.float64),
                           torch.eye(3, dtype=torch.float64)], dim=0)
    mix = MixtureOfTangentGaussianSO3(U_inits, sigma_init=1e-2)
    # Imbalanced weights via logits.
    mix.logits.data = torch.tensor([1.0, -1.0], dtype=torch.float64)
    U, w = mix.sample(64)
    assert U.shape == (64, 3, 3)
    assert w.shape == (64,)
    assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=torch.float64),
                         atol=1e-9)


def test_mixture_gradient_flows_through_render():
    from laue_torch.synthetic import default_truth, make_model
    torch.manual_seed(0)
    truth = default_truth(1, strain=False)
    U_inits = torch.stack([truth.U[0], truth.U[0]], dim=0)
    mix = MixtureOfTangentGaussianSO3(U_inits, sigma_init=1e-2)
    model = make_model(strain_mode="voigt", rotation="matrix",
                       n_pix=192, h_max=6, psf_sigma=2.5,
                       energy_image=False)
    img = mix.render(model, truth.lat, truth.P, truth.R, M=8)
    loss = (img ** 2).mean()
    loss.backward()
    for n, p in mix.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"nan grad for {n}"


# ── Nye tensor ────────────────────────────────────────────────────────────

def test_matrix_to_rotvec_identity():
    R = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    rv = matrix_to_rotvec(R)
    assert torch.allclose(rv, torch.zeros(1, 3, dtype=torch.float64))


def test_matrix_to_rotvec_axis_angle_round_trip():
    rvec = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    R = rodrigues_to_matrix(rvec)
    rv = matrix_to_rotvec(R)
    assert torch.allclose(rv, rvec, atol=1e-12)


def test_nye_tensor_recovers_analytical_linear_gradient():
    R_field, alpha_truth = synthetic_linear_gradient_field(
        n_voxels=15, axis_index=2, rate_per_voxel_deg=0.1, spacing=1.0)
    alpha = nye_tensor(R_field, spacing=1.0)
    # Interior voxel must match analytic to machine precision.
    err = (alpha[7] - alpha_truth).abs().max().item()
    assert err < 1e-10, f"interior Nye error too large: {err}"


def test_nye_tensor_handles_2d_field():
    # 2-D field with the same gradient along axis 0; constant along axis 1.
    R_1d, _ = synthetic_linear_gradient_field(
        n_voxels=5, axis_index=2, rate_per_voxel_deg=0.1, spacing=1.0)
    # Stack along a second spatial axis (constant orientation along that axis).
    R_2d = R_1d.unsqueeze(1).expand(5, 4, 3, 3).contiguous()
    alpha = nye_tensor(R_2d, spacing=(1.0, 1.0))
    # α[..., 2, 0] = rate; α[..., 2, 1] = 0 (constant along axis 1).
    expected_x = math.radians(0.1)
    assert abs(alpha[2, 2, 2, 0].item() - expected_x) < 1e-10
    assert abs(alpha[2, 2, 2, 1].item()) < 1e-10
