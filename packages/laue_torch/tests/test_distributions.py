"""Unit tests for laue_torch.distributions."""

from __future__ import annotations

import math

import torch

from laue_torch import (
    GaussianStrain,
    IndependentVoxelDistribution,
    LaueForwardModel,
    TangentGaussianSO3,
)
from laue_torch.geometry import sixd_to_matrix


def test_strain_sampling_moments_match_parameters():
    torch.manual_seed(0)
    eps_mean = torch.tensor([1e-3, -2e-3, 5e-4, 1e-4, -1e-4, 2e-4],
                            dtype=torch.float64)
    sigma = 5e-4
    dist = GaussianStrain(eps_init=eps_mean, sigma_init=sigma)
    samples = dist.sample(20_000)
    # Mean.
    assert torch.allclose(samples.mean(0), eps_mean, atol=1e-5)
    # Diagonal covariance — samples are zero-correlated since off-diag is 0
    # at init; expected diag = sigma².
    diag = samples.var(0, unbiased=False)
    assert torch.allclose(diag, torch.full_like(diag, sigma * sigma),
                          rtol=0.1)


def test_so3_sampling_concentrates_at_mean():
    """For a small isotropic spread, samples cluster near the mean and
    the average sampled rotation matrix is close to U_mean."""
    torch.manual_seed(0)
    U_mean = torch.eye(3, dtype=torch.float64)
    dist = TangentGaussianSO3(U_init=U_mean, sigma_init=0.01)  # ~0.6° σ
    U_samples = dist.sample(5000)
    # Frobenius mean of samples ≈ U_mean (for small spread).
    frob_mean = U_samples.mean(0)
    assert (frob_mean - U_mean).abs().max() < 0.01


def test_so3_covariance_recovers_init():
    dist = TangentGaussianSO3(sigma_init=2e-3)
    cov = dist.covariance()
    expected = (2e-3) ** 2 * torch.eye(3, dtype=torch.float64)
    assert torch.allclose(cov, expected, atol=1e-12)


def test_gradient_flows_through_voxel_render():
    """Gradient of an image-MSE loss must reach every distribution param."""
    from laue_torch.synthetic import default_truth, make_model

    torch.manual_seed(0)
    truth = default_truth(1, strain=False)
    orient = TangentGaussianSO3(U_init=truth.U[0], sigma_init=0.01)
    strain = GaussianStrain(sigma_init=5e-4)
    voxel = IndependentVoxelDistribution(orient, strain)

    model = make_model(strain_mode="voigt", rotation="matrix",
                       n_pix=192, h_max=6, psf_sigma=2.5,
                       energy_image=False)
    img = voxel.render(model, truth.lat, truth.P, truth.R, M=16)
    target = torch.zeros_like(img)
    loss = ((img - target) ** 2).mean()
    loss.backward()
    grads = {n: p.grad for n, p in voxel.named_parameters()}
    for n, g in grads.items():
        assert g is not None, f"no gradient for {n}"
        assert torch.isfinite(g).all(), f"non-finite grad in {n}"
        assert g.abs().sum() > 0, f"zero gradient in {n}"


def test_voxel_render_changes_with_orientation_spread():
    """Wider spread → smoother/broader rendered image (lower max pixel)."""
    from laue_torch.synthetic import default_truth, make_model

    torch.manual_seed(0)
    truth = default_truth(1, strain=False)
    # Need a detector + hkl set that actually produces spots for the
    # default truth orientation; the toy 192×192 setup leaves the detector
    # outside the diffraction cone.
    model = make_model(strain_mode="voigt", rotation="matrix",
                       n_pix=384, h_max=8, psf_sigma=2.0, px_size=0.0008,
                       energy_image=False)

    def render_with_spread(sigma_deg):
        orient = TangentGaussianSO3(U_init=truth.U[0],
                                    sigma_init=math.radians(sigma_deg))
        strain = GaussianStrain(sigma_init=1e-6)
        voxel = IndependentVoxelDistribution(orient, strain)
        with torch.no_grad():
            return voxel.render(model, truth.lat, truth.P, truth.R, M=64)

    img_narrow = render_with_spread(0.01)
    img_broad = render_with_spread(2.0)
    assert img_narrow.max() > 0, "smoke test: narrow render produced no spots"
    # Broader distribution → samples cover more orientations → spots
    # spread out → individual pixel maxima drop.
    assert img_broad.max() < img_narrow.max() * 0.7
