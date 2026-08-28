"""Per-voxel orientation- and strain-distribution functions.

Each voxel is modelled as a joint distribution ``p(U, ε)`` over orientations
``U ∈ SO(3)`` and Voigt-6 strains ``ε ∈ R^6``.  The Laue diffraction pattern
is the Monte-Carlo average

    I(detector) ≈ (1/M) Σ_g render( U_g, ε_g )    with   (U_g, ε_g) ~ p(U, ε).

All distributions are reparameterized so the forward map is differentiable
end-to-end, and the parameters of ``p`` can be optimized by Adam against
the observed peak shape.

For v1 we use a factored Gaussian model:

  - **Orientation**: tangent-space Gaussian on SO(3) — sample
    ``δ ~ N(0, Σ_orient)`` in the body frame, then ``U = U_mean · exp(skew(δ))``.
    Valid for mosaic widths ≲10°.
  - **Strain**: multivariate normal on Voigt-6.

Cross-correlated `p(U, ε)` (e.g. twin variants where strain is conditional
on orientation mode) is a follow-up — same forward kernel, different prior.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .geometry import rodrigues_to_matrix, sixd_to_matrix


# ── Cholesky-parameterized covariance ──────────────────────────────────────

class CholeskyCov(nn.Module):
    """Lower-triangular Cholesky factor with positive diagonal.

    Stores ``log(diag)`` and the strict lower-triangle entries as
    independent parameters.  ``Σ = L Lᵀ`` is positive-definite by
    construction.
    """

    def __init__(self, dim: int, init_scale: float = 1e-3,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self.dim = dim
        self.log_diag = nn.Parameter(
            torch.full((dim,), math.log(init_scale), dtype=dtype))
        n_off = dim * (dim - 1) // 2
        self.off_diag = nn.Parameter(torch.zeros(n_off, dtype=dtype))
        tril_idx = torch.tril_indices(dim, dim, offset=-1)
        self.register_buffer("tril_idx", tril_idx)

    def L(self) -> Tensor:
        L = torch.diag(self.log_diag.exp())
        if self.off_diag.numel() > 0:
            L = L.clone()
            L[self.tril_idx[0], self.tril_idx[1]] = self.off_diag
        return L

    def cov(self) -> Tensor:
        L = self.L()
        return L @ L.T

    def sample(self, N: int, generator: Optional[torch.Generator] = None) -> Tensor:
        L = self.L()
        z = torch.randn(N, self.dim, dtype=L.dtype, device=L.device,
                        generator=generator)
        return z @ L.T                                    # (N, dim)


# ── Tangent-space Gaussian on SO(3) ────────────────────────────────────────

class TangentGaussianSO3(nn.Module):
    """Gaussian distribution on SO(3) via a tangent-space parameterization.

    Sample as ``U = U_mean · exp(skew(δ))`` with ``δ ~ N(0, Σ)`` in the
    body frame.  The mean is stored as a 6D continuous representation
    (Zhou et al. 2019) so it stays on SO(3) under unconstrained gradient
    updates.
    """

    def __init__(
        self,
        U_init: Optional[Tensor] = None,
        sigma_init: float = 1e-3,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if U_init is None:
            U_init = torch.eye(3, dtype=dtype)
        if U_init.shape[-2:] != (3, 3):
            raise ValueError(f"U_init must be 3×3, got {tuple(U_init.shape)}")
        d6 = torch.cat([U_init[..., :, 0], U_init[..., :, 1]], dim=-1)
        self.mean_d6 = nn.Parameter(d6.to(dtype))
        self.cov = CholeskyCov(3, init_scale=sigma_init, dtype=dtype)

    def mean(self) -> Tensor:
        return sixd_to_matrix(self.mean_d6)

    def sample(self, N: int, generator: Optional[torch.Generator] = None) -> Tensor:
        delta = self.cov.sample(N, generator=generator)        # (N, 3) tangent
        U_pert = rodrigues_to_matrix(delta)                    # (N, 3, 3)
        U_mean = self.mean()                                   # (3, 3)
        return U_mean.unsqueeze(0) @ U_pert                    # (N, 3, 3)

    def covariance(self) -> Tensor:
        return self.cov.cov()


# ── Multivariate Gaussian on Voigt-6 strain ────────────────────────────────

class GaussianStrain(nn.Module):
    """Voigt-6 strain ``ε ~ N(ε_mean, Σ)``.

    Reparameterized sampling: ``ε = ε_mean + L · z``, ``z ~ N(0, I)``,
    where ``L`` is the lower-triangular Cholesky factor of ``Σ``.
    """

    def __init__(
        self,
        eps_init: Optional[Tensor] = None,
        sigma_init: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if eps_init is None:
            eps_init = torch.zeros(6, dtype=dtype)
        if eps_init.shape != (6,):
            raise ValueError(f"eps_init must be (6,), got {tuple(eps_init.shape)}")
        self.mean = nn.Parameter(eps_init.to(dtype))
        self.cov = CholeskyCov(6, init_scale=sigma_init, dtype=dtype)

    def sample(self, N: int, generator: Optional[torch.Generator] = None) -> Tensor:
        eta = self.cov.sample(N, generator=generator)          # (N, 6)
        return self.mean.unsqueeze(0) + eta                     # (N, 6)

    def covariance(self) -> Tensor:
        return self.cov.cov()


# ── Independent voxel distribution ─────────────────────────────────────────

class MixtureOfTangentGaussianSO3(nn.Module):
    """Mixture of K tangent-Gaussian SO(3) kernels with learnable mixing weights.

    Use case: voxels containing **multiple orientation modes** (e.g. twin
    parent + variants, low-angle subgrains, sub-beam-size grains).
    Each kernel is an independent :class:`TangentGaussianSO3`; the
    mixing weights ``π = softmax(logits)`` are learned alongside the
    kernel parameters.

    Sampling is **stratified** — kernel ``k`` always contributes
    ``M / K`` (± 1) phantom samples, each with weight ``π_k / m_k`` so
    that the total contribution from kernel ``k`` is ``π_k``. This is
    deterministic in the kernel allocation (no Gumbel-softmax needed)
    and analytically differentiable in ``π``.
    """

    def __init__(
        self,
        U_inits: Tensor,
        sigma_init: float = 1e-2,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if U_inits.dim() != 3 or U_inits.shape[-2:] != (3, 3):
            raise ValueError(f"U_inits must be (K, 3, 3), got {tuple(U_inits.shape)}")
        self.K = U_inits.shape[0]
        self.kernels = nn.ModuleList([
            TangentGaussianSO3(U_init=U_inits[k], sigma_init=sigma_init, dtype=dtype)
            for k in range(self.K)
        ])
        self.logits = nn.Parameter(torch.zeros(self.K, dtype=dtype))

    def weights(self) -> Tensor:
        return torch.softmax(self.logits, dim=0)

    def means(self) -> Tensor:
        return torch.stack([k.mean() for k in self.kernels], dim=0)         # (K, 3, 3)

    def sample(self, M: int, generator: Optional[torch.Generator] = None
               ) -> tuple[Tensor, Tensor]:
        """Draw ``M`` total phantom samples from the mixture.

        Returns ``(U_samples, weights)`` of shapes ``(M, 3, 3)`` and
        ``(M,)``; ``weights`` sums to 1.
        """
        w = self.weights()
        per_k = M // self.K
        extra = M % self.K
        all_U: list[Tensor] = []
        all_w: list[Tensor] = []
        for k in range(self.K):
            m_k = per_k + (1 if k < extra else 0)
            if m_k == 0:
                continue
            U_k = self.kernels[k].sample(m_k, generator=generator)          # (m_k, 3, 3)
            w_k = (w[k] / m_k).expand(m_k)
            all_U.append(U_k)
            all_w.append(w_k)
        return torch.cat(all_U, dim=0), torch.cat(all_w, dim=0)

    def render(
        self,
        model,
        lat: Tensor,
        P: Tensor,
        R: Tensor,
        *,
        M: int = 128,
        E_range: tuple[float, float] = (5.0, 30.0),
        generator: Optional[torch.Generator] = None,
        per_spot_intensity: Optional[Tensor] = None,
        psf_sigma: Optional[Tensor] = None,
        psf_eta: Optional[Tensor] = None,
    ) -> Tensor:
        """Render the mixture-image via the multi-grain forward kernel."""
        if model.rotation != "matrix":
            raise ValueError("model.rotation must be 'matrix'")
        if model.strain_mode != "voigt":
            raise ValueError("model.strain_mode must be 'voigt'")
        if model.reduce != "sum":
            raise ValueError("model.reduce must be 'sum'")
        U, w = self.sample(M, generator=generator)
        eps = torch.zeros(M, 6, dtype=lat.dtype, device=lat.device)
        psi = per_spot_intensity
        if psi is not None and psi.dim() == 1:
            psi = psi.unsqueeze(0).expand(U.shape[0], -1)
        return model(U, lat, P, R, strain=eps, weights=w, E_range=E_range,
                     per_spot_intensity=psi, psf_sigma=psf_sigma,
                     psf_eta=psf_eta)


class MixtureOfVoxelDistributions(nn.Module):
    r"""Mixture of K voxel distributions, each with its own (orientation,
    strain) pair plus a learnable mixing weight.

    Use case: twin variants where strain is **variant-conditional** ---
    each parent / twin component has its own characteristic elastic
    strain (e.g.\ deformation twins with shear strain across the
    interface).  This generalises :class:`MixtureOfTangentGaussianSO3`
    by adding a per-kernel strain distribution.

    Each kernel ``k`` is an :class:`IndependentVoxelDistribution`
    (orientation $\times$ strain).  Sampling is stratified --- kernel
    ``k`` contributes ``M / K \pm 1`` phantom samples, each with weight
    ``\pi_k / m_k``.
    """

    def __init__(
        self,
        components: list["IndependentVoxelDistribution"],
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if not components:
            raise ValueError("need at least one voxel-distribution component")
        self.K = len(components)
        self.components = nn.ModuleList(components)
        self.logits = nn.Parameter(torch.zeros(self.K, dtype=dtype))

    def weights(self) -> Tensor:
        return torch.softmax(self.logits, dim=0)

    def sample(self, M: int, generator: Optional[torch.Generator] = None
               ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(U_samples, eps_samples, weights)``."""
        w = self.weights()
        per_k = M // self.K
        extra = M % self.K
        all_U: list[Tensor] = []
        all_eps: list[Tensor] = []
        all_w: list[Tensor] = []
        for k in range(self.K):
            m_k = per_k + (1 if k < extra else 0)
            if m_k == 0:
                continue
            U_k, eps_k = self.components[k].sample(m_k, generator=generator)
            w_k = (w[k] / m_k).expand(m_k)
            all_U.append(U_k)
            all_eps.append(eps_k)
            all_w.append(w_k)
        return torch.cat(all_U, 0), torch.cat(all_eps, 0), torch.cat(all_w, 0)

    def render(
        self,
        model,
        lat: Tensor,
        P: Tensor,
        R: Tensor,
        *,
        M: int = 128,
        E_range: tuple[float, float] = (5.0, 30.0),
        generator: Optional[torch.Generator] = None,
        per_spot_intensity: Optional[Tensor] = None,
        psf_sigma: Optional[Tensor] = None,
        psf_eta: Optional[Tensor] = None,
    ) -> Tensor:
        if model.rotation != "matrix":
            raise ValueError("model.rotation must be 'matrix'")
        if model.strain_mode != "voigt":
            raise ValueError("model.strain_mode must be 'voigt'")
        if model.reduce != "sum":
            raise ValueError("model.reduce must be 'sum'")
        U, eps, w = self.sample(M, generator=generator)
        # Per-spot intensity is per-reflection (intrinsic |F_hkl|² or
        # observation-derived target).  Same for every phantom sample, so
        # we just expand a (H,) tensor to (M, H) here.
        psi = per_spot_intensity
        if psi is not None and psi.dim() == 1:
            psi = psi.unsqueeze(0).expand(U.shape[0], -1)
        return model(U, lat, P, R, strain=eps, weights=w, E_range=E_range,
                     per_spot_intensity=psi, psf_sigma=psf_sigma,
                     psf_eta=psf_eta)


class IndependentVoxelDistribution(nn.Module):
    """Factored ``p(U, ε) = p(U) · p(ε)``.

    Use as the learnable module driving voxel-level peak-shape fitting:
    parameters are ``(U_mean, Σ_orient, ε_mean, Σ_strain)``; the Monte-
    Carlo rendered image is differentiable in all of them.

    Cross-correlated `p(U, ε)` priors (twin variants, plastic-deformation
    coupling) can replace this without changing the forward kernel.
    """

    def __init__(
        self,
        orient: TangentGaussianSO3,
        strain: GaussianStrain,
    ):
        super().__init__()
        self.orient = orient
        self.strain = strain

    def sample(self, N: int, generator: Optional[torch.Generator] = None):
        U = self.orient.sample(N, generator=generator)          # (N, 3, 3)
        eps = self.strain.sample(N, generator=generator)        # (N, 6)
        return U, eps

    def render(
        self,
        model,
        lat: Tensor,
        P: Tensor,
        R: Tensor,
        *,
        M: int = 64,
        E_range: tuple[float, float] = (5.0, 30.0),
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Monte-Carlo render of the voxel image with M phantom samples.

        ``model.rotation`` must be ``"matrix"`` and ``model.strain_mode``
        must be ``"voigt"``.  ``model.reduce='sum'`` is used implicitly.
        """
        if model.rotation != "matrix":
            raise ValueError(
                f"VoxelDistribution.render expects model.rotation='matrix' "
                f"(got {model.rotation!r}); pass orientation samples directly.")
        if model.strain_mode != "voigt":
            raise ValueError(
                f"VoxelDistribution.render expects model.strain_mode='voigt' "
                f"(got {model.strain_mode!r}).")
        if model.reduce != "sum":
            raise ValueError(
                f"VoxelDistribution.render expects model.reduce='sum' "
                f"(got {model.reduce!r}).")
        U, eps = self.sample(M, generator=generator)
        weights = torch.full((M,), 1.0 / M, dtype=lat.dtype, device=lat.device)
        return model(U, lat, P, R, strain=eps, weights=weights, E_range=E_range)
