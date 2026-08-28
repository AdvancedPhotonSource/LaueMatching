"""Streak footprint: orientation spread -> pixel-space covariance.

A grain with an orientation distribution does not make a point spot.  Perturbing
the orientation by a small tangent vector omega (axis-angle, radians) moves each
reflection on the detector by ``J omega`` to first order, where

    J = d(pixel) / d(omega)     shape (2, 3), units pixels per radian.

So an orientation covariance ``Sigma_omega`` (3x3, rad^2) produces a pixel-space
covariance

    Sigma_px = J Sigma_omega J^T + sigma_psf^2 I     shape (2, 2), px^2.

That is the footprint the reflection actually paints, and it is what
``rasterize.anisotropic_gaussian_splat`` consumes.  On the Ti-64 ID6 frame
J is ~5000 px/rad, so the measured sigma_par = 0.22 deg gives a ~40 px streak
while sigma_perp = 0.043 deg keeps it ~8 px wide.

ANGLE UNITS: everything here is in RADIANS (laue_torch internal convention).
Callers holding degrees -- the measured spreads are quoted in degrees -- must
convert.  Pixel units are pixels.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

__all__ = [
    "combined_pixel_covariance",
    "finite_difference_jacobian",
    "pixel_covariance",
    "pixel_jacobian",
    "spread_covariance",
    "strain_jacobian",
    "tangent_rotation",
]

# A projection function maps one (3, 3) orientation matrix to (H, 2) pixel
# coordinates for that grain's reflections.  Reflections that miss the detector
# must still occupy a row (NaN is fine) so the row count is stable in omega.
ProjectFn = Callable[[Tensor], Tensor]
# A strained projection additionally takes a deviatoric-5 strain vector.
StrainedProjectFn = Callable[[Tensor, Tensor], Tensor]


def tangent_rotation(omega: Tensor) -> Tensor:
    """Rotation matrix from a tangent (axis-angle) vector, via ``matrix_exp``.

    We deliberately do NOT use :func:`geometry.rodrigues_to_matrix` here.  That
    function ends in ``torch.where(near_zero, eye, R)``, and at ``rvec = 0`` the
    selected branch is the CONSTANT identity -- so its derivative at the origin
    is identically zero.  Every Jacobian in this module is evaluated exactly at
    omega = 0, which is precisely where that branch bites: the streaks would come
    out perfectly round and the anisotropy would silently vanish.

    ``matrix_exp`` of the skew-symmetric generator is analytic at the origin and
    differentiates correctly there.  It agrees with ``rodrigues_to_matrix``
    elsewhere (asserted in the tests).
    """
    zero = torch.zeros_like(omega[..., 0])
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    skew = torch.stack(
        [
            torch.stack([zero, -wz, wy], dim=-1),
            torch.stack([wz, zero, -wx], dim=-1),
            torch.stack([-wy, wx, zero], dim=-1),
        ],
        dim=-2,
    )
    return torch.linalg.matrix_exp(skew)


def _perturbed(project_fn: ProjectFn, om: Tensor, omega: Tensor) -> Tensor:
    """Project with the orientation rotated by tangent vector ``omega``."""
    return project_fn(tangent_rotation(omega) @ om)


def pixel_jacobian(project_fn: ProjectFn, om: Tensor) -> Tensor:
    """d(pixel)/d(omega) at omega = 0, in pixels per radian.

    project_fn : (3, 3) orientation -> (H, 2) pixel coordinates.
    om         : (3, 3) orientation matrix.

    Returns (H, 2, 3).

    Uses forward-mode autodiff: the input is 3-dimensional and the output is
    2H-dimensional, so three JVPs give the exact Jacobian -- far cheaper than
    reverse mode here, and exact unlike finite differences.
    """
    if om.shape[-2:] != (3, 3):
        raise ValueError(f"om must be (3, 3), got {tuple(om.shape)}")
    omega0 = torch.zeros(3, dtype=om.dtype, device=om.device)

    try:
        from torch.func import jacfwd
    except ImportError as exc:  # pragma: no cover - torch < 2.0
        raise ImportError(
            "pixel_jacobian needs torch.func (torch >= 2.0); use "
            "finite_difference_jacobian on older torch"
        ) from exc

    jac = jacfwd(lambda w: _perturbed(project_fn, om, w))(omega0)  # (H, 2, 3)
    if jac.shape[-1] != 3 or jac.shape[-2] != 2:
        raise ValueError(
            f"project_fn must return (H, 2); jacobian came back {tuple(jac.shape)}"
        )
    return jac


def finite_difference_jacobian(
    project_fn: ProjectFn, om: Tensor, eps: float = 1e-5
) -> Tensor:
    """Central-difference d(pixel)/d(omega), pixels per radian -> (H, 2, 3).

    The independent oracle for :func:`pixel_jacobian` in the tests.  Also usable
    directly when ``project_fn`` is not differentiable (e.g. wraps numpy).
    """
    cols = []
    for i in range(3):
        d = torch.zeros(3, dtype=om.dtype, device=om.device)
        d[i] = eps
        plus = _perturbed(project_fn, om, d)
        minus = _perturbed(project_fn, om, -d)
        cols.append((plus - minus) / (2.0 * eps))       # (H, 2)
    return torch.stack(cols, dim=-1)                     # (H, 2, 3)


def strain_jacobian(project_fn: StrainedProjectFn, om: Tensor) -> Tensor:
    """d(pixel)/d(deviatoric strain) at zero strain -> (H, 2, 5), px per unit strain.

    WHY THIS TERM EXISTS.  A rigid rotation moves every reflection of a grain by
    the same rotation, so ``pixel_jacobian`` produces almost the same streak
    length for all of them (measured on Ti-64 ID6: predicted widths vary only
    x1.2 across a grain's reflections).  The DATA vary x3.8, and a variance
    decomposition put 68% of that variance WITHIN grains (F=1.04, p=0.46) --
    reflections of one grain differ as much as reflections of different grains.
    A per-grain sigma cannot produce that; an hkl-dependent mechanism must.

    Deviatoric strain is exactly that mechanism: it rotates each plane normal by
    an amount set by q-hat's orientation relative to the strain tensor, so
    different hkl of ONE grain broaden differently.

    *** DO NOT ADD A STRAIN-SPREAD PARAMETER TO THE POLYCHROMATIC JOINT FIT. ***
    Orientation and deviatoric strain are DEGENERATE in white-beam Laue:
    ``report/report.tex`` reports rho(U, eps) ~ +-0.98 from an H^-1 coupling
    analysis, surviving calibrant, multi-grain and energy-resolved setups, and
    calls it a physical degeneracy of polychromatic Laue rather than an
    algorithmic one.  A free sigma_eps is therefore unidentifiable -- it is
    simply absorbed into the rotation spread.  Measured confirmation on Ti-64
    ID6: |J_omega| varies 1.09x within a grain and |J_eps| varies 1.12x, i.e.
    the two mechanisms are indistinguishable from the data.

    This function is kept because it is genuinely useful for DIAGNOSING that
    degeneracy, and for monochromatic or energy-resolved work where the
    coupling is broken -- not as a term to switch on here.

    ONLY THE 5 DEVIATORIC COMPONENTS ARE USED, and that is a physical statement,
    not a convenience: in white-beam Laue the diffracted direction depends only
    on the DIRECTION of q, not its magnitude, so a hydrostatic strain moves no
    spot at all -- it only shifts the energy that satisfies Bragg.  Including a
    dilatational component would add an exactly unidentifiable parameter.

    project_fn : (om, eps5) -> (H, 2) pixels, with ``eps5`` in the
                 ``geometry.deviatoric5_to_symmetric`` layout
                 (e11, e22, e23, e13, e12).
    """
    if om.shape[-2:] != (3, 3):
        raise ValueError(f"om must be (3, 3), got {tuple(om.shape)}")
    eps0 = torch.zeros(5, dtype=om.dtype, device=om.device)

    try:
        from torch.func import jacfwd
    except ImportError as exc:  # pragma: no cover - torch < 2.0
        raise ImportError("strain_jacobian needs torch.func (torch >= 2.0)") from exc

    jac = jacfwd(lambda e: project_fn(om, e))(eps0)          # (H, 2, 5)
    if jac.shape[-1] != 5 or jac.shape[-2] != 2:
        raise ValueError(
            f"project_fn must return (H, 2); jacobian came back {tuple(jac.shape)}"
        )
    return jac


def combined_pixel_covariance(
    terms: "list[tuple[Tensor, Tensor]]",
    psf_sigma: float | Tensor,
    jitter: float = 0.0,
) -> Tensor:
    """Sum independent broadening mechanisms into one pixel covariance.

        Sigma_px = sum_i  J_i Sigma_i J_i^T  +  sigma_psf^2 I

    terms : list of (jac, cov) pairs.  ``jac`` is (..., 2, k) and ``cov`` is
            (k, k) or (..., k, k), for any k -- 3 for rotation, 5 for deviatoric
            strain, 1 for a scalar mechanism.  Mechanisms are assumed
            INDEPENDENT, which is why they simply add; a rotation-strain
            correlation would need a joint block instead.

    Reduces to :func:`pixel_covariance` for a single term.
    """
    if not terms:
        raise ValueError("need at least one (jac, cov) term")
    total = None
    for jac, cov in terms:
        if jac.shape[-2] != 2:
            raise ValueError(f"each jac must be (..., 2, k), got {tuple(jac.shape)}")
        k = jac.shape[-1]
        if cov.shape[-2:] != (k, k):
            raise ValueError(
                f"cov {tuple(cov.shape)} does not match jac's {k} parameters"
            )
        c = cov.expand(*jac.shape[:-2], k, k) if cov.dim() == 2 else cov
        contrib = jac @ c @ jac.transpose(-1, -2)
        total = contrib if total is None else total + contrib
    total = 0.5 * (total + total.transpose(-1, -2))
    psf = torch.as_tensor(psf_sigma, dtype=total.dtype, device=total.device)
    eye = torch.eye(2, dtype=total.dtype, device=total.device)
    return total + (psf**2 + jitter) * eye


def spread_covariance(
    sigma_par: Tensor | float,
    sigma_perp: Tensor | float,
    axis: Tensor,
) -> Tensor:
    """Axially symmetric orientation covariance, radians^2 -> (..., 3, 3).

    Variance ``sigma_par^2`` along ``axis`` and ``sigma_perp^2`` in the plane
    perpendicular to it:

        Sigma_omega = sigma_perp^2 I + (sigma_par^2 - sigma_perp^2) n n^T

    This is the anisotropic spread measured on the Ti-64 streaks (sigma_par
    ~8.5x the PSF floor, sigma_perp at it).  ``axis`` need not be normalized;
    it is normalized here.  Batched over leading dimensions of ``axis``.
    """
    n = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dtype, device = n.dtype, n.device
    sp = torch.as_tensor(sigma_par, dtype=dtype, device=device)
    sq = torch.as_tensor(sigma_perp, dtype=dtype, device=device)
    eye = torch.eye(3, dtype=dtype, device=device).expand(*n.shape[:-1], 3, 3)
    outer = n.unsqueeze(-1) * n.unsqueeze(-2)                     # (..., 3, 3)
    return (sq**2).unsqueeze(-1).unsqueeze(-1) * eye + (
        (sp**2 - sq**2).unsqueeze(-1).unsqueeze(-1) * outer
    )


def pixel_covariance(
    jac: Tensor,
    cov_omega: Tensor,
    psf_sigma: float | Tensor,
    jitter: float = 0.0,
) -> Tensor:
    """Sigma_px = J Sigma_omega J^T + sigma_psf^2 I  -> (..., 2, 2), px^2.

    jac       : (..., 2, 3) pixels per radian.
    cov_omega : (..., 3, 3) or (3, 3) orientation covariance, rad^2.
    psf_sigma : instrument point-spread sigma in PIXELS (1.06 px measured on Si
                for the 34-ID-E detector).  The PSF floor is what keeps the
                result positive-definite when the spread is zero.
    jitter    : optional extra variance added to the diagonal, px^2.

    The result is symmetrized explicitly: ``J Sigma J^T`` is symmetric in exact
    arithmetic but drifts at float32, and the splat requires a valid covariance.
    """
    if jac.shape[-2:] != (2, 3):
        raise ValueError(f"jac must be (..., 2, 3), got {tuple(jac.shape)}")
    if cov_omega.dim() == 2:
        cov_omega = cov_omega.expand(*jac.shape[:-2], 3, 3)
    cov = jac @ cov_omega @ jac.transpose(-1, -2)                 # (..., 2, 2)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    psf = torch.as_tensor(psf_sigma, dtype=jac.dtype, device=jac.device)
    floor = (psf**2 + jitter) * torch.eye(
        2, dtype=jac.dtype, device=jac.device
    )
    return cov + floor
