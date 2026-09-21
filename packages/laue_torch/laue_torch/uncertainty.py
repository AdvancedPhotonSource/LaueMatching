"""Laplace approximation to the posterior over recovered model parameters.

After optimisation has converged on a maximum-a-posteriori estimate
``θ_MAP``, the negative-log-likelihood is locally quadratic.  The
Laplace posterior covariance is

.. math::
    \\Sigma_\\mathrm{post} \\approx (H + \\Sigma_\\mathrm{prior}^{-1})^{-1},

with ``H`` the Hessian of the negative log-likelihood at θ_MAP.  With
a flat (improper) prior, ``Σ_post = H^{-1}`` and per-parameter marginal
posterior std follows from ``diag(Σ_post)``.

Usage: pass any **scalar** loss as a function of a single flat
parameter tensor.  The caller is responsible for making the loss
differentiable in θ (for the closure used by Adam, just pack/unpack
into the original module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor


@dataclass
class LaplacePosterior:
    """Result of a Laplace approximation."""
    theta: Tensor               # the MAP point (flat tensor)
    hessian: Tensor             # (n, n) symmetric Hessian
    cov: Tensor                 # (n, n) posterior covariance ≈ H^{-1}
    sigma: Tensor               # (n,) per-parameter marginal std
    eigvals: Tensor             # (n,) Hessian eigenvalues (ascending)
    cond_number: float          # λ_max / λ_min (effective)
    rank_eff: int               # number of eigenvalues > eigval_floor
    # False at a saddle: the Laplace approximation is then invalid and the
    # affected ``sigma`` entries are NaN. None if the installed midas_invert
    # does not report it.
    is_positive_definite: Optional[bool] = None
    n_negative_eigvals: Optional[int] = None


def laplace_posterior(
    loss_fn: Callable[[Tensor], Tensor],
    theta_flat: Tensor,
    *,
    noise_variance: float = 1.0,
    pinv_rtol: float = 1e-9,
) -> LaplacePosterior:
    """Compute the Laplace posterior at ``theta_flat``.

    ``loss_fn`` is the **per-data-point** mean-squared error loss as a
    function of the flat parameter vector.  ``noise_variance`` scales
    the Hessian to convert from MSE-loss curvature to log-likelihood
    curvature under a Gaussian noise model with that variance.

    A reasonable default is to set ``noise_variance`` to the converged
    value of ``loss_fn(theta_flat)``, which is the empirical Bayes /
    plug-in estimate.

    The numerical core (Hessian -> pinv covariance + eigen-diagnostics) is the
    shared ``midas_invert.laplace_uncertainty``; this wrapper preserves the
    laue_torch ``LaplacePosterior`` dataclass API its callers depend on.
    """
    from midas_invert import laplace_uncertainty as _laplace
    res = _laplace(loss_fn, theta_flat, noise_var=noise_variance, pinv_rtol=pinv_rtol)
    return LaplacePosterior(
        theta=theta_flat.detach().clone(),
        hessian=res["hessian"].detach(),
        cov=res["cov"].detach(),
        sigma=res["sigma"].detach(),
        eigvals=res["eigvals"].detach(),
        cond_number=res["cond_number"],
        rank_eff=res["rank_eff"],
        is_positive_definite=res.get("is_positive_definite"),
        n_negative_eigvals=res.get("n_negative_eigvals"),
    )


def laplace_posterior_from_residuals(
    residual_fn: Callable[[Tensor], Tensor],
    theta_flat: Tensor,
    *,
    pinv_rtol: float = 1e-9,
) -> LaplacePosterior:
    """Laplace posterior for a least-squares fit, on the right curvature scale.

    ``residual_fn(theta)`` returns the residual vector (model - data over the
    N data points the fit used). Under i.i.d. Gaussian noise of variance s^2
    the negative log-likelihood is ``SSR / (2 s^2)``, whose Hessian is
    ``H(0.5 * SSR) / s^2``. :func:`laplace_posterior` computes
    ``H(loss) / noise_variance``, so this passes ``loss = 0.5 * SSR`` and the
    plug-in ``noise_variance = SSR / N`` at ``theta_flat``.

    Passing the per-point MSE instead (``laplace_posterior``'s documented
    usage) gives a Hessian N/2 times too small, i.e. every sigma sqrt(N/2)
    times too large. ``tests/test_laplace_curvature_scale.py`` pins this
    helper against the empirical scatter of repeated noisy fits.
    """
    theta0 = theta_flat.detach()
    with torch.no_grad():
        r0 = residual_fn(theta0).reshape(-1)
    n = r0.numel()
    if n == 0:
        raise ValueError("residual_fn returned no residuals")
    noise_var = float((r0 ** 2).sum().item()) / n
    if not noise_var > 0.0:
        raise ValueError("zero residual at theta: the plug-in noise variance is 0, "
                         "so there is no Laplace width to report")

    def loss_fn(theta: Tensor) -> Tensor:
        return 0.5 * (residual_fn(theta).reshape(-1) ** 2).sum()

    return laplace_posterior(loss_fn, theta0, noise_variance=noise_var,
                             pinv_rtol=pinv_rtol)


def credible_interval_halfwidth(sigma: Tensor, conf: float = 0.683) -> Tensor:
    """Half-width of a marginal Gaussian credible interval at ``conf``."""
    from math import sqrt
    try:
        from scipy.special import erfinv  # type: ignore
        z = sqrt(2.0) * float(erfinv(conf))
    except ImportError:
        # Fallback for common confidences.
        z = {0.683: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(conf, 1.0)
    return z * sigma
