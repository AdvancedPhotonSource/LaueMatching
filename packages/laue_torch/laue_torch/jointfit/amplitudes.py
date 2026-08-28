"""Non-negative amplitude solve -- the inner half of the separable fit.

Given the per-reflection basis (``design.build_basis``) the model is linear in
the amplitudes,

    minimize_{a >= 0}  || B^T a - y ||^2

so the amplitudes can be solved exactly at every outer step instead of being
optimized jointly with the orientations.  That is variable projection: the outer
problem keeps only ~6 nonlinear parameters per grain (orientation + spread)
rather than one extra parameter per reflection.

Non-negativity is not cosmetic.  It is the constraint that stops a spurious
grain from "explaining" intensity by subtracting a negative contribution
somewhere else -- without it the free amplitudes would make the model able to
fit essentially anything.

GRADIENTS.  By default the returned amplitudes are DETACHED and the outer
gradient is taken at fixed ``a*``.  That is correct, not an approximation: at
the inner optimum the objective's derivative with respect to the free
amplitudes vanishes, and the clamped ones stay clamped under a small parameter
perturbation, so the envelope theorem gives

    d/dtheta [ min_a f(theta, a) ]  =  (partial f / partial theta) |_{a = a*}

Pass ``differentiable=True`` to unroll the solver instead (a much larger graph;
only needed if strict complementarity is in doubt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["AmplitudeSolution", "gram_and_rhs", "solve_amplitudes", "solve_nnls"]


@dataclass
class AmplitudeSolution:
    """Result of an amplitude solve.

    amplitudes : (K,) non-negative amplitudes.
    n_iter     : iterations actually run.
    converged  : whether the change fell below ``tol`` before ``max_iter``.
                 Reported rather than asserted -- a silently truncated solve
                 would look like a poor model fit and be misread as physics.
    max_step   : the final iteration's largest amplitude change.
    """

    amplitudes: Tensor
    n_iter: int
    converged: bool
    max_step: float


def gram_and_rhs(basis: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """G = B B^T  and  b = B y, for the quadratic form 0.5 a^T G a - b^T a.

    basis  : (K, P) per-reflection basis rows.
    target : (P,) or (nx, ny) observed intensity over the same ROI.
    """
    y = target.reshape(-1)
    if basis.shape[-1] != y.shape[0]:
        raise ValueError(
            f"basis has {basis.shape[-1]} pixels but target has {y.shape[0]}"
        )
    return basis @ basis.transpose(-1, -2), basis @ y


def solve_nnls(
    gram: Tensor,
    rhs: Tensor,
    ridge: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-12,
    init: Tensor | None = None,
) -> AmplitudeSolution:
    """min_{a >= 0} 0.5 a^T G a - b^T a, by projected-gradient FISTA.

    ridge : Tikhonov term added to the diagonal.  Two reflections that land on
            top of each other make G singular and their split is genuinely
            undetermined; a small ridge picks the minimum-norm split instead of
            letting the solver wander.
    """
    if gram.shape[-1] != gram.shape[-2]:
        raise ValueError(f"gram must be square, got {tuple(gram.shape)}")
    if gram.shape[-1] != rhs.shape[-1]:
        raise ValueError(
            f"gram is {tuple(gram.shape)} but rhs is {tuple(rhs.shape)}"
        )
    k = gram.shape[-1]
    if k == 0:
        return AmplitudeSolution(rhs.new_zeros(0), 0, True, 0.0)

    G = gram
    if ridge:
        G = G + ridge * torch.eye(k, dtype=G.dtype, device=G.device)

    # Lipschitz constant of the gradient = largest eigenvalue of G.
    lam_max = torch.linalg.eigvalsh(G).max().clamp_min(1e-30)
    step = 1.0 / lam_max

    a = torch.zeros(k, dtype=G.dtype, device=G.device) if init is None \
        else init.clamp_min(0).clone()
    z = a.clone()
    t = 1.0
    converged = False
    max_step = math.inf
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        grad = z @ G - rhs
        a_new = (z - step * grad).clamp_min(0.0)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        delta = a_new - a
        # Adaptive restart: if momentum starts fighting the descent direction,
        # drop it.  Without this FISTA can oscillate on the near-degenerate
        # Gram matrices that overlapping reflections produce.
        if float((z - a_new) @ delta) > 0:
            t_new = 1.0
            z = a_new.clone()
        else:
            z = a_new + ((t - 1.0) / t_new) * delta
        max_step = float(delta.abs().max())
        a, t = a_new, t_new
        if max_step < tol:
            converged = True
            break

    return AmplitudeSolution(a, n_iter, converged, max_step)


def solve_amplitudes(
    basis: Tensor,
    target: Tensor,
    ridge: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-12,
    differentiable: bool = False,
) -> AmplitudeSolution:
    """Solve for non-negative per-reflection amplitudes over one ROI.

    basis          : (K, P) from ``design.build_basis``.
    target         : (P,) or (nx, ny) observed intensity.
    differentiable : unroll the solver into the graph.  Default False, which
                     detaches ``a*``; see the module docstring on why that
                     yields the correct outer gradient anyway.
    """
    if differentiable:
        gram, rhs = gram_and_rhs(basis, target)
        return solve_nnls(gram, rhs, ridge=ridge, max_iter=max_iter, tol=tol)

    with torch.no_grad():
        gram, rhs = gram_and_rhs(basis, target)
        sol = solve_nnls(gram, rhs, ridge=ridge, max_iter=max_iter, tol=tol)
    return sol


def residual(basis: Tensor, target: Tensor, amplitudes: Tensor) -> Tensor:
    """Sum of squared residuals for the fitted model, differentiable in ``basis``.

    This is the quantity the outer loop minimizes.  ``amplitudes`` is normally
    the detached solution from :func:`solve_amplitudes`.
    """
    y = target.reshape(-1)
    model = amplitudes @ basis
    return ((model - y) ** 2).sum()
