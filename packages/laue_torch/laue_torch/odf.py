"""ODF / orientation-mixture helpers.

Downstream consumer of :class:`laue_torch.forward.LaueForwardModel`.
The forward model knows how to render a batch of grains with per-grain
weights; this module supplies the (U, w) bag from various distributional
parameterizations.
"""

from __future__ import annotations

from typing import Optional

import math
import torch
from torch import Tensor, nn

from .geometry import quat_to_matrix


class DiscreteOrientationMixture(nn.Module):
    """Bag of `K` orientation candidates with learnable mixture weights.

    Parameters
    ----------
    U_init : (K, 3, 3)
        Initial orientation matrices (e.g. random samples on SO(3) or a
        coarse uniform grid).
    train_orientations : bool
        If True, also refine the orientations via 6D continuous params.
    weight_param : "softmax" | "softplus" | "raw"
        How to map raw logits to non-negative mixture weights:

        - softmax: `w = softmax(logits)` — sums to 1, dense ODF.
        - softplus: `w = softplus(logits)` — non-negative, scale-free.
        - raw: `w = logits` — unconstrained (caller handles priors).
    """

    def __init__(
        self,
        U_init: Tensor,
        train_orientations: bool = False,
        weight_param: str = "softmax",
    ):
        super().__init__()
        if U_init.dim() != 3 or U_init.shape[-2:] != (3, 3):
            raise ValueError(f"U_init must be (K, 3, 3), got {tuple(U_init.shape)}")
        self.K = U_init.shape[0]
        self.weight_param = weight_param
        self.logits = nn.Parameter(torch.zeros(self.K, dtype=U_init.dtype))
        if train_orientations:
            # Encode as 6D continuous representation for smooth refinement.
            d6 = torch.cat([U_init[:, :, 0], U_init[:, :, 1]], dim=-1)
            self.d6 = nn.Parameter(d6.clone())
            self.register_buffer("U_init", U_init.clone())
        else:
            self.register_buffer("U_buf", U_init.clone())
            self.d6 = None

    def orientations(self) -> Tensor:
        if self.d6 is not None:
            from .geometry import sixd_to_matrix
            return sixd_to_matrix(self.d6)
        return self.U_buf

    def weights(self) -> Tensor:
        if self.weight_param == "softmax":
            return torch.softmax(self.logits, dim=0)
        if self.weight_param == "softplus":
            return torch.nn.functional.softplus(self.logits)
        return self.logits

    def entropy(self) -> Tensor:
        """Shannon entropy of the mixture weights — useful as a sparsity prior."""
        w = self.weights()
        w_safe = w / w.sum().clamp_min(1e-30)
        return -(w_safe * (w_safe.clamp_min(1e-30)).log()).sum()

    def l1(self) -> Tensor:
        """L1 norm of weights — alternative sparsity prior."""
        return self.weights().abs().sum()


def random_so3(K: int, dtype: torch.dtype = torch.float64,
               device: str = "cpu", seed: Optional[int] = None) -> Tensor:
    """Uniform random rotations on SO(3) via random unit quaternions."""
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        q = torch.randn(K, 4, dtype=dtype, device=device, generator=gen)
    else:
        q = torch.randn(K, 4, dtype=dtype, device=device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    return quat_to_matrix(q)


def fibonacci_so3(K: int, dtype: torch.dtype = torch.float64,
                  device: str = "cpu") -> Tensor:
    """Quasi-uniform rotations from the Hopf fibration / Fibonacci grid on S^3."""
    i = torch.arange(K, dtype=dtype, device=device)
    phi = (1 + 5 ** 0.5) / 2
    u1 = i / K
    u2 = (i / phi) % 1.0
    u3 = (i * phi) % 1.0
    s1 = torch.sqrt(1 - u1)
    s2 = torch.sqrt(u1)
    q = torch.stack([
        s1 * torch.sin(2 * math.pi * u2),
        s1 * torch.cos(2 * math.pi * u2),
        s2 * torch.sin(2 * math.pi * u3),
        s2 * torch.cos(2 * math.pi * u3),
    ], dim=-1)
    return quat_to_matrix(q)
