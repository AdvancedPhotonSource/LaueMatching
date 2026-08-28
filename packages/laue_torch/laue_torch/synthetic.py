"""Synthetic truth fixtures and a model factory for tests and tutorials.

These were previously in ``laue_torch/experiments/utils.py``, which was a
research scratch area that is not part of the distributed package. The three
helpers here are the only pieces of it that shipped code and the test suite
actually depend on, so they live in the package proper.

The numbers are the canonical FCC Cu-like sample on the ``params_sim.txt``
detector, with orientations taken from ``fourOrientations.csv`` -- a stable
seed, not a physically meaningful configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .forward import LaueForwardModel

__all__ = ["Truth", "default_truth", "fcc_hkls", "make_model"]


def fcc_hkls(h_max: int = 8) -> Tensor:
    """All FCC-allowed (h, k, l) with |h|, |k|, |l| <= ``h_max``, excluding 000.

    FCC extinction: h, k, l must be all even or all odd.
    """
    out = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if not ((h % 2 == k % 2) and (k % 2 == l % 2)):
                    continue
                out.append([h, k, l])
    return torch.tensor(out, dtype=torch.long)


@dataclass
class Truth:
    """Ground-truth parameters for a synthetic Laue pattern."""

    lat: Tensor
    P: Tensor
    R: Tensor
    U: Tensor             # (G, 3, 3) rotation matrices
    eps: Tensor | None    # (G, 6) Voigt strain or None


def default_truth(n_grains: int = 1, strain: bool = False,
                  dtype: torch.dtype = torch.float64) -> Truth:
    """The canonical synthetic sample: FCC Cu-like, up to 4 grains."""
    lat = torch.tensor([0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0], dtype=dtype)
    P = torch.tensor([0.028745, 0.002788, 0.513115], dtype=dtype)
    R = torch.tensor([-1.20131258, -1.21399082, -1.21881158], dtype=dtype)
    # Use the canonical fourOrientations.csv rotations as a stable seed.
    base = torch.tensor([
        [[0.867151, 0.494088, 0.062670],
         [-0.052670, 0.216095, -0.974957],
         [-0.495254, 0.842135, 0.213410]],
        [[0.960281, -0.278115, 0.022653],
         [0.039600, 0.216183, 0.975556],
         [-0.276212, -0.935911, 0.218609]],
        [[0.781022, 0.604762, 0.155372],
         [-0.143788, 0.418195, -0.896878],
         [-0.607626, 0.677975, 0.413520]],
        [[0.642588, -0.523001, 0.560226],
         [0.762020, 0.355870, -0.541898],
         [0.077683, 0.774291, 0.627745]],
    ], dtype=dtype)
    U = base[:n_grains]
    eps = None
    if strain:
        eps = torch.tensor([
            [1.0e-3, -1.5e-3, 5.0e-4, 2.0e-4, -1.0e-4, 3.0e-4],
            [5.0e-4, 8.0e-4, -1.0e-3, -1.5e-4, 2.0e-4, -1.0e-4],
            [-2.0e-3, 1.0e-3, 1.0e-3, 1.0e-4, 1.5e-4, -2.0e-4],
            [3.0e-4, -3.0e-4, 0.0,    -2.0e-4, 1.0e-4, 1.5e-4],
        ], dtype=dtype)[:n_grains]
    return Truth(lat=lat, P=P, R=R, U=U, eps=eps)


def make_model(
    *,
    strain_mode: str = "none",
    rotation: str = "rodrigues",
    detector_rotation: str = "rodrigues",
    n_pix: int = 768,
    px_size: float = 0.0006,
    psf_sigma: float = 3.0,
    h_max: int = 8,
    energy_image: bool = True,
    hard: bool = False,
    tau_z: float = 5e-3,
    tau_px: float = 2.0,
    tau_E: float = 0.3,
    reduce: str = "sum",
) -> LaueForwardModel:
    """Build a :class:`LaueForwardModel` on the canonical synthetic detector."""
    return LaueForwardModel(
        hkls=fcc_hkls(h_max=h_max),
        n_pix=(n_pix, n_pix),
        px_size=(px_size, px_size),
        psf_sigma=psf_sigma,
        render_window=int(2 * math.ceil(3 * psf_sigma) + 1),
        rotation=rotation,
        detector_rotation=detector_rotation,
        strain_mode=strain_mode,
        hard=hard,
        tau_z=tau_z,
        tau_px=tau_px,
        tau_E=tau_E,
        reduce=reduce,
        energy_image=energy_image,
    )
