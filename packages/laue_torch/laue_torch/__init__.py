"""Differentiable PyTorch forward projection for LaueMatching."""

__version__ = "0.1.3"

from .geometry import (
    HC_KEV_NM,
    quat_to_matrix,
    rodrigues_to_matrix,
    sixd_to_matrix,
    to_rotation_matrix,
    reciprocal_matrix,
    voigt_to_symmetric,
    deviatoric5_to_symmetric,
    strain_to_B,
)
from .rasterize import gaussian_splat, hard_window, soft_window
from .forward import ForwardAux, LaueForwardModel
from .io import LaueParams, generate_hkls, load_orientations, parse_params
from .odf import DiscreteOrientationMixture, fibonacci_so3, random_so3
from .distributions import (
    CholeskyCov,
    GaussianStrain,
    IndependentVoxelDistribution,
    MixtureOfTangentGaussianSO3,
    MixtureOfVoxelDistributions,
    TangentGaussianSO3,
)
from .uncertainty import (
    LaplacePosterior,
    credible_interval_halfwidth,
    laplace_posterior,
)
from . import nye

__all__ = [
    "HC_KEV_NM",
    "CholeskyCov",
    "DiscreteOrientationMixture",
    "ForwardAux",
    "GaussianStrain",
    "IndependentVoxelDistribution",
    "LaueForwardModel",
    "LaueParams",
    "MixtureOfTangentGaussianSO3",
    "MixtureOfVoxelDistributions",
    "TangentGaussianSO3",
    "deviatoric5_to_symmetric",
    "fibonacci_so3",
    "gaussian_splat",
    "generate_hkls",
    "hard_window",
    "load_orientations",
    "parse_params",
    "quat_to_matrix",
    "random_so3",
    "reciprocal_matrix",
    "rodrigues_to_matrix",
    "sixd_to_matrix",
    "soft_window",
    "strain_to_B",
    "to_rotation_matrix",
    "voigt_to_symmetric",
]
