"""Differentiable JAX Laue forward projection (port of laue_torch).

Single-framework JAX implementation so the Laue forward composes with JAX-CPFEM
in one autodiff graph (Plan A of jax_cpfem_followup_plan.md, bridge Option 7.1).

Enable float64 to match the laue_torch reference::

    import jax; jax.config.update("jax_enable_x64", True)
"""

__version__ = "0.1.0"

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
from .rasterize import gaussian_splat, hard_window, soft_window, pseudo_voigt_splat
from .forward import laue_forward

__all__ = [
    "HC_KEV_NM",
    "deviatoric5_to_symmetric",
    "gaussian_splat",
    "hard_window",
    "laue_forward",
    "pseudo_voigt_splat",
    "quat_to_matrix",
    "reciprocal_matrix",
    "rodrigues_to_matrix",
    "sixd_to_matrix",
    "soft_window",
    "strain_to_B",
    "to_rotation_matrix",
    "voigt_to_symmetric",
]
