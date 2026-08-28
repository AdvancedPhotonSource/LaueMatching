"""Differentiable coded-aperture transmission for Laue diffraction.

Adds depth-resolved Laue microdiffraction support to ``laue_torch`` following:

* Gürsoy, Sheyfer, Wojcik, Liu, Tischler.  *J. Appl. Cryst.* **55**,
  1104–1110 (2022) — coded-aperture acquisition + reconstruction.
* Gürsoy, Sheyfer, Wojcik, Liu, Tischler.  *Rev. Sci. Instrum.* **94**,
  013702 (2023) — 6-DOF digital autofocusing of the coded-aperture
  geometry.

The :class:`CodedApertureMask` module computes the transmission
:math:`T(\\mathrm{ray}, p, E) \\in [0,1]` of a diffracted ray through a
binary de Bruijn–coded absorber (typically Au on Si\\ :sub:`3`\\ N\\ :sub:`4`)
as a function of the aperture pose (6 DOF), scan offset along the beam,
and per-ray energy.  All operations are torch-differentiable so the
mask pose and bar parameters can be jointly refined with orientation
and strain via autograd.

See ``laue_torch/implementation_plan_coded_aperture.md`` for the full
implementation roadmap and validation targets.
"""

from .absorption import mu_au, mu_si3n4
from .mask import CodedApertureMask, build_de_bruijn_sequence
from .mask2d import (
    CodedApertureMask2D,
    build_mura_pattern,
    build_pinhole_pattern,
    build_random_binary_pattern,
)
from .mask_spectral import CodedApertureMaskSpectral
from .autofocus import AutofocusResult, autofocus_geometry
from .landscape import LandscapeReport, autofocus_hessian, format_report
from .baseline_twostage import (
    TwoStagePixelResult,
    two_stage_pixel_reconstruct,
    two_stage_scan_reconstruct,
)
from .io_h5 import (
    DEFAULT_LAYOUT,
    load_mask_h5,
    load_voxel_h5,
    save_mask_h5,
    save_voxel_h5,
)

__all__ = [
    "AutofocusResult",
    "CodedApertureMask",
    "CodedApertureMask2D",
    "CodedApertureMaskSpectral",
    "DEFAULT_LAYOUT",
    "LandscapeReport",
    "TwoStagePixelResult",
    "autofocus_geometry",
    "autofocus_hessian",
    "build_de_bruijn_sequence",
    "build_mura_pattern",
    "build_pinhole_pattern",
    "build_random_binary_pattern",
    "format_report",
    "load_mask_h5",
    "load_voxel_h5",
    "mu_au",
    "mu_si3n4",
    "save_mask_h5",
    "save_voxel_h5",
    "two_stage_pixel_reconstruct",
    "two_stage_scan_reconstruct",
]
