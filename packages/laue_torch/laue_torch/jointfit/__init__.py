"""Joint (non-greedy) multi-grain fitting.

The peel pipeline is a greedy matching pursuit on PEAKS: index, forward-project,
mask a disc, re-index the residual.  Measurements on a two-phase alloy frame
with overlapping grains showed that forces every pixel to exactly one winner,
when overlapping grain contributions physically ADD -- and that no mask shape
repairs it.

This subpackage fits grains JOINTLY against continuous image intensity instead.
Each reflection carries its own free non-negative amplitude, so absolute
intensities -- which would need |F|^2 * I0(E), and I0(E) is unavailable -- are
never predicted.  Orientation information comes from POSITION and SHAPE only,
which are spectrum-independent.

Amplitudes enter linearly, so the problem is separable: a non-negative linear
solve inside, gradient descent on orientation and spread outside.

Nothing in the existing peel/index path is modified by this package.
"""

from .amplitudes import (
    AmplitudeSolution,
    gram_and_rhs,
    residual,
    solve_amplitudes,
    solve_nnls,
)
from .design import ROI, build_basis, rois_from_labels, spots_in_roi, suggested_window
from .model import FitReport, JointGrainFit, SpotGeometry, fit_grains
from .select import BicReport, SelectionResult, model_bic, prune_grains
from .synthetic import (
    SyntheticScene,
    make_projection,
    make_scene,
    perturb_orientations,
    random_orientations,
)
from .footprint import (
    combined_pixel_covariance,
    finite_difference_jacobian,
    pixel_covariance,
    pixel_jacobian,
    spread_covariance,
    strain_jacobian,
    tangent_rotation,
)

__all__ = [
    "AmplitudeSolution",
    "BicReport",
    "FitReport",
    "SelectionResult",
    "model_bic",
    "prune_grains",
    "JointGrainFit",
    "ROI",
    "SpotGeometry",
    "SyntheticScene",
    "build_basis",
    "finite_difference_jacobian",
    "fit_grains",
    "gram_and_rhs",
    "make_projection",
    "make_scene",
    "perturb_orientations",
    "pixel_covariance",
    "pixel_jacobian",
    "random_orientations",
    "residual",
    "rois_from_labels",
    "solve_amplitudes",
    "solve_nnls",
    "spots_in_roi",
    "combined_pixel_covariance",
    "spread_covariance",
    "strain_jacobian",
    "suggested_window",
    "tangent_rotation",
]
