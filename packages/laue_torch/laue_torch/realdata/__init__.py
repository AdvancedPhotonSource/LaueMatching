"""Tools for running per-voxel ODF/SDF refinement on real LaueMatching output.

Pipeline overview
-----------------

The real-data analysis is a three-stage pipeline that wraps the existing
``laue_torch`` ODF refinement around the LaueMatching post-processed H5
output:

1. :class:`io.LaueScanLoader` --- adapter that reads the LaueMatching
   ``/entry/results/...`` group and the matching detector image,
   exposing them per-voxel as ``VoxelMeasurement`` records.

2. :class:`driver.VoxelODFRefiner` --- runs the per-voxel ODF refinement
   given a measurement (image + indexer-supplied seed orientation) and
   returns the recovered ODF parameters with Laplace posterior.

3. :func:`plots.plot_orientation_map` and friends --- per-voxel 2-D map
   visualisation of any of the recovered scalar / tensor fields:
   orientation, mosaic spread, GND density tensor, posterior σ.

The default ``LaueScanLoader`` follows LaueMatching's
``laue_postprocess.py`` H5 layout (``/entry/results/orientations``,
``/entry/results/filtered_orientations``, ``/entry/data/raw_data``,
etc.).  If your data layout differs, subclass and override
:meth:`load_voxel`.
"""

from .io import (
    LaueScanLoader,
    VoxelMeasurement,
    load_lauematching_h5,
)
from .driver import VoxelODFRefiner, VoxelODFResult
from .plots import (
    plot_orientation_map,
    plot_sigma_map,
    plot_gnd_map,
)
from .geometry import (
    Crystal,
    GeoN,
    make_lauematching_params,
    parse_crystal,
    parse_geon_xml,
    write_lauematching_config,
)
from .multi_grain import MultiGrainResult, MultiGrainVoxelRefiner
from .depth_resolved import (
    CodedApertureVoxelMeasurement,
    DepthResolvedVoxelPosterior,
    DepthResolvedVoxelRefiner,
    DepthResolvedVoxelResult,
)
from .reference_grain import (
    ReferenceGrainParallaxRefiner,
    ReferenceGrainResult,
    TwoSourceMeasurement,
)
from .coded_aperture_loader import CodedApertureScanLoader
from .multi_voxel_tv import MultiVoxelTVRefiner, MultiVoxelTVResult

__all__ = [
    "LaueScanLoader",
    "VoxelMeasurement",
    "load_lauematching_h5",
    "VoxelODFRefiner",
    "VoxelODFResult",
    "MultiGrainResult",
    "MultiGrainVoxelRefiner",
    "CodedApertureScanLoader",
    "CodedApertureVoxelMeasurement",
    "DepthResolvedVoxelPosterior",
    "DepthResolvedVoxelRefiner",
    "DepthResolvedVoxelResult",
    "ReferenceGrainParallaxRefiner",
    "ReferenceGrainResult",
    "TwoSourceMeasurement",
    "MultiVoxelTVRefiner",
    "MultiVoxelTVResult",
    "plot_orientation_map",
    "plot_sigma_map",
    "plot_gnd_map",
]
