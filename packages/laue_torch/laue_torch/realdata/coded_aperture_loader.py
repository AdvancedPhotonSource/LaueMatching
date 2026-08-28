"""Iterator over a directory of per-voxel coded-aperture H5 files.

Mirrors the existing :class:`LaueScanLoader` pattern but produces
:class:`CodedApertureVoxelMeasurement` records (frame stacks +
mask scan positions + seed orientation + seed depth).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

from ..coded_aperture.io_h5 import DEFAULT_LAYOUT, load_voxel_h5, load_mask_h5
from .depth_resolved import CodedApertureVoxelMeasurement


class CodedApertureScanLoader:
    """Iterate per-voxel H5 files for a coded-aperture scan.

    Parameters
    ----------
    paths_or_dir
        Directory of ``*.h5`` files (one per voxel) or an explicit list
        of paths.
    pattern
        Glob used when ``paths_or_dir`` is a directory.  Files are
        sorted by name (alphabetical = voxel order for typical scan
        filenames).
    layout
        Dataset-path mapping; overrides :data:`DEFAULT_LAYOUT` for
        non-standard schemas.  Useful for adapting to whatever H5
        format the real-data partner ships without subclassing.
    dtype
        Torch dtype for the materialised tensors.

    Notes
    -----
    The mask is *not* part of the per-voxel iteration: it is one object
    shared across all voxels in a scan.  Read it once with
    :meth:`load_mask` (which pulls from the first voxel file by
    default) or :func:`laue_torch.coded_aperture.load_mask_h5` directly.
    """

    def __init__(
        self,
        paths_or_dir,
        *,
        pattern: str = "*.h5",
        layout: Optional[dict] = None,
        dtype: torch.dtype = torch.float64,
    ):
        if isinstance(paths_or_dir, (str, Path)):
            p = Path(paths_or_dir)
            if p.is_dir():
                self.paths = sorted(p.glob(pattern))
            else:
                self.paths = [p]
        else:
            self.paths = [Path(x) for x in paths_or_dir]
        if not self.paths:
            raise FileNotFoundError(
                f"No H5 files matched {paths_or_dir} (pattern {pattern!r})"
            )
        self.layout = dict(layout) if layout is not None else dict(DEFAULT_LAYOUT)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterable[CodedApertureVoxelMeasurement]:
        for idx, path in enumerate(self.paths):
            yield self.load_voxel(idx, path)

    def load_voxel(self, voxel_index: int, path: Path) -> CodedApertureVoxelMeasurement:
        return load_voxel_h5(
            path, voxel_index=voxel_index,
            layout=self.layout, dtype=self.dtype,
        )

    def load_mask(self, *, source: Optional[Path] = None,
                  make_geometry_learnable: bool = False):
        """Read the (shared) mask from the first voxel file (or ``source``)."""
        src = Path(source) if source is not None else self.paths[0]
        return load_mask_h5(
            src,
            mask_group=self.layout["mask_group"],
            make_geometry_learnable=make_geometry_learnable,
            dtype=self.dtype,
        )
