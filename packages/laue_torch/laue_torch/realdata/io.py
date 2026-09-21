"""I/O adapter for LaueMatching post-processed H5 output.

The default :class:`LaueScanLoader` reads the per-image ``*.output.h5``
that ``laue_index.pipeline.laue_postprocess`` (streaming) and
``RunImage`` write:

* ``/entry/results/orientations`` --- (N, M) array of indexed solutions,
  one indexer solution-table row per solution. Two layouts exist (column
  map in ``laue_index.records.SOLUTION_FORMATS``): 34 columns with
  GrainNr in col 0 and the row-major 3x3 orientation matrix in cols
  22..30 (RunImage), or 35 columns with ImageNr prepended and the matrix
  in cols 23..31 (stream). The layout is picked from the column count.
* ``/entry/results/filtered_orientations`` --- quality-filtered subset,
  same layout.
* ``/entry/data/raw_data`` --- the raw detector image.
* ``/entry/data/input_blurred`` --- the preprocessed (background-
  subtracted, blurred) image used for indexing.
* ``/entry/data/component_centers`` --- (N, 4) array of detected
  spot centres ``(label, x, y, area)``, x = column, y = row.
* attrs on ``/entry/results``: ``image_nr``, ``source_file``,
  ``source_frame``.

AXIS ORDER. The images are stored as real detector frames,
``image[row, col]`` = ``[Y, X]``; the forward model renders ``img[X, Y]``.
The loader returns the image AS STORED and records the layout in
``VoxelMeasurement.axis_order``: always set, to ``"YX"`` unless the file
carries an ``<entry>/axis_order`` marker saying otherwise. A hand-built
``VoxelMeasurement`` has ``axis_order=None`` and the refiners refuse it. The refiners
(:class:`~laue_torch.realdata.driver.VoxelODFRefiner`,
:class:`~laue_torch.realdata.multi_grain.MultiGrainVoxelRefiner`) do the
transpose at their entry point, so this is the ONE place the conversion
happens. ``spots_xy`` is already ``(X, Y)`` and is not transposed.

For a scan, each voxel is its own H5 file (one per frame); the
loader takes a directory of such files and yields a sequence of
:class:`VoxelMeasurement` records.

If your data layout differs, subclass and override :meth:`load_voxel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..io import (
    AXIS_ORDER_DETECTOR,
    read_axis_order,
    solution_orientation_columns,
)


@dataclass
class VoxelMeasurement:
    """One voxel's worth of measurement.

    ``U_seed_list`` holds one or more candidate orientations from the
    upstream indexer; the refiner can pick the best (e.g. by image
    residual) or refine each into a separate mode of a mixture model.
    """
    voxel_index: int
    image: Tensor                        # detector image, layout given by axis_order
    U_seed_list: Tensor                  # (K, 3, 3) — indexer-supplied orientations
    metadata: dict                       # arbitrary auxiliary info (e.g. source_file)
    spots_xy: Optional[Tensor] = None    # (S, 2) — detected spot centres (X, Y) (optional)
    voxel_position_um: Optional[tuple[float, float, float]] = None
    # "YX": image[row, col] = (NrPxY, NrPxX), a real frame.
    # "XY": img[X, Y] = (NrPxX, NrPxY), the forward model's own layout.
    # No default: None makes the refiners raise, because a wrong guess is
    # silent on a square detector (VoxelODFRefiner did not transpose in
    # 0.1.3, so a guessed "YX" would silently flip a laue_torch render).
    # LaueScanLoader always sets it: from the file's marker, else "YX".
    # Consumers convert with ``laue_torch.io.to_model_layout``.
    axis_order: Optional[str] = None


class LaueScanLoader:
    """Iterates the H5 files produced by ``laue_postprocess.py``.

    Constructor takes a directory or a list of file paths. If a
    directory is given, all ``*.h5`` files matching ``pattern`` are
    used and sorted by name (alphabetical order is usually voxel
    order for typical scan filenames).

    Parameters
    ----------
    paths_or_dir : str or Path or sequence of paths
    pattern : glob pattern, used when ``paths_or_dir`` is a directory.
    image_dataset : H5 dataset path for the detector image (default:
        ``/entry/data/input_blurred`` --- the preprocessed image used
        for indexing). Use ``/entry/data/raw_data`` to refine against
        the raw frame instead.
    use_filtered : if True, use ``/entry/results/filtered_orientations``;
        else use the unfiltered ``/entry/results/orientations``.
    orientation_columns : (lo, hi) col indices that carry the row-major
        3×3 orientation matrix in the orientations array. Default
        ``None`` picks it from the column count: (22, 31) for the
        34-column RunImage layout, (23, 32) for the 35-column stream
        layout (``laue_index.records.SOLUTION_FORMATS``); any other
        count raises. Pass a tuple only for a non-indexer layout.

    The image is returned as stored (detector ``[row, col]``), with the
    layout recorded in ``VoxelMeasurement.axis_order``; see the module
    docstring.
    """

    def __init__(
        self,
        paths_or_dir,
        *,
        pattern: str = "*.h5",
        image_dataset: str = "/entry/data/input_blurred",
        use_filtered: bool = True,
        orientation_columns: Optional[tuple[int, int]] = None,
    ):
        if isinstance(paths_or_dir, (str, Path)):
            p = Path(paths_or_dir)
            if p.is_dir():
                self.paths = sorted(p.glob(pattern))
            else:
                self.paths = [p]
        else:
            self.paths = [Path(p) for p in paths_or_dir]
        self.image_dataset = image_dataset
        self.use_filtered = use_filtered
        self.orientation_columns = orientation_columns
        if not self.paths:
            raise FileNotFoundError(
                f"No H5 files matched {paths_or_dir} (pattern {pattern!r})")

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterable[VoxelMeasurement]:
        for idx, path in enumerate(self.paths):
            yield self.load_voxel(idx, path)

    def load_voxel(self, voxel_index: int, path: Path) -> VoxelMeasurement:
        """Load a single voxel from its H5 file."""
        import h5py
        with h5py.File(path, "r") as hf:
            results_group = "/entry/results"
            ds_name = ("filtered_orientations" if self.use_filtered
                       else "orientations")
            if results_group + "/" + ds_name not in hf:
                raise KeyError(f"{path}: missing {results_group}/{ds_name}")
            orient_arr = np.asarray(hf[results_group + "/" + ds_name])
            if orient_arr.size == 0:
                # No solution for this frame (postprocess writes shape (0,)).
                U_seed_list = torch.zeros((0, 3, 3), dtype=torch.float64)
            else:
                if orient_arr.ndim == 1:
                    orient_arr = orient_arr[None, :]
                if self.orientation_columns is None:
                    try:
                        lo, hi = solution_orientation_columns(orient_arr.shape[1])
                    except ValueError as exc:
                        raise ValueError(f"{path}: {results_group}/{ds_name}: {exc}") from None
                else:
                    lo, hi = self.orientation_columns
                if orient_arr.ndim != 2 or orient_arr.shape[1] < hi or hi - lo != 9:
                    raise ValueError(
                        f"{path}: orientations array shape {orient_arr.shape} "
                        f"cannot supply columns [{lo}, {hi}); "
                        f"is your `orientation_columns` config correct?")
                U_seed_list = torch.tensor(
                    orient_arr[:, lo:hi].reshape(-1, 3, 3), dtype=torch.float64)

            if self.image_dataset not in hf:
                raise KeyError(f"{path}: missing {self.image_dataset}")
            image = torch.tensor(np.asarray(hf[self.image_dataset]),
                                 dtype=torch.float64)
            # Honour an <entry>/axis_order marker (laue_torch-written files);
            # LaueMatching indexer output has none and is detector layout.
            entry = "/" + self.image_dataset.strip("/").split("/")[0]
            axis_order = read_axis_order(hf, entry + "/axis_order",
                                         default=AXIS_ORDER_DETECTOR)

            spots_xy = None
            cc_path = "/entry/data/component_centers"
            if cc_path in hf:
                cc = np.asarray(hf[cc_path])
                if cc.ndim == 2 and cc.shape[1] >= 3:
                    spots_xy = torch.tensor(cc[:, 1:3], dtype=torch.float64)

            metadata = dict(hf[results_group].attrs)
            metadata["source_file"] = str(path)

        return VoxelMeasurement(
            voxel_index=voxel_index,
            image=image,
            U_seed_list=U_seed_list,
            spots_xy=spots_xy,
            metadata=metadata,
            axis_order=axis_order,
        )


def load_lauematching_h5(path) -> VoxelMeasurement:
    """Convenience: load a single LaueMatching H5 file as one voxel."""
    loader = LaueScanLoader([path])
    return next(iter(loader))
