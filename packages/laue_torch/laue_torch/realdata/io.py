"""I/O adapter for LaueMatching post-processed H5 output.

The default :class:`LaueScanLoader` reads the structure that
``scripts/laue_postprocess.py`` writes:

* ``/entry/results/orientations`` --- (N, M) array of indexed solutions.
  Per the RunImage / Stream convention, each row carries (at least)
  GrainNr in column 0 and orientation matrix elements in columns 1..9
  (row-major 3×3); extra columns may carry quality metrics.
* ``/entry/results/filtered_orientations`` --- quality-filtered subset.
* ``/entry/data/raw_data`` --- the raw detector image.
* ``/entry/data/input_blurred`` --- the preprocessed (background-
  subtracted, blurred) image used for indexing.
* ``/entry/data/component_centers`` --- (N, 4) array of detected
  spot centres ``(label, x, y, area)``.
* attrs on ``/entry/results``: ``image_nr``, ``source_file``,
  ``source_frame``.

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


@dataclass
class VoxelMeasurement:
    """One voxel's worth of measurement.

    ``U_seed_list`` holds one or more candidate orientations from the
    upstream indexer; the refiner can pick the best (e.g. by image
    residual) or refine each into a separate mode of a mixture model.
    """
    voxel_index: int
    image: Tensor                        # (Nx, Ny) — the preprocessed detector image
    U_seed_list: Tensor                  # (K, 3, 3) — indexer-supplied orientations
    metadata: dict                       # arbitrary auxiliary info (e.g. source_file)
    spots_xy: Optional[Tensor] = None    # (S, 2) — detected spot centres (optional)
    voxel_position_um: Optional[tuple[float, float, float]] = None


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
        (1, 10) follows the RunImage convention (GrainNr in col 0,
        9 matrix elements in cols 1–9).
    """

    def __init__(
        self,
        paths_or_dir,
        *,
        pattern: str = "*.h5",
        image_dataset: str = "/entry/data/input_blurred",
        use_filtered: bool = True,
        orientation_columns: tuple[int, int] = (1, 10),
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
        lo, hi = self.orientation_columns
        with h5py.File(path, "r") as hf:
            results_group = "/entry/results"
            ds_name = ("filtered_orientations" if self.use_filtered
                       else "orientations")
            if results_group + "/" + ds_name not in hf:
                raise KeyError(f"{path}: missing {results_group}/{ds_name}")
            orient_arr = np.asarray(hf[results_group + "/" + ds_name])
            if orient_arr.ndim != 2 or orient_arr.shape[1] < hi:
                raise ValueError(
                    f"{path}: orientations array shape {orient_arr.shape} "
                    f"has fewer than {hi} columns; "
                    f"is your `orientation_columns` config correct?")
            U_seed_list = torch.tensor(
                orient_arr[:, lo:hi].reshape(-1, 3, 3), dtype=torch.float64)

            if self.image_dataset not in hf:
                raise KeyError(f"{path}: missing {self.image_dataset}")
            image = torch.tensor(np.asarray(hf[self.image_dataset]),
                                 dtype=torch.float64)

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
        )


def load_lauematching_h5(path) -> VoxelMeasurement:
    """Convenience: load a single LaueMatching H5 file as one voxel."""
    loader = LaueScanLoader([path])
    return next(iter(loader))
