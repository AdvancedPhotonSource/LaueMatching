"""HDF5 I/O helpers for coded-aperture data.

Defines a *default* H5 layout for per-voxel coded-aperture scans plus
round-trip helpers for :class:`CodedApertureMask` and
:class:`CodedApertureVoxelMeasurement`.

Layout (one H5 per voxel; matches the existing LaueMatching ``/entry``
convention):

::

  /entry/
      attrs: voxel_index, source_file
      data/
          frames           (M, Nx, Ny) float64 — raw or pre-processed
                                                  detector frames
          scan_offsets_um  (M,)         float64 — coded-aperture scan
                                                  positions, µm
      results/
          U_seed           (3, 3)       float64 — seed orientation
                                                  from upstream indexer
          z_seed_um        scalar       float64 — initial depth guess
                                                  along the beam, µm
      mask/                              (mirrors CodedApertureMask state)
          sequence         (L,)         int64
          bar_widths_um    (L,)         float64
          au_thickness_um  scalar       float64
          sub_thickness_um scalar       float64
          position_um      (3,)         float64
          rotvec           (3,)         float64
          attrs: edge_softness_um

If your real data uses a different schema, supply your own loader by
subclassing :class:`laue_torch.realdata.CodedApertureScanLoader` and
overriding ``load_voxel`` — or pass ``dataset_paths={...}`` to the
default loader to remap dataset names without writing code.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    from .mask import CodedApertureMask
    from ..realdata.depth_resolved import CodedApertureVoxelMeasurement


DEFAULT_LAYOUT = {
    "frames": "/entry/data/frames",
    "scan_offsets_um": "/entry/data/scan_offsets_um",
    "U_seed": "/entry/results/U_seed",
    "z_seed_um": "/entry/results/z_seed_um",
    "mask_group": "/entry/mask",
}


# ── Mask round-trip ────────────────────────────────────────────────────────


def save_mask_h5(mask, path, *, mask_group: str = "/entry/mask") -> None:
    """Write a :class:`CodedApertureMask` to an H5 group."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as hf:
        if mask_group in hf:
            del hf[mask_group]
        g = hf.create_group(mask_group)
        g.create_dataset("sequence", data=mask.sequence.detach().cpu().numpy().astype(np.int64))
        g.create_dataset("bar_widths_um", data=mask.bar_widths_um.detach().cpu().numpy())
        g.create_dataset("au_thickness_um", data=float(mask.au_thickness_um.item()))
        g.create_dataset("sub_thickness_um", data=float(mask.sub_thickness_um.item()))
        g.create_dataset("position_um", data=mask.position_um.detach().cpu().numpy())
        g.create_dataset("rotvec", data=mask.rotvec.detach().cpu().numpy())
        g.attrs["edge_softness_um"] = float(mask.edge_softness_um)


def load_mask_h5(path, *, mask_group: str = "/entry/mask",
                 make_geometry_learnable: bool = False,
                 dtype: torch.dtype = torch.float64):
    """Read a :class:`CodedApertureMask` from an H5 group."""
    import h5py
    from .mask import CodedApertureMask

    path = Path(path)
    with h5py.File(path, "r") as hf:
        if mask_group not in hf:
            raise KeyError(f"{path}: no mask group at {mask_group!r}")
        g = hf[mask_group]
        sequence = torch.tensor(np.asarray(g["sequence"]), dtype=torch.int64)
        bar_widths_um = torch.tensor(np.asarray(g["bar_widths_um"]), dtype=dtype)
        au_thickness_um = float(np.asarray(g["au_thickness_um"]))
        sub_thickness_um = float(np.asarray(g["sub_thickness_um"]))
        position_um = torch.tensor(np.asarray(g["position_um"]), dtype=dtype)
        rotvec = torch.tensor(np.asarray(g["rotvec"]), dtype=dtype)
        edge_softness_um = float(g.attrs.get("edge_softness_um", 0.5))

    return CodedApertureMask(
        sequence=sequence,
        bar_widths_um=bar_widths_um,
        au_thickness_um=au_thickness_um,
        sub_thickness_um=sub_thickness_um,
        position_um=position_um,
        rotvec=rotvec,
        edge_softness_um=edge_softness_um,
        make_geometry_learnable=make_geometry_learnable,
        dtype=dtype,
    )


# ── Voxel-measurement round-trip ──────────────────────────────────────────


def save_voxel_h5(measurement, mask, path,
                  *, layout: dict = DEFAULT_LAYOUT) -> None:
    """Write a voxel measurement + the mask state to an H5 file.

    A full coded-aperture scan is a directory of such files, one per
    voxel — the loader iterates them via :class:`CodedApertureScanLoader`.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        entry = hf.create_group("entry") if "entry" not in hf else hf["entry"]
        entry.attrs["voxel_index"] = int(measurement.voxel_index)
        if measurement.metadata.get("source_file"):
            entry.attrs["source_file"] = str(measurement.metadata["source_file"])

        hf.create_dataset(
            layout["frames"],
            data=measurement.frame_stack.detach().cpu().numpy(),
        )
        hf.create_dataset(
            layout["scan_offsets_um"],
            data=measurement.scan_offsets_um.detach().cpu().numpy(),
        )
        hf.create_dataset(
            layout["U_seed"],
            data=measurement.U_seed.detach().cpu().numpy(),
        )
        hf.create_dataset(
            layout["z_seed_um"],
            data=float(measurement.z_seed_um),
        )

    save_mask_h5(mask, path, mask_group=layout["mask_group"])


def load_voxel_h5(path,
                  *, voxel_index: int = 0,
                  layout: dict = DEFAULT_LAYOUT,
                  dtype: torch.dtype = torch.float64):
    """Read a voxel measurement from H5 (without the mask)."""
    import h5py
    from ..realdata.depth_resolved import CodedApertureVoxelMeasurement

    path = Path(path)
    with h5py.File(path, "r") as hf:
        for key in ("frames", "scan_offsets_um", "U_seed", "z_seed_um"):
            if layout[key] not in hf:
                raise KeyError(f"{path}: missing {layout[key]!r}")
        frame_stack = torch.tensor(np.asarray(hf[layout["frames"]]), dtype=dtype)
        scan_offsets = torch.tensor(np.asarray(hf[layout["scan_offsets_um"]]), dtype=dtype)
        U_seed = torch.tensor(np.asarray(hf[layout["U_seed"]]), dtype=dtype)
        z_seed_um = float(np.asarray(hf[layout["z_seed_um"]]))
        metadata = {"source_file": str(path)}
        if "entry" in hf:
            for k in hf["entry"].attrs:
                metadata[k] = hf["entry"].attrs[k]
        # Fall back to attrs-supplied index if present (override caller arg).
        vi = int(metadata.get("voxel_index", voxel_index))

    return CodedApertureVoxelMeasurement(
        voxel_index=vi,
        frame_stack=frame_stack,
        scan_offsets_um=scan_offsets,
        U_seed=U_seed,
        z_seed_um=z_seed_um,
        metadata=metadata,
    )
