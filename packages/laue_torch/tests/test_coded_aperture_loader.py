"""Round-trip: save/load coded-aperture H5, iterate via the scan loader,
refine via the depth refiner.

This is the Phase 5 integration gate — it does *not* assert tight
recovery (that's Phase 2 / Phase 3 territory).  It does verify that:

* :func:`save_voxel_h5` writes the documented schema,
* :class:`CodedApertureScanLoader` reads it back into
  :class:`CodedApertureVoxelMeasurement` records exactly equal to what
  was written,
* :func:`load_mask_h5` reads the mask back to bit-identity,
* the loader output plugs into :class:`DepthResolvedVoxelRefiner`
  without further plumbing.

The H5 schema validated here is the default; real-data partners can
remap dataset names via the ``layout`` argument to the loader.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import (
    CodedApertureMask,
    build_de_bruijn_sequence,
    load_mask_h5,
    save_voxel_h5,
)
from laue_torch.io import LaueParams, generate_hkls
from laue_torch.realdata import (
    CodedApertureScanLoader,
    CodedApertureVoxelMeasurement,
    DepthResolvedVoxelRefiner,
)


DTYPE = torch.float64


def _toy_params() -> LaueParams:
    return LaueParams(
        sg_num=225,
        symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.0016, px_y=0.0016, n_pix_x=256, n_pix_y=256,
        E_lo=5.0, E_hi=12.0, psf_sigma=2.0,
    )


def _toy_mask() -> CodedApertureMask:
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2),
        bar_widths_um=12.0,
        au_thickness_um=6.0,
        sub_thickness_um=0.0,
        position_um=torch.tensor([3.0, 1.0, 500.0], dtype=DTYPE),
        rotvec=torch.tensor([0.05, -0.03, 0.02], dtype=DTYPE),
        edge_softness_um=2.0,
        make_geometry_learnable=False,
        dtype=DTYPE,
    )


def _toy_orientation() -> torch.Tensor:
    q = torch.tensor(
        [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
        dtype=DTYPE,
    )
    return quat_to_orient_mat(q).reshape(3, 3)


def _synth_voxel(vi: int, z_um: float, mask: CodedApertureMask,
                  params: LaueParams, hkls: torch.Tensor):
    model = LaueForwardModel(
        hkls=hkls,
        n_pix=(params.n_pix_x, params.n_pix_y),
        px_size=(params.px_x, params.px_y),
        psf_sigma=params.psf_sigma,
        rotation="matrix",
        detector_rotation="rodrigues",
        strain_mode="none",
        hard=False,
    )
    t = params.to_tensors(dtype=DTYPE)
    U_truth = _toy_orientation()
    scan_offsets = torch.linspace(-36.0, 36.0, 8, dtype=DTYPE)
    with torch.no_grad():
        frames = model.forward_stack(
            U_truth.unsqueeze(0), t["lattice"], t["P"], t["R"],
            coded_aperture=mask,
            scan_offsets_um=scan_offsets,
            source_xyz=torch.tensor([0.0, 0.0, z_um * 1e-6], dtype=DTYPE),
            E_range=(params.E_lo, params.E_hi),
        ).detach()
    return CodedApertureVoxelMeasurement(
        voxel_index=vi,
        frame_stack=frames,
        scan_offsets_um=scan_offsets,
        U_seed=U_truth,
        z_seed_um=0.0,
        metadata={"source_file": "synthetic"},
    )


def test_h5_roundtrip(tmp_path: Path):
    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num,
        lattice_nm=params.lattice,
        E_hi_keV=params.E_hi,
    )
    mask = _toy_mask()

    # Synthesise 3 voxels and write them to a scratch directory.
    voxels = [_synth_voxel(vi, z, mask, params, hkls)
              for vi, z in enumerate([-2.0, 0.0, 2.0])]
    scan_dir = tmp_path / "scan"
    for v in voxels:
        save_voxel_h5(v, mask, scan_dir / f"voxel_{v.voxel_index:03d}.h5")

    # ── round-trip: loader vs. original ──
    loader = CodedApertureScanLoader(scan_dir, dtype=DTYPE)
    assert len(loader) == 3
    loaded = list(loader)
    assert len(loaded) == 3
    for orig, got in zip(voxels, loaded):
        assert got.voxel_index == orig.voxel_index
        assert torch.equal(got.frame_stack, orig.frame_stack), "frame_stack drift"
        assert torch.equal(got.scan_offsets_um, orig.scan_offsets_um)
        assert torch.equal(got.U_seed, orig.U_seed)
        assert got.z_seed_um == orig.z_seed_um
        assert "source_file" in got.metadata

    # ── mask round-trip ──
    loaded_mask = loader.load_mask()
    assert torch.equal(loaded_mask.sequence, mask.sequence)
    assert torch.equal(loaded_mask.bar_widths_um, mask.bar_widths_um)
    assert torch.equal(loaded_mask.position_um, mask.position_um)
    assert torch.equal(loaded_mask.rotvec, mask.rotvec)
    assert loaded_mask.au_thickness_um.item() == mask.au_thickness_um.item()
    assert loaded_mask.sub_thickness_um.item() == mask.sub_thickness_um.item()
    assert loaded_mask.edge_softness_um == mask.edge_softness_um


def test_loader_plugs_into_refiner(tmp_path: Path):
    """Loaded voxels flow straight into the existing depth-resolved refiner.

    Don't assert tight recovery — that's Phase 2's job.  Here we only
    check that one Adam step runs without crashing and that the result
    dataclass has the expected fields.
    """
    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num,
        lattice_nm=params.lattice,
        E_hi_keV=params.E_hi,
    )
    mask = _toy_mask()

    voxels = [_synth_voxel(0, 0.0, mask, params, hkls)]
    scan_dir = tmp_path / "scan"
    save_voxel_h5(voxels[0], mask, scan_dir / "voxel_000.h5")

    loader = CodedApertureScanLoader(scan_dir, dtype=DTYPE)
    refiner = DepthResolvedVoxelRefiner(
        params=params,
        mask=loader.load_mask(),
        hkls=hkls,
        n_steps=3,
        lr_z=1.0,
        lr_rot=1e-3,
    )
    voxel = next(iter(loader))
    result = refiner.refine(voxel)

    assert result.voxel_index == 0
    assert result.n_steps == 3
    assert isinstance(result.final_loss, float)
    assert result.U_refined.shape == (3, 3)
    assert isinstance(result.z_um, float)
    assert isinstance(result.dt_s, float) and result.dt_s >= 0.0


def test_loader_supports_custom_layout(tmp_path: Path):
    """Custom dataset paths via the ``layout`` argument."""
    import h5py

    custom_layout = {
        "frames": "/myscan/raw_stack",
        "scan_offsets_um": "/myscan/p_um",
        "U_seed": "/myscan/u_seed",
        "z_seed_um": "/myscan/z_seed",
        "mask_group": "/myscan/mask",
    }

    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num,
        lattice_nm=params.lattice,
        E_hi_keV=params.E_hi,
    )
    mask = _toy_mask()
    voxel = _synth_voxel(0, 0.0, mask, params, hkls)
    path = tmp_path / "voxel_000.h5"
    save_voxel_h5(voxel, mask, path, layout=custom_layout)

    # Confirm the custom paths are what got written.
    with h5py.File(path, "r") as hf:
        assert custom_layout["frames"] in hf
        assert custom_layout["mask_group"] in hf

    loader = CodedApertureScanLoader(tmp_path, layout=custom_layout, dtype=DTYPE)
    out = next(iter(loader))
    assert torch.equal(out.frame_stack, voxel.frame_stack)
    assert torch.equal(out.U_seed, voxel.U_seed)
