"""Detector axis order across every laue_torch entry point (handbook invariant 38).

The forward model renders ``img[X, Y]`` (X = detector column); a real frame is
``image[row, col]`` = ``[Y, X]``. On a square detector a missing or doubled
transpose raises nothing and fits every spot against the wrong pixel -- a
synthetic control built that way scored 0-2 truth hits instead of 17-61.

Every test here uses a NON-SQUARE detector (Nx = 96 columns, Ny = 64 rows), so
a layout mistake changes the array shape and cannot pass by symmetry. The
pinned behaviour: a spot at detector row r, column c of a real-layout frame
lines up with a prediction at X = c, Y = r.
"""
from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.io import LaueParams, generate_hkls, to_model_layout

NX, NY = 96, 64          # columns, rows
DT = torch.float64


def _params() -> LaueParams:
    return LaueParams(
        sg_num=225, symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.006, px_y=0.006, n_pix_x=NX, n_pix_y=NY,
        E_lo=5.0, E_hi=30.0, psf_sigma=1.0,
    )


def _U(q=(0.56153266089081, -0.1069242896544219,
          -0.7939419137346801, 0.2071340258144413)) -> torch.Tensor:
    q = torch.tensor(q, dtype=DT)
    return quat_to_orient_mat(q / q.norm()).reshape(3, 3)


def _model(p: LaueParams, hkls: torch.Tensor) -> LaueForwardModel:
    t = p.to_tensors()
    return LaueForwardModel(
        hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"], psf_sigma=p.psf_sigma,
        rotation="matrix", detector_rotation="rodrigues", strain_mode="voigt",
        hard=False, reduce="sum")


def _render(p: LaueParams, U: torch.Tensor):
    """Model-layout render [X, Y] plus the per-reflection aux."""
    hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
    t = p.to_tensors()
    img, aux = _model(p, hkls)(U.unsqueeze(0), t["lattice"], t["P"], t["R"],
                               strain=torch.zeros(1, 6, dtype=DT),
                               return_aux=True)
    return img.detach(), aux


# ── the convention itself ──────────────────────────────────────────────────

def test_real_frame_spot_at_row_r_col_c_is_prediction_at_X_c_Y_r():
    p = _params()
    img_xy, aux = _render(p, _U())
    assert img_xy.shape == (NX, NY)
    real = img_xy.T.contiguous()                       # what a detector records
    assert real.shape == (NY, NX)
    row, col = np.unravel_index(int(real.argmax()), real.shape)
    on = aux.mask > 0.5
    px = aux.px[on].round().long().tolist()
    py = aux.py[on].round().long().tolist()
    assert (col, row) in set(zip(px, py)), \
        "brightest pixel of the real-layout frame is not at any predicted (X, Y) = (col, row)"
    back = to_model_layout(real, "YX", (NX, NY))
    assert torch.equal(back, img_xy)


def test_to_model_layout_rejects_a_wrongly_declared_layout():
    real = torch.zeros(NY, NX, dtype=DT)
    with pytest.raises(ValueError, match="axis_order"):
        to_model_layout(real, "XY", (NX, NY))
    with pytest.raises(ValueError, match="axis_order"):
        to_model_layout(real, "row-col", (NX, NY))


# ── cli.main ───────────────────────────────────────────────────────────────

def test_cli_writes_axis_order_XY(tmp_path):
    from laue_torch.cli import main
    cfg = tmp_path / "params.txt"
    cfg.write_text(
        "LatticeParameter 0.35238 0.35238 0.35238 90 90 90\n"
        "SpaceGroup 225\n"
        "Symmetry F\n"
        "P_Array 0.028745 0.002788 0.513115\n"
        "R_Array -1.20131258 -1.21399082 -1.21881158\n"
        f"PxX 0.006\nPxY 0.006\nNrPxX {NX}\nNrPxY {NY}\n"
        "Elo 5\nEhi 30\nSimulationSmoothingWidth 1\n")
    ori = tmp_path / "orient.txt"
    np.savetxt(ori, _U().numpy().reshape(1, 9))
    out = tmp_path / "sim.h5"
    assert main(["-configFile", str(cfg), "-orientationFile", str(ori),
                 "-outputFile", str(out)]) == 0
    with h5py.File(out, "r") as hf:
        assert hf["/entry1/axis_order"][()] == b"XY"
        assert hf["/entry1/data/data"].shape == (NX, NY)


# ── LaueScanLoader -> VoxelODFRefiner ──────────────────────────────────────

def _write_output_h5(path, real_frame: np.ndarray, U: np.ndarray, n_cols: int = 35):
    """A laue_postprocess-style .output.h5: junk everywhere except the matrix."""
    om = 23 if n_cols == 35 else 22
    row = np.full(n_cols, 777.0)
    row[om:om + 9] = U.reshape(9)
    with h5py.File(path, "w") as hf:
        g = hf.create_group("/entry/results")
        g.create_dataset("orientations", data=row[None, :])
        g.create_dataset("filtered_orientations", data=row[None, :])
        g.attrs["image_nr"] = 1
        hf.create_dataset("/entry/data/input_blurred", data=real_frame)


def test_scan_loader_and_voxel_refiner_line_up_on_non_square(tmp_path):
    from laue_torch.realdata import LaueScanLoader, VoxelODFRefiner
    p = _params()
    U = _U()
    refiner = VoxelODFRefiner(p, sigma_init_deg=1e-4, psf_sigma=p.psf_sigma,
                              n_steps=1, M_render=4, compute_posterior=False)
    t = refiner.tensors
    with torch.no_grad():
        img_xy = refiner.model(U.unsqueeze(0), t["lattice"], t["P"], t["R"],
                               strain=torch.zeros(1, 6, dtype=DT))
    real = img_xy.T.numpy().copy()                        # (NY, NX), as stored
    _write_output_h5(tmp_path / "image_00001.output.h5", real, U.numpy())

    (voxel,) = list(LaueScanLoader(tmp_path))
    assert voxel.axis_order == "YX"
    assert tuple(voxel.image.shape) == (NY, NX)          # returned as stored
    assert torch.allclose(voxel.U_seed_list[0], U)

    res = refiner.refine(voxel)
    # n_steps=1: final_loss is the loss at the seed. Aligned, the tiny-spread
    # render reproduces the frame; transposed it could not even be subtracted.
    # Square 80x80 copy of this geometry: aligned 1.4e-8 * signal, transposed 1.9 * signal.
    signal = float((img_xy ** 2).mean())
    assert res.final_loss < 1e-3 * signal, (res.final_loss, signal)


def test_voxel_refiner_rejects_real_frame_declared_model_layout(tmp_path):
    from laue_torch.realdata import VoxelMeasurement, VoxelODFRefiner
    p = _params()
    refiner = VoxelODFRefiner(p, n_steps=1, M_render=2, compute_posterior=False)
    m = VoxelMeasurement(voxel_index=0, image=torch.zeros(NY, NX, dtype=DT),
                         U_seed_list=_U().unsqueeze(0), metadata={},
                         axis_order="XY")
    with pytest.raises(ValueError, match="axis_order"):
        refiner.refine(m)


def test_voxel_measurement_without_axis_order_is_refused():
    """VoxelODFRefiner did not transpose in 0.1.3, so a hand-built
    VoxelMeasurement holding a laue_torch render would be silently transposed
    on a square detector by any default. There is none: None raises."""
    from laue_torch.realdata import VoxelMeasurement, VoxelODFRefiner
    p = dataclasses.replace(_params(), n_pix_x=NY, n_pix_y=NY)      # square
    refiner = VoxelODFRefiner(p, n_steps=1, M_render=2, compute_posterior=False)
    m = VoxelMeasurement(voxel_index=0, image=torch.zeros(NY, NY, dtype=DT),
                         U_seed_list=_U().unsqueeze(0), metadata={})
    assert m.axis_order is None
    with pytest.raises(ValueError, match="axis_order is None"):
        refiner.refine(m)


def test_scan_loader_honours_axis_order_marker(tmp_path):
    from laue_torch.realdata import LaueScanLoader
    _write_output_h5(tmp_path / "a.output.h5", np.zeros((NX, NY)), _U().numpy())
    with h5py.File(tmp_path / "a.output.h5", "a") as hf:
        hf.create_dataset("/entry/axis_order", data=np.bytes_("XY"))
    (voxel,) = list(LaueScanLoader(tmp_path))
    assert voxel.axis_order == "XY"


# ── MultiGrainVoxelRefiner ─────────────────────────────────────────────────

def test_multi_grain_refine_lines_up_on_non_square():
    from laue_torch.realdata import MultiGrainVoxelRefiner
    p = _params()
    U = _U()
    ref = MultiGrainVoxelRefiner(p, sigma_init_deg=1e-4, n_steps=1, M_render=4)
    t = ref.tensors
    with torch.no_grad():
        img_xy = ref.model(U.unsqueeze(0), t["lattice"], t["P"], t["R"],
                           strain=torch.zeros(1, 6, dtype=DT))
    res = ref.refine(img_xy.T.contiguous(), U.unsqueeze(0), axis_order="YX")
    # Measured on a square 80x80 copy of this geometry (where a wrong transpose
    # cannot fail on shape): aligned 1.1e-4 * peak^2, transposed 3.1e-3 * peak^2.
    peak_sq = float(img_xy.max() ** 2)
    assert res.final_loss < 1e-3 * peak_sq, (res.final_loss, peak_sq)
    with pytest.raises(ValueError, match="axis_order"):
        ref.refine(img_xy.T.contiguous(), U.unsqueeze(0), axis_order="XY")
    # No default: an undeclared layout raises even where the shape would fit.
    with pytest.raises(ValueError, match="axis_order is None"):
        ref.refine(img_xy.T.contiguous(), U.unsqueeze(0))
    with pytest.raises(ValueError, match="axis_order is None"):
        ref.refine(img_xy, U.unsqueeze(0))


# ── ReferenceGrainParallaxRefiner ──────────────────────────────────────────

def test_reference_grain_accepts_declared_real_layout():
    from laue_torch.realdata import (ReferenceGrainParallaxRefiner,
                                     TwoSourceMeasurement)
    p = _params()
    hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
    U = _U()
    img_xy, _ = _render(p, U)
    ref = ReferenceGrainParallaxRefiner(p, hkls=hkls, n_steps=0)
    common = dict(U_reference=U, z_reference_um=0.0, U_sample_seed=U)
    r_xy = ref.refine(TwoSourceMeasurement(image=img_xy, **common))
    r_yx = ref.refine(TwoSourceMeasurement(image=img_xy.T.contiguous(),
                                           axis_order="YX", **common))
    assert r_xy.initial_loss == r_yx.initial_loss
    with pytest.raises(ValueError, match="axis_order"):
        ref.refine(TwoSourceMeasurement(image=img_xy.T.contiguous(), **common))


# ── coded-aperture io_h5 ───────────────────────────────────────────────────

def _ca_measurement(frames: torch.Tensor):
    from laue_torch.realdata import CodedApertureVoxelMeasurement
    return CodedApertureVoxelMeasurement(
        voxel_index=0, frame_stack=frames,
        scan_offsets_um=torch.arange(frames.shape[0], dtype=DT),
        U_seed=_U(), z_seed_um=0.0)


def _ca_mask():
    from laue_torch.coded_aperture import CodedApertureMask, build_de_bruijn_sequence
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=3, alphabet=2),
        bar_widths_um=12.0, au_thickness_um=6.0, sub_thickness_um=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DT),
        rotvec=torch.zeros(3, dtype=DT), edge_softness_um=2.0,
        make_geometry_learnable=False, dtype=DT)


def test_coded_aperture_h5_writes_and_honours_axis_order(tmp_path):
    from laue_torch.coded_aperture import load_voxel_h5, save_voxel_h5
    frames_xy = torch.rand(2, NX, NY, dtype=DT)
    path = tmp_path / "v.h5"
    save_voxel_h5(_ca_measurement(frames_xy), _ca_mask(), path)
    with h5py.File(path, "r") as hf:
        assert hf["/entry/axis_order"][()] == b"XY"
    assert torch.equal(load_voxel_h5(path).frame_stack, frames_xy)

    # A partner file in detector layout, declared as such, loads as model layout.
    with h5py.File(path, "a") as hf:
        del hf["/entry/data/frames"]
        hf.create_dataset("/entry/data/frames",
                          data=frames_xy.transpose(-1, -2).numpy())
        del hf["/entry/axis_order"]
        hf.create_dataset("/entry/axis_order", data=np.bytes_("YX"))
    assert torch.equal(load_voxel_h5(path).frame_stack, frames_xy)
