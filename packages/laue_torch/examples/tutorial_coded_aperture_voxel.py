"""End-to-end coded-aperture tutorial (Phase 5 deliverable).

Synthesises a small coded-aperture scan, writes it to a directory of
per-voxel H5 files in the default schema, then runs the full
:mod:`laue_torch.coded_aperture` + :mod:`laue_torch.realdata` pipeline
exactly as a user would on real data:

    1. ``CodedApertureScanLoader`` reads the voxel directory.
    2. ``load_mask_h5`` reads the (shared) coded-aperture mask state.
    3. ``autofocus_geometry`` refines the mask pose on a Si calibration
       scan.
    4. ``DepthResolvedVoxelRefiner`` recovers per-voxel ``(z, U)``.
    5. ``plot_depth_profile`` + ``plot_autofocus_convergence`` save
       diagnostic PNGs.

When Dina's real H5 layout arrives this script becomes the
parity-comparison entry point: swap the ``synthesise_scan`` block for
``CodedApertureScanLoader(<real-data-dir>)`` and the rest of the
pipeline is unchanged.

Runs in well under 5 minutes on CPU.  Outputs go to
``laue_torch/examples/tutorial_coded_aperture_out/``.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import (
    CodedApertureMask,
    autofocus_geometry,
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
from laue_torch.realdata.plots import (
    plot_autofocus_convergence,
    plot_depth_profile,
)


DTYPE = torch.float64
OUT = Path(__file__).resolve().parent / "tutorial_coded_aperture_out"


def synth_params() -> LaueParams:
    return LaueParams(
        sg_num=225,
        symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.0016, px_y=0.0016, n_pix_x=512, n_pix_y=512,
        E_lo=5.0, E_hi=15.0, psf_sigma=2.0,
    )


def synth_orientation() -> torch.Tensor:
    q = torch.tensor(
        [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
        dtype=DTYPE,
    )
    return quat_to_orient_mat(q).reshape(3, 3)


def synth_mask(*, position_um, rotvec) -> CodedApertureMask:
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2),
        bar_widths_um=12.0,
        au_thickness_um=6.0,
        sub_thickness_um=0.0,
        position_um=position_um,
        rotvec=rotvec,
        edge_softness_um=4.0,
        make_geometry_learnable=False,
        dtype=DTYPE,
    )


def synthesise_scan(out_dir: Path) -> dict:
    """Generate a tiny synthetic coded-aperture scan to <out_dir>/.

    Returns a metadata dict so the rest of the tutorial can compare
    refined values against the known truth.  As of the strain-enabled
    Phase 2+ refiner, the synthetic now also imposes a known
    deviatoric strain on each voxel so the tutorial exercises the
    full ``(z, U, ε)`` recovery path.
    """
    params = synth_params()
    hkls = generate_hkls(
        sg_num=params.sg_num,
        lattice_nm=params.lattice,
        E_hi_keV=params.E_hi,
    )

    # Truth pose + sample
    pos_truth = torch.tensor([5.0, 3.0, 500.0], dtype=DTYPE)
    rotvec_truth = torch.tensor([0.08, -0.04, 0.06], dtype=DTYPE)
    mask_truth = synth_mask(position_um=pos_truth, rotvec=rotvec_truth)
    U_truth = synth_orientation()

    # 4 voxels along the beam (would be "depth slices" of a thin sample).
    # Each voxel is given a shared deviatoric strain (matches the
    # common scenario of a single-phase strained crystal — strain
    # varies slowly across the probed region).
    voxel_zs_um = [-3.0, -1.0, 1.0, 3.0]
    strain_truth = torch.tensor(
        [4.0e-4, -2.0e-4, 1.5e-4, -1.0e-4, 0.8e-4],
        dtype=DTYPE,
    )
    scan_offsets_um = torch.linspace(-36.0, 36.0, 12, dtype=DTYPE)

    # Forward model with strain enabled (the refiner will also use
    # ``strain_mode="deviatoric"`` so the two pipelines match).
    model = LaueForwardModel(
        hkls=hkls,
        n_pix=(params.n_pix_x, params.n_pix_y),
        px_size=(params.px_x, params.px_y),
        psf_sigma=params.psf_sigma,
        rotation="matrix",
        detector_rotation="rodrigues",
        strain_mode="deviatoric",
        hard=False,
    )
    t = params.to_tensors(dtype=DTYPE)

    out_dir.mkdir(parents=True, exist_ok=True)
    for vi, z in enumerate(voxel_zs_um):
        with torch.no_grad():
            frame_stack = model.forward_stack(
                U_truth.unsqueeze(0), t["lattice"], t["P"], t["R"],
                strain=strain_truth.unsqueeze(0),
                coded_aperture=mask_truth,
                scan_offsets_um=scan_offsets_um,
                source_xyz=torch.tensor([0.0, 0.0, z * 1e-6], dtype=DTYPE),
                E_range=(params.E_lo, params.E_hi),
            ).detach()
        measurement = CodedApertureVoxelMeasurement(
            voxel_index=vi,
            frame_stack=frame_stack,
            scan_offsets_um=scan_offsets_um,
            U_seed=U_truth,
            z_seed_um=0.0,        # purposely off so the refiner has something to do
        )
        save_voxel_h5(measurement, mask_truth, out_dir / f"voxel_{vi:03d}.h5")

    return dict(
        params=params,
        hkls=hkls,
        U_truth=U_truth,
        pos_truth=pos_truth,
        rotvec_truth=rotvec_truth,
        voxel_zs_um=voxel_zs_um,
        strain_truth=strain_truth,
        n_voxels=len(voxel_zs_um),
    )


def run_pipeline(scan_dir: Path, truth: dict) -> None:
    params = truth["params"]
    hkls = truth["hkls"]

    # ─── Stage 1: read the scan ─────────────────────────────────────────
    loader = CodedApertureScanLoader(scan_dir, dtype=DTYPE)
    measurements = list(loader)
    print(f"[load] {len(measurements)} voxels from {scan_dir}")

    # The mask state is replicated in every voxel H5; read it from the
    # first file.  In a real deployment a separate calibration file is
    # cleaner — load it directly with ``load_mask_h5(calib_path)``.
    mask_truth_from_h5 = loader.load_mask()

    # ─── Stage 2: autofocus on the same data (Si calibration in the real
    #             world; here we just demonstrate the call shape) ──────
    pos_init = truth["pos_truth"] + torch.tensor([2.0, -1.5, 1.0], dtype=DTYPE)
    rotvec_init = truth["rotvec_truth"] + torch.tensor(
        [0.005, 0.003, -0.004], dtype=DTYPE,
    )
    mask_init = synth_mask(position_um=pos_init, rotvec=rotvec_init)

    autofocus_result = autofocus_geometry(
        measurements, mask_init,
        params=params, hkls=hkls,
        n_steps=400,
        lr_pos_um=0.5,
        lr_rot_rad=2e-3,
        refine_U=False,
    )
    print(
        f"[autofocus] loss {autofocus_result.initial_loss:.3e} → "
        f"{autofocus_result.final_loss:.3e}  in {autofocus_result.dt_s:.1f} s"
    )
    plot_autofocus_convergence(
        autofocus_result,
        pos_truth=truth["pos_truth"].tolist(),
        rotvec_truth=truth["rotvec_truth"].tolist(),
        out_path=OUT / "autofocus_convergence.png",
    )

    # ─── Stage 3: per-voxel depth + orientation + strain refinement ─────
    refiner = DepthResolvedVoxelRefiner(
        params=params,
        mask=autofocus_result.refined_mask,
        hkls=hkls,
        n_steps=300,
        lr_z=2.0,
        lr_rot=2e-3,
        lr_strain=5e-5,
        mask_edge_softness_um=4.0,
        strain_mode="deviatoric",
        refine_strain=True,
    )
    results = [refiner.refine(v) for v in measurements]
    posteriors = [refiner.posterior(v, r) for v, r in zip(measurements, results)]
    for res, post in zip(results, posteriors):
        z_truth = truth["voxel_zs_um"][res.voxel_index]
        print(
            f"[voxel {res.voxel_index}] z {res.z_init_um:+.2f} → "
            f"{res.z_um:+.3f} µm (truth {z_truth:+.2f}, "
            f"σ_post {post.z_sigma_um:.3e} µm), "
            f"loss {res.initial_loss:.3e} → {res.final_loss:.3e}"
        )
    plot_depth_profile(
        results,
        z_truth_um=truth["voxel_zs_um"],
        out_path=OUT / "depth_profile.png",
    )

    # ─── Stage 4: strain recovery summary ───────────────────────────────
    strain_truth = truth["strain_truth"].numpy()
    print("\n[strain] recovered deviatoric ε (Voigt-5) vs truth:")
    print(f"  truth = {strain_truth.tolist()}")
    for res in results:
        if res.strain is None:
            continue
        eps = res.strain.cpu().numpy()
        print(f"  voxel {res.voxel_index}: ε = {eps.tolist()}")

    _plot_strain_recovery(results, strain_truth, OUT / "strain_recovery.png")
    print(
        f"[plots] wrote {OUT}/autofocus_convergence.png + "
        f"depth_profile.png + strain_recovery.png"
    )


def _plot_strain_recovery(results, strain_truth, out_path):
    """Bar plot of recovered deviatoric strain components across voxels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_vox = len(results)
    n_comp = strain_truth.shape[0]
    eps = np.stack(
        [r.strain.cpu().numpy() for r in results if r.strain is not None],
        axis=0,
    )                                              # (n_vox, n_comp)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_comp)
    width = 0.8 / (n_vox + 1)
    ax.bar(x - 0.4, strain_truth, width=width, label="truth",
           color="black", alpha=0.7)
    for vi in range(n_vox):
        ax.bar(x - 0.4 + (vi + 1) * width, eps[vi], width=width,
               label=f"voxel {results[vi].voxel_index}")
    ax.set_xticks(x)
    ax.set_xticklabels(["e1", "e2", "e3", "e4", "e5"])
    ax.set_ylabel("deviatoric strain (Voigt-5)")
    ax.set_title("Strain recovery on synthetic 4-voxel coded-aperture scan")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scan_dir = OUT / "scan"
    if scan_dir.exists():
        shutil.rmtree(scan_dir)
    truth = synthesise_scan(scan_dir)
    run_pipeline(scan_dir, truth)


if __name__ == "__main__":
    main()
