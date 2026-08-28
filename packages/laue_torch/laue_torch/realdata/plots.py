"""Per-voxel map plotters for real-data ODF refinement output.

For 1-D scans (linear voxel sequence) the plots show line plots vs
voxel index.  For 2-D scans (a known ``(n_x, n_y)`` grid) the plots
are colour-mapped images.  The grid shape is supplied at call time;
the loader does not infer it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional

import math
import numpy as np


def _scalar_field(results, getter) -> np.ndarray:
    """Build a 1-D array of per-voxel scalars."""
    return np.array([getter(r) for r in results], dtype=float)


def _reshape(field: np.ndarray, grid_shape: Optional[Sequence[int]]):
    if grid_shape is None:
        return field, False
    arr = field.reshape(grid_shape)
    return arr, True


def plot_sigma_map(
    results,
    *,
    grid_shape: Optional[Sequence[int]] = None,
    out_path: str | Path,
    title: str = r"recovered $\sigma_U$ [deg]",
    show_posterior: bool = False,
):
    """Plot per-voxel recovered ``sigma_U`` (and optionally posterior σ)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = _scalar_field(results, lambda r: r.sigma_U_deg)
    if show_posterior:
        post = _scalar_field(results, lambda r: r.posterior_sigma_U_deg)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axs = axes
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        axs = [ax]

    arr, is_2d = _reshape(sigmas, grid_shape)
    if is_2d:
        im = axs[0].imshow(arr, origin="lower", cmap="viridis")
        plt.colorbar(im, ax=axs[0], fraction=0.046)
    else:
        axs[0].plot(np.arange(len(arr)), arr, "o-")
        axs[0].set_xlabel("voxel index")
        axs[0].set_ylabel(r"$\sigma_U$ [deg]")
    axs[0].set_title(title)

    if show_posterior:
        arr, is_2d = _reshape(post, grid_shape)
        if is_2d:
            im = axs[1].imshow(arr, origin="lower", cmap="magma")
            plt.colorbar(im, ax=axs[1], fraction=0.046)
        else:
            axs[1].plot(np.arange(len(arr)), arr, "o-")
            axs[1].set_xlabel("voxel index")
            axs[1].set_ylabel(r"$\sigma_U$ posterior $\sigma$ [deg]")
        axs[1].set_title(r"Laplace posterior $\sigma$ on $\sigma_U$")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_orientation_map(
    results,
    *,
    grid_shape: Optional[Sequence[int]] = None,
    out_path: str | Path,
    reference_index: int = 0,
    title: str = "cubic misorientation to reference voxel [deg]",
):
    """Plot per-voxel cubic misorientation to a chosen reference voxel.

    Useful for visualising intragranular orientation gradients within
    a single grain.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from ..symmetry import cubic_misorientation_deg

    U_field = torch.stack([r.U_mean for r in results], dim=0)
    U_ref = U_field[reference_index].unsqueeze(0)
    miso = cubic_misorientation_deg(U_field, U_ref).numpy()

    arr, is_2d = _reshape(miso, grid_shape)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if is_2d:
        im = ax.imshow(arr, origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.plot(np.arange(len(arr)), arr, "o-")
        ax.set_xlabel("voxel index")
        ax.set_ylabel("cubic miso to ref [deg]")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_gnd_map(
    results,
    *,
    grid_shape: Sequence[int],
    spacing_um: float | tuple[float, ...] = 1.0,
    burgers_m: float = 2.5e-10,
    out_path: str | Path,
    title: str = r"GND density $\rho_\mathrm{GND}$ [m$^{-2}$]",
):
    """Compute and plot the per-voxel GND density from the recovered
    ``U_mean`` field.

    Requires a 2-D or 3-D scan with known ``grid_shape``; the rotation
    field is reshaped, Nye's tensor is computed by central differences,
    and the Frobenius-norm GND density is plotted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from ..nye import nye_tensor, gnd_density

    U_flat = torch.stack([r.U_mean for r in results], dim=0)
    U_field = U_flat.reshape(*grid_shape, 3, 3)
    if isinstance(spacing_um, (int, float)):
        spacing = (float(spacing_um),) * len(grid_shape)
    else:
        spacing = tuple(float(s) for s in spacing_um)
    alpha = nye_tensor(U_field, spacing=spacing)
    rho = gnd_density(alpha, burgers_m=burgers_m).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if len(grid_shape) == 1:
        ax.plot(np.arange(rho.shape[0]), rho, "o-")
        ax.set_xlabel("voxel index")
        ax.set_ylabel(r"$\rho_\mathrm{GND}$ [m$^{-2}$]")
        ax.set_yscale("log")
    elif len(grid_shape) == 2:
        im = ax.imshow(rho, origin="lower", cmap="inferno",
                       norm=plt.matplotlib.colors.LogNorm())
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\rho_\mathrm{GND}$ [m$^{-2}$]")
    else:
        # 3-D: project to mid-z slice.
        mid = grid_shape[2] // 2
        im = ax.imshow(rho[..., mid], origin="lower", cmap="inferno",
                       norm=plt.matplotlib.colors.LogNorm())
        plt.colorbar(im, ax=ax, fraction=0.046,
                     label=r"$\rho_\mathrm{GND}$ [m$^{-2}$]")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Coded-aperture-specific plots (Phase 5) ────────────────────────────────


def plot_depth_profile(
    results,
    *,
    out_path: str | Path,
    z_truth_um: Optional[Sequence[float]] = None,
    title: str = "recovered depth along beam [um]",
):
    """Plot recovered per-voxel depth ``z`` from
    :class:`DepthResolvedVoxelResult` records.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_um = _scalar_field(results, lambda r: r.z_um)
    z_init = _scalar_field(results, lambda r: r.z_init_um)
    idx = np.arange(len(z_um))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(idx, z_init, "x--", color="0.7", label="seed (initial guess)")
    ax.plot(idx, z_um, "o-", color="C0", label="refined")
    if z_truth_um is not None:
        z_truth = np.asarray(z_truth_um, dtype=float)
        if z_truth.shape != z_um.shape:
            raise ValueError(
                f"z_truth_um shape {z_truth.shape} != results shape {z_um.shape}")
        ax.plot(idx, z_truth, "k--", lw=1.0, label="truth")
    ax.set_xlabel("voxel index")
    ax.set_ylabel("z [um]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_autofocus_convergence(
    autofocus_result,
    *,
    out_path: str | Path,
    pos_truth: Optional[Sequence[float]] = None,
    rotvec_truth: Optional[Sequence[float]] = None,
    title: str = "coded-aperture autofocus: pose refinement",
):
    """One-shot summary of a digital-autofocus run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos_init = autofocus_result.pose_position_init.detach().cpu().numpy()
    rot_init = autofocus_result.pose_rotvec_init.detach().cpu().numpy()
    mask = autofocus_result.refined_mask
    pos_fin = mask.position_um.detach().cpu().numpy()
    rot_fin = mask.rotvec.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels_pos = ["pos_x", "pos_y", "pos_z (heave)"]
    labels_rot = ["rotvec_x", "rotvec_y", "rotvec_z"]
    x = np.arange(3)

    for ax, init, fin, truth_vec, labels, units in [
        (axes[0], pos_init, pos_fin, pos_truth, labels_pos, "um"),
        (axes[1], rot_init, rot_fin, rotvec_truth, labels_rot, "rad"),
    ]:
        w = 0.3
        ax.bar(x - w/2, init, w, label="initial", color="0.7")
        ax.bar(x + w/2, fin, w, label="refined", color="C0")
        if truth_vec is not None:
            ax.plot(x, np.asarray(truth_vec, dtype=float),
                    "k_", markersize=22, mew=2, label="truth")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(units)
        ax.legend()
    axes[0].set_title("position")
    axes[1].set_title("rotation (rotvec)")
    fig.suptitle(
        f"{title}\nloss: {autofocus_result.initial_loss:.3e} -> "
        f"{autofocus_result.final_loss:.3e}   "
        f"({autofocus_result.n_steps} steps, {autofocus_result.dt_s:.1f} s)"
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
