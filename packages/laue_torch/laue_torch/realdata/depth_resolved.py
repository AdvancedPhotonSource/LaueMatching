"""Phase 2: depth-resolved single-voxel refiner for coded-aperture data.

Given a frame stack rendered through a coded-aperture mask and a seed
orientation, recover the per-voxel depth ``z`` along the beam axis plus
a refined orientation.

This is the differentiable analogue of the two-stage solver from
Gürsoy *et al.* (*J. Appl. Cryst.* **55**, 2022): instead of (a) finding
the coded-aperture position ``p`` by exhaustive search, (b) solving for
the signal footprint ``s`` by NNLS, and (c) ray-tracing ``(p, s)`` back
to a depth, we minimise

.. math::
   L(z, U) = \\sum_{m=1}^{M} \\| f(U, z, p_m) - d_m \\|_2^2

in one joint Adam loop, with the differentiable ``LaueForwardModel`` +
:class:`CodedApertureMask` (Phase 1) providing the rendering map
:math:`f` and ``torch.autograd`` providing the gradients.

The voxel-level strain and the SO(3) tangent-Gaussian spread (``σ_U``
of the existing :class:`VoxelODFRefiner`) are deferred to later phases
— Phase 2 deliberately scopes to the simplest end-to-end depth
recovery so the loss landscape and convergence properties can be
characterised cleanly.

See ``laue_torch/implementation_plan_coded_aperture.md`` §2 Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import time

import torch
from torch import Tensor, nn

from midas_stress.orientation import (
    axis_angle_to_orient_mat,
    quat_to_orient_mat,
)

from ..coded_aperture import CodedApertureMask
from ..forward import LaueForwardModel
from ..io import LaueParams
from ..uncertainty import LaplacePosterior, laplace_posterior


def _quat_to_rotmat(q: Tensor) -> Tensor:
    """Unit-normalised quaternion → (3, 3) rotation matrix.

    Canonical primitive: ``midas_stress.orientation.quat_to_orient_mat``
    (see memory ``feedback_orientation_from_midas_stress``).  It expects
    unit quaternions and returns the flat 9-element row-major form
    (memory ``feedback_midas_stress_miso_radians``); we normalise the
    quaternion and reshape here so the refinement code can pass an
    unconstrained 4-vector parameter.
    """
    q = q / torch.linalg.norm(q).clamp_min(1e-30)
    return quat_to_orient_mat(q).reshape(3, 3)


@dataclass
class CodedApertureVoxelMeasurement:
    """One voxel's coded-aperture frame stack + indexer seed.

    Attributes
    ----------
    voxel_index
        Index into a scan.  Used downstream for plotting / aggregation.
    frame_stack
        ``(M, Nx, Ny)`` tensor — the M observed images, one per coded-
        aperture scan position.
    scan_offsets_um
        ``(M,)`` tensor — the mask scan positions (in µm) matching
        ``frame_stack``.
    U_seed
        ``(3, 3)`` rotation matrix — initial orientation.  Typically
        from a single-frame Laue index of the unmasked sum-image, or
        from a prior voxel in a scan.
    z_seed_um
        Initial depth guess [µm] along the beam axis.  Default 0
        (sample-center plane).
    metadata
        Arbitrary auxiliary info propagated to the result.
    """

    voxel_index: int
    frame_stack: Tensor
    scan_offsets_um: Tensor
    U_seed: Tensor
    z_seed_um: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class DepthResolvedVoxelResult:
    voxel_index: int
    U_refined: Tensor               # (3, 3) refined orientation
    z_um: float                     # refined depth (along beam, µm)
    z_init_um: float
    final_loss: float
    initial_loss: float
    n_steps: int
    dt_s: float
    # Per-voxel deviatoric strain (5-vector) or full Voigt strain (6-vector);
    # ``None`` when strain was not refined.  Components follow the
    # ``laue_torch.geometry.{voigt_to_symmetric, deviatoric5_to_symmetric}``
    # conventions.
    strain: Optional[Tensor] = None
    strain_mode: str = "none"       # "none" | "voigt" | "deviatoric"
    metadata: dict = field(default_factory=dict)


@dataclass
class DepthResolvedVoxelPosterior:
    """Laplace posterior at the converged ``(z, U, ε)``.

    ``z_sigma_um`` and ``rot_sigma_deg`` are marginal 1-σ widths in
    native physical units.  ``cov`` is the full posterior covariance
    of the *tangent-space* parameter vector
    ``[z_perturb, ω_x, ω_y, ω_z, ε_1, ..., ε_n]``; consult that
    matrix for parameter correlations.
    """

    voxel_index: int
    z_sigma_um: float
    rot_sigma_deg: tuple[float, float, float]
    strain_sigma: Optional[Tensor]      # (n_strain,) or ``None``
    strain_mode: str
    cov: Tensor                         # (n, n) symmetric posterior cov
    eigvals: Tensor                     # ascending
    cond_number: float
    rank_eff: int
    noise_variance: float


class DepthResolvedVoxelRefiner:
    """Per-voxel joint ``(z, U)`` refinement on a coded-aperture frame stack.

    Parameters
    ----------
    params
        Parsed :class:`LaueParams`; supplies the detector pose and
        lattice for the underlying ``LaueForwardModel``.
    mask
        The (already-calibrated) coded-aperture mask.  Frozen for this
        phase — joint mask refinement is Phase 3 ("autofocusing").
    hkls
        ``(H, 3)`` integer reflection list, e.g. from
        :func:`laue_torch.io.generate_hkls`.
    n_steps
        Adam iterations.
    lr_z
        Learning rate for ``z`` [µm units].  Coded-aperture depth has
        a much smaller natural scale than orientation, so we use two
        param groups with separate learning rates.
    lr_rot
        Learning rate for the orientation perturbation.
    psf_sigma
        Geometric PSF in pixels; default reads from ``params.psf_sigma``.
    mask_edge_softness_um
        Optional override of the mask's edge softness during refinement
        (the calibration-time value can be too tight for early Adam
        steps — wider edges = wider basin of attraction).  If ``None``,
        the mask's own value is used unchanged.
    strain_mode
        ``"none"`` (default), ``"voigt"`` (6-vector, full strain), or
        ``"deviatoric"`` (5-vector, trace-free).  Polychromatic Laue is
        formally insensitive to the hydrostatic component of strain (it
        is degenerate with lattice scale), so ``"deviatoric"`` is the
        recommended mode for real samples.  Picking ``"none"`` keeps
        ``DepthResolvedVoxelRefiner`` bit-identical to the pre-strain
        (Phase 2) implementation.
    lr_strain
        Learning rate for the strain perturbation.  Strain components
        are dimensionless ~10⁻³, so a learning rate of ``5e-5`` gives
        the same effective step size as ``lr_rot`` for orientation.
    refine_strain
        Convenience switch: when ``False`` (default) the strain
        component is held at zero even if ``strain_mode`` is non-``none``.
        Set to ``True`` for joint ``(z, U, ε)`` refinement.
    """

    def __init__(
        self,
        params: LaueParams,
        *,
        mask: CodedApertureMask,
        hkls: Tensor,
        n_steps: int = 200,
        lr_z: float = 2.0,
        lr_rot: float = 1e-3,
        lr_strain: float = 5.0e-5,
        psf_sigma: Optional[float] = None,
        mask_edge_softness_um: Optional[float] = None,
        E_range: Optional[tuple[float, float]] = None,
        strain_mode: str = "none",
        refine_strain: bool = False,
    ):
        if strain_mode not in ("none", "voigt", "deviatoric"):
            raise ValueError(
                f"strain_mode must be one of 'none' | 'voigt' | 'deviatoric'; "
                f"got {strain_mode!r}"
            )
        if refine_strain and strain_mode == "none":
            raise ValueError(
                "refine_strain=True requires strain_mode in {'voigt','deviatoric'}"
            )

        self.params = params
        self.mask = mask
        self.hkls = hkls
        self.n_steps = int(n_steps)
        self.lr_z = float(lr_z)
        self.lr_rot = float(lr_rot)
        self.lr_strain = float(lr_strain)
        self.strain_mode = strain_mode
        self.refine_strain = bool(refine_strain)
        self.E_range = E_range or (params.E_lo, params.E_hi)

        sigma = psf_sigma if psf_sigma is not None else params.psf_sigma
        self.model = LaueForwardModel(
            hkls=hkls,
            n_pix=(params.n_pix_x, params.n_pix_y),
            px_size=(params.px_x, params.px_y),
            psf_sigma=sigma,
            rotation="matrix",
            detector_rotation="rodrigues",
            strain_mode=strain_mode,
            hard=False,
        )

        if mask_edge_softness_um is not None:
            self.mask.edge_softness_um = float(mask_edge_softness_um)

    # ── core refinement ───────────────────────────────────────────────────

    def refine(self, voxel: CodedApertureVoxelMeasurement) -> DepthResolvedVoxelResult:
        t = self.params.to_tensors()
        dtype = voxel.frame_stack.dtype
        device = voxel.frame_stack.device

        lat = t["lattice"].to(dtype=dtype, device=device)
        P = t["P"].to(dtype=dtype, device=device)
        R = t["R"].to(dtype=dtype, device=device)

        U_seed = voxel.U_seed.to(dtype=dtype, device=device)
        # Parameterise the orientation perturbation as a quaternion δ_q
        # that multiplies the seed: ``U(δ_q) = quat_to_matrix(δ_q) · U_seed``.
        # ``quat_to_matrix`` is smooth at the identity quaternion (1,0,0,0),
        # unlike the Rodrigues parameterisation which has a ``torch.where``
        # branch at the zero rotvec that detaches the gradient (see memory
        # ``project_aa_grad_at_zero``).  Quaternion is unit-normalised inside
        # ``quat_to_matrix``, so we can leave it unconstrained during Adam.
        delta_quat = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        )
        z_um = nn.Parameter(
            torch.tensor(float(voxel.z_seed_um), dtype=dtype, device=device)
        )

        # Strain parameter (None when strain_mode='none', else zero-init).
        # The forward model interprets the shape based on its own
        # ``strain_mode`` attribute: 6-vec (voigt) or 5-vec (deviatoric).
        n_strain = 0
        strain: Optional[Tensor] = None
        if self.strain_mode == "voigt":
            n_strain = 6
        elif self.strain_mode == "deviatoric":
            n_strain = 5
        if n_strain > 0:
            strain_init = torch.zeros(n_strain, dtype=dtype, device=device)
            if self.refine_strain:
                strain = nn.Parameter(strain_init)
            else:
                strain = strain_init  # constant zero — not optimised

        opt_groups = [
            {"params": [z_um], "lr": self.lr_z},
            {"params": [delta_quat], "lr": self.lr_rot},
        ]
        if isinstance(strain, nn.Parameter):
            opt_groups.append({"params": [strain], "lr": self.lr_strain})
        opt = torch.optim.Adam(opt_groups)

        scan_offsets = voxel.scan_offsets_um.to(dtype=dtype, device=device)
        target = voxel.frame_stack.to(dtype=dtype, device=device)

        def _loss() -> Tensor:
            dR = _quat_to_rotmat(delta_quat)
            U = (dR @ U_seed).unsqueeze(0)
            # source position in lab frame [m] — beam axis is +z.
            src_xyz = torch.stack([
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                z_um * 1.0e-6,
            ])
            strain_arg = (strain.unsqueeze(0) if strain is not None and n_strain > 0
                          else None)
            pred = self.model.forward_stack(
                U, lat, P, R,
                strain=strain_arg,
                coded_aperture=self.mask,
                scan_offsets_um=scan_offsets,
                source_xyz=src_xyz,
                E_range=self.E_range,
            )
            return (pred - target).pow(2).mean()

        t0 = time.perf_counter()
        with torch.no_grad():
            initial_loss = float(_loss().item())

        for _step in range(self.n_steps):
            opt.zero_grad()
            L = _loss()
            L.backward()
            opt.step()

        with torch.no_grad():
            final_loss = float(_loss().item())
            U_refined = (_quat_to_rotmat(delta_quat) @ U_seed).detach()
            z_final = float(z_um.detach().item())
            strain_final = strain.detach().clone() if strain is not None else None

        return DepthResolvedVoxelResult(
            voxel_index=voxel.voxel_index,
            U_refined=U_refined,
            z_um=z_final,
            z_init_um=float(voxel.z_seed_um),
            final_loss=final_loss,
            initial_loss=initial_loss,
            n_steps=self.n_steps,
            dt_s=time.perf_counter() - t0,
            strain=strain_final,
            strain_mode=self.strain_mode,
            metadata=dict(voxel.metadata),
        )

    # ── Laplace posterior (Phase 7 add) ───────────────────────────────────

    def posterior(
        self,
        voxel: "CodedApertureVoxelMeasurement",
        result: "DepthResolvedVoxelResult",
        *,
        noise_variance: Optional[float] = None,
    ) -> "DepthResolvedVoxelPosterior":
        """Laplace approximation at the converged ``(z, U, ε)``.

        Builds a closure mapping the *tangent-space* parameters
        ``(z [µm], ω₃ [rad axis-angle around the refined U], ε)`` to
        the MSE loss, computes the Hessian via autograd at the
        converged state (tangent vector = 0), and inverts it to a
        Gaussian posterior covariance via the canonical
        :func:`laue_torch.uncertainty.laplace_posterior`.

        Parameters
        ----------
        voxel
            Same measurement object the refiner converged on.
        result
            The :class:`DepthResolvedVoxelResult` returned by
            :meth:`refine`.
        noise_variance
            Plug-in pixel-noise variance (``σ_pixel²``).  ``None``
            (default) uses the empirical Bayes estimate
            ``noise_variance = result.final_loss`` --- the MSE at
            convergence.  Pass an explicit value when you have a
            calibrated detector-noise estimate.

        Returns
        -------
        :class:`DepthResolvedVoxelPosterior` with per-parameter
        marginal standard deviations (``z`` in µm, rotation in deg,
        strain in dimensionless strain units) and the full posterior
        covariance for diagnostic plotting.
        """
        dtype = voxel.frame_stack.dtype
        device = voxel.frame_stack.device

        t = self.params.to_tensors(dtype=dtype, device=str(device))
        lat = t["lattice"]
        P = t["P"]
        R = t["R"]

        U_conv = result.U_refined.to(dtype=dtype, device=device)
        z_conv = float(result.z_um)
        eps_conv: Optional[Tensor]
        if result.strain is not None and result.strain_mode != "none":
            eps_conv = result.strain.to(dtype=dtype, device=device)
        else:
            eps_conv = None

        scan_offsets = voxel.scan_offsets_um.to(dtype=dtype, device=device)
        target = voxel.frame_stack.to(dtype=dtype, device=device)

        n_strain = 0
        if self.strain_mode == "voigt":
            n_strain = 6
        elif self.strain_mode == "deviatoric":
            n_strain = 5

        # Flat parameter layout:
        #   theta[0]        : Δz_um
        #   theta[1:4]      : quaternion *imaginary* part v_xyz around the
        #                     converged U (with w = √(1 − |v|²); always
        #                     unit-norm, no gauge eigenvalue)
        #   theta[4:4+n_e]  : Δε (strain perturbation)
        # Reporting note: a small ‖v‖ in this parametrisation maps to
        # a rotation by ≈ 2·‖v‖ radians, so we convert ``sigma_v`` to
        # rotation-angle σ via that factor below.
        n_total = 1 + 3 + n_strain

        def _build_loss(theta: Tensor) -> Tensor:
            z_perturb = theta[0]
            v = theta[1:4]
            # Unit quaternion (w, x, y, z) with w = √(1 − |v|²).  Smooth
            # at v = 0; the rotation matrix follows via the canonical
            # midas_stress.quat_to_orient_mat path.
            v_norm_sq = (v * v).sum().clamp_max(1.0 - 1.0e-12)
            w = torch.sqrt(1.0 - v_norm_sq)
            q = torch.stack([w, v[0], v[1], v[2]])
            dR = quat_to_orient_mat(q).reshape(3, 3)
            U = (dR @ U_conv).unsqueeze(0)
            z_total_um = z_conv + z_perturb
            src_xyz = torch.stack([
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                z_total_um * 1.0e-6,
            ])
            if n_strain > 0:
                eps_perturb = theta[4:4 + n_strain]
                strain_total = (eps_conv if eps_conv is not None
                                else torch.zeros(n_strain, dtype=dtype, device=device))
                strain_total = strain_total + eps_perturb
                strain_arg = strain_total.unsqueeze(0)
            else:
                strain_arg = None
            pred = self.model.forward_stack(
                U, lat, P, R,
                strain=strain_arg,
                coded_aperture=self.mask,
                scan_offsets_um=scan_offsets,
                source_xyz=src_xyz,
                E_range=self.E_range,
            )
            return (pred - target).pow(2).mean()

        theta_at_conv = torch.zeros(n_total, dtype=dtype, device=device)
        nv = float(noise_variance) if noise_variance is not None else max(
            result.final_loss, 1.0e-30,
        )
        # Compute the Hessian directly and invert with SVD-based pinv
        # — the canonical :func:`laplace_posterior` calls
        # ``torch.linalg.eigvalsh`` which fails on near-singular Hessians
        # (the position-rotation degeneracy we document in §3 of the
        # paper produces near-zero eigenvalues that trip the LAPACK
        # divide-and-conquer eigensolver).  SVD handles those gracefully.
        H_loss = torch.autograd.functional.hessian(
            _build_loss, theta_at_conv.clone(), vectorize=False,
        )
        H_sym = 0.5 * (H_loss + H_loss.T) / nv

        # Tikhonov ridge: typical Hessians of this problem have
        # near-zero eigenvalues (the position-orientation degeneracy
        # we quantify in §3) that crash both ``eigvalsh`` and ``svd``.
        # Add a ridge proportional to the diagonal scale so
        # well-conditioned directions are unaffected.
        diag_scale = float(H_sym.diag().abs().max().item())
        ridge = max(diag_scale * 1.0e-8, 1.0e-30)
        H_sym = H_sym + ridge * torch.eye(n_total, dtype=dtype, device=device)

        # SVD-based eigenanalysis + pseudoinverse.  ``torch.linalg.svd``
        # uses a Jacobi-style backend on small problems and converges
        # where eigvalsh's divide-and-conquer does not.
        H_for_svd = H_sym.to(torch.float64)
        try:
            Uvec, S, Vh = torch.linalg.svd(H_for_svd, full_matrices=False)
        except torch._C._LinAlgError:
            # Last-resort: numpy backend uses a different LAPACK path
            # that sometimes converges where torch's does not.
            import numpy as np
            Un, Sn, Vhn = np.linalg.svd(H_for_svd.detach().cpu().numpy(),
                                         full_matrices=False)
            Uvec = torch.from_numpy(Un).to(H_for_svd)
            S = torch.from_numpy(Sn).to(H_for_svd)
            Vh = torch.from_numpy(Vhn).to(H_for_svd)
        # Symmetric H ⇒ singular values = absolute eigenvalues.  Recover
        # signed eigenvalues from the U/V projection sign.
        # For numerical PSD enforcement (the Hessian *should* be PSD at
        # the MAP up to optimizer slack), we clamp negative values to 0.
        signs = torch.sign((Uvec * Vh.T).sum(dim=0))
        eigvals_signed = (S * signs).to(dtype)
        # Pseudoinverse via SVD: drop singular values below a relative tol.
        s_floor = float(S.max().item()) * 1.0e-9
        S_inv = torch.where(S > s_floor, 1.0 / S, torch.zeros_like(S))
        cov = (Vh.T @ torch.diag(S_inv) @ Uvec.T).to(dtype)
        cov = 0.5 * (cov + cov.T)
        sigma = cov.diag().clamp_min(0.0).sqrt()

        rank_eff = int((S > s_floor).sum().item())
        eigvals_sorted, _ = torch.sort(eigvals_signed)
        max_e = float(eigvals_sorted.max().item())
        min_e = max(float(eigvals_sorted.min().item()), s_floor)
        cond_number = float(max_e / min_e) if min_e > 0 else float("inf")

        lap = LaplacePosterior(
            theta=theta_at_conv.detach(),
            hessian=H_sym.detach(),
            cov=cov,
            sigma=sigma,
            eigvals=eigvals_sorted,
            cond_number=cond_number,
            rank_eff=rank_eff,
        )

        # Project to physical units.  Around v = 0, the unit quaternion
        # (w, v) with w = √(1 − |v|²) implements a rotation by
        # 2·arcsin(‖v‖) ≈ 2·‖v‖ rad for small ‖v‖.  Per-axis σ on v
        # therefore maps to per-axis rotation σ by a factor of 2.
        sigma = lap.sigma.detach().cpu()
        z_sigma_um = float(sigma[0].item())
        rot_sigma_rad = 2.0 * sigma[1:4]
        rot_sigma_deg = tuple(float(math.degrees(s.item())) for s in rot_sigma_rad)
        if n_strain > 0:
            strain_sigma = sigma[4:4 + n_strain].clone()
        else:
            strain_sigma = None

        return DepthResolvedVoxelPosterior(
            voxel_index=result.voxel_index,
            z_sigma_um=z_sigma_um,
            rot_sigma_deg=rot_sigma_deg,
            strain_sigma=strain_sigma,
            strain_mode=self.strain_mode,
            cov=lap.cov.detach().cpu(),
            eigvals=lap.eigvals.detach().cpu(),
            cond_number=lap.cond_number,
            rank_eff=lap.rank_eff,
            noise_variance=nv,
        )
