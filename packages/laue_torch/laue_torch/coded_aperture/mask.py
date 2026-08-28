"""Differentiable coded-aperture mask transmission.

The :class:`CodedApertureMask` module computes the X-ray transmission
:math:`T \\in [0,1]` of a diffracted ray through a binary coded absorber
(Au bars on a Si\\ :sub:`3`\\ N\\ :sub:`4` substrate) as a function of:

* **ray geometry** — origin and unit-vector direction in the lab frame
* **wavelength** — per-ray, via NIST μ(λ) lookup
* **scan offset** — translation of the aperture along its bar axis (the
  scan direction in the Gürsoy *et al.* setup)
* **mask parameters** — pose (position + Rodrigues rotvec), per-bar
  widths, Au thickness, Si\\ :sub:`3`\\ N\\ :sub:`4` substrate thickness.

All parameters are torch-differentiable.  Bar boundaries are smoothed
with sigmoids of width ``edge_softness_um`` so gradients flow through
the discrete binary sequence.

Lab/aperture frame convention
------------------------------

At zero rotation, the aperture's intrinsic axes coincide with the lab
axes:

* ``u`` (intrinsic x) — bar-array axis = scan direction
* ``v`` (intrinsic y) — bar long axis (perpendicular within the plane)
* ``n`` (intrinsic z) — aperture plane normal

The 6-DOF pose maps the aperture from this canonical frame into the lab
frame.  Surge/sway/heave/yaw/pitch/roll (Gürsoy *et al.* RSI 2023)
correspond to ``(position[0], position[1], position[2])`` and the three
components of ``rotvec``, with mapping documented in
``laue_torch/implementation_plan_coded_aperture.md``.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor, nn

from midas_stress.orientation import axis_angle_to_orient_mat

from .absorption import mu_au, mu_si3n4

_RAD2DEG = 180.0 / math.pi
_EPS_ROTVEC = 1.0e-30


def _rotvec_to_matrix(rotvec: Tensor) -> Tensor:
    """Axis-angle-vector (axis · θ_rad) → (3, 3) rotation matrix.

    Wraps ``midas_stress.orientation.axis_angle_to_orient_mat`` — the
    canonical MIDAS primitive (see memory ``feedback_orientation_from_midas_stress``).

    Note: like every axis-angle-vector or Rodrigues parameterisation,
    the gradient at ``rotvec = 0`` is structurally zero (the axis is
    undefined there).  For refinement starting from the identity,
    compose with a quaternion-delta parameter — see
    :class:`laue_torch.realdata.DepthResolvedVoxelRefiner` for the
    pattern.  Initialise the mask with a non-zero rotvec (the
    calibrated pose) for the gradient w.r.t. ``rotvec`` to flow.
    """
    norm = torch.linalg.norm(rotvec).clamp_min(_EPS_ROTVEC)
    axis = rotvec / norm
    angle_deg = norm * _RAD2DEG
    return axis_angle_to_orient_mat(axis, angle_deg)


def build_de_bruijn_sequence(order: int = 8, alphabet: int = 2) -> Tensor:
    """Construct a binary (or higher-alphabet) de Bruijn sequence B(k, n).

    Returns an ``int64`` 1-D tensor of length ``alphabet**order`` such
    that every length-``order`` substring (taken cyclically) appears
    exactly once.  For ``order=8, alphabet=2`` this gives the 256-bit
    sequence used by Gürsoy *et al.*

    Implemented by the standard recursive Lyndon-word algorithm
    (Frank Ruskey, *Combinatorial Generation*).
    """
    k = int(alphabet)
    n = int(order)
    a = [0] * (k * n)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return torch.tensor(seq, dtype=torch.int64)


class CodedApertureMask(nn.Module):
    """Differentiable Beer–Lambert transmission of a coded-aperture mask.

    Parameters
    ----------
    sequence
        1-D integer tensor of 0/1 values; the binary coding pattern.
    bar_widths_um
        Either a scalar (uniform bar width), a 2-tensor
        ``(w_zero_bit, w_one_bit)`` to give 0-bits and 1-bits different
        widths (the Gürsoy *et al.* default — 7.5 µm for 0-bits, 15 µm
        nominal for 1-bits, with the *measured* 1-bit width somewhat
        larger due to electroplating overgrowth), or a per-bar 1-D
        tensor of length ``len(sequence)``.
    au_thickness_um
        Au absorber thickness, normal to the aperture plane.  Two
        accepted forms:

        * **scalar** (default ~4.6 µm) — uniform thickness everywhere
          the sequence is 1; effectively binary.  This is the
          Gürsoy *et al.* JAC 2022 configuration.

        * **(L,)** — *per-bar* thickness, enabling **multi-level coded
          apertures**.  Each bar may have its own Au column height
          (e.g. ``[0, 5, 10, 20]`` µm for a 4-level mask).  The
          sequence still acts as a binary gate: where ``sequence[k] = 0``
          the bar contributes zero thickness regardless of
          ``au_thickness_um[k]``.  This generalisation enables the
          paper's open future-work (§IV, RSI 2023): simultaneous
          encoding of depth *and* energy via thickness-dependent
          Beer–Lambert absorption.
    sub_thickness_um
        Si\\ :sub:`3`\\ N\\ :sub:`4` substrate thickness, ~3 µm.
    position_um
        3-tensor — aperture-center position in the lab frame [µm].
        For the published 34-ID-E setup, ~``(0, 1000, 0)`` if the
        aperture is 1 mm above the sample along the lab y-axis.  The
        caller is responsible for matching the convention of the
        accompanying ``LaueForwardModel`` instance.
    rotvec
        3-tensor — Rodrigues rotation vector mapping the canonical
        aperture frame (u=x, v=y, n=z) into the lab frame.
    edge_softness_um
        Sigmoid width of bar boundaries (smoothing scale for the binary
        sequence indicator).  Much smaller than the smallest bar width;
        default 0.5 µm.  This is a *physical* smoothing approximating
        the small but finite penumbra at each bar edge, so leaving it
        nonzero in production is acceptable.
    make_geometry_learnable
        If True, ``position_um`` and ``rotvec`` are registered as
        ``nn.Parameter``; otherwise they are buffers.  Per-bar widths
        and thicknesses follow the same flag.  Default False so the
        mask is frozen until a calibration loop opts in.
    """

    def __init__(
        self,
        sequence: Tensor,
        bar_widths_um: Union[float, Tensor],
        *,
        au_thickness_um: Union[float, Tensor] = 4.6,
        sub_thickness_um: float = 3.0,
        position_um: Optional[Tensor] = None,
        rotvec: Optional[Tensor] = None,
        edge_softness_um: float = 0.5,
        make_geometry_learnable: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if sequence.dim() != 1:
            raise ValueError(f"sequence must be 1-D, got shape {tuple(sequence.shape)}")
        L = int(sequence.numel())
        seq_bits = sequence.to(dtype=dtype)
        if torch.any((seq_bits != 0) & (seq_bits != 1)):
            raise ValueError("sequence must be binary (0/1)")
        self.register_buffer("sequence", seq_bits)
        self.L = L

        # Per-bar widths.  Three accepted forms:
        #   scalar  → uniform
        #   (2,)    → (w_zero_bit, w_one_bit), broadcast via the sequence
        #   (L,)    → per-bar override
        if isinstance(bar_widths_um, (int, float)):
            widths = torch.full((L,), float(bar_widths_um), dtype=dtype)
        else:
            bw = torch.as_tensor(bar_widths_um, dtype=dtype)
            if bw.dim() == 0:
                widths = bw.expand(L).clone()
            elif bw.dim() == 1 and bw.numel() == 2:
                w0, w1 = bw[0], bw[1]
                widths = torch.where(seq_bits > 0.5, w1.expand(L), w0.expand(L))
            elif bw.dim() == 1 and bw.numel() == L:
                widths = bw.clone()
            else:
                raise ValueError(
                    f"bar_widths_um must be scalar, (2,) or ({L},); "
                    f"got shape {tuple(bw.shape)}"
                )

        # Accept scalar (Phase 0; uniform binary) or (L,) per-bar thickness
        # (Phase 6; multi-level / energy-encoding masks).  Storage is always
        # a tensor; downstream code branches on ``self.au_thickness_um.dim()``.
        if isinstance(au_thickness_um, (int, float)):
            au_th = torch.as_tensor(float(au_thickness_um), dtype=dtype)
        else:
            au_th_t = torch.as_tensor(au_thickness_um, dtype=dtype)
            if au_th_t.dim() == 0:
                au_th = au_th_t.clone()
            elif au_th_t.dim() == 1 and au_th_t.numel() == L:
                au_th = au_th_t.clone()
            else:
                raise ValueError(
                    f"au_thickness_um must be scalar or shape ({L},); "
                    f"got {tuple(au_th_t.shape)}"
                )
        sub_th = torch.as_tensor(float(sub_thickness_um), dtype=dtype)
        pos = (position_um if position_um is not None
               else torch.zeros(3, dtype=dtype))
        rot = (rotvec if rotvec is not None
               else torch.zeros(3, dtype=dtype))
        pos = torch.as_tensor(pos, dtype=dtype)
        rot = torch.as_tensor(rot, dtype=dtype)
        if pos.shape != (3,):
            raise ValueError(f"position_um must be shape (3,), got {tuple(pos.shape)}")
        if rot.shape != (3,):
            raise ValueError(f"rotvec must be shape (3,), got {tuple(rot.shape)}")

        def _register(name: str, tensor: Tensor) -> None:
            if make_geometry_learnable:
                self.register_parameter(name, nn.Parameter(tensor.clone()))
            else:
                self.register_buffer(name, tensor.clone())

        _register("bar_widths_um", widths)
        _register("au_thickness_um", au_th)
        _register("sub_thickness_um", sub_th)
        _register("position_um", pos)
        _register("rotvec", rot)

        self.edge_softness_um = float(edge_softness_um)
        self.dtype = dtype

    # ── helpers ─────────────────────────────────────────────────────────────

    @property
    def total_width_um(self) -> Tensor:
        """Total physical width of the coded pattern along the bar axis."""
        return self.bar_widths_um.sum()

    def bar_edges_um(self) -> Tensor:
        """Bar boundaries in the intrinsic ``u`` coordinate.

        Length ``L+1``: ``edges[k]`` is the left edge of bar ``k``,
        ``edges[L]`` is the right edge of bar ``L-1``.  Centered on
        ``u = 0`` (so ``edges[0] = -W/2``, ``edges[L] = +W/2``).
        """
        W = self.total_width_um
        cum = torch.cat([self.bar_widths_um.new_zeros(1),
                         torch.cumsum(self.bar_widths_um, dim=0)])
        return cum - W / 2.0

    def _smooth_bar_indicator(self, u_query_um: Tensor) -> Tensor:
        """Per-bar smooth indicator ``(...,L)`` ∈ [0, 1].

        Internal helper shared by :meth:`au_coverage` (binary view)
        and :meth:`au_thickness_at` (multi-level / Phase 6 view).
        """
        tau = self.edge_softness_um
        edges = self.bar_edges_um()                                     # (L+1,)
        u = u_query_um.unsqueeze(-1)                                    # (..., 1)
        left = torch.sigmoid((u - edges[:-1]) / tau)                    # (..., L)
        right = torch.sigmoid((edges[1:] - u) / tau)                    # (..., L)
        return left * right                                              # (..., L)

    def au_coverage(self, u_query_um: Tensor) -> Tensor:
        """Smoothed Au *presence* indicator at intrinsic positions ``u_query_um``.

        Returns a tensor of shape ``u_query_um.shape`` valued in [0, 1],
        equal to 1 inside a 1-bit, 0 inside a 0-bit, and varying smoothly
        across bar boundaries with sigmoid scale ``edge_softness_um``.

        This is the **binary** view of the mask — for the Phase 6
        multi-level Au-thickness encoding, use
        :meth:`au_thickness_at`.  The two views coincide (up to a
        scalar factor of ``au_thickness_um``) only when the Au
        thickness is scalar.
        """
        indicator = self._smooth_bar_indicator(u_query_um)
        seq = self.sequence.to(indicator.dtype)
        return (indicator * seq).sum(dim=-1)

    def au_thickness_at(self, u_query_um: Tensor) -> Tensor:
        """Effective Au column thickness [µm] at intrinsic positions ``u_query_um``.

        This is the quantity that enters Beer–Lambert at each ray:

        .. math::
           T_{\\mathrm{Au}}(u) =
               \\exp(-\\mu_{\\mathrm{Au}}(\\lambda)\\,
                     t_{\\mathrm{eff}}(u)\\,/\\,\\cos\\theta)

        where ``t_eff(u) = Σ_k (au_thickness_per_bar[k] · sequence[k]) ·
        indicator_k(u)``.  Scalar ``au_thickness_um`` reduces to
        ``t_eff(u) = au_thickness_um · au_coverage(u)`` — exactly the
        Phase 0 behaviour.
        """
        indicator = self._smooth_bar_indicator(u_query_um)              # (..., L)
        seq = self.sequence.to(indicator.dtype)
        if self.au_thickness_um.dim() == 0:
            per_bar = self.au_thickness_um * seq                         # (L,)
        else:
            per_bar = self.au_thickness_um * seq                         # (L,)
        return (indicator * per_bar).sum(dim=-1)

    # ── intrinsic frame ────────────────────────────────────────────────────

    def aperture_axes_lab(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return (u_hat, v_hat, n_hat) — aperture basis in the lab frame."""
        R = _rotvec_to_matrix(self.rotvec)                              # (3,3)
        u_hat = R[:, 0]
        v_hat = R[:, 1]
        n_hat = R[:, 2]
        return u_hat, v_hat, n_hat

    # ── forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        ray_origin_um: Tensor,
        ray_direction: Tensor,
        wavelength_A: Tensor,
        scan_offset_um: Union[float, Tensor] = 0.0,
    ) -> Tensor:
        """Compute transmission ∈ [0, 1] for a batch of rays.

        Parameters
        ----------
        ray_origin_um
            ``(N, 3)`` or ``(3,)`` — ray origin in lab frame, in
            **micrometers**.  Typically the diffracted source point.
        ray_direction
            ``(N, 3)`` — unit vectors of the diffracted rays.  Need not
            be exactly normalised; the math depends only on the line.
        wavelength_A
            ``(N,)`` or scalar — wavelength in Ångström for μ(λ) lookup.
        scan_offset_um
            scalar or ``(N,)`` — translation of the aperture along its
            intrinsic ``u`` axis [µm].  Positive offset shifts the coded
            sequence in the +u direction (equivalently, query the
            sequence at ``u_intrinsic - p``).

        Returns
        -------
        transmission : ``(N,)`` tensor in [0, 1]
        """
        if ray_direction.dim() != 2 or ray_direction.shape[-1] != 3:
            raise ValueError(
                f"ray_direction must be (N, 3); got {tuple(ray_direction.shape)}")
        N = ray_direction.shape[0]
        if ray_origin_um.dim() == 1:
            ray_origin_um = ray_origin_um.unsqueeze(0).expand(N, 3)
        elif ray_origin_um.shape != (N, 3):
            raise ValueError(
                f"ray_origin_um must broadcast to (N, 3)=({N}, 3); "
                f"got {tuple(ray_origin_um.shape)}")

        wl = torch.as_tensor(wavelength_A, dtype=ray_direction.dtype,
                              device=ray_direction.device)
        if wl.dim() == 0:
            wl = wl.expand(N)

        if isinstance(scan_offset_um, (int, float)):
            p = torch.as_tensor(float(scan_offset_um),
                                 dtype=ray_direction.dtype,
                                 device=ray_direction.device)
        else:
            p = scan_offset_um.to(ray_direction.dtype)
        if p.dim() == 0:
            p = p.expand(N)

        u_hat, _, n_hat = self.aperture_axes_lab()

        # Plane: (X - center) · n = 0;  ray: X = O + t · d
        center = self.position_um
        denom = (ray_direction * n_hat).sum(dim=-1)                     # (N,)
        # Clamp away from zero — rays parallel to the aperture plane are
        # unphysical (would never hit it). They get full transmission
        # below via mask saturation; their gradient through `t` is
        # killed by the clamp.
        denom_safe = torch.where(
            denom.abs() > 1e-9,
            denom,
            torch.full_like(denom, 1e-9),
        )
        offset_to_center = (center - ray_origin_um)                      # (N, 3)
        t = (offset_to_center * n_hat).sum(dim=-1) / denom_safe         # (N,)

        intersection = ray_origin_um + t.unsqueeze(-1) * ray_direction  # (N, 3)
        rel = intersection - center                                      # (N, 3)
        u_intrinsic = (rel * u_hat).sum(dim=-1)                          # (N,)

        u_query = u_intrinsic - p                                         # (N,)
        # Effective Au column at u — handles scalar (binary) and per-bar
        # (multi-level) thickness uniformly.  For scalar this matches the
        # historical ``coverage(u) * au_thickness_um`` exactly.
        t_au_normal = self.au_thickness_at(u_query)                      # (N,) µm

        # Beer–Lambert path lengths (oblique incidence increases the
        # effective traversal).  |cos_inc| = |d · n| / |d|.
        norm_d = torch.linalg.norm(ray_direction, dim=-1).clamp_min(1e-30)
        cos_inc = denom.abs() / norm_d
        cos_inc_safe = cos_inc.clamp_min(1e-6)
        path_au = t_au_normal / cos_inc_safe                              # (N,)
        path_sub = self.sub_thickness_um / cos_inc_safe                  # (N,)

        mu_au_val = mu_au(wl)                                            # (N,) 1/µm
        mu_sub_val = mu_si3n4(wl)                                        # (N,) 1/µm

        absorb_au = mu_au_val * path_au
        absorb_sub = mu_sub_val * path_sub
        return torch.exp(-(absorb_au + absorb_sub))
