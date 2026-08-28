"""Multi-material 1D coded aperture (Design #2: spectrally-coded mask).

Where the standard ``CodedApertureMask`` uses bars of a single
absorber (Au), this class arranges bars of *different elements*
along the scan axis.  Each element has its own absorption-coefficient
curve μ(λ), so each diffracted Bragg ray — with energy fixed by
Bragg's law — sees a transmission that depends jointly on:

1. Which bar the ray pierces (via depth-encoded geometry, as in the
   standard 1D mask).
2. Which *material* that bar is made of (per-bar choice).
3. The ray's energy / wavelength (from Bragg's law).

The result is a single-shot encoding of *both* depth and energy
information, addressing the open future-work bullet of Gürsoy *et al.*
JAC 2022 §IV.

Materials are referenced by their atomic symbol; absorption is
computed by :func:`midas_hkls.absorption.linear_absorption_coefficient`
which threads NIST XCOM tables under the canonical MIDAS primitive.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import torch
from torch import Tensor, nn

from midas_hkls.absorption import linear_absorption_coefficient

from .absorption import mu_si3n4, _CM_PER_UM  # noqa: F401 — re-use unit factor
from .mask import _rotvec_to_matrix


__all__ = ["CodedApertureMaskSpectral"]


class CodedApertureMaskSpectral(nn.Module):
    """Multi-material 1D coded aperture for spectral depth encoding.

    Parameters
    ----------
    bar_materials
        ``L``-long sequence of element symbols (e.g.
        ``["Cu", "Mo", "Au", "Pt", "Cu", ...]``).  Must be in the
        ``midas_hkls`` NIST table.
    bar_thicknesses_um
        Scalar (uniform) or ``(L,)`` per-bar thickness.
    bar_widths_um
        Scalar (uniform) or ``(L,)`` per-bar width.
    sub_thickness_um
        Si₃N₄ substrate thickness.
    position_um, rotvec
        Mask pose in the lab frame (same convention as
        :class:`CodedApertureMask`).
    edge_softness_um
        Sigmoid smoothing scale at bar boundaries.
    """

    def __init__(
        self,
        bar_materials: Sequence[str],
        bar_thicknesses_um: Union[float, Tensor],
        *,
        bar_widths_um: Union[float, Tensor] = 12.0,
        sub_thickness_um: float = 0.0,
        position_um: Tensor = None,
        rotvec: Tensor = None,
        edge_softness_um: float = 0.5,
        make_geometry_learnable: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        materials = list(bar_materials)
        L = len(materials)
        if L < 2:
            raise ValueError(f"need ≥ 2 bars; got {L}")
        self.bar_materials = materials
        self.L = L

        # Per-bar thickness (always stored as (L,) tensor).
        if isinstance(bar_thicknesses_um, (int, float)):
            t_arr = torch.full((L,), float(bar_thicknesses_um), dtype=dtype)
        else:
            t = torch.as_tensor(bar_thicknesses_um, dtype=dtype)
            if t.dim() == 0:
                t_arr = t.expand(L).clone()
            elif t.dim() == 1 and t.numel() == L:
                t_arr = t.clone()
            else:
                raise ValueError(
                    f"bar_thicknesses_um must be scalar or shape ({L},); "
                    f"got {tuple(t.shape)}"
                )

        # Per-bar width (always (L,) tensor).
        if isinstance(bar_widths_um, (int, float)):
            w_arr = torch.full((L,), float(bar_widths_um), dtype=dtype)
        else:
            w = torch.as_tensor(bar_widths_um, dtype=dtype)
            if w.dim() == 0:
                w_arr = w.expand(L).clone()
            elif w.dim() == 1 and w.numel() == L:
                w_arr = w.clone()
            else:
                raise ValueError(
                    f"bar_widths_um must be scalar or shape ({L},); "
                    f"got {tuple(w.shape)}"
                )

        sub_th = torch.as_tensor(float(sub_thickness_um), dtype=dtype)
        pos = position_um if position_um is not None else torch.zeros(3, dtype=dtype)
        rot = rotvec if rotvec is not None else torch.zeros(3, dtype=dtype)
        pos = torch.as_tensor(pos, dtype=dtype)
        rot = torch.as_tensor(rot, dtype=dtype)

        def _register(name, t):
            if make_geometry_learnable:
                self.register_parameter(name, nn.Parameter(t.clone()))
            else:
                self.register_buffer(name, t.clone())

        _register("bar_thicknesses_um", t_arr)
        _register("bar_widths_um", w_arr)
        _register("sub_thickness_um", sub_th)
        _register("position_um", pos)
        _register("rotvec", rot)

        self.edge_softness_um = float(edge_softness_um)
        self.dtype = dtype

    @property
    def total_width_um(self) -> Tensor:
        return self.bar_widths_um.sum()

    def bar_edges_um(self) -> Tensor:
        W = self.total_width_um
        cum = torch.cat([self.bar_widths_um.new_zeros(1),
                         torch.cumsum(self.bar_widths_um, dim=0)])
        return cum - W / 2.0

    def aperture_axes_lab(self) -> tuple[Tensor, Tensor, Tensor]:
        R = _rotvec_to_matrix(self.rotvec)
        return R[:, 0], R[:, 1], R[:, 2]

    def _smooth_bar_indicators(self, u_query_um: Tensor) -> Tensor:
        """Per-bar smooth indicator ``(..., L)`` summing to ~1 inside the mask."""
        tau = self.edge_softness_um
        edges = self.bar_edges_um()
        u = u_query_um.unsqueeze(-1)
        left = torch.sigmoid((u - edges[:-1]) / tau)
        right = torch.sigmoid((edges[1:] - u) / tau)
        return left * right

    def absorbance_at(self, u_query_um: Tensor, wavelength_A: Tensor) -> Tensor:
        """Per-ray material-aware Au-column equivalent absorbance.

        Returns the unitless ``Σ_k indicator_k(u) · μ_{mat_k}(λ) · t_k``
        — a scalar quantity per ray that enters Beer–Lambert directly.
        """
        indicators = self._smooth_bar_indicators(u_query_um)         # (..., L)
        # Per-bar μ at each wavelength.  Vectorise over (N_rays, L).
        # Build an (N_rays, L) μ-table by stacking per-bar look-ups.
        # midas_hkls.linear_absorption_coefficient is per-element;
        # cache one call per unique element to amortise.
        unique_mats = list(dict.fromkeys(self.bar_materials))
        mu_per_mat = {
            mat: linear_absorption_coefficient(mat, wavelength_A) * 1e-4
            for mat in unique_mats
        }
        # Build per-bar μ tensor of shape (N_rays, L)
        mu_per_bar_columns = []
        for mat in self.bar_materials:
            mu_per_bar_columns.append(mu_per_mat[mat])
        mu_per_bar = torch.stack(mu_per_bar_columns, dim=-1)        # (..., L)
        # mu_per_bar values are per-ray; broadcast against per-bar thickness.
        per_bar_absorb = (indicators * mu_per_bar
                          * self.bar_thicknesses_um.unsqueeze(0))
        return per_bar_absorb.sum(dim=-1)

    def forward(
        self,
        ray_origin_um: Tensor,
        ray_direction: Tensor,
        wavelength_A: Tensor,
        scan_offset_um: Union[float, Tensor] = 0.0,
    ) -> Tensor:
        """Transmission ∈ [0, 1] per ray (Beer–Lambert with per-bar material)."""
        if ray_direction.dim() != 2 or ray_direction.shape[-1] != 3:
            raise ValueError(
                f"ray_direction must be (N, 3); got {tuple(ray_direction.shape)}"
            )
        N = ray_direction.shape[0]
        if ray_origin_um.dim() == 1:
            ray_origin_um = ray_origin_um.unsqueeze(0).expand(N, 3)

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
        center = self.position_um
        denom = (ray_direction * n_hat).sum(dim=-1)
        denom_safe = torch.where(
            denom.abs() > 1e-9, denom, torch.full_like(denom, 1e-9),
        )
        offset = center - ray_origin_um
        t = (offset * n_hat).sum(dim=-1) / denom_safe
        intersection = ray_origin_um + t.unsqueeze(-1) * ray_direction
        rel = intersection - center
        u_intrinsic = (rel * u_hat).sum(dim=-1) - p

        absorbance_au = self.absorbance_at(u_intrinsic, wl)
        norm_d = torch.linalg.norm(ray_direction, dim=-1).clamp_min(1e-30)
        cos_inc = denom.abs() / norm_d
        cos_inc_safe = cos_inc.clamp_min(1e-6)
        path_sub = self.sub_thickness_um / cos_inc_safe
        mu_sub_val = mu_si3n4(wl)
        absorb_sub = mu_sub_val * path_sub

        return torch.exp(-(absorbance_au / cos_inc_safe + absorb_sub))
