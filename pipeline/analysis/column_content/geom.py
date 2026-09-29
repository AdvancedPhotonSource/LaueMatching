"""Laue geometry, measured kernel and cloud statistics for the column-content fit (reference implementation).

Ported from the sampleH Zn analysis (2026-09; the sampleH mixture script). Geometry comes from laue_material.Phase and a LaueMatching
params file; projection matches the indexer's (asserted in the sampleH work against laue_torch's forward model to 3.6e-10 px).

CONVENTIONS (the transpose trap has bitten this code family twice): raw frames are indexed [row = y, col = x];
px is the x (column) coordinate, py the y (row) coordinate.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # pipeline/analysis
from laue_material import Phase  # noqa: E402

HC_NM = 1.2398419739
DT = torch.float64


def tangent_rotation(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vectors (radians) -> (..., 3, 3) rotation matrices (Rodrigues); differentiable, safe at 0."""
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K) + s * K + (1 - c) * (K @ K)


class Geom:
    """Detector geometry as torch tensors from a LaueMatching params file."""

    def __init__(self, params_path: str, phase: str, device: str = "cpu", e_range=None):
        """e_range (keV) defaults to the params file's Elo/Ehi, the band synthetic.render_columns renders with."""
        ph = Phase(params_path, name=phase)
        self.ph, self.device = ph, device
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=DT, device=device)  # noqa: E731
        self.B = t(ph.B)                                   # Phase builds B with its space group (R-axes embedding)
        self.ki, self.roti, self.P = t(ph.ki), t(ph.roti), t(ph.P)
        self.dx, self.dy = float(ph.dx), float(ph.dy)
        self.nx, self.ny = int(ph.npx_x), int(ph.npx_y)
        self.elo, self.ehi = (ph.Elo, ph.Ehi) if e_range is None else (float(e_range[0]), float(e_range[1]))

    def project(self, oms: torch.Tensor, hk: torch.Tensor):
        """oms (K,3,3), hk (n,3) -> px, py, E (keV), ok, each (K, n)."""
        q = torch.einsum("kij,jl,nl->kni", oms, self.B, hk)
        ql = q.norm(dim=2)
        qh = q / ql.clamp_min(1e-12)[..., None]
        kf = self.ki - 2.0 * qh[..., 2:3] * qh
        xd = kf @ self.roti.T
        z = xd[..., 2]
        zs = torch.where(z.abs() < 1e-12, torch.ones_like(z), z)
        xs = xd * (self.P[2] / zs)[..., None]
        px = (xs[..., 0] - self.P[0]) / self.dx + 0.5 * (self.nx - 1)
        py = (xs[..., 1] - self.P[1]) / self.dy + 0.5 * (self.ny - 1)
        st = -qh[..., 2]
        E = HC_NM * ql / (4 * math.pi * st.clamp_min(1e-12))
        ok = ((ql > 1e-9) & (z > 0) & (st > 1e-9) & (px >= 0) & (px < self.nx - 1)
              & (py >= 0) & (py < self.ny - 1) & (E > self.elo) & (E < self.ehi))
        return px, py, E, ok

    def mean_q_axis(self, om: torch.Tensor, hk: torch.Tensor) -> torch.Tensor:
        """Unit mean scattering-vector direction of these reflections (folded onto +z): the fit's blind axis."""
        q = (om @ self.B @ hk.T).T
        qh = q / q.norm(dim=1, keepdim=True)
        qh = qh * torch.where(qh[:, 2:3] < 0, -1.0, 1.0)
        m = qh.mean(0)
        return m / m.norm()


class EmpKernel:
    """The MEASURED instrument kernel (fine grid from a calibrant, e.g. a thin Si crystal): pixel-integrated by 3x3
    sub-sampling, bilinear in the spot position (differentiable). With ``scale`` given, widths stretch and flux stays 1.
    File keys: Kf (fine grid, indexed [y, x]), hw (half-width), hf (pitch)."""

    def __init__(self, path: str, device: str = "cpu", scale: float = 1.0):
        z = np.load(path)
        self.hw, self.hf, self.scale = float(z["hw"]), float(z["hf"]), float(scale)
        self.K = torch.as_tensor(z["Kf"], dtype=DT, device=device)[None, None]
        self.sub = [(-1.0 + i) / 3.0 for i in range(3)]
        self.path = path

    def __call__(self, dx: torch.Tensor, dy: torch.Tensor, scale=None) -> torch.Tensor:
        import torch.nn.functional as Fnn
        sc = self.scale if scale is None else scale
        acc = torch.zeros_like(dx)
        for oy in self.sub:
            for ox in self.sub:
                gx = (dx + ox) / (sc * self.hw)
                gy = (dy + oy) / (sc * self.hw)
                grid = torch.stack([gx, gy], -1).reshape(1, -1, 1, 2)
                acc = acc + Fnn.grid_sample(self.K, grid, mode="bilinear", padding_mode="zeros",
                                            align_corners=True).reshape(dx.shape)
        return acc / 9.0 if scale is None else acc / (9.0 * sc * sc)


def cloud_summary(omegas: torch.Tensor, w: torch.Tensor, blind: torch.Tensor) -> dict:
    """Weighted cloud statistics (deg), split by the blind axis (mean q direction)."""
    wn = w / w.sum().clamp_min(1e-30)
    mu = (wn[:, None] * omegas).sum(0)
    d = omegas - mu
    S = (wn[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    b = blind / blind.norm()
    Pp = torch.eye(3, dtype=DT, device=omegas.device) - torch.outer(b, b)
    perp = float(torch.sqrt(torch.clamp(torch.trace(Pp @ S @ Pp), min=0.0)))
    along = float(torch.sqrt(torch.clamp(b @ S @ b, min=0.0)))
    occ = int((w > 0.01 * float(w.max())).sum()) if float(w.max()) > 0 else 0
    return dict(extent_perp_deg=math.degrees(perp), extent_blind_deg=math.degrees(along),
                mean_offset_deg=math.degrees(float(mu.norm())), n_occupied=occ)


def truth_summary(truth_oms: np.ndarray, weights: np.ndarray, om_ref: np.ndarray, blind: np.ndarray) -> dict:
    """The same statistics for a known set of sub-orientations, as tangent offsets from om_ref."""
    from scipy.spatial.transform import Rotation
    om_ref = np.asarray(om_ref, float).reshape(3, 3)
    w = torch.as_tensor(np.asarray(weights, float), dtype=DT)
    rv = torch.as_tensor(np.stack([Rotation.from_matrix(np.asarray(M, float) @ om_ref.T).as_rotvec()
                                   for M in truth_oms]), dtype=DT)
    return cloud_summary(rv, w, torch.as_tensor(np.asarray(blind, float), dtype=DT))


def require_degrees(fn, *, batch: bool = False, where: str = "misorientation"):
    """Raise unless ``fn`` returns DEGREES: probe fn(I, Rz(5 deg)) (batch: fn(I, [Rz])) must lie in (4, 6).
    midas_stress returns RADIANS, which would make a 1 deg threshold mean 57 deg."""
    c, s = math.cos(math.radians(5.0)), math.sin(math.radians(5.0))
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    v = float(np.asarray(fn(np.eye(3), rz[None] if batch else rz), float).ravel()[0])
    if not 4.0 < v < 6.0:
        raise ValueError(f"{where} must return DEGREES: a 5 deg rotation gave {v:.6g} "
                         "(radians? wrap with np.degrees, or use laue_material.Phase.misorientation)")
