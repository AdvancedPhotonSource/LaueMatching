"""laue_torch.nye: curvature vs Nye tensor, frames, slip-system solve, rotvec.

* ``lattice_curvature`` returns kappa_ij = d omega_i / d x_j (what the old
  ``nye_tensor`` computed and called alpha). Nye's tensor is
  ``nye_alpha(kappa) = kappa^T - tr(kappa) I``; the two differ whenever
  tr(kappa) != 0 or kappa is not symmetric.
* ``frame="crystal"`` expresses the grid (x_j) index in the reference
  crystal frame as well, so both indices live in one frame.
* ``slip_system_gnd`` on the 12 FCC systems (rank 8) returns the
  MIN-NORM densities; the old normal-equation solve on the singular 12x12
  Gram matrix returned an arbitrary member of the null-space family.
* ``matrix_to_rotvec`` has the identity Jacobian at R = I (was 0).
"""
from __future__ import annotations

import math
import warnings

import pytest
import torch

from laue_torch.geometry import rodrigues_to_matrix
from laue_torch.nye import (
    fcc_slip_systems,
    lattice_curvature,
    matrix_to_rotvec,
    nye_alpha,
    nye_tensor,
    slip_system_gnd,
    synthetic_linear_gradient_field,
)

DT = torch.float64


def _twist_field(n=9, rate_deg=0.2, U=None):
    """Rotation about lab z increasing along grid axis 2 (z): a pure twist,
    kappa = rate * e_z (x) e_z, tr kappa = rate."""
    U = torch.eye(3, dtype=DT) if U is None else U
    ang = (torch.arange(n, dtype=DT) - n // 2) * math.radians(rate_deg)
    rv = torch.zeros(n, 3, dtype=DT)
    rv[:, 2] = ang
    R1 = rodrigues_to_matrix(rv) @ U                          # (n, 3, 3)
    return R1[None, None].expand(3, 3, n, 3, 3).contiguous()   # (x, y, z, 3, 3)


def test_nye_alpha_differs_from_curvature_when_trace_nonzero():
    R = _twist_field()
    kappa = lattice_curvature(R, spacing=1.0)[1, 1, 4]
    r = math.radians(0.2)
    assert float(kappa[2, 2]) == pytest.approx(r, rel=1e-9)
    assert float(torch.trace(kappa)) == pytest.approx(r, rel=1e-9)
    alpha = nye_alpha(kappa)
    expect = torch.diag(torch.tensor([-r, -r, 0.0], dtype=DT))
    assert torch.allclose(alpha, expect, atol=1e-12)
    assert not torch.allclose(alpha, kappa, atol=1e-6)


def test_nye_alpha_formula_on_random_kappa():
    k = torch.randn(4, 3, 3, dtype=DT, generator=torch.Generator().manual_seed(0))
    a = nye_alpha(k)
    tr = k.diagonal(dim1=-2, dim2=-1).sum(-1)
    ref = k.transpose(-1, -2) - tr[:, None, None] * torch.eye(3, dtype=DT)
    assert torch.allclose(a, ref)


def test_crystal_frame_rotates_the_grid_index():
    """Same physical gradient, crystal rotated by U: the crystal-frame
    curvature is U^T kappa_lab U; the mixed-frame one is U^T kappa_lab."""
    U = rodrigues_to_matrix(torch.tensor([0.3, -0.5, 0.7], dtype=DT))
    R = _twist_field(U=U)
    k_c = lattice_curvature(R, spacing=1.0)[1, 1, 4]
    k_m = lattice_curvature(R, spacing=1.0, frame="mixed")[1, 1, 4]
    r = math.radians(0.2)
    k_lab = torch.zeros(3, 3, dtype=DT)
    k_lab[2, 2] = r
    assert torch.allclose(k_m, U.T @ k_lab, atol=1e-10)
    assert torch.allclose(k_c, U.T @ k_lab @ U, atol=1e-10)
    # A frame-invariant (the trace) survives only in the one-frame version.
    assert float(torch.trace(k_c)) == pytest.approx(r, rel=1e-8)


def test_nye_tensor_is_a_deprecated_alias_of_the_old_quantity():
    R, k_truth = synthetic_linear_gradient_field(n_voxels=15, rate_per_voxel_deg=0.1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old = nye_tensor(R, spacing=1.0)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert torch.allclose(old, lattice_curvature(R, spacing=1.0, frame="mixed"))
    assert (old[7] - k_truth).abs().max() < 1e-10


def test_default_reference_is_the_centre_voxel():
    R = _twist_field(n=9)
    k_default = lattice_curvature(R, spacing=1.0)
    k_centre = lattice_curvature(R, spacing=1.0, R_ref=R[1, 1, 4])
    assert torch.allclose(k_default, k_centre)


def test_slip_system_gnd_is_min_norm_on_rank_deficient_fcc():
    S = fcc_slip_systems()
    flat = S.reshape(12, 9)
    assert int(torch.linalg.matrix_rank(flat)) == 8
    alpha = 2.0 * S[0]
    rho = slip_system_gnd(alpha, S, burgers_m=1.0)
    ref = torch.linalg.pinv(flat.T) @ alpha.reshape(9)
    assert torch.allclose(rho, ref, atol=1e-10)
    assert float(rho.norm()) <= 2.0 + 1e-10          # min-norm <= the obvious 2 e_0
    assert torch.allclose((rho[:, None, None] * S).sum(0), alpha, atol=1e-10)


def test_matrix_to_rotvec_jacobian_at_identity():
    J = torch.autograd.functional.jacobian(
        lambda r: matrix_to_rotvec(rodrigues_to_matrix(r)), torch.zeros(3, dtype=DT))
    assert torch.isfinite(J).all()
    assert torch.allclose(J, torch.eye(3, dtype=DT), atol=1e-8)


def test_matrix_to_rotvec_values():
    for rv in ([0.1, 0.2, 0.3], [1e-9, -2e-9, 3e-9], [0.0, 0.0, 0.0], [1.0, -2.0, 0.5]):
        r = torch.tensor(rv, dtype=DT)
        assert torch.allclose(matrix_to_rotvec(rodrigues_to_matrix(r)), r, atol=1e-12)


def test_plot_gnd_map_uses_nye_alpha_in_per_metre(tmp_path):
    """rho = ||alpha||_F / b with alpha = kappa^T - tr(kappa) I and the grid
    spacing converted from um to m (it was used in um, 1e6 too small)."""
    pytest.importorskip("matplotlib")
    from laue_torch.realdata import plot_gnd_map

    R = _twist_field(n=5)[:, :, :, :, :].reshape(3, 3, 5, 3, 3)

    class Res:
        def __init__(self, U):
            self.U_mean = U
    results = [Res(U) for U in R.reshape(-1, 3, 3)]
    rho = plot_gnd_map(results, grid_shape=(3, 3, 5), spacing_um=2.0,
                       burgers_m=2.5e-10, out_path=tmp_path / "g.png")
    r = math.radians(0.2) / 2.0e-6                    # rad per metre
    expect = math.sqrt(2.0) * r / 2.5e-10             # ||diag(-r, -r, 0)||_F / b
    assert float(rho[1, 1, 2]) == pytest.approx(expect, rel=1e-6)
