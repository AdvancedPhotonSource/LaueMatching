"""Wire-scan (differential-aperture) DAXM depth gating."""
import pytest
import torch

from laue_torch.wire_scan import (
    depth_centroid_strain,
    integrated_profile,
    recover_depth_strain,
    triangulate_depths,
    visibility_matrix,
    wire_scan,
)

DT = torch.float64
R0, WIDTH = 100.0, 1.5


def _grid():
    return torch.linspace(90.0, 110.0, 240, dtype=DT)


@pytest.mark.unit
def test_visibility_is_cumulative():
    V = visibility_matrix(5)
    assert V.shape == (6, 5)
    assert float(V[0].sum()) == 5.0 and float(V[-1].sum()) == 0.0


@pytest.mark.unit
def test_triangulation_recovers_per_depth_profiles():
    r = _grid()
    eps = torch.tensor([0.0, 0.01, 0.02, 0.03, 0.04], dtype=DT)
    M = wire_scan(eps, r, r0=R0, width=WIDTH)
    eps_rec = depth_centroid_strain(triangulate_depths(M), r, r0=R0)
    assert torch.allclose(eps_rec, eps, atol=1e-4)


@pytest.mark.unit
def test_integrated_profile_is_depth_degenerate_but_wire_is_not():
    r = _grid()
    eps_a = torch.tensor([0.0, 0.01, 0.02, 0.03, 0.04], dtype=DT)
    eps_b = eps_a.flip(0)
    assert torch.allclose(integrated_profile(eps_a, r, r0=R0, width=WIDTH),
                          integrated_profile(eps_b, r, r0=R0, width=WIDTH), atol=1e-9)
    assert not torch.allclose(wire_scan(eps_a, r, r0=R0, width=WIDTH),
                              wire_scan(eps_b, r, r0=R0, width=WIDTH), atol=1e-6)


@pytest.mark.unit
def test_recover_depth_strain_gradient_and_nonmonotonic():
    r = _grid()
    for eps_true in (torch.linspace(0.0, 0.04, 7, dtype=DT),
                     torch.tensor([0.0, 0.03, 0.01, 0.04, 0.02, 0.0], dtype=DT)):
        M = wire_scan(eps_true, r, r0=R0, width=WIDTH)
        out = recover_depth_strain(M, r, r0=R0, width=WIDTH,
                                   n_depth=eps_true.numel(), steps=1500, lr=0.01)
        assert torch.allclose(out["eps"], eps_true, atol=2e-3), out["eps"]
