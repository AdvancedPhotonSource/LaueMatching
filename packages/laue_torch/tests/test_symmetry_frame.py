"""laue_torch's symmetry helpers work in laue_torch's crystal frame.

``geometry.reciprocal_matrix`` puts a along Cartesian x (as the C indexer
does); ``midas_stress`` builds its operators with a* along x. For trigonal
cells the frames differ by 30 deg about c, so midas_stress's 2-folds land in
the wrong place and a symmetry-equivalent pair reads as misoriented (the C1
defect, fixed the same way in laue_index.lattice and laue_material).
"""
import math

import pytest
import torch

from laue_torch import symmetry

HEX = (0.476, 0.476, 1.299, 90.0, 90.0, 120.0)


def _rot(axis, deg):
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64)
    t = math.radians(deg)
    return torch.eye(3, dtype=torch.float64) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


# A true 2-fold in laue_torch's frame: along a (x) for -3m1, perpendicular (y) for -31m.
@pytest.mark.parametrize("sg,axis", [(167, (1, 0, 0)), (164, (1, 0, 0)), (150, (1, 0, 0)),
                                     (162, (0, 1, 0)), (149, (0, 1, 0))])
def test_trigonal_symmetry_equivalent_pair_is_zero_apart(sg, axis):
    U = _rot((0.3, -0.5, 0.8), 37.0)
    V = U @ _rot(axis, 180)
    assert float(symmetry.misorientation_deg(U[None], V[None], sg, lattice=HEX)[0]) < 1e-6


@pytest.mark.parametrize("sg", [150, 167, 162])
def test_operators_are_lattice_symmetries_in_this_frame(sg):
    from laue_torch.geometry import reciprocal_matrix
    B = reciprocal_matrix(torch.tensor(HEX, dtype=torch.float64)).detach()
    A = 2 * math.pi * torch.linalg.inv(B).T
    for S in symmetry.symmetry_operators(sg, lattice=HEX):
        M = torch.linalg.inv(A) @ S @ A
        assert torch.allclose(M, M.round(), atol=1e-6)


def test_nearest_variant_returns_the_equivalent_orientation():
    U = _rot((0.3, -0.5, 0.8), 37.0)
    V = U @ _rot((1, 0, 0), 180)          # a true 2-fold for SG 167
    W = symmetry.nearest_variant(V, U, 167, lattice=HEX)
    assert torch.allclose(W, U, atol=1e-9)


@pytest.mark.parametrize("sg", [194, 225])
def test_hex_and_cubic_unchanged(sg):
    U = _rot((0.3, -0.5, 0.8), 37.0)
    V = _rot((-0.2, 0.1, 0.9), 71.0)
    lat = HEX if sg == 194 else (0.36, 0.36, 0.36, 90.0, 90.0, 90.0)
    from midas_stress import misorientation_om_batch
    ref = float(misorientation_om_batch(U[None], V[None], sg)[0]) * 180 / math.pi
    assert abs(float(symmetry.misorientation_deg(U[None], V[None], sg, lattice=lat)[0]) - ref) < 1e-9
