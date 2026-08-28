"""Contract test: laue_torch.symmetry vs the midas_stress primitive it wraps.

``laue_torch.symmetry`` does not implement symmetry folding -- it delegates to
``midas_stress.misorientation_om_batch`` and converts RADIANS to DEGREES. That
makes two things worth locking down, because neither fails loudly on its own:

1. **The unit convention.** midas_stress returns radians; every caller here
   wants degrees. If that conversion is ever dropped, misorientations come out
   57x too small -- which reads as excellent convergence, not as a bug.

2. **The symmetry convention.** If a future midas_stress changes how it folds
   into the fundamental zone, every misorientation this package reports shifts
   silently.

The reference values below were produced by the ORIGINAL hand-rolled
implementation before it was deleted, at ``torch.manual_seed(12345)``. The
replacement reproduced them to 7.1e-14 deg; a wider 500-pair sweep and a
set-comparison of the two symmetry groups (24/24 operators matched, all proper)
agreed to 4.6e-13 deg.

The three production call sites (realdata/driver.py, realdata/plots.py,
realdata/multi_grain.py) are lazy imports inside methods that no other test
exercises, so the shape/dtype assertions here are their only coverage.
"""

from __future__ import annotations

import math

import pytest
import torch

from laue_torch.symmetry import (
    CUBIC_SPACE_GROUP,
    cubic_misorientation_deg,
    misorientation_deg,
)

DT = torch.float64

#: Produced by the pre-migration implementation at torch.manual_seed(12345).
#: Do not regenerate these from the current code -- that would defeat the test.
REFERENCE_DEG = [
    58.246559073, 43.319247551, 53.740396590, 45.425156204,
    47.541609860, 48.008206020, 30.178837633, 48.193824032,
]


def _rand_rot(n: int) -> torch.Tensor:
    """n random rotation matrices from normalised random quaternions."""
    q = torch.randn(n, 4, dtype=DT)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


@pytest.fixture
def pair():
    torch.manual_seed(12345)
    return _rand_rot(8), _rand_rot(8)


def test_matches_pre_migration_reference(pair):
    """Locks symmetry convention AND units against the original implementation."""
    M1, M2 = pair
    got = cubic_misorientation_deg(M1, M2)
    expected = torch.tensor(REFERENCE_DEG, dtype=DT)
    assert torch.allclose(got, expected, atol=1e-8), (
        f"midas_stress misorientation drifted from the pre-migration reference.\n"
        f"  expected {expected.tolist()}\n  got      {got.tolist()}"
    )


def test_result_is_degrees_not_radians(pair):
    """A cubic misorientation can exceed 2*pi in degrees but never in radians.

    The maximum cubic-cubic misorientation is ~62.8 deg = 1.10 rad. Values
    above 2*pi are therefore only possible if the conversion happened.
    """
    M1, M2 = pair
    got = cubic_misorientation_deg(M1, M2)
    assert got.max() > 2 * math.pi, (
        "misorientation looks like radians -- the rad->deg conversion in "
        "laue_torch.symmetry has been dropped"
    )
    # And nothing may exceed the cubic fundamental-zone bound.
    assert got.max() <= 62.9


def test_identity_and_exact_symmetry_operator_are_zero():
    """The two cases where a symmetry-folding bug actually shows."""
    eye = torch.eye(3, dtype=DT).unsqueeze(0)
    assert cubic_misorientation_deg(eye, eye).item() == pytest.approx(0.0, abs=1e-9)

    # A 90 deg rotation about z is a cubic symmetry operator: modulo cubic
    # symmetry it is indistinguishable from the identity.
    rot90_z = torch.tensor([[[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]]], dtype=DT)
    assert cubic_misorientation_deg(eye, rot90_z).item() == pytest.approx(0.0, abs=1e-9)


def test_small_angle_is_recovered():
    """0.01 deg must come back as 0.01 deg -- the 57x units trap, concretely."""
    axis = torch.tensor([1.0, 1.0, 1.0], dtype=DT)
    axis = axis / axis.norm()
    ang = math.radians(0.01)
    K = torch.tensor([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]], dtype=DT)
    R = (torch.eye(3, dtype=DT) + math.sin(ang) * K
         + (1 - math.cos(ang)) * (K @ K)).unsqueeze(0)
    eye = torch.eye(3, dtype=DT).unsqueeze(0)
    assert cubic_misorientation_deg(eye, R).item() == pytest.approx(0.01, abs=1e-7)


def test_space_group_is_honoured(pair):
    """Hexagonal must NOT reduce to the same answer as cubic.

    The old helper hardcoded the 24 cubic operators, so hcp (Ti-64) was
    silently folded under the wrong point group. Guards against a regression
    that ignores the space_group argument.
    """
    M1, M2 = pair
    cubic = misorientation_deg(M1, M2, CUBIC_SPACE_GROUP)
    hexagonal = misorientation_deg(M1, M2, 194)
    assert not torch.allclose(cubic, hexagonal), (
        "space_group appears to be ignored -- cubic and hexagonal agree exactly"
    )


@pytest.mark.parametrize("shape_a,shape_b,expected", [
    ((1, 3, 3), (1, 3, 3), (1,)),    # realdata/driver.py, realdata/multi_grain.py
    ((8, 3, 3), (1, 3, 3), (8,)),    # realdata/plots.py -- broadcast against a reference
])
def test_production_call_site_shapes(shape_a, shape_b, expected):
    """The exact shapes the three realdata call sites pass in."""
    torch.manual_seed(0)
    M1 = _rand_rot(shape_a[0])
    M2 = _rand_rot(shape_b[0])
    out = cubic_misorientation_deg(M1, M2)
    assert tuple(out.shape) == expected
    assert out.dtype == DT
    out.numpy()          # plots.py does this
    if expected == (1,):
        float(out.item())  # driver.py and multi_grain.py do this


def test_dtype_and_device_preserved():
    """midas_stress promises tensors back on the input's dtype/device."""
    torch.manual_seed(1)
    M1, M2 = _rand_rot(4), _rand_rot(4)
    out = cubic_misorientation_deg(M1, M2)
    assert out.device == M1.device
    assert isinstance(out, torch.Tensor)
