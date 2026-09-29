"""C2: the lattice setting for the seven rhombohedral space groups.

``reciprocal_matrix(lattice, sg_num=...)`` picks the Cartesian embedding from
the SUPPLIED PARAMETERS, not from the space-group number alone:

* hexagonal axes (a, a, c, 90, 90, 120) -> the standard branch (a along x);
  this is what midas_hkls (and so ``generate_hkls``) uses for SG 167;
* rhombohedral axes (a, a, a, alpha, alpha, alpha != 90) -> the rhombohedral
  embedding with the 3-fold along Cartesian [111], as C ``calcRecipArray``;
* anything else for SG 146/148/155/160/161/166/167 raises.

``sg_num=None`` keeps the old behaviour (standard branch for everything).

The reference below is an independent numpy port of C ``calcRecipArray``
(packages/laue_index/c_src/LaueMatchingHeaders.h), with the branch passed in
explicitly, so this test does not depend on the C or laue_index fix landing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from laue_torch.geometry import RHOMBOHEDRAL_SPACE_GROUPS, reciprocal_matrix

DEG = math.pi / 180.0


def _c_calc_recip_array(lat, rhomb: bool) -> np.ndarray:
    """numpy port of C calcRecipArray (columns a*, b*, c*), branch explicit."""
    a, b, c, alpha, beta, gamma = lat
    ca, cb, cg = math.cos(alpha * DEG), math.cos(beta * DEG), math.cos(gamma * DEG)
    sg = math.sin(gamma * DEG)
    phi = math.sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    Vc = a * b * c * phi
    pv = 2 * math.pi / Vc
    if not rhomb:
        a0, a1, a2 = a, 0.0, 0.0
        b0, b1, b2 = b * cg, b * sg, 0.0
        c0 = c * cb
        c1 = c * (ca - cb * cg) / sg
        c2 = c * phi / sg
    else:
        p = math.sqrt(1.0 + 2 * ca)
        q = math.sqrt(1.0 - ca)
        pmq = (a / 3.0) * (p - q)
        p2q = (a / 3.0) * (p + 2 * q)
        a0, a1, a2 = p2q, pmq, pmq
        b0, b1, b2 = pmq, p2q, pmq
        c0, c1, c2 = pmq, pmq, p2q
    r = np.zeros((3, 3))
    r[0, 0] = (b1 * c2 - b2 * c1) * pv
    r[1, 0] = (b2 * c0 - b0 * c2) * pv
    r[2, 0] = (b0 * c1 - b1 * c0) * pv
    r[0, 1] = (c1 * a2 - c2 * a1) * pv
    r[1, 1] = (c2 * a0 - c0 * a2) * pv
    r[2, 1] = (c0 * a1 - c1 * a0) * pv
    r[0, 2] = (a1 * b2 - a2 * b1) * pv
    r[1, 2] = (a2 * b0 - a0 * b2) * pv
    r[2, 2] = (a0 * b1 - a1 * b0) * pv
    return r


# Calcite-like cell (nm): hexagonal axes and the equivalent rhombohedral axes.
A_H, C_H = 0.4990, 1.7061
A_R = math.sqrt(3 * A_H ** 2 + C_H ** 2) / 3.0
ALPHA_R = 2 * math.degrees(math.asin(3 * A_H / (2 * math.sqrt(3 * A_H ** 2 + C_H ** 2))))

CASES = [
    ("225", 225, (0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0), False),
    ("194", 194, (0.2950, 0.2950, 0.4683, 90.0, 90.0, 120.0), False),
    ("167-hex", 167, (A_H, A_H, C_H, 90.0, 90.0, 120.0), False),
    ("167-rhomb", 167, (A_R, A_R, A_R, ALPHA_R, ALPHA_R, ALPHA_R), True),
]


def test_rhombohedral_space_group_set():
    assert set(RHOMBOHEDRAL_SPACE_GROUPS) == {146, 148, 155, 160, 161, 166, 167}


@pytest.mark.parametrize("tag,sg,lat,rhomb", CASES, ids=[c[0] for c in CASES])
def test_parity_with_c_formula(tag, sg, lat, rhomb):
    got = reciprocal_matrix(torch.tensor(lat, dtype=torch.float64), sg_num=sg).numpy()
    ref = _c_calc_recip_array(lat, rhomb)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10 * np.abs(ref).max())


def test_167_hex_axes_c_star_is_2pi_over_c():
    B = reciprocal_matrix(torch.tensor(CASES[2][2], dtype=torch.float64), sg_num=167)
    assert float(torch.linalg.norm(B[:, 2])) == pytest.approx(2 * math.pi / C_H, rel=1e-12)


def test_167_rhomb_and_hex_describe_the_same_lattice():
    """Rhombohedral (111) is hexagonal (003): |a*+b*+c*|_R = 3 |c*|_H, and it
    lies along Cartesian [111] (the C embedding's 3-fold)."""
    B_r = reciprocal_matrix(torch.tensor(CASES[3][2], dtype=torch.float64), sg_num=167)
    B_h = reciprocal_matrix(torch.tensor(CASES[2][2], dtype=torch.float64), sg_num=167)
    g111 = B_r.sum(dim=1)
    assert float(torch.linalg.norm(g111)) == pytest.approx(
        3 * float(torch.linalg.norm(B_h[:, 2])), rel=1e-9)
    ones = torch.ones(3, dtype=torch.float64) / math.sqrt(3.0)
    assert float(torch.dot(g111 / torch.linalg.norm(g111), ones)) == pytest.approx(1.0, abs=1e-12)


def test_sg_none_keeps_the_old_formula():
    lat = torch.tensor(CASES[3][2], dtype=torch.float64)
    np.testing.assert_allclose(reciprocal_matrix(lat).numpy(),
                               _c_calc_recip_array(CASES[3][2], False), atol=1e-12)


@pytest.mark.parametrize("lat", [
    (A_H, A_H, C_H, 90.0, 90.0, 90.0),          # tetragonal cell under a trigonal SG
    (A_R, A_R, 1.1 * A_R, ALPHA_R, ALPHA_R, ALPHA_R),
    (A_R, A_R, A_R, 90.0, 90.0, 90.0),          # alpha = 90 is a cube, not rhombohedral
])
def test_other_settings_raise(lat):
    with pytest.raises(ValueError, match="rhombohedral|hexagonal"):
        reciprocal_matrix(torch.tensor(lat, dtype=torch.float64), sg_num=166)


def test_rhomb_branch_is_differentiable():
    lat = torch.tensor(CASES[3][2], dtype=torch.float64, requires_grad=True)
    B = reciprocal_matrix(lat, sg_num=167)
    B.sum().backward()
    assert torch.isfinite(lat.grad).all() and lat.grad.abs().sum() > 0


def test_batched_lattice():
    lat = torch.tensor([CASES[2][2], CASES[2][2]], dtype=torch.float64)
    B = reciprocal_matrix(lat, sg_num=167)
    assert B.shape == (2, 3, 3)


def test_forward_model_threads_sg_num():
    from laue_torch import LaueForwardModel
    m = LaueForwardModel(hkls=torch.tensor([[1, 1, 1]]), n_pix=(64, 64),
                         px_size=(1e-4, 1e-4), sg_num=167)
    assert m.sg_num == 167
    U = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    lat = torch.tensor(CASES[3][2], dtype=torch.float64)
    _, aux = m(U, lat, torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64),
               torch.zeros(3, dtype=torch.float64), return_aux=True)
    expect = float(torch.linalg.norm(reciprocal_matrix(lat, sg_num=167).sum(dim=1)))
    assert float(aux.qlen[0]) == pytest.approx(expect, rel=1e-12)
