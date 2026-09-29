"""laue_jax.geometry: smooth Rodrigues at 0, and the C2 lattice-setting rule.

No repo checkout needed (unlike the torch parity gate).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from laue_jax.geometry import RHOMBOHEDRAL_SPACE_GROUPS, reciprocal_matrix, rodrigues_to_matrix

DEG = math.pi / 180.0


# ── Rodrigues at the origin ────────────────────────────────────────────────

def test_rodrigues_grad_at_zero_is_the_analytic_limit():
    """dR[0,1]/d rvec at rvec = 0 is (0, 0, -1). The old jnp.where switch
    returned NaN (0/0 through the unselected branch)."""
    g = jax.grad(lambda r: rodrigues_to_matrix(r)[0, 1])(jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(g), [0.0, 0.0, -1.0], atol=1e-12)


def test_rodrigues_jacfwd_and_float32_finite_at_zero():
    J = jax.jacfwd(rodrigues_to_matrix)(jnp.zeros(3))
    assert np.all(np.isfinite(np.asarray(J)))
    # dR/dr_k at 0 is the generator [e_k]x.
    np.testing.assert_allclose(np.asarray(J[1, 0, 2]), 1.0, atol=1e-12)
    grad32 = jax.grad(lambda r: rodrigues_to_matrix(r).sum())(jnp.zeros(3, dtype=jnp.float32))
    assert np.all(np.isfinite(np.asarray(grad32)))
    J32 = jax.jacfwd(rodrigues_to_matrix)(jnp.zeros(3, dtype=jnp.float32))
    assert np.all(np.isfinite(np.asarray(J32)))


def test_rodrigues_values_unchanged_away_from_zero():
    r = jnp.array([0.3, -0.7, 1.1])
    th = float(jnp.linalg.norm(r))
    k = np.asarray(r) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ref = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    np.testing.assert_allclose(np.asarray(rodrigues_to_matrix(r)), ref, atol=1e-13)
    np.testing.assert_allclose(np.asarray(rodrigues_to_matrix(jnp.zeros(3))), np.eye(3), atol=0)


# ── C2: lattice setting ────────────────────────────────────────────────────

def _c_calc_recip_array(lat, rhomb: bool) -> np.ndarray:
    """numpy port of C calcRecipArray (LaueMatchingHeaders.h), branch explicit."""
    a, b, c, alpha, beta, gamma = lat
    ca, cb, cg = math.cos(alpha * DEG), math.cos(beta * DEG), math.cos(gamma * DEG)
    sg = math.sin(gamma * DEG)
    phi = math.sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    pv = 2 * math.pi / (a * b * c * phi)
    if not rhomb:
        a0, a1, a2 = a, 0.0, 0.0
        b0, b1, b2 = b * cg, b * sg, 0.0
        c0, c1, c2 = c * cb, c * (ca - cb * cg) / sg, c * phi / sg
    else:
        p, q = math.sqrt(1.0 + 2 * ca), math.sqrt(1.0 - ca)
        pmq, p2q = (a / 3.0) * (p - q), (a / 3.0) * (p + 2 * q)
        a0, a1, a2 = p2q, pmq, pmq
        b0, b1, b2 = pmq, p2q, pmq
        c0, c1, c2 = pmq, pmq, p2q
    return np.array([
        [(b1 * c2 - b2 * c1), (c1 * a2 - c2 * a1), (a1 * b2 - a2 * b1)],
        [(b2 * c0 - b0 * c2), (c2 * a0 - c0 * a2), (a2 * b0 - a0 * b2)],
        [(b0 * c1 - b1 * c0), (c0 * a1 - c1 * a0), (a0 * b1 - a1 * b0)],
    ]) * pv


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
    got = np.asarray(reciprocal_matrix(jnp.asarray(lat), sg_num=sg))
    ref = _c_calc_recip_array(lat, rhomb)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10 * np.abs(ref).max())


def test_sg_none_keeps_the_old_formula():
    got = np.asarray(reciprocal_matrix(jnp.asarray(CASES[3][2])))
    np.testing.assert_allclose(got, _c_calc_recip_array(CASES[3][2], False), atol=1e-12)


def test_other_settings_raise():
    with pytest.raises(ValueError, match="rhombohedral|hexagonal"):
        reciprocal_matrix(jnp.asarray((A_H, A_H, C_H, 90.0, 90.0, 90.0)), sg_num=167)


def test_matches_torch_if_available():
    torch = pytest.importorskip("torch")
    from laue_torch.geometry import reciprocal_matrix as rm_t
    for _, sg, lat, _ in CASES:
        a = rm_t(torch.tensor(lat, dtype=torch.float64), sg_num=sg).numpy()
        b = np.asarray(reciprocal_matrix(jnp.asarray(lat), sg_num=sg))
        np.testing.assert_allclose(a, b, atol=1e-13)


def test_pseudo_voigt_splat_matches_torch_with_taper():
    torch = pytest.importorskip("torch")
    from laue_torch.rasterize import pseudo_voigt_splat as pv_t
    from laue_jax.rasterize import pseudo_voigt_splat as pv_j
    px = np.array([20.3, 31.49]); py = np.array([18.7, 25.1]); it = np.array([1.0, 0.5])
    for eta, window in ((0.5, 13), (0.5, 7), (1.0, 17), (0.0, 13)):
        a = pv_t(torch.tensor(px), torch.tensor(py), torch.tensor(it), n_pix=(48, 40),
                 sigma=1.0, window=window, eta=eta).numpy()
        b = np.asarray(pv_j(jnp.asarray(px), jnp.asarray(py), jnp.asarray(it),
                            (48, 40), 1.0, window, eta=eta))
        np.testing.assert_allclose(a, b, atol=1e-13)
