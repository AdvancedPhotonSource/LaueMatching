"""Continuous-L rod forward model: cross-checked against LaueForwardModel.

The whole point of ``jointfit.fault_rod`` is that its physics must be the
SAME physics as ``LaueForwardModel`` (duplicated for the continuous-L case,
not re-derived) -- otherwise a stacking-fault rod prediction would silently
live in a different convention than every other forward-model consumer in
this package. ``test_fault_rod_matches_forward_model`` is the test that
keeps that promise honest.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.forward import LaueForwardModel
from laue_torch.geometry import rodrigues_to_matrix
from laue_torch.jointfit import (
    RodPoint,
    rod_accessible_mask,
    rod_forward,
    rod_local_jacobian,
)

DTYPE = torch.float64


def _random_setup(seed: int):
    g = torch.Generator().manual_seed(seed)
    rvec = torch.randn(3, generator=g, dtype=DTYPE) * 0.3
    U = rodrigues_to_matrix(rvec)
    lattice = torch.tensor([0.352, 0.352, 0.352, 90.0, 90.0, 90.0], dtype=DTYPE)
    P = torch.tensor([0.01, -0.005, 0.5], dtype=DTYPE)
    R = torch.eye(3, dtype=DTYPE)
    n_pix = (2000, 2000)
    px_size = (1.72e-4, 1.72e-4)
    return U, lattice, P, R, n_pix, px_size


@pytest.mark.parametrize("hkl", [(1, 1, -1), (2, 0, -2), (-1, 3, -2), (2, 2, -4)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fault_rod_matches_forward_model(hkl, seed):
    """At an integer L, rod_forward must reproduce LaueForwardModel exactly.

    hkl signs are chosen so every (hkl, seed) combination is genuinely
    forward-scattering (sin_theta bounded away from 0) at this orientation
    -- checked explicitly below. A degenerate point (sin_theta clamped to
    its floor by both implementations) would make this test pass by both
    sides agreeing on the same meaningless huge "energy" rather than by
    actually cross-validating the physics; this was caught happening for
    10 of 12 combinations under the original (unsigned) hkl choice.
    """
    U, lattice, P, R, n_pix, px_size = _random_setup(seed)
    h, k, l = hkl

    model = LaueForwardModel(
        hkls=torch.tensor([[h, k, l]], dtype=torch.long),
        n_pix=n_pix, px_size=px_size, hard=True, detector_rotation="matrix",
    )
    _, aux = model.forward(
        U=U, lattice=lattice, P=P, R=R,
        E_range=(-1e6, 1e6),  # wide open: masking must not hide a mismatch
        return_aux=True,
    )

    pt = rod_forward(U, lattice, h, k, torch.tensor(float(l), dtype=DTYPE), P, R, n_pix, px_size)

    assert pt.sin_theta.item() > 1e-6, (
        "test point is not genuinely forward-scattering -- this combination "
        "would pass by both implementations agreeing on clamped garbage, "
        "not by validating the physics; pick a different hkl/seed"
    )
    assert aux.energy[0].item() == pytest.approx(pt.energy.item(), rel=1e-9)
    assert aux.px[0].item() == pytest.approx(pt.px.item(), rel=1e-9, abs=1e-6)
    assert aux.py[0].item() == pytest.approx(pt.py.item(), rel=1e-9, abs=1e-6)


def test_rod_forward_batches_over_L():
    """A batch of L values returns a same-shaped batch of every field."""
    # Deliberately forward-scattering by construction, not by luck: with
    # U = I and a cubic lattice, q = (2*pi/a) * (h, k, L), so qhat_z < 0
    # (sin_theta = -qhat_z > 0, the physical condition) whenever L < 0.
    # A RANDOM orientation is not guaranteed to put any given (h, k) row
    # into forward-scattering geometry at all -- most don't -- so this test
    # does not gamble on one.
    lattice = torch.tensor([0.352, 0.352, 0.352, 90.0, 90.0, 90.0], dtype=DTYPE)
    U = torch.eye(3, dtype=DTYPE)
    P = torch.tensor([0.0, 0.0, 0.5], dtype=DTYPE)
    R = torch.eye(3, dtype=DTYPE)
    # A large virtual detector here: this test checks the masking LOGIC
    # (z > 0, energy window), not realistic detector coverage -- at this
    # (h, k) most of this L range scatters at a large angle that would miss
    # any real detector, confirmed by hand before widening n_pix.
    n_pix = (200_000, 200_000)
    px_size = (1.72e-4, 1.72e-4)

    L = torch.linspace(-6.5, -0.5, 13, dtype=DTYPE)
    pt = rod_forward(U, lattice, 1, 1, L, P, R, n_pix, px_size)
    assert isinstance(pt, RodPoint)
    for field in (pt.q, pt.qlen, pt.sin_theta, pt.energy, pt.px, pt.py, pt.z):
        assert field.shape[: L.dim()] == L.shape

    accessible = rod_accessible_mask(pt, n_pix, E_range=(1.0, 200.0))
    assert accessible.any(), "a row constructed to be forward-scattering should have accessible points"
    assert bool((pt.z[accessible] > 0).all())


def test_rod_forward_matches_single_point_slice():
    """Batched and single-L calls must agree pointwise (no hidden broadcasting bug)."""
    # U = I, cubic, negative L: deliberately forward-scattering (see
    # test_rod_forward_batches_over_L for why), not a seed gamble.
    lattice = torch.tensor([0.352, 0.352, 0.352, 90.0, 90.0, 90.0], dtype=DTYPE)
    U = torch.eye(3, dtype=DTYPE)
    P = torch.tensor([0.0, 0.0, 0.5], dtype=DTYPE)
    R = torch.eye(3, dtype=DTYPE)
    n_pix = (2000, 2000)
    px_size = (1.72e-4, 1.72e-4)

    L_batch = torch.tensor([-1.0, -4.0, -7.0], dtype=DTYPE)
    pt_batch = rod_forward(U, lattice, 1, 3, L_batch, P, R, n_pix, px_size)
    for i, l0 in enumerate(L_batch):
        pt_single = rod_forward(U, lattice, 1, 3, l0, P, R, n_pix, px_size)
        assert pt_single.sin_theta.item() > 1e-6, "test point must be genuinely forward-scattering"
        assert pt_batch.energy[i].item() == pytest.approx(pt_single.energy.item(), rel=1e-9)
        assert pt_batch.px[i].item() == pytest.approx(pt_single.px.item(), rel=1e-9, abs=1e-6)


def test_rod_local_jacobian_matches_finite_difference():
    # Same deliberately-forward-scattering construction as
    # test_rod_forward_batches_over_L: a random seed is not guaranteed to
    # put an arbitrary (h, k, L0) into valid geometry (sin_theta > 0), and a
    # degenerate point (sin_theta clamped to its floor) would make both the
    # autodiff and finite-difference derivatives agree on a meaningless flat
    # branch instead of testing anything -- caught by checking sin_theta
    # explicitly below, not just trusting the two derivatives to agree.
    lattice = torch.tensor([0.352, 0.352, 0.352, 90.0, 90.0, 90.0], dtype=DTYPE)
    U = torch.eye(3, dtype=DTYPE)
    P = torch.tensor([0.0, 0.0, 0.5], dtype=DTYPE)
    R = torch.eye(3, dtype=DTYPE)
    n_pix = (2000, 2000)
    px_size = (1.72e-4, 1.72e-4)
    h, k, L0 = 3, 1, -4.3

    sanity = rod_forward(U, lattice, h, k, torch.tensor(L0, dtype=DTYPE), P, R, n_pix, px_size)
    assert sanity.sin_theta.item() > 1e-6, "test point must be genuinely forward-scattering"

    result = rod_local_jacobian(U, lattice, h, k, L0, P, R, n_pix, px_size)

    eps = 1e-5
    plus = rod_forward(U, lattice, h, k, torch.tensor(L0 + eps, dtype=DTYPE), P, R, n_pix, px_size)
    minus = rod_forward(U, lattice, h, k, torch.tensor(L0 - eps, dtype=DTYPE), P, R, n_pix, px_size)
    dE_fd = (plus.energy.item() - minus.energy.item()) / (2 * eps)
    dpx_fd = (plus.px.item() - minus.px.item()) / (2 * eps)
    dpy_fd = (plus.py.item() - minus.py.item()) / (2 * eps)

    assert result["dE_dL"] == pytest.approx(dE_fd, rel=1e-4, abs=1e-6)
    assert result["dpx_dL"] == pytest.approx(dpx_fd, rel=1e-4, abs=1e-3)
    assert result["dpy_dL"] == pytest.approx(dpy_fd, rel=1e-4, abs=1e-3)


def test_rod_accessible_mask_excludes_backscatter_and_off_energy():
    U, lattice, P, R, n_pix, px_size = _random_setup(2)
    L = torch.linspace(-40.0, 40.0, 400, dtype=DTYPE)
    pt = rod_forward(U, lattice, 2, 2, L, P, R, n_pix, px_size)
    mask = rod_accessible_mask(pt, n_pix, E_range=(11.0, 90.0))

    # every masked-in point genuinely satisfies each individual condition
    Nx, Ny = n_pix
    assert bool(((pt.z[mask] > 0)).all())
    assert bool((pt.energy[mask] >= 11.0).all()) and bool((pt.energy[mask] <= 90.0).all())
    assert bool((pt.px[mask] >= 0).all()) and bool((pt.px[mask] <= Nx - 1).all())
    # a wide L sweep should NOT be accessible everywhere (the mask must bind)
    assert not bool(mask.all())


def test_rod_forward_rejects_bad_shapes():
    U, lattice, P, R, n_pix, px_size = _random_setup(0)
    with pytest.raises(ValueError):
        rod_forward(U[:2, :2], lattice, 1, 1, torch.tensor(1.0, dtype=DTYPE), P, R, n_pix, px_size)
    with pytest.raises(ValueError):
        rod_forward(U, lattice[:4], 1, 1, torch.tensor(1.0, dtype=DTYPE), P, R, n_pix, px_size)
