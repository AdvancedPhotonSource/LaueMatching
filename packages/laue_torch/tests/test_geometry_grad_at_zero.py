"""REGRESSION: rodrigues_to_matrix must be differentiable AT the origin.

`rodrigues_to_matrix` used to end in ``torch.where(near_zero, eye, R)``.
``torch.where`` is a hard switch: autograd propagates only through the selected
branch, so at ``rvec = 0`` the derivative came back identically zero instead of
the analytic Rodrigues limit.

Why that matters, concretely. The natural way to refine an orientation is to
compose a delta onto a seed::

    U = rodrigues_to_matrix(dr) @ U0        # dr starts at ZERO

The first gradient is zero, so the optimiser never takes a step and the fit
returns its input while reporting success. Measured 2026-08-28 during an
optimiser comparison: an Adam refinement ran 300 steps and returned the seed
orientation to the last bit, on every seed.

The same defect is recorded in ``midas_grain_odf.odf.axis_angle_to_matrix``;
``midas_pf_odf.inversion._aa_to_R`` carries the smooth formulation this now uses.
"""

from __future__ import annotations

import math

import pytest
import torch

from laue_torch.geometry import rodrigues_to_matrix

DT = torch.float64


def test_gradient_at_exactly_zero_is_the_skew_generator():
    """dR/drvec at the origin is the skew generator: dR[0,1]/drvec[2] = -1."""
    aa = torch.zeros(3, dtype=DT, requires_grad=True)
    g = torch.autograd.grad(rodrigues_to_matrix(aa)[0, 1], aa)[0]
    assert abs(float(g[2]) + 1.0) < 1e-10, f"expected -1, got {g[2]} (full {g})"

    aa2 = torch.zeros(3, dtype=DT, requires_grad=True)
    g2 = torch.autograd.grad(rodrigues_to_matrix(aa2)[1, 0], aa2)[0]
    assert abs(float(g2[2]) - 1.0) < 1e-10, f"expected +1, got {g2[2]}"


@pytest.mark.parametrize("mag", [0.0, 1e-14, 1e-13, 1e-12, 1e-10, 1e-6])
def test_no_dead_zone_near_the_origin(mag):
    """The old implementation was dead for |rvec| < 1e-12 and fine above it."""
    aa = torch.tensor([0.0, 0.0, mag], dtype=DT, requires_grad=True)
    g = torch.autograd.grad(rodrigues_to_matrix(aa)[0, 1], aa)[0]
    assert abs(float(g[2]) + math.cos(mag)) < 1e-9, (
        f"|rvec|={mag:g} gave d/drvec_z = {float(g[2])}, expected {-math.cos(mag)}"
    )


def test_still_a_correct_rotation():
    """Fixing the gradient must not change the VALUE. Checked against scipy."""
    scipy_spatial = pytest.importorskip("scipy.spatial.transform")
    Rotation = scipy_spatial.Rotation
    torch.manual_seed(0)
    for _ in range(8):
        v = torch.randn(3, dtype=DT) * 1.5
        R = rodrigues_to_matrix(v)
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=DT), atol=1e-12)
        assert abs(float(torch.det(R)) - 1.0) < 1e-12
        ref = torch.tensor(Rotation.from_rotvec(v.numpy()).as_matrix(), dtype=DT)
        assert torch.allclose(R, ref, atol=1e-11)


def test_identity_at_zero():
    R = rodrigues_to_matrix(torch.zeros(3, dtype=DT))
    assert torch.allclose(R, torch.eye(3, dtype=DT), atol=1e-15)


def test_gradcheck_at_the_origin():
    """The strict check, at the point that used to fail."""
    aa = torch.zeros(3, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(rodrigues_to_matrix, (aa,), eps=1e-6,
                                    atol=1e-8, rtol=1e-6)


def test_a_delta_composed_onto_a_seed_can_actually_move():
    """The end-to-end shape of the failure: refine by composing a delta.

    Without a gradient at the origin this loop is a no-op and the 'refined'
    orientation equals the seed exactly.
    """
    torch.manual_seed(3)
    q = torch.randn(4, dtype=DT); q = q / q.norm()
    w, x, y, z = q
    U0 = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=DT)
    target = rodrigues_to_matrix(torch.tensor([0.01, -0.02, 0.03], dtype=DT)) @ U0

    dr = torch.zeros(3, dtype=DT, requires_grad=True)
    opt = torch.optim.Adam([dr], lr=5e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = ((rodrigues_to_matrix(dr) @ U0 - target) ** 2).sum()
        loss.backward()
        opt.step()

    assert float(dr.abs().sum()) > 1e-6, "optimiser never moved off the origin"
    assert float(loss) < 1e-12, f"did not converge, loss={float(loss):.3e}"
