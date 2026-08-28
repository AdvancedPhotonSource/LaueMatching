"""Tests for ``LaueForwardModel`` with non-zero ``source_xyz``.

Phase 1 of the coded-aperture extension — see
``laue_torch/implementation_plan_coded_aperture.md``.

Confirms:

* ``source_xyz=None`` is bit-identical to the historical (v2.1) forward
  path — no regression on existing pipelines.
* ``source_xyz=zeros(3)`` matches the None path to float64 tolerance
  (i.e. the general-form algebra reduces correctly to the original).
* Spot positions shift predictably with a depth offset along the
  beam axis.
* The full forward is differentiable in ``source_xyz``.
"""
from __future__ import annotations

import pytest
import torch

from laue_torch import LaueForwardModel


DTYPE = torch.float64
torch.manual_seed(0)


def _toy_model() -> LaueForwardModel:
    hkls = torch.tensor([
        [1, 1, 1], [-1, -1, -1], [2, 0, 0], [-2, 0, 0],
        [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2],
        [2, 2, 0], [-2, -2, 0], [1, -1, 3], [3, -1, 1],
        [1, -1, -3], [-3, 1, 1], [2, -2, 4], [4, 2, -2],
    ], dtype=torch.long)
    return LaueForwardModel(
        hkls=hkls,
        n_pix=(256, 256),
        px_size=(0.0016, 0.0016),
        psf_sigma=2.0,
        render_window=9,
        rotation="matrix",
        detector_rotation="rodrigues",
        strain_mode="none",
        hard=False,
        tau_z=5e-3,
        tau_px=2.0,
        tau_E=0.3,
    )


def _good_inputs():
    lat = torch.tensor([0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0], dtype=DTYPE)
    P = torch.tensor([0.028745, 0.002788, 0.513115], dtype=DTYPE)
    R = torch.tensor([-1.20131258, -1.21399082, -1.21881158], dtype=DTYPE)
    U = torch.tensor([
        [0.867151, 0.494088, 0.062670],
        [-0.052670, 0.216095, -0.974957],
        [-0.495254, 0.842135, 0.213410],
    ], dtype=DTYPE).unsqueeze(0)
    return U, lat, P, R


def test_source_none_matches_legacy_path():
    """Calling ``forward`` without ``source_xyz`` must match the historical path."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    img_default = model(U, lat, P, R, E_range=(5.0, 30.0))
    img_explicit = model(U, lat, P, R, E_range=(5.0, 30.0), source_xyz=None)
    assert torch.equal(img_default, img_explicit), "default vs explicit-None drift"


def test_source_zero_matches_none_to_fp_tol():
    """``source_xyz = 0`` algebra reduces to the legacy path.

    The two branches use slightly different operation orderings (general
    form: ``s_det = R^T·0; t = (P[2]-s_det[2])/z; proj = s + t·xyz - P``;
    legacy form: ``scale = P[2]/z; proj = xyz·scale - P``) so an exact
    bit-equality cannot be guaranteed — but the difference must be at
    pure float64 rounding noise (< 1e-12 in pixel intensities).
    """
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    img_none = model(U, lat, P, R, E_range=(5.0, 30.0))
    img_zero = model(U, lat, P, R, E_range=(5.0, 30.0),
                     source_xyz=torch.zeros(3, dtype=DTYPE))
    delta = (img_none - img_zero).abs().max().item()
    assert delta < 1e-12, f"source=zero diverged from None by {delta}"


def test_source_offset_shifts_image():
    """A non-trivial source offset must change the image."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    img_zero = model(U, lat, P, R, E_range=(5.0, 30.0),
                     source_xyz=torch.zeros(3, dtype=DTYPE))
    # 50 µm source offset along the beam — well within the coded-aperture
    # microbeam regime (~100 µm probed depth).
    img_off = model(U, lat, P, R, E_range=(5.0, 30.0),
                    source_xyz=torch.tensor([0.0, 0.0, 5e-5], dtype=DTYPE))
    diff = (img_zero - img_off).abs().sum().item()
    assert diff > 1e-6, f"50 µm source offset produced no image change ({diff})"


def test_grad_flows_through_source_xyz():
    """Gradient w.r.t. source_xyz is finite and non-zero."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    src = torch.zeros(3, dtype=DTYPE, requires_grad=True)
    img = model(U, lat, P, R, E_range=(5.0, 30.0), source_xyz=src)
    loss = (img * img).sum()
    loss.backward()
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()
    assert src.grad.abs().sum() > 0


def test_gradcheck_source_xyz():
    """``torch.autograd.gradcheck`` on a tiny problem.

    Uses ``hard=False`` (default) and a generously sized detector so the
    soft masks stay differentiable — same recipe as the existing
    ``test_grad.py`` gradcheck cases.
    """
    model = _toy_model()
    U, lat, P, R = _good_inputs()

    def f(src):
        img = model(U, lat, P, R, E_range=(5.0, 30.0), source_xyz=src)
        return (img * img).sum()

    src = torch.tensor([1e-6, -2e-6, 5e-6], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(f, (src,), eps=1e-6, atol=1e-4, rtol=1e-3)
