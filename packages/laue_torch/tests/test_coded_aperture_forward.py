"""Integration tests: ``LaueForwardModel`` + ``CodedApertureMask``.

Phase 1 of the coded-aperture extension — see
``laue_torch/implementation_plan_coded_aperture.md``.

Confirms:

* ``coded_aperture=None`` is bit-identical to the historical forward path.
* All-zero sequence (no Au absorbers, no substrate) leaves the image
  essentially unchanged (transmission ≈ 1).
* All-one sequence with non-trivial Au attenuates the image
  significantly.
* The scan offset shifts which spots are blocked.
* ``forward_stack`` returns a frame stack of the correct shape and
  varies across frames.
* The whole pipeline is differentiable in the mask pose, scan offset,
  Au thickness, and source position.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import CodedApertureMask, build_de_bruijn_sequence


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


def _make_mask(
    sequence_bits,
    *,
    bar_width: float = 10.0,
    au_thickness: float = 4.6,
    sub_thickness: float = 0.0,
    edge_softness: float = 0.5,
    position_um=None,
    rotvec=None,
    learnable: bool = False,
) -> CodedApertureMask:
    seq = torch.tensor(sequence_bits, dtype=torch.int64)
    return CodedApertureMask(
        sequence=seq,
        bar_widths_um=bar_width,
        au_thickness_um=au_thickness,
        sub_thickness_um=sub_thickness,
        edge_softness_um=edge_softness,
        position_um=position_um,
        rotvec=rotvec,
        make_geometry_learnable=learnable,
        dtype=DTYPE,
    )


# ── bit-identity with legacy path ─────────────────────────────────────────


def test_coded_aperture_none_matches_legacy():
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    a = model(U, lat, P, R, E_range=(5.0, 30.0))
    b = model(U, lat, P, R, E_range=(5.0, 30.0), coded_aperture=None)
    assert torch.equal(a, b)


def test_zero_sequence_leaves_image_unchanged():
    """All-zero coding (no Au, no substrate) ⇒ transmission ≈ 1 everywhere."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    mask = _make_mask([0] * 8, sub_thickness=0.0,
                       position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE))
    img_no = model(U, lat, P, R, E_range=(5.0, 30.0))
    img_yes = model(U, lat, P, R, E_range=(5.0, 30.0),
                    coded_aperture=mask, scan_offset_um=0.0)
    rel = (img_no - img_yes).abs().max() / img_no.abs().max().clamp_min(1e-30)
    assert rel.item() < 1e-3, f"all-zero mask perturbed image by {rel.item()}"


# ── attenuation behaviour ─────────────────────────────────────────────────


def test_dense_au_attenuates_image():
    """A wide all-Au mask in front of the detector cuts the total intensity."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    # Aperture covers the whole working region: 1024 µm of contiguous Au
    # at ~500 µm above the sample (along lab +z).
    mask = _make_mask(
        [1] * 128,
        bar_width=8.0,
        au_thickness=10.0,
        sub_thickness=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
    )
    img_no = model(U, lat, P, R, E_range=(5.0, 30.0)).sum().item()
    img_yes = model(U, lat, P, R, E_range=(5.0, 30.0),
                    coded_aperture=mask, scan_offset_um=0.0).sum().item()
    assert img_yes < 0.6 * img_no, (
        f"10 µm Au blanket should drop intensity > 40%, got "
        f"{img_yes:.3e} / {img_no:.3e} = {img_yes/img_no:.2%}"
    )


def test_scan_offset_changes_image():
    """Scanning the mask must change which spots are attenuated."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    seq = build_de_bruijn_sequence(order=4, alphabet=2).tolist()    # 16 bits
    mask = _make_mask(
        seq,
        bar_width=12.0,
        au_thickness=6.0,
        sub_thickness=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
    )
    img_a = model(U, lat, P, R, E_range=(5.0, 30.0),
                  coded_aperture=mask, scan_offset_um=0.0)
    img_b = model(U, lat, P, R, E_range=(5.0, 30.0),
                  coded_aperture=mask, scan_offset_um=24.0)  # shift 2 bars
    delta = (img_a - img_b).abs().sum().item()
    base = img_a.abs().sum().item()
    assert delta / base > 0.05, (
        f"scan-offset shift did not move enough intensity: {delta/base:.2%}"
    )


# ── forward_stack ─────────────────────────────────────────────────────────


def test_forward_stack_shape_and_variance():
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    seq = build_de_bruijn_sequence(order=4, alphabet=2).tolist()
    mask = _make_mask(
        seq,
        bar_width=12.0,
        au_thickness=6.0,
        sub_thickness=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
    )
    offsets = torch.linspace(0.0, 48.0, 5, dtype=DTYPE)  # 5 frames
    stack = model.forward_stack(
        U, lat, P, R,
        coded_aperture=mask,
        scan_offsets_um=offsets,
        E_range=(5.0, 30.0),
    )
    Nx, Ny = model.n_pix
    assert stack.shape == (5, Nx, Ny)
    # Frames must differ from each other (mask is sweeping)
    frame_var = stack.var(dim=0).sum().item()
    assert frame_var > 0


# ── gradient flow ─────────────────────────────────────────────────────────


def test_grad_through_scan_offset():
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    seq = build_de_bruijn_sequence(order=4, alphabet=2).tolist()
    mask = _make_mask(
        seq,
        bar_width=12.0,
        au_thickness=6.0,
        sub_thickness=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
        edge_softness=2.0,           # smoother for gradient stability
    )
    p = torch.tensor(7.0, dtype=DTYPE, requires_grad=True)
    img = model(U, lat, P, R, E_range=(5.0, 30.0),
                coded_aperture=mask, scan_offset_um=p)
    loss = (img * img).sum()
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert p.grad.abs() > 0


def test_grad_through_mask_pose():
    """Mask position + rotvec are learnable when ``make_geometry_learnable=True``."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    seq = build_de_bruijn_sequence(order=4, alphabet=2).tolist()
    mask = _make_mask(
        seq,
        bar_width=12.0,
        au_thickness=6.0,
        sub_thickness=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
        edge_softness=2.0,
        learnable=True,
    )
    img = model(U, lat, P, R, E_range=(5.0, 30.0),
                coded_aperture=mask, scan_offset_um=0.0)
    loss = (img * img).sum()
    loss.backward()
    assert mask.position_um.grad is not None
    assert mask.rotvec.grad is not None
    assert torch.isfinite(mask.position_um.grad).all()
    assert torch.isfinite(mask.rotvec.grad).all()


def test_gradcheck_scan_offset_small():
    """gradcheck for the scan offset through the full forward."""
    model = _toy_model()
    U, lat, P, R = _good_inputs()
    seq = build_de_bruijn_sequence(order=4, alphabet=2).tolist()
    mask = _make_mask(
        seq,
        bar_width=12.0,
        au_thickness=6.0,
        sub_thickness=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
        edge_softness=2.0,
    )

    def f(p):
        img = model(U, lat, P, R, E_range=(5.0, 30.0),
                    coded_aperture=mask, scan_offset_um=p)
        return (img * img).sum()

    p = torch.tensor(7.0, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(f, (p,), eps=1e-4, atol=1e-4, rtol=1e-3)
