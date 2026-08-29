"""The simulator must put spots where it says it does.

`GenerateSimulation.py` used to write each spot with
``self.img[int(py), int(px)] = I`` -- `int()` FLOORS, so every spot landed up to
a pixel low-and-left, a mean 0.42 px in a fixed direction. Nothing caught it:
the C indexer samples intensity at integer pixels and is nearly blind to a
sub-pixel shift, so its answers barely moved. A sub-pixel-sensitive refiner
fitting the same image converged 0.0077 deg from the true orientation, which was
misread as a deficiency of that refiner's forward model until an inverse-crime
control localised it to the generator.

These tests pin the two properties that make the raster trustworthy for
sub-pixel work. They are cheap and need no config file, orientation database or
detector data.
"""
from __future__ import annotations

import numpy as np
import pytest

from laue_index.pipeline import add_to_path

add_to_path()
from GenerateSimulation import DiffractionSimulator  # noqa: E402

SIGMA = 2.0
N = 96


class _Raster(DiffractionSimulator):
    """Only the rasteriser: no config parsing, no diffraction geometry."""

    def __init__(self, n: int = N, sigma: float = SIGMA):
        self.img = np.zeros((n, n), dtype=np.float64)
        self.params = {"gaussWidth": sigma}


def _centroid(img):
    total = img.sum()
    ys, xs = np.indices(img.shape)
    return (ys * img).sum() / total, (xs * img).sum() / total


def _sigma(img, cy, cx):
    total = img.sum()
    ys, xs = np.indices(img.shape)
    vy = ((ys - cy) ** 2 * img).sum() / total
    vx = ((xs - cx) ** 2 * img).sum() / total
    return np.sqrt(0.5 * (vy + vx))


_PHASES = [(fy, fx) for fy in (0.0, 0.13, 0.5, 0.87, 0.99)
           for fx in (0.0, 0.29, 0.5, 0.73, 0.95)]


@pytest.mark.parametrize("fy,fx", _PHASES)
def test_centroid_lands_on_the_float_position(fy, fx):
    """The property the old code violated, at every sub-pixel phase.

    1 px is ~0.015 deg of crystal rotation at a typical Laue geometry, so a
    tolerance of 1e-5 px keeps the induced orientation error below 1e-7 deg --
    four orders under anything measurable.
    """
    r = _Raster()
    py, px = N // 2 + fy, N // 2 + fx
    r.splat_spot(py, px, 10_000.0)
    cy, cx = _centroid(r.img)
    assert np.hypot(cy - py, cx - px) < 1e-5, (
        f"spot asked for ({py}, {px}) rendered at ({cy}, {cx})")


def test_width_does_not_depend_on_sub_pixel_phase():
    """A bilinear splat would pass the centroid test and fail this one.

    Its effective sigma is sqrt(sigma^2 + f(1-f)), up to +3% at a pixel corner --
    a systematic that a width-sensitive fit would silently absorb.
    """
    widths = []
    for fy, fx in ((0.0, 0.0), (0.5, 0.5), (0.5, 0.0), (0.25, 0.75)):
        r = _Raster()
        py, px = N // 2 + fy, N // 2 + fx
        r.splat_spot(py, px, 10_000.0)
        widths.append(_sigma(r.img, *_centroid(r.img)))
    assert max(widths) - min(widths) < 1e-3
    assert abs(np.mean(widths) - SIGMA) < 5e-3


def test_intensity_is_conserved_and_spots_add():
    """The old line ASSIGNED, so one spot could silently erase another."""
    r = _Raster()
    r.splat_spot(48.4, 48.6, 10_000.0)
    assert r.img.sum() == pytest.approx(10_000.0, rel=1e-12)
    r.splat_spot(48.4, 48.6, 10_000.0)
    assert r.img.sum() == pytest.approx(20_000.0, rel=1e-12)


def test_a_spot_at_the_detector_edge_is_clipped_not_wrapped():
    r = _Raster()
    r.splat_spot(1.3, 1.7, 10_000.0)
    assert r.img.sum() > 0
    assert np.isfinite(r.img).all()
    # Nothing may appear at the opposite edge.
    assert r.img[-8:, :].sum() == 0.0 and r.img[:, -8:].sum() == 0.0


def test_degenerate_width_still_places_the_centroid_correctly():
    """gaussWidth 0 takes the bilinear branch; it must not fall back to floor."""
    r = _Raster(sigma=0.0)
    py, px = 48.25, 48.75
    r.splat_spot(py, px, 10_000.0)
    cy, cx = _centroid(r.img)
    assert np.hypot(cy - py, cx - px) < 1e-12
    assert r.img.sum() == pytest.approx(10_000.0, rel=1e-12)
