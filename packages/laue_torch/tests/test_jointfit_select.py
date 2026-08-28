"""Grain-count model selection.

The peel answers "how many grains?" by accretion and never asks whether a grain
is worth its parameters.  These tests pin the replacement: recover exactly the
true grain count from a padded model, in both decoy regimes.

Two regimes matter and they exercise different machinery:
  * RANDOM decoys earn zero amplitude -> removed as INACTIVE (definitional);
  * NEAR-DUPLICATE decoys under noise DO earn amplitude -> must be removed on
    BIC grounds, which is the real test of the criterion.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.jointfit import (
    JointGrainFit,
    make_scene,
    model_bic,
    prune_grains,
    random_orientations,
)
from laue_torch.jointfit.synthetic import perturb_orientations


def _scene(noise=0.0):
    return make_scene(
        n_grains=3, n_reflections=8, n_pix=(256, 256), noise=noise, seed=1,
        overlap_pixels=[(90.0, 110.0), (170.0, 60.0)],
    )


def _model(scene, oms, axes):
    return JointGrainFit(
        oms.clone(), scene.project_fn, psf_sigma=scene.psf_sigma,
        sigma_par=scene.sigma_par, sigma_perp=scene.sigma_perp,
        axis_init=axes.clone(),
    )


def _true_model(scene):
    return _model(scene, scene.orientations, scene.axes)


def _padded(scene, decoys, decoy_axes=None):
    oms = torch.cat([scene.orientations, decoys], dim=0)
    ax = torch.cat(
        [scene.axes,
         scene.axes[:1].expand(decoys.shape[0], 3) if decoy_axes is None else decoy_axes],
        dim=0,
    )
    return _model(scene, oms, ax)


# ── BIC ingredients ────────────────────────────────────────────────────────


def test_bic_report_is_self_consistent():
    sc = _scene()
    rois, imgs = [sc.full_roi()], [sc.image]
    rep = model_bic(_true_model(sc), rois, imgs)
    assert rep.n_pixels == sc.full_roi().n_pixels
    assert rep.n_active_grains == 3
    assert bool(rep.active.all())
    assert rep.n_effective_params > 3 * 6      # amplitudes on top of 6/grain
    assert rep.rss >= 0 and math.isfinite(rep.bic)


def test_bic_prefers_the_truth_over_a_missing_grain():
    sc = _scene()
    rois, imgs = [sc.full_roi()], [sc.image]
    full = model_bic(_true_model(sc), rois, imgs).bic
    missing = model_bic(_model(sc, sc.orientations[:2], sc.axes[:2]), rois, imgs).bic
    assert full < missing, f"BIC failed to require the 3rd grain: {full} vs {missing}"


def test_inactive_grains_cost_no_parameters():
    """A zero-amplitude grain is unidentifiable, so it must not be charged.

    If it were charged, BIC would look decisive when it was only counting.
    """
    sc = _scene()
    rois, imgs = [sc.full_roi()], [sc.image]
    truth = model_bic(_true_model(sc), rois, imgs)
    padded = model_bic(
        _padded(sc, random_orientations(3, seed=99, dtype=sc.orientations.dtype)),
        rois, imgs,
    )
    assert padded.n_active_grains == 3
    assert padded.n_effective_params == truth.n_effective_params
    assert padded.bic == pytest.approx(truth.bic, rel=1e-12)


# ── pruning: both decoy regimes ────────────────────────────────────────────


@pytest.mark.parametrize("noise", [0.0, 0.02, 0.1])
def test_prune_removes_random_decoys(noise):
    sc = _scene(noise)
    decoys = random_orientations(3, seed=99, dtype=sc.orientations.dtype)
    sel = prune_grains(_padded(sc, decoys), [sc.full_roi()], [sc.image])
    assert sel.keep == [0, 1, 2], f"kept {sel.keep}"
    assert sorted(sel.inactive) == [3, 4, 5], (
        "random decoys should be removed as inactive, not on BIC grounds"
    )


@pytest.mark.parametrize("noise", [0.02, 0.1])
def test_prune_removes_near_duplicate_decoys_on_bic_grounds(noise):
    """The hard case: decoys that DO carry amplitude.

    Under noise a 1 deg rotated copy of each true grain absorbs some residual,
    so stage 1 cannot touch it and BIC has to do the work.
    """
    sc = _scene(noise)
    decoys = perturb_orientations(sc.orientations, angle_rad=math.pi / 180, seed=7)
    sel = prune_grains(
        _padded(sc, decoys, decoy_axes=sc.axes), [sc.full_roi()], [sc.image]
    )
    assert sel.keep == [0, 1, 2], f"kept {sel.keep}"
    assert sorted(sel.dropped) == [3, 4, 5], (
        f"near-duplicates must be dropped by BIC, got inactive={sel.inactive} "
        f"dropped={sel.dropped}"
    )
    assert sel.bic <= sel.start_bic


def test_prune_keeps_every_real_grain():
    """The failure that would matter most: deleting a true grain."""
    sc = _scene(0.02)
    sel = prune_grains(_true_model(sc), [sc.full_roi()], [sc.image])
    assert sel.keep == [0, 1, 2]
    assert sel.dropped == [] and sel.inactive == []


def test_prune_respects_max_drops():
    sc = _scene(0.02)
    decoys = perturb_orientations(sc.orientations, angle_rad=math.pi / 180, seed=7)
    sel = prune_grains(
        _padded(sc, decoys, decoy_axes=sc.axes), [sc.full_roi()], [sc.image],
        max_drops=1,
    )
    assert len(sel.dropped) <= 1


def test_prune_never_empties_the_model():
    """Even fed nothing but decoys, selection must not return an empty list."""
    sc = _scene()
    rand = random_orientations(3, seed=555, dtype=sc.orientations.dtype)
    m = _model(sc, rand, sc.axes)
    sel = prune_grains(m, [sc.full_roi()], [sc.image])
    assert len(sel.keep) >= 1
