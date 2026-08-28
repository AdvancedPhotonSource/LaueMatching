"""Recovery on synthetic ground truth -- the gate before any real data.

The Ti-64 investigation produced several confident results that dissolved under
a proper control, so the claims this package will eventually make about real
data are stated here as tests against fields we built:

  1. at the true parameters the residual is ~0;
  2. from perturbed seeds the fit moves TOWARD the truth;
  3. a SPURIOUS grain gets amplitude ~0 instead of being credited with a
     fragment -- the specific failure that inflated the peel's grain support;
  4. random orientations do NOT fit as well as the true ones, i.e. the free
     per-reflection amplitudes are not powerful enough to fit anything.

(4) is the null for this whole approach.  Free amplitudes make the model very
flexible, and without this test a good-looking fit on real data would be
uninterpretable.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.jointfit import (
    JointGrainFit,
    fit_grains,
    make_scene,
    random_orientations,
)
from laue_torch.jointfit.synthetic import perturb_orientations


def _model(scene, orientations=None, **kw):
    return JointGrainFit(
        (scene.orientations if orientations is None else orientations).clone(),
        scene.project_fn,
        psf_sigma=scene.psf_sigma,
        sigma_par=scene.sigma_par,
        sigma_perp=scene.sigma_perp,
        axis_init=kw.pop("axis_init", None),
        **kw,
    )


def _misorientation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation angle between two orientation matrices, radians (no symmetry)."""
    r = a @ b.transpose(-1, -2)
    tr = r.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))


@pytest.fixture(scope="module")
def scene():
    # Includes an overlap column: every grain gets a reflection at the same two
    # detector positions, so those clouds have several true contributors.
    return make_scene(
        n_grains=3, n_reflections=8, n_pix=(256, 256), noise=0.0, seed=1,
        overlap_pixels=[(90.0, 110.0), (170.0, 60.0)],
    )


# ── the scene itself is what we think it is ────────────────────────────────


def test_scene_renders_streaks_not_blobs(scene):
    assert scene.amplitudes.numel() > 0
    assert float(scene.image.max()) > 0
    # Ti-64-like footprint: elongated, ~19 px along the streak.
    from laue_torch.jointfit import pixel_covariance, pixel_jacobian, spread_covariance
    jac = torch.nan_to_num(pixel_jacobian(scene.project_fn, scene.orientations[0]),
                           nan=0.0)
    cov = pixel_covariance(jac, spread_covariance(scene.sigma_par, scene.sigma_perp,
                                                  scene.axes[0]), scene.psf_sigma)
    sig = torch.linalg.eigvalsh(cov).clamp_min(0).sqrt()
    ratio = (sig[:, 1] / sig[:, 0]).max()
    assert float(ratio) > 3.0, f"footprints are too round: {ratio}"


def test_overlap_pixels_are_shared_by_all_grains(scene):
    """The hard configuration is actually present, not just requested."""
    hits = 0
    for i in range(scene.n_grains):
        pix = scene.project_fn(scene.orientations[i])
        d = torch.linalg.norm(
            pix - torch.tensor([90.0, 110.0], dtype=pix.dtype), dim=-1
        )
        hits += int(bool((d[torch.isfinite(d)] < 2.0).any()))
    assert hits == scene.n_grains, f"only {hits} grains reach the shared cloud"


# ── 1. residual vanishes at the truth ──────────────────────────────────────


def test_residual_is_negligible_at_truth(scene):
    m = _model(scene, axis_init=scene.axes.clone())
    loss, info = m.loss([scene.full_roi()], [scene.image])
    signal = float((scene.image ** 2).sum())
    assert float(loss) / signal < 1e-10, f"loss {float(loss)} vs signal {signal}"
    assert info["unconverged_solves"] == 0


def test_amplitudes_recovered_at_truth(scene):
    """Per-grain amplitude must match the ground-truth total for that grain."""
    m = _model(scene, axis_init=scene.axes.clone())
    got = m.grain_amplitudes([scene.full_roi()], [scene.image])
    assert got.shape == (scene.n_grains,)
    assert bool((got > 0).all()), f"a real grain got zero amplitude: {got}"
    assert float(got.sum()) == pytest.approx(float(scene.amplitudes.sum()), rel=0.05)


# ── 2. the fit moves toward the truth from a perturbed seed ────────────────


def test_fit_reduces_loss_and_improves_orientation(scene):
    start = perturb_orientations(scene.orientations, angle_rad=0.004, seed=5)
    m = _model(scene, orientations=start, axis_init=scene.axes.clone(),
               fit_spread=False)
    rois, imgs = [scene.full_roi()], [scene.image]

    before = float(m.loss(rois, imgs)[0])
    err_before = _misorientation(m.orientations().detach(), scene.orientations)
    report = fit_grains(m, rois, imgs, n_iter=60, lr=2e-3)
    err_after = _misorientation(m.orientations().detach(), scene.orientations)

    assert report.loss < before, f"loss rose: {before} -> {report.loss}"
    assert report.loss < 0.25 * before, (
        f"loss barely moved: {before:.4e} -> {report.loss:.4e}"
    )
    assert float(err_after.mean()) < float(err_before.mean()), (
        f"orientation error grew: {err_before.mean()} -> {err_after.mean()}"
    )
    assert report.unconverged_solves == 0


# ── 3. a spurious grain is not credited ────────────────────────────────────


def _with_decoys(scene, decoys):
    """Model with the true grains (TRUE spread axes) plus decoy grains.

    Passing the real grains' true axes matters: with the default axis every
    grain renders the wrong footprint, the fit is bad for everyone, and a decoy
    test then passes for the wrong reason.
    """
    combined = torch.cat([scene.orientations, decoys], dim=0)
    axes = torch.cat([scene.axes, scene.axes[:1].expand(decoys.shape[0], 3)], dim=0)
    return _model(scene, orientations=combined, axis_init=axes.clone())


def test_random_spurious_grain_gets_exactly_zero_amplitude(scene):
    """The core claim: an unnecessary grain is not credited with a fragment.

    Under the greedy peel a random extra orientation picks up unmasked
    fragments of a big cloud and looks supported.  Here it competes against the
    grains that actually produced the intensity, and non-negativity denies it a
    negative correction elsewhere, so it gets nothing at all.
    """
    extra = random_orientations(2, seed=4242, dtype=scene.orientations.dtype)
    amps = _with_decoys(scene, extra).grain_amplitudes(
        [scene.full_roi()], [scene.image]
    )
    real, fake = amps[: scene.n_grains], amps[scene.n_grains:]
    assert float(real.min()) > 0, f"a real grain got nothing: {real}"
    assert float(fake.max()) == 0.0, (
        f"spurious grains were credited: real={real.tolist()} fake={fake.tolist()}"
    )


@pytest.mark.parametrize("deg", [0.25, 1.0, 4.0])
def test_near_duplicate_decoy_is_not_credited(scene, deg):
    """The HARD decoy: a slightly rotated copy of a real grain.

    Random orientations are an easy null -- their spots land nowhere near real
    intensity.  The failure mode that actually inflated the peel is a decoy
    sitting ON a real cloud, so this rotates a true grain by a fraction of the
    streak width (0.25 deg is a 2.2 px shift inside a ~19 px streak) and checks
    it still earns nothing.  This is the ghost-decoy lesson from the Si twin:
    a null built only from random orientations proves much less than it seems.
    """
    decoy = perturb_orientations(
        scene.orientations[:1], angle_rad=deg * math.pi / 180, seed=7
    )
    amps = _with_decoys(scene, decoy).grain_amplitudes(
        [scene.full_roi()], [scene.image]
    )
    real, fake = amps[: scene.n_grains], amps[scene.n_grains]
    assert float(fake) < 1e-3 * float(real.min()), (
        f"{deg} deg decoy was credited {float(fake)} against real {real.tolist()}"
    )


@pytest.mark.parametrize("noise,max_ratio", [(0.02, 0.005), (0.1, 0.02), (0.5, 0.05)])
def test_decoy_credit_stays_small_under_noise(noise, max_ratio):
    """With noise a near-duplicate absorbs some residual -- bound how much.

    Measured: the decoy takes 0.07% of a real grain's amplitude at noise 0.02,
    0.36% at 0.1, and 1.8% at 0.5 (where the residual is already 18% of signal).
    It scales with noise, as it must, but never approaches being credited as a
    grain.  Random decoys stay at exactly zero throughout.
    """
    sc = make_scene(
        n_grains=3, n_reflections=8, n_pix=(256, 256), noise=noise, seed=1,
        overlap_pixels=[(90.0, 110.0), (170.0, 60.0)],
    )
    decoy = perturb_orientations(sc.orientations[:1], angle_rad=math.pi / 180, seed=7)
    amps = _with_decoys(sc, decoy).grain_amplitudes([sc.full_roi()], [sc.image])
    ratio = float(amps[3]) / float(amps[:3].min())
    assert ratio < max_ratio, f"decoy took {ratio:.4f} of a real grain at noise {noise}"

    rand = random_orientations(1, seed=4242, dtype=sc.orientations.dtype)
    r_amps = _with_decoys(sc, rand).grain_amplitudes([sc.full_roi()], [sc.image])
    assert float(r_amps[3]) == 0.0, "random decoy picked up noise"


def test_adding_spurious_grains_barely_improves_the_fit(scene):
    """Extra grains must not buy a materially better residual.

    If they did, the free amplitudes would be flexible enough to reward adding
    grains indefinitely and model selection could never work.
    """
    rois, imgs = [scene.full_roi()], [scene.image]
    base = float(_model(scene, axis_init=scene.axes.clone()).loss(rois, imgs)[0])

    extra = random_orientations(3, seed=99, dtype=scene.orientations.dtype)
    padded = float(_with_decoys(scene, extra).loss(rois, imgs)[0])

    signal = float((scene.image ** 2).sum())
    assert (base - padded) / signal < 1e-9, (
        f"3 spurious grains improved the fit by {(base - padded) / signal:.2e} "
        "of signal energy"
    )


# ── 4. the null: random orientations must NOT fit ──────────────────────────


def test_random_orientations_fit_far_worse_than_the_truth(scene):
    """The null for the whole approach.

    Free per-reflection amplitudes make the model flexible; this checks the
    flexibility is not unlimited.  A random-orientation fit must leave most of
    the signal unexplained.
    """
    rois, imgs = [scene.full_roi()], [scene.image]
    signal = float((scene.image ** 2).sum())
    true_loss = float(_model(scene, axis_init=scene.axes.clone()).loss(rois, imgs)[0])

    worst = 0.0
    for s in range(5):
        rand = random_orientations(scene.n_grains, seed=1000 + s,
                                   dtype=scene.orientations.dtype)
        loss = float(_model(scene, orientations=rand).loss(rois, imgs)[0])
        worst = max(worst, loss / signal)
        assert loss > 100 * max(true_loss, 1e-12), (
            f"random orientations fit nearly as well as the truth "
            f"(seed {s}): {loss:.3e} vs {true_loss:.3e}"
        )
    # And they leave most of the signal on the table in absolute terms.
    assert worst > 0.5, f"random fits explained too much: residual {worst:.2f} of signal"


def test_noise_floor_is_respected(scene):
    """With noise, the residual at truth should sit near the noise energy."""
    noisy = make_scene(
        n_grains=3, n_reflections=8, n_pix=(256, 256), noise=0.02, seed=1,
        overlap_pixels=[(90.0, 110.0), (170.0, 60.0)],
    )
    m = _model(noisy, axis_init=noisy.axes.clone())
    loss = float(m.loss([noisy.full_roi()], [noisy.image])[0])
    noise_energy = float(((noisy.image - noisy.clean) ** 2).sum())
    assert loss < 3.0 * noise_energy, f"loss {loss:.3e} vs noise {noise_energy:.3e}"
