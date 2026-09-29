"""Calibration of the spatial nulls (group A3, invariant 10).

Every map-to-map statistic in the chain now takes its p-value from ONE shared
toroidal-shift null (``raster.toroidal_shift_null``, NaN-aware). Each test here
builds a KNOWN-NULL synthetic -- two independent, spatially smoothed fields with
30% of positions missing -- in the shape of one script's statistic and checks:

* the new null's false-positive rate at p < 0.05 is near 0.05, and
* the null it replaces fails the same test (fail-before evidence): a label or
  value permutation that ignores autocorrelation flags far more than 5%.

200 repetitions each, kept under ~30 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None

REPS = 200
NSHIFT = 99
LO, HI = 0.02, 0.09          # acceptable false-positive rate at alpha 0.05


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


@pytest.fixture
def raster():
    d = str(_analysis_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    import raster as r
    return r


def _field(rng, shape=(48, 48), sigma=4.0):
    f = ndi.gaussian_filter(rng.normal(size=shape), sigma, mode="wrap")
    return (f - f.mean()) / f.std()


def _missing(rng, shape, frac=0.3):
    return rng.random(shape) < frac


def _perm_p(stat_vals_fn, rng, n=NSHIFT):
    obs, draw = stat_vals_fn
    null = np.array([draw(rng) for _ in range(n)])
    return ((np.abs(null) >= abs(obs)).sum() + 1) / (n + 1)


def _rate(ps):
    return float(np.mean(np.asarray(ps) < 0.05))


# ---------------------------------------------------------------------------
# optical_overlay / separate_layers / reg_refine measured point: map correlation
# ---------------------------------------------------------------------------
def test_map_correlation_null_is_calibrated(raster):
    rng = np.random.default_rng(0)
    new, old = [], []
    for _ in range(REPS):
        a = _field(rng); b = _field(rng)
        a[_missing(rng, a.shape)] = np.nan
        res = raster.toroidal_shift_null(lambda f: raster.nan_corr(f, b), a, n=NSHIFT, rng=rng)
        new.append(res["p"])
        ok = np.isfinite(a)
        av, bv = a[ok], b[ok]
        old.append(_perm_p((raster.nan_corr(av, bv),
                            lambda r: raster.nan_corr(av, r.permutation(bv))), rng))
    print(f"\nRATE new {_rate(new):.3f} old {_rate(old):.3f}")
    assert LO <= _rate(new) <= HI, _rate(new)
    assert _rate(old) > 0.25, _rate(old)          # the permutation null fails


# ---------------------------------------------------------------------------
# substrate_deposit B: pedestal mean where a binary map is present vs absent
# ---------------------------------------------------------------------------
def test_presence_mean_difference_null_is_calibrated(raster):
    rng = np.random.default_rng(1)
    new, old = [], []
    for _ in range(REPS):
        ped = _field(rng)
        ped[_missing(rng, ped.shape)] = np.nan
        present = _field(rng) > 0.3              # a smooth blob map, independent

        def diff(f):
            ok = np.isfinite(f)
            pr = present[ok]
            if pr.sum() < 3 or (~pr).sum() < 3:
                return np.nan
            return f[ok][~pr].mean() - f[ok][pr].mean()

        new.append(raster.toroidal_shift_null(diff, ped, n=NSHIFT, rng=rng)["p"])
        ok = np.isfinite(ped); pv, pr = ped[ok], present[ok]
        old.append(_perm_p((pv[~pr].mean() - pv[pr].mean(),
                            lambda r: (lambda s: pv[~s].mean() - pv[s].mean())(r.permutation(pr))),
                           rng))
    print(f"\nRATE new {_rate(new):.3f} old {_rate(old):.3f}")
    assert LO <= _rate(new) <= HI, _rate(new)
    assert _rate(old) > 0.25, _rate(old)


# ---------------------------------------------------------------------------
# substrate_deposit C: per-cluster footprint vs mean pedestal over the footprint
# ---------------------------------------------------------------------------
def _blobs(rng, shape, n=30):
    """n spatially compact 'clusters' of varying size (label map, -1 = none)."""
    lab = np.full(shape, -1)
    for k in range(n):
        cy, cx = rng.integers(shape[0]), rng.integers(shape[1])
        r = rng.integers(1, 7)
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = k
    return lab


def test_footprint_vs_pedestal_null_is_calibrated(raster):
    rng = np.random.default_rng(2)
    new, old = [], []
    for _ in range(REPS):
        ped = _field(rng)
        ped[_missing(rng, ped.shape)] = np.nan
        lab = _blobs(rng, ped.shape)
        ids = [k for k in np.unique(lab) if k >= 0]
        fp = np.array([np.log10((lab == k).sum()) for k in ids])

        def corr(f):
            m = np.array([np.nanmean(np.where(lab == k, f, np.nan)) if np.isfinite(
                f[lab == k]).any() else np.nan for k in ids])
            return raster.nan_corr(fp, m)

        new.append(raster.toroidal_shift_null(corr, ped, n=NSHIFT, rng=rng)["p"])
        m0 = np.array([np.nanmean(np.where(lab == k, ped, np.nan)) if np.isfinite(
            ped[lab == k]).any() else np.nan for k in ids])
        ok = np.isfinite(m0)
        old.append(_perm_p((raster.nan_corr(fp[ok], m0[ok]),
                            lambda r: raster.nan_corr(fp[ok], r.permutation(m0[ok]))), rng))
    print(f"\nRATE new {_rate(new):.3f} old {_rate(old):.3f}")
    assert LO <= _rate(new) <= HI, _rate(new)
    # per-cluster values are few and the fields smooth over a few clusters: the
    # permutation null is only mildly optimistic here; record, do not overclaim
    assert _rate(old) >= _rate(new) - 0.02, (_rate(old), _rate(new))


# ---------------------------------------------------------------------------
# big_grain_split_test: |median(field | lobe 2) - median(field | lobe 1)|
# ---------------------------------------------------------------------------
def test_lobe_median_difference_null_is_calibrated(raster):
    rng = np.random.default_rng(3)
    new, old = [], []
    for _ in range(REPS):
        mis = _field(rng)                          # a smooth single-grain gradient
        lab = _blobs(rng, mis.shape, n=2)
        l1, l2 = lab == 0, lab == 1
        if l1.sum() < 5 or l2.sum() < 5:
            l1 = np.zeros(mis.shape, bool); l1[5:15, 5:15] = True
            l2 = np.zeros(mis.shape, bool); l2[30:40, 30:40] = True
        occ = l1 | l2
        field = np.where(occ, mis, np.nan)

        def dmed(f):
            a, b = f[l1], f[l2]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 3 or len(b) < 3:
                return np.nan
            return abs(np.median(b) - np.median(a))

        # shift the misorientation FIELD (the whole map, not just the cluster)
        # against the fixed lobe masks
        new.append(raster.toroidal_shift_null(dmed, mis, n=NSHIFT, rng=rng,
                                              alternative="greater")["p"])
        d1, d2 = field[l1], field[l2]
        both = np.concatenate([d1, d2]); n1 = len(d1)
        obs = abs(np.median(d2) - np.median(d1))
        null = [abs(np.median(p[n1:]) - np.median(p[:n1]))
                for p in (rng.permutation(both) for _ in range(NSHIFT))]
        old.append(((np.array(null) >= obs).sum() + 1) / (NSHIFT + 1))
    print(f"\nRATE new {_rate(new):.3f} old {_rate(old):.3f}")
    assert LO <= _rate(new) <= HI, _rate(new)
    assert _rate(old) > 0.5, _rate(old)


# ---------------------------------------------------------------------------
# reg_refine: the BEST of a registration grid
# ---------------------------------------------------------------------------
def test_best_of_grid_null_is_calibrated(raster):
    """The maximum |r| over a grid of registrations, against an independent map.
    The old script reported the best of 1,260 with no null at all; its implicit
    null is the naive Pearson test of that best value, which fails badly."""
    from scipy.stats import pearsonr
    rng = np.random.default_rng(4)
    shifts = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]
    new, old = [], []
    for _ in range(REPS):
        foot = _field(rng)
        foot[_missing(rng, foot.shape)] = np.nan
        black = (_field(rng) > 0).astype(float)

        def best(f):
            return min(raster.nan_corr(f, np.roll(black, s, axis=(0, 1))) for s in shifts)

        new.append(raster.toroidal_shift_null(best, foot, n=NSHIFT, rng=rng,
                                              alternative="less")["p"])
        b = best(foot)
        ok = np.isfinite(foot)
        n = int(ok.sum())
        t = b * np.sqrt((n - 2) / max(1 - b * b, 1e-12))
        from scipy.stats import t as tdist
        old.append(2 * tdist.sf(abs(t), n - 2))
    print(f"\nRATE new {_rate(new):.3f} old {_rate(old):.3f}")
    assert LO <= _rate(new) <= HI, _rate(new)
    assert _rate(old) > 0.25, _rate(old)


def test_toroidal_null_detects_a_real_association(raster):
    """Power check: a genuinely shared structure is still found."""
    rng = np.random.default_rng(9)
    hits = 0
    for _ in range(40):
        common = _field(rng)
        a = common + 0.5 * _field(rng); b = common + 0.5 * _field(rng)
        a[_missing(rng, a.shape)] = np.nan
        hits += raster.toroidal_shift_null(lambda f: raster.nan_corr(f, b), a,
                                           n=NSHIFT, rng=rng)["p"] < 0.05
    assert hits >= 30, hits


# ---------------------------------------------------------------------------
# every listed script uses the shared null
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["substrate_deposit.py", "optical_overlay.py", "reg_refine.py",
                                  "big_grain_split_test.py", "separate_layers.py"])
def test_scripts_use_the_shared_toroidal_null(name):
    src = (_analysis_dir() / name).read_text()
    assert "toroidal_shift_null(" in src, name


# ---------------------------------------------------------------------------
# variant_coherence: variants carried by contiguous alpha clusters
# ---------------------------------------------------------------------------
def _cluster_raster(rng, shape=(30, 30), n=40):
    """Contiguous clusters tiling part of the raster (label per position, -1 none)."""
    lab = _blobs(rng, shape, n=n)
    gi, gj = np.nonzero(lab >= 0)
    return gi, gj, lab[gi, gj]


def test_variant_coherence_null_is_calibrated(raster, monkeypatch):
    monkeypatch.setenv("LAUE_CONNECTIVITY", "8")
    rng = np.random.default_rng(6)
    offs = raster.neighbour_offsets()
    new, old = [], []
    for _ in range(REPS):
        gi, gj, cl = _cluster_raster(rng)
        var_of = rng.integers(0, 12, cl.max() + 1)        # random variant per cluster
        var = var_of[cl]
        maj = raster.majority_map(var, gi, gj, (30, 30), 12)
        obs = raster.neighbour_agreement(maj, offs)[0]
        null = [raster.neighbour_agreement(raster.majority_map(
            raster.permute_cluster_labels(var, cl, rng), gi, gj, (30, 30), 12), offs)[0]
            for _ in range(NSHIFT)]
        new.append(((np.array(null) >= obs).sum() + 1) / (NSHIFT + 1))
        vals = maj[maj >= 0]
        sh = []
        for _ in range(NSHIFT):
            m = maj.copy(); m[maj >= 0] = rng.permutation(vals)
            sh.append(raster.neighbour_agreement(m, offs)[0])
        old.append(((np.array(sh) >= obs).sum() + 1) / (NSHIFT + 1))
    print(f"\nRATE new {_rate(new):.3f} old {_rate(old):.3f}")
    assert LO <= _rate(new) <= HI, _rate(new)
    assert _rate(old) > 0.5, _rate(old)


def test_permute_cluster_labels_keeps_footprints(raster):
    rng = np.random.default_rng(0)
    cl = np.array([0, 0, 1, 1, 1, 2, -1])
    var = np.array([3, 3, 5, 5, 5, 7, 4])
    out = raster.permute_cluster_labels(var, cl, rng)
    for c in (0, 1, 2):
        assert len(set(out[cl == c])) == 1          # one label per cluster still
    assert sorted(out[[0, 2, 5]].tolist()) == [3, 5, 7]
    assert out[6] == 4                              # unclustered instance untouched
