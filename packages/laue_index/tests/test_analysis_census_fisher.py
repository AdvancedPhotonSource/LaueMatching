"""Calibration of the alpha-exclusion census combination (group A3).

Known-null synthetic: clustered peak fields; alpha grains and beta candidates are
arc-shaped predicted patterns (Laue zones), independent of each other; beta
candidates are SELECTED the way the validator selects them (analytic Poisson
p < 1e-4 on all peaks); a "cluster" is one selected beta from each of three
different frames.

* OLD: Fisher over the analytic Poisson p of the alpha-unclaimed hits -- the
  combination the census used. Must fail (far more than 1% below p = 0.01).
* NEW: exclusion_stats -- empirical p against the matched exclusion null (the
  frame's alpha set replaced by random alpha patterns, K times; ties randomised),
  Fisher over distinct frames. Must be near alpha.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.stats import combine_pvalues, poisson

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None

N, TOL = 512, 8.0


@pytest.fixture
def es():
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout")
    d = str(Path(_REPO_ROOT) / "pipeline" / "analysis")
    if d not in sys.path:
        sys.path.insert(0, d)
    import exclusion_stats
    return exclusion_stats


def _patterns(rng, m, per_arc=10, arcs=4):
    """m arc-shaped patterns; returns list of (n_i, 2) arrays on the detector."""
    c = rng.uniform(-200, 700, (m, arcs, 1, 2))
    r = rng.uniform(150, 500, (m, arcs, 1))
    a = rng.uniform(0, 2 * np.pi, (m, arcs, 1)) + rng.uniform(-0.6, 0.6, (m, arcs, per_arc))
    pts = np.stack([c[..., 0] + r * np.cos(a), c[..., 1] + r * np.sin(a)], -1).reshape(m, -1, 2)
    ok = (pts >= 0).all(-1) & (pts < N).all(-1)
    return pts, ok


def _frame(rng):
    cen = rng.uniform(40, N - 40, (15, 2))
    pk = np.clip(np.concatenate([c + rng.normal(0, 15, (10, 2)) for c in cen]), 0, N - 1)
    return pk


def _alpha_claims(es, tree, npk, rng, n_alpha=3):
    pts, ok = _patterns(rng, n_alpha)
    return es.claim_mask(tree, npk, [p[o] for p, o in zip(pts, ok)], TOL)


def test_census_combination_is_calibrated_and_the_old_one_is_not(es):
    rng = np.random.default_rng(0)
    K, M, NFR = 99, 5000, 150
    sel = {}
    for fi in range(NFR):
        pk = _frame(rng); npk = len(pk); tree = cKDTree(pk)
        claimed = _alpha_claims(es, tree, npk, rng)
        nulls = [_alpha_claims(es, tree, npk, rng) for _ in range(K)]
        pts, ok = _patterns(rng, M)
        d, j = tree.query(pts.reshape(-1, 2))
        d = d.reshape(M, -1); j = j.reshape(M, -1)
        hit = (d < TOL) & ok
        h = hit.sum(1); npred = ok.sum(1)
        lam = npred * npk * np.pi * TOL ** 2 / (N * N)
        keep = np.flatnonzero((npred > 0) & (poisson.sf(h - 1, lam) < 1e-4))
        out = []
        for i in keep:
            u = int((hit[i] & ~claimed[j[i]]).sum())
            un = [int((hit[i] & ~nm[j[i]]).sum()) for nm in nulls]
            nu = int((~claimed).sum())
            lam_u = npred[i] * max(nu, 1) * np.pi * TOL ** 2 / (N * N)
            out.append((poisson.sf(u - 1, lam_u), es.empirical_p(u, un, rng), int(h[i])))
        if out:
            sel[fi] = out
    frames = sorted(sel)
    n_sel = sum(len(v) for v in sel.values())
    print(f"\nselected betas: {n_sel} on {len(frames)} frames")
    assert len(frames) >= 20, f"too few frames with a selected beta ({len(frames)})"
    old, new = [], []
    for _ in range(400):
        fs = rng.choice(frames, 3, replace=False)
        items = [sel[f][rng.integers(len(sel[f]))] for f in fs]
        old.append(combine_pvalues([i[0] for i in items], method="fisher")[1])
        new.append(es.cluster_combine(fs, [i[1] for i in items], [i[2] for i in items])[1])
    old, new = np.array(old), np.array(new)
    print(f"\nRATE old p<0.01 {np.mean(old < 0.01):.3f}; new p<0.01 {np.mean(new < 0.01):.3f}, "
          f"p<0.05 {np.mean(new < 0.05):.3f}")
    assert np.mean(old < 0.01) > 0.2                    # fail-before evidence
    assert 0.02 <= np.mean(new < 0.05) <= 0.09
    assert np.mean(new < 0.01) <= 0.03


def test_empirical_p_is_uniform_for_a_discrete_statistic(es):
    rng = np.random.default_rng(1)
    ps = [es.empirical_p(rng.poisson(1.0), rng.poisson(1.0, 99), rng) for _ in range(4000)]
    ps = np.array(ps)
    for a in (0.05, 0.2, 0.5):
        assert abs(np.mean(ps < a) - a) < 0.025, (a, np.mean(ps < a))


def test_cluster_combine_uses_distinct_frames(es):
    n, p = es.cluster_combine(["a", "a", "b"], [0.5, 0.01, 0.2], [10, 3, 8])
    assert n == 2                              # two frames, not three instances
    # frame a is represented by its highest-nhit instance (p 0.5), not its smallest p
    assert p == pytest.approx(combine_pvalues([0.5, 0.2], method="fisher")[1])
