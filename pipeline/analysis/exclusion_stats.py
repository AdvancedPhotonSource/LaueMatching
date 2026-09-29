"""Calibrated significance for the alpha-exclusion census (beta_alpha_exclusion_census.py).

THE PROBLEM. The census scored each beta instance by how many of its predicted
reflections land on peaks no validated alpha grain claims, gave that count an
analytic Poisson p, and Fisher-combined those p across a cluster's instances. Two
things break that:

1. SELECTION. Every beta instance in the census was already selected by the
   validator (Poisson p < 1e-4 on ALL peaks), so its hits are extreme by
   construction and its unclaimed-hit p is not uniform under the null.
2. DEPENDENCE / CLUSTERING. Peaks cluster on the detector and a pattern's hits
   come in groups, so even conditional on the hit count the unclaimed count is
   over-dispersed relative to any fixed-rate model.

On a known-null synthetic (clustered peaks, arc-shaped patterns, alpha claims
unrelated to beta, betas selected at p < 1e-4, three-frame clusters) the old
combination put 98% of null clusters below p = 0.01. A binomial model conditional
on the hit count still gave 15%; a beta-binomial with the over-dispersion
measured on unselected draws gave 0% (no power). Neither is kept.

THE FIX: a MATCHED EXCLUSION NULL, conditional on the instance itself. The beta
instance's own hits are fixed (so its selection is conditioned on, not modelled);
what is randomised is the alpha CLAIM: the frame's validated alpha grains are
replaced by the same number of alpha grains in random orientations (same phase,
same reflection structure), ``K`` times, and the instance's unclaimed count is
recomputed against each. ``empirical_p`` compares the observed count with those
``K`` values, with randomised tie-breaking so the p is uniform under the null (a
discrete ``>=`` rule is conservative to the point of no power: 0.3% at 0.05).
A cluster's p is the Fisher combination over its DISTINCT FRAMES, one instance per
frame (the one with the highest nhit, i.e. chosen by the validator's statistic, not
by its p). On the synthetic: 5.5% at 0.05.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import combine_pvalues


def random_rotations(rng, n):
    """``n`` uniformly random rotation matrices, shape (n, 3, 3)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
                     np.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
                     np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1)], -2)


def claim_mask(tree, npeaks, patterns, tol):
    """Peaks claimed (nearest peak within ``tol``) by any of the (n, 2) ``patterns``."""
    claimed = np.zeros(npeaks, bool)
    for pr in patterns:
        if len(pr):
            d, j = tree.query(pr)
            claimed[j[d < tol]] = True
    return claimed


def unclaimed_hits(tree, pr, tol, claimed):
    """Predicted reflections of ``pr`` whose nearest peak is within ``tol`` and unclaimed."""
    if not len(pr):
        return 0
    d, j = tree.query(pr)
    return int(((d < tol) & ~claimed[j]).sum())


def empirical_p(u_obs, u_null, rng):
    """Upper-tail p of ``u_obs`` against the null sample ``u_null``, ties randomised.

    ``(#{null > obs} + U * (#{null == obs} + 1)) / (K + 1)`` with U ~ Uniform(0, 1):
    exactly uniform under the null for any discrete statistic.
    """
    u_null = np.asarray(u_null)
    gt = int((u_null > u_obs).sum())
    eq = int((u_null == u_obs).sum())
    return float((gt + rng.random() * (eq + 1)) / (len(u_null) + 1))


def cluster_combine(frames, pvals, nhit):
    """``(n_distinct_frames, combined_p)`` for one cluster.

    One instance per distinct frame -- the highest ``nhit`` there -- then Fisher.
    A single frame returns its own p.
    """
    frames = np.asarray(frames); pvals = np.asarray(pvals, float); nhit = np.asarray(nhit)
    best = {}
    for f, p, h in zip(frames, pvals, nhit):
        if f not in best or h > best[f][1]:
            best[f] = (p, h)
    ps = np.array([v[0] for v in best.values()])
    if len(ps) == 1:
        return 1, float(ps[0])
    ps = np.clip(ps, 1e-300, 1.0)
    return len(ps), float(combine_pvalues(ps, method="fisher")[1])
