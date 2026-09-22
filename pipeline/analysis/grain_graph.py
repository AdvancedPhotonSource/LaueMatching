"""Grains by spatial flood-fill: the EBSD-style definition, and the metrics that judge it.

A grain is a connected set of instances: two instances are joined when their raster
positions are equal or neighbours (``raster.connectivity()``, 8 by default) AND their
symmetry-reduced misorientation is below ``theta_deg``. Components with at least
``min_positions`` distinct positions are grains.

Why this and not orientation clustering followed by a spatial split (the definition every
sampleH grain count used until 2026-09-21, "D0"): D0 FRAGMENTS. Complete-linkage clustering
at 1 deg cuts a grain whose internal spread exceeds 1 deg into pieces, and those pieces then
sit on the same positions within a fraction of a degree of each other -- measured on
sampleH, about half of all D0 grains shared positions with another grain within 5 deg, and
D0 grain counts moved 3.5x over tolerance / minimum-size choices. Flood-fill cannot
fragment that way, but it can CHAIN along a slow orientation gradient (the reason D0 was
adopted); :func:`grain_metrics` measures both, and the choice of definition is recorded in
the campaign's ``PREREGISTER_grain_definition.md``.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from raster import connectivity


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _neighbour_offsets():
    if connectivity() == 8:
        return [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    return [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def flood_fill_grains(oms, row, col, misorientation, theta_deg, min_positions=5):
    """Grains as lists of instance indices.

    ``oms`` (n, 3, 3); ``row``, ``col`` integer raster indices; ``misorientation(A, Bs)``
    returns symmetry-reduced misorientation in DEGREES of one matrix against a stack
    (``laue_material.Phase.misorientation``).
    """
    n = len(oms)
    at = defaultdict(list)
    for i, rc in enumerate(zip(row.tolist(), col.tolist())):
        at[rc].append(i)
    uf = _UF(n)
    offs = _neighbour_offsets()
    for i in range(n):
        r, c = int(row[i]), int(col[i])
        cand = [j for dr, dc in offs for j in at.get((r + dr, c + dc), ()) if j > i]
        if not cand:
            continue
        d = np.asarray(misorientation(oms[i], oms[cand]), float)
        for j in np.asarray(cand)[d < theta_deg]:
            uf.union(i, int(j))
    comp = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    out = []
    for members in comp.values():
        if len(set(zip(row[members].tolist(), col[members].tolist()))) >= min_positions:
            out.append(members)
    return out


def medoid(oms, members, misorientation, sample=200, seed=0):
    """Member minimising summed misorientation to a random sample of the grain."""
    members = list(members)
    rng = np.random.default_rng(seed)
    ref = members if len(members) <= sample else list(rng.choice(members, sample, replace=False))
    best, bestv = members[0], np.inf
    cands = members if len(members) <= sample else ref
    for m in cands:
        v = float(np.sum(misorientation(oms[m], oms[ref])))
        if v < bestv:
            best, bestv = m, v
    return best


def grain_metrics(oms, row, col, grains, misorientation, frag_deg=5.0, chain_deg=5.0):
    """F (fragmentation), Ch (area-weighted chaining) and per-grain summaries."""
    pos = [set(zip(row[g].tolist(), col[g].tolist())) for g in grains]
    reps = [medoid(oms, g, misorientation) for g in grains]
    spread95 = []
    for g, r in zip(grains, reps):
        d = np.asarray(misorientation(oms[r], oms[g]), float)
        spread95.append(float(np.percentile(d, 95)))
    by_pos = defaultdict(set)
    for k, p in enumerate(pos):
        for rc in p:
            by_pos[rc].add(k)
    fragmented = np.zeros(len(grains), bool)
    for k, p in enumerate(pos):
        others = set().union(*(by_pos[rc] for rc in p)) - {k}
        if others:
            o = sorted(others)
            d = np.asarray(misorientation(oms[reps[k]], oms[[reps[j] for j in o]]), float)
            if np.any(d < frag_deg):
                fragmented[k] = True
    area = np.array([len(p) for p in pos], float)
    chained = np.array(spread95) > chain_deg
    return {
        "F": float(fragmented.mean()) if len(grains) else float("nan"),
        "Ch": float(area[chained].sum() / area.sum()) if len(grains) else float("nan"),
        "npos": area, "reps": reps, "spread95": np.array(spread95),
    }


def merge_overlapping(grains, oms, row, col, misorientation, merge_deg=5.0):
    """Transitively merge grains that share >= 1 raster position and whose representatives
    (first member, as ``conclusions.py``) are within ``merge_deg``.

    This is definition D2 of the campaign's ``PREREGISTER_grain_definition.md``: D0 (orientation
    clustering + spatial split) followed by this merge. **It CHAINS and is not adopted.** It
    passed the fragmentation / 95th-percentile chaining screen on sampleH (0.019 / 0.035), but
    adversarial review found merges on a single shared pixel (median 2.85 deg per hop) and one
    merged "grain" of 24 pieces spanning 10 deg; the percentile metric could not see that and
    the maximum span did. Kept as a tool, with that warning. See handbook invariant 39.
    """
    pos = [set(zip(row[g].tolist(), col[g].tolist())) for g in grains]
    by = defaultdict(set)
    for k, p in enumerate(pos):
        for rc in p:
            by[rc].add(k)
    uf = _UF(len(grains))
    for k, p in enumerate(pos):
        others = sorted(o for o in set().union(*(by[rc] for rc in p)) if o > k)
        if not others:
            continue
        d = np.asarray(misorientation(oms[grains[k][0]], oms[[grains[o][0] for o in others]]), float)
        for o, dd in zip(others, d):
            if dd < merge_deg:
                uf.union(k, o)
    comp = defaultdict(list)
    for k, g in enumerate(grains):
        comp[uf.find(k)] += list(g)
    return list(comp.values())
