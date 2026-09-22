"""pipeline/analysis/grain_graph.py: flood-fill grains and the fragmentation/chaining metrics.

Synthetic, and checks both behaviours the definition is chosen on: it does not fragment a
grain with a gentle gradient (D0's failure), and it DOES chain along a gradient (the reason
the chaining metric exists) -- so the metric must see it.
"""
import sys

import numpy as np
import pytest

from conftest import _REPO_ROOT

pytestmark = pytest.mark.skipif(_REPO_ROOT is None, reason="needs the repo's pipeline/analysis")
if _REPO_ROOT is not None:
    sys.path.insert(0, str(_REPO_ROOT / "pipeline" / "analysis"))


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def miso(A, Bs):
    """Plain (no symmetry) rotation angle, degrees -- enough for synthetic tests."""
    Bs = np.asarray(Bs).reshape(-1, 3, 3)
    tr = np.einsum("ij,nij->n", A, Bs)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


@pytest.fixture(autouse=True)
def _conn(monkeypatch):
    monkeypatch.setenv("LAUE_CONNECTIVITY", "8")


def test_two_distinct_grains_side_by_side_stay_separate():
    import grain_graph as G
    rows, cols, oms = [], [], []
    for r in range(4):
        for c in range(8):
            rows.append(r); cols.append(c)
            oms.append(np.eye(3) if c < 4 else rot([0, 0, 1], 20))
    g = G.flood_fill_grains(np.array(oms), np.array(rows), np.array(cols), miso, 1.0)
    assert sorted(len(x) for x in g) == [16, 16]


def test_a_gentle_gradient_is_one_grain_not_fragments():
    import grain_graph as G
    rows, cols, oms = [], [], []
    for r in range(3):
        for c in range(10):
            rows.append(r); cols.append(c); oms.append(rot([1, 0, 0], 0.3 * c))   # 2.7 deg end to end
    g = G.flood_fill_grains(np.array(oms), np.array(rows), np.array(cols), miso, 1.0)
    assert len(g) == 1 and len(g[0]) == 30


def test_a_long_gradient_chains_and_the_metric_sees_it():
    import grain_graph as G
    rows, cols, oms = [], [], []
    for c in range(40):
        for r in range(3):
            rows.append(r); cols.append(c); oms.append(rot([1, 0, 0], 0.5 * c))   # 19.5 deg end to end
    oms, rows, cols = np.array(oms), np.array(rows), np.array(cols)
    g = G.flood_fill_grains(oms, rows, cols, miso, 1.0)
    assert len(g) == 1
    m = G.grain_metrics(oms, rows, cols, g, miso)
    assert m["Ch"] == 1.0 and m["spread95"][0] > 5


def test_fragmentation_metric_flags_pieces_on_the_same_positions():
    import grain_graph as G
    rows, cols = np.repeat(np.arange(3), 5), np.tile(np.arange(5), 3)
    oms = np.array([np.eye(3)] * 15)
    pieces = [list(range(15)), list(range(15))]          # two "grains" on identical positions
    m = G.grain_metrics(np.concatenate([oms, oms]), np.concatenate([rows, rows]),
                        np.concatenate([cols, cols]), [pieces[0], [i + 15 for i in pieces[1]]], miso)
    assert m["F"] == 1.0


def test_min_positions_counts_distinct_positions_not_instances():
    import grain_graph as G
    oms = np.array([np.eye(3)] * 8)
    rows = np.array([0, 0, 0, 0, 1, 1, 1, 1]); cols = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    assert G.flood_fill_grains(oms, rows, cols, miso, 1.0, min_positions=5) == []
    assert len(G.flood_fill_grains(oms, rows, cols, miso, 1.0, min_positions=4)) == 1


def test_merge_overlapping_joins_pieces_on_shared_positions_only_within_the_angle():
    import grain_graph as G
    rows, cols = np.repeat(np.arange(3), 5), np.tile(np.arange(5), 3)
    oms = np.concatenate([[np.eye(3)] * 15, [rot([0, 0, 1], 2)] * 15, [rot([0, 0, 1], 20)] * 15])
    r = np.concatenate([rows] * 3); c = np.concatenate([cols] * 3)
    grains = [list(range(15)), list(range(15, 30)), list(range(30, 45))]
    out = G.merge_overlapping(grains, oms, r, c, miso, merge_deg=5.0)
    assert sorted(len(g) for g in out) == [15, 30]          # 0 and 2 deg merge; 20 deg stays apart


def test_merge_overlapping_needs_a_shared_position():
    import grain_graph as G
    oms = np.array([np.eye(3)] * 10)
    r = np.zeros(10, int); c = np.arange(10)
    out = G.merge_overlapping([list(range(5)), list(range(5, 10))], oms, r, c, miso)
    assert len(out) == 2
