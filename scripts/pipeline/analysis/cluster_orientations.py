"""Symmetry-aware orientation clustering that scales to a full raster.

The greedy loop in parentbeta_validate.py is O(n_clusters x n_instances x n_sym)
and is fine for the ~1e3-1e4 instances a test scan produces. A full 201x201
raster yields ~2e5 instances, where it does not finish.

Approach here: represent each orientation as a unit quaternion, take a canonical
fundamental-zone representative, and index them with a KD-tree. For two
orientations, the misorientation angle satisfies

    cos(theta/2) = |q1 . q2|      ->      ||q1 - q2||^2 = 2 - 2|q1 . q2|

so a misorientation cut of theta_max becomes a Euclidean radius
r = sqrt(2 - 2 cos(theta_max/2)) in quaternion space. Symmetry is handled by
querying the tree with all n_sym variants of each orientation (and both signs of
q), which avoids the fundamental-zone boundary problem that plain FZ-reduction
would introduce: two orientations 0.1 deg apart can land on opposite sides of a
zone boundary and be missed.

Clusters are the connected components of that neighbour graph, which is also a
better definition than the greedy one -- greedy assignment depends on iteration
order, connected components do not.

usage: cluster_orientations.py <validated.npz> <out.npz> [tol_deg] [phase]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from laue_material import Phase


def om_to_quat(oms: np.ndarray) -> np.ndarray:
    """(n,3,3) rotation matrices -> (n,4) unit quaternions (w,x,y,z)."""
    m = np.asarray(oms, float)
    n = len(m)
    q = np.empty((n, 4))
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    a = t > 0
    s = np.sqrt(np.maximum(t[a] + 1.0, 1e-30)) * 2
    q[a, 0] = 0.25 * s
    q[a, 1] = (m[a, 2, 1] - m[a, 1, 2]) / s
    q[a, 2] = (m[a, 0, 2] - m[a, 2, 0]) / s
    q[a, 3] = (m[a, 1, 0] - m[a, 0, 1]) / s

    b = (~a) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + m[b, 0, 0] - m[b, 1, 1] - m[b, 2, 2], 1e-30)) * 2
    q[b, 0] = (m[b, 2, 1] - m[b, 1, 2]) / s
    q[b, 1] = 0.25 * s
    q[b, 2] = (m[b, 0, 1] + m[b, 1, 0]) / s
    q[b, 3] = (m[b, 0, 2] + m[b, 2, 0]) / s

    c = (~a) & (~b) & (m[:, 1, 1] >= m[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + m[c, 1, 1] - m[c, 0, 0] - m[c, 2, 2], 1e-30)) * 2
    q[c, 0] = (m[c, 0, 2] - m[c, 2, 0]) / s
    q[c, 1] = (m[c, 0, 1] + m[c, 1, 0]) / s
    q[c, 2] = 0.25 * s
    q[c, 3] = (m[c, 1, 2] + m[c, 2, 1]) / s

    d = (~a) & (~b) & (~c)
    s = np.sqrt(np.maximum(1.0 + m[d, 2, 2] - m[d, 0, 0] - m[d, 1, 1], 1e-30)) * 2
    q[d, 0] = (m[d, 1, 0] - m[d, 0, 1]) / s
    q[d, 1] = (m[d, 0, 2] + m[d, 2, 0]) / s
    q[d, 2] = (m[d, 1, 2] + m[d, 2, 1]) / s
    q[d, 3] = 0.25 * s

    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # fix the sign convention so q and -q do not read as different orientations
    flip = q[:, 0] < 0
    q[flip] *= -1
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def cluster(oms: np.ndarray, sym_ops: np.ndarray, tol_deg: float,
            verbose: bool = True) -> np.ndarray:
    """Connected-component clusters under symmetry-reduced misorientation."""
    n = len(oms)
    if n == 0:
        return np.zeros(0, int)
    q = om_to_quat(oms)
    qs = om_to_quat(sym_ops)                      # (nsym,4)
    r = np.sqrt(max(2.0 - 2.0 * np.cos(np.radians(tol_deg) / 2.0), 0.0))
    if verbose:
        print(f"  {n} instances, {len(qs)} sym ops, tol {tol_deg} deg -> "
              f"quaternion radius {r:.6f}", flush=True)

    tree = cKDTree(q)
    rows, cols = [], []
    for k, s in enumerate(qs):
        # The symmetry operator multiplies on the RIGHT: the pipeline's
        # misorientation is min_S angle(A^T B S) (verified numerically against
        # laue_material.misorientation -- the left-handed form A^T S B differs by
        # up to 78 deg and silently splits real grains).
        # Both signs: q and -q are the same rotation but different points in R^4.
        for sgn in (1.0, -1.0):
            qq = quat_mul(q, np.broadcast_to(sgn * s, q.shape))
            pairs = tree.query_ball_point(qq, r, workers=-1)
            for i, js in enumerate(pairs):
                if js:
                    rows.extend([i] * len(js))
                    cols.extend(js)
        if verbose:
            print(f"    sym op {k+1}/{len(qs)}: {len(rows)} edges so far", flush=True)

    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    if verbose:
        print(f"  -> {ncomp} clusters", flush=True)
    return labels


def miso_matrix(oms: np.ndarray, sym_ops: np.ndarray) -> np.ndarray:
    """Full n x n symmetry-reduced misorientation in degrees, vectorised.

    Generalises the pipeline's own one-vs-many form

        np.einsum('ij,kj,mki->m', S, A, Bs)

    to all pairs, one contraction per symmetry operator:

        np.einsum('ij,pkj,qki->pq', S, A, A)

    Generalising the existing formula (rather than deriving a new one) preserves the
    right-multiply convention -- the left-handed form differs by up to 78 deg and
    silently splits real grains.

    NOTE ON PRECISION. This uses ``sym_ops`` directly, whereas
    ``laue_material.misorientation`` routes through midas_stress, whose symmetry
    quaternions are 5-decimal rounded. Measured over 59,700 real Zn pairs the two differ
    by 0.0247 deg worst case (median 0.00025), which flips **0** pairs across a 1.0 deg
    cut, 9 across 0.5 deg and 14 across 0.3 deg (0.02%). Negligible against a typical
    tolerance sweep, but a real floor on tight-cut counts -- do not quote 0.3 deg results
    as exact.

    Speed matters here: n separate misorientation calls per component did not finish a
    60k-instance sweep in 84 minutes; this form completes it in 26 s.
    """
    best = None
    for S in np.asarray(sym_ops):
        tr = np.einsum('ij,pkj,qki->pq', S, oms, oms, optimize=True)
        d = np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
        best = d if best is None else np.minimum(best, d)
    np.fill_diagonal(best, 0.0)
    return 0.5 * (best + best.T)


def cluster_diameter(oms: np.ndarray, sym_ops: np.ndarray, tol_deg: float,
                     verbose: bool = True) -> np.ndarray:
    """Diameter-constrained clustering: every member within tol of every OTHER member.

    Connected components link A-B and B-C at 0.9 deg each even when A-C are 1.8 deg
    apart, so a "1.0 deg cluster" can span several degrees. On sampleH a 924-position cluster
    had 2.65 deg median internal spread and failed a raw-image test at chance, while each
    of its positions indexed correctly on its own orientation -- the signature of
    over-merging. Complete linkage removes chaining by construction and is deterministic.
    On sampleH it halved the tolerance sensitivity of the grain count (4.28x -> 2.02x).

    Diameter clusters are subsets of the connected components at the same tolerance, so
    the components bound the work and only their internal matrices are built.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    base = cluster(oms, sym_ops, tol_deg, verbose=False)
    labels = np.full(len(oms), -1, np.int64)
    nxt = 0
    viol = 0
    for c in range(base.max() + 1):
        idx = np.where(base == c)[0]
        if len(idx) == 1:
            labels[idx] = nxt; nxt += 1
            continue
        M = miso_matrix(oms[idx], sym_ops)
        fl = fcluster(linkage(squareform(M, checks=False), method="complete"),
                      t=tol_deg, criterion="distance")
        for v in np.unique(fl):
            sel = fl == v
            if sel.sum() > 1 and M[np.ix_(sel, sel)].max() > tol_deg + 1e-6:
                viol += 1
            labels[idx[sel]] = nxt; nxt += 1
    if verbose:
        print(f"  -> {nxt} diameter clusters from {base.max()+1} connected components "
              f"({viol} diameter violations)", flush=True)
    return labels


def _greedy(oms, ph, tol):
    """The original greedy assignment, for the equivalence gate only."""
    labels = np.full(len(oms), -1)
    cid = 0
    for i in range(len(oms)):
        if labels[i] >= 0:
            continue
        un = np.where(labels < 0)[0]
        d = ph.misorientation(oms[i], oms[un])
        labels[un[d < tol]] = cid
        cid += 1
    return labels


def selftest(ph, n=400, tol=1.0, seed=0):
    """Synthetic gate: planted clusters must be recovered, and the result must
    agree with the greedy method it replaces on a set small enough to run both."""
    rng = np.random.default_rng(seed)

    def rand_om():
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    def small_rot(deg):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        t = np.radians(deg)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)

    # 8 planted grains, well separated, each with a tight spread
    centres = [rand_om() for _ in range(8)]
    oms, truth = [], []
    for gi, c in enumerate(centres):
        for _ in range(n // 8):
            oms.append(small_rot(rng.uniform(0, 0.25)) @ c)
            truth.append(gi)
    oms = np.array(oms); truth = np.array(truth)

    lab = cluster(oms, ph.sym_ops, tol, verbose=False)
    ok_planted = all(len(np.unique(lab[truth == gi])) == 1 for gi in range(8))
    merged = len(np.unique(lab)) != 8
    print(f"  planted-cluster recovery: each grain in one cluster = {ok_planted}, "
          f"n_clusters = {len(np.unique(lab))} (expect 8)")

    # THE test that matters: the neighbour relation the tree finds must agree,
    # pair for pair, with the pipeline's own misorientation. Gating against
    # laue_material rather than a reimplementation is what catches a
    # wrong-sided symmetry convention -- an earlier version of this file applied
    # the operator on the left (A^T S B) and passed a hand-rolled symmetry check
    # that shared the same error.
    sub = oms[:200]
    n_bad = 0
    for i in range(len(sub)):
        d = ph.misorientation(sub[i], sub)
        truth_nb = set(np.where(d < tol)[0].tolist())
        qs_ = om_to_quat(ph.sym_ops)
        qsub = om_to_quat(sub)
        r = np.sqrt(max(2.0 - 2.0 * np.cos(np.radians(tol) / 2.0), 0.0))
        found = set()
        for s in qs_:
            for sgn in (1.0, -1.0):
                qq = quat_mul(qsub, np.broadcast_to(sgn * s, qsub.shape))
                found |= set(np.where(np.linalg.norm(qq - qsub[i], axis=1) < r)[0].tolist())
        if truth_nb != found:
            n_bad += 1
    print(f"  neighbour sets identical to laue_material.misorientation: "
          f"{len(sub) - n_bad}/{len(sub)} instances")

    # crystal-symmetry equivalence, using the side the pipeline uses (A -> A S)
    S = ph.sym_ops[len(ph.sym_ops) // 2]
    oms2 = np.concatenate([oms, np.einsum('nij,jk->nik', oms[:20], S)])
    lab2 = cluster(oms2, ph.sym_ops, tol, verbose=False)
    sym_ok = all(lab2[i] == lab2[len(oms) + i] for i in range(20))
    print(f"  symmetry-equivalent orientations land in the same cluster: {sym_ok}")

    # connected components may MERGE greedy clusters (chaining) but must never
    # split one: greedy links everything within tol of a seed, which is a subset
    # of the transitive closure.
    sub2 = oms[:150]
    lg = _greedy(sub2, ph, tol)
    lk = cluster(sub2, ph.sym_ops, tol, verbose=False)
    greedy_refines = all(
        len(np.unique(lk[np.where(lg == v)[0]])) == 1 for v in np.unique(lg))
    print(f"  every greedy cluster stays intact under the KD-tree method: {greedy_refines}")
    print(f"  greedy {len(np.unique(lg))} clusters, kdtree {len(np.unique(lk))} "
          f"clusters on 150 instances")

    assert ok_planted and not merged and sym_ok and greedy_refines and n_bad == 0, \
        "selftest FAILED"
    print("  cluster_orientations selftest OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        ph = Phase.load(sys.argv[2] if len(sys.argv) > 2 else "zn")
        selftest(ph)
        sys.exit(0)

    argv = [a for a in sys.argv[1:] if a != "--diameter"]
    diameter = "--diameter" in sys.argv
    src, dst = argv[0], argv[1]
    tol = float(argv[2]) if len(argv) > 2 else 1.0
    phase = argv[3] if len(argv) > 3 else os.environ.get("LAUE_PHASE", "zn")
    ph = Phase.load(phase)

    z = np.load(src, allow_pickle=True)
    oms = z["oms"]
    mode = "diameter (complete linkage)" if diameter else "connected components"
    print(f"clustering {len(oms)} instances from {src}  [{mode}, tol {tol} deg]", flush=True)
    t0 = time.time()
    labels = (cluster_diameter(oms, ph.sym_ops, tol) if diameter
              else cluster(oms, ph.sym_ops, tol))
    print(f"  {time.time()-t0:.1f}s", flush=True)

    out = {k: z[k] for k in z.files if k != "labels"}
    out["labels"] = labels
    np.savez(dst, **out)
    cnt = np.bincount(labels)
    print(f"clusters: {len(cnt)}; sizes: max {cnt.max()}, "
          f"n>=2 {(cnt>=2).sum()}, n>=10 {(cnt>=10).sum()}, singletons {(cnt==1).sum()}")
    print(f"wrote {dst}")
