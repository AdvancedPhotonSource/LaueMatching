"""Is the 'retained-beta anchor' actual corroboration, or a birthday-problem artifact?

parentbeta_reconstruct reports, for each inferred parent, the misorientation to the
nearest directly-indexed beta cluster (size>=2) and calls it CONSISTENT below 2 deg.
With 6677 beta clusters to choose from, a close match may simply be what chance gives.

Null: draw random orientations and measure the SAME statistic -- misorientation to the
nearest beta cluster of size>=2. If the parents' anchors sit inside this distribution,
the anchor is not evidence.
"""
import os
import numpy as np

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
NDRAW = 3000

def rmat(ax, deg):
    u = np.asarray(ax, float); u /= np.linalg.norm(u); t = np.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(t)*K + (1-np.cos(t))*(K@K)

CUB = [np.eye(3)]
for ax, d in [([1,0,0],90),([1,0,0],180),([1,0,0],270),([0,1,0],90),([0,1,0],180),([0,1,0],270),
              ([0,0,1],90),([0,0,1],180),([0,0,1],270),([1,1,0],180),([1,-1,0],180),([1,0,1],180),
              ([-1,0,1],180),([0,1,1],180),([0,1,-1],180),([1,1,1],120),([1,1,1],240),
              ([1,-1,1],120),([1,-1,1],240),([-1,1,1],120),([-1,1,1],240),([1,1,-1],120),([1,1,-1],240)]:
    CUB.append(rmat(ax, d))
CUB = np.array(CUB)

def cubmiso(A, Bs):
    best = np.full(len(Bs), 999.)
    for S in CUB:
        tr = np.einsum('ij,kj,mki->m', S, A, Bs)
        best = np.minimum(best, np.degrees(np.arccos(np.clip((tr-1)/2, -1, 1))))
    return best

def rand_om(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

z = np.load(f"{W}/peel_map/{PREFIX}_beta_validated.npz", allow_pickle=True)
oms, lab = z["oms"], z["labels"]
counts = np.bincount(lab[lab >= 0])
reps = []
for c in np.where(counts >= 2)[0]:
    reps.append(oms[np.where(lab == c)[0][0]])
reps = np.array(reps)
print(f"beta clusters with size>=2 available as anchors: {len(reps):,}")

rng = np.random.default_rng(7)
d = np.array([cubmiso(rand_om(rng), reps).min() for _ in range(NDRAW)])
print(f"\nNULL: nearest beta cluster for a RANDOM orientation ({NDRAW:,} draws)")
print(f"  mean {d.mean():.2f} deg   median {np.median(d):.2f}   "
      f"5th pct {np.percentile(d,5):.2f}   1st pct {np.percentile(d,1):.2f}   min {d.min():.2f}")
for anchor in (1.41, 1.74, 3.27, 4.40):
    print(f"  P(random orientation lands within {anchor:.2f} deg) = {100*(d <= anchor).mean():.1f}%")
