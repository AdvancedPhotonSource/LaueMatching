"""How much does the grain count depend on the clustering tolerance?

Grains are formed by grouping validated instances within a misorientation tolerance.
That tolerance is a chosen knob, and the largest 'grain' turned out to be two
crystallites 0.44 deg apart merged at 1.0 deg -- so the counts are sensitive to it.

Tightening the tolerance can only SPLIT existing clusters, never merge them, so we
recluster within each existing cluster instead of from scratch: cost is sum(n_i^2)
rather than N^2, which is orders of magnitude cheaper and exact.

Reports, per tolerance: total grains, grains recurring at >=5 positions (the tier the
reports stand behind), and the largest grain's size.

usage: tolerance_sensitivity.py [phase]
"""
import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "alpha"
TOLS = [0.2, 0.3, 0.4, 0.5, 0.75, 1.0]

z = np.load(f"{W}/peel_map/{PREFIX}_{PHASE}_validated.npz", allow_pickle=True)
oms, X, Z, lab = z["oms"], z["X"].astype(float), z["Z"].astype(float), z["labels"]

# Symmetry follows the space group of the phase, not its name.
from laue_material import Phase
OPS = Phase.load(PHASE).sym_ops

def miso(A, Bs):
    best = np.full(len(Bs), 999.)
    for S in OPS:
        tr = np.einsum('ij,kj,mki->m', S, A, Bs)
        best = np.minimum(best, np.degrees(np.arccos(np.clip((tr-1)/2, -1, 1))))
    return best

def npos_of(idx):
    return len(set(zip(np.round(X[idx], 4), np.round(Z[idx], 4))))

base = [np.where(lab == c)[0] for c in range(lab.max()+1)]
print(f"{PREFIX} {PHASE}: {len(oms)} instances, {len(base)} clusters at 1.0 deg\n")
print(f"{'tol':>6} {'grains':>8} {'>=5 pos':>9} {'largest':>9}")
rows = []
for tol in TOLS:
    groups = []
    for idx in base:
        if len(idx) == 1 or tol >= 1.0:
            groups.append(idx); continue
        sub = oms[idx]
        lab2 = np.full(len(idx), -1); cid = 0
        for i in range(len(idx)):
            if lab2[i] >= 0: continue
            un = np.where(lab2 < 0)[0]
            lab2[un[miso(sub[i], sub[un]) < tol]] = cid; cid += 1
        for c in range(cid):
            groups.append(idx[lab2 == c])
    pos = np.array([npos_of(g) for g in groups])
    rows.append((tol, len(groups), int((pos >= 5).sum()), int(pos.max())))
    print(f"{tol:>6.2f} {len(groups):>8,} {int((pos>=5).sum()):>9,} {int(pos.max()):>9,}")

rows = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
ax[0].plot(rows[:, 0], rows[:, 1], "-o", color="#4269d0", lw=2, label="all grains")
ax[0].plot(rows[:, 0], rows[:, 2], "-o", color="#e8843c", lw=2, label=r"grains at $\geq$5 positions")
ax[0].set_xlabel("clustering tolerance (deg)"); ax[0].set_ylabel("number of grains")
ax[0].set_yscale("log"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.25, lw=.5)
ax[0].set_title("A · grain count vs the tolerance knob", fontsize=10)
ax[1].plot(rows[:, 0], rows[:, 3], "-o", color="#b33a3a", lw=2)
ax[1].set_xlabel("clustering tolerance (deg)"); ax[1].set_ylabel("largest grain (beam positions)")
ax[1].grid(alpha=.25, lw=.5)
ax[1].set_title("B · size of the largest grain", fontsize=10)
fig.suptitle(f"{PREFIX} {PHASE}: how much the reported counts depend on one chosen tolerance",
             fontsize=12)
fig.savefig(f"{W}/figures/{PREFIX}_{PHASE}_tolerance.png", dpi=150)
print(f"\nsaved {PREFIX}_{PHASE}_tolerance.png")
