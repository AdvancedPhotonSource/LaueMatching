"""Is the largest 'grain' one grain, or many laths sharing an orientation?

A validated cluster groups every instance within 1.0 deg. With no depth resolution
that can merge physically distinct laths which happen to share an orientation --
notably several laths of the SAME Burgers variant of the same prior-beta parent,
which are crystallographically identical to well within the tolerance.

Three discriminators, each with its own null:

 1. SPATIAL CONNECTIVITY. Map the cluster's beam positions and count 4-connected
    components. One compact blob is consistent with a single grain (or a column of
    stacked grains, which is indistinguishable without depth). Many scattered
    patches point to separate laths sharing an orientation.
    Null: the same number of positions scattered at random over the occupied map.

 2. FILL FRACTION of the component's bounding box. A real grain section should
    largely fill its own outline; a merged family need not.

 3. INTERNAL MISORIENTATION STRUCTURE. Spread about the cluster mean, and whether
    that spread is smooth (one deformed grain) or splits into sub-modes (distinct
    laths). Compared against the 1.0 deg clustering tolerance that built it.

usage: big_grain_diagnostic.py [rank] [nshuffle]
  rank 0 = largest cluster, 1 = second largest, ...
"""
import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
SQ2 = np.sqrt(2.0)
RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSHUF = int(sys.argv[2]) if len(sys.argv) > 2 else 400

z = np.load(f"{W}/peel_map/{PREFIX}_alpha_validated.npz", allow_pickle=True)
oms, X, Z, lab = z["oms"], z["X"].astype(float), z["Z"].astype(float), z["labels"]
counts = np.bincount(lab[lab >= 0])
order = np.argsort(counts)[::-1]
cid = int(order[RANK])
sel = lab == cid
print(f"{PREFIX}: cluster #{cid} (rank {RANK}) has {sel.sum()} instances")

# --- build the occupancy grid ------------------------------------------------
Xu = np.unique(np.round(X, 4)); Zu = np.unique(np.round(Z, 4))
xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
nx, nz = len(Xu), len(Zu)
occupied = np.zeros((nz, nx), bool)      # positions where ANY validated alpha exists
for x, zc in zip(X, Z):
    occupied[zi[round(zc, 4)], xi[round(x, 4)]] = True
grain = np.zeros((nz, nx), bool)
for x, zc in zip(X[sel], Z[sel]):
    grain[zi[round(zc, 4)], xi[round(x, 4)]] = True
npos = int(grain.sum())
print(f"  occupies {npos} distinct beam positions of {int(occupied.sum())} occupied "
      f"({100*npos/occupied.sum():.1f}% of the mapped area)")

# --- 1. connectivity vs a scattered null -------------------------------------
lab_cc, ncc = ndi.label(grain)
sizes = np.bincount(lab_cc.ravel())[1:]
big = sizes.max()
print(f"\n1. CONNECTIVITY")
print(f"   components: {ncc}; largest holds {big}/{npos} positions ({100*big/npos:.1f}%)")
print(f"   components with >=10 positions: {(sizes >= 10).sum()}")

rng = np.random.default_rng(0)
occ_idx = np.argwhere(occupied)
null_ncc, null_big = [], []
for _ in range(NSHUF):
    pick = occ_idx[rng.choice(len(occ_idx), npos, replace=False)]
    m = np.zeros((nz, nx), bool); m[pick[:, 0], pick[:, 1]] = True
    l2, n2 = ndi.label(m)
    s2 = np.bincount(l2.ravel())[1:]
    null_ncc.append(n2); null_big.append(s2.max())
null_ncc = np.array(null_ncc); null_big = np.array(null_big)
print(f"   NULL (same count scattered at random): components "
      f"{null_ncc.mean():.0f} +/- {null_ncc.std():.0f}, largest blob "
      f"{null_big.mean():.0f} +/- {null_big.std():.0f}")
verdict1 = ("CLUSTERED — far more contiguous than chance"
            if big > null_big.mean() + 5*null_big.std() else
            "NOT distinguishable from scattered")
print(f"   -> {verdict1}")

# --- 2. fill fraction of the largest component -------------------------------
main = lab_cc == (np.argmax(sizes) + 1)
rows = np.where(main.any(axis=1))[0]; cols = np.where(main.any(axis=0))[0]
bh, bw = int(np.ptp(rows)) + 1, int(np.ptp(cols)) + 1
fill = main.sum() / (bh * bw)
print(f"\n2. SHAPE OF THE LARGEST COMPONENT")
print(f"   bounding box {bw} x {bh} positions "
      f"({bw*0.25:.1f} x {bh*0.25*SQ2/SQ2:.1f} um in sample frame), fill {100*fill:.0f}%")

# --- 3. internal misorientation ----------------------------------------------
def rmat(ax, deg):
    u = np.asarray(ax, float); u /= np.linalg.norm(u); t = np.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(t)*K + (1-np.cos(t))*(K@K)
HEX = np.array([rmat([0, 0, 1], 60*k) for k in range(6)] +
               [rmat([np.cos(np.radians(a)), np.sin(np.radians(a)), 0], 180)
                for a in (0, 30, 60, 90, 120, 150)])
def miso(A, Bs):
    best = np.full(len(Bs), 999.)
    for S in HEX:
        tr = np.einsum('ij,kj,mki->m', S, A, Bs)
        best = np.minimum(best, np.degrees(np.arccos(np.clip((tr-1)/2, -1, 1))))
    return best
sub = oms[sel]
ref = sub[0]
d = miso(ref, sub)
print(f"\n3. INTERNAL MISORIENTATION (vs the cluster's reference orientation)")
print(f"   mean {d.mean():.2f} deg, median {np.median(d):.2f}, 95th {np.percentile(d,95):.2f}, "
      f"max {d.max():.2f}  (clustering tolerance was 1.0 deg)")

# --- figure ------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
xs = (Xu - Xu.min()); zs = (Zu - Zu.min())*SQ2
ext = [xs.min()-0.125, xs.max()+0.125, zs.min()-0.09, zs.max()+0.09]

ax[0].imshow(occupied, origin="lower", extent=ext, cmap="Greys", alpha=.25, aspect="equal")
shown = np.where(main, 2, np.where(grain, 1, 0)).astype(float)
shown[shown == 0] = np.nan
ax[0].imshow(shown, origin="lower", extent=ext, cmap="coolwarm", vmin=1, vmax=2, aspect="equal")
ax[0].set_title(f"A · where this one cluster appears\n{npos} positions, "
                f"{ncc} component(s); red = largest", fontsize=10)
ax[0].set_xlabel(r"sample-surface X ($\mu$m)"); ax[0].set_ylabel(r"sample-surface Z ($\mu$m)")

ax[1].hist(null_big, bins=30, color="#9aa7b1", edgecolor="white", lw=.4)
ax[1].axvline(big, color="#4269d0", lw=2.5)
ax[1].annotate(f"observed {big}", xy=(big, ax[1].get_ylim()[1]*.7), xytext=(-8, 0),
               textcoords="offset points", ha="right", color="#4269d0",
               fontsize=11, fontweight="bold")
ax[1].set_xlabel("largest connected blob (positions)")
ax[1].set_ylabel(f"random scatters (of {NSHUF})")
ax[1].set_title("B · contiguity vs a scattered null\n"
                f"null {null_big.mean():.0f} $\\pm$ {null_big.std():.0f}", fontsize=10)
ax[1].grid(alpha=.25, lw=.5)

ax[2].hist(d, bins=40, color="#e8843c", edgecolor="white", lw=.3)
ax[2].axvline(1.0, color="#555", ls="--", lw=1.4)
ax[2].text(1.02, ax[2].get_ylim()[1]*.9, "clustering\ntolerance", fontsize=8.5, color="#555", va="top")
ax[2].set_xlabel("misorientation from cluster reference (deg)")
ax[2].set_ylabel("instances")
ax[2].set_title(f"C · internal spread\nmedian {np.median(d):.2f}$^\\circ$, max {d.max():.2f}$^\\circ$",
                fontsize=10)
ax[2].grid(alpha=.25, lw=.5)

fig.suptitle(f"{PREFIX}: is the largest validated $\\alpha$ cluster one grain? "
             f"{npos} positions, {ncc} connected component(s)", fontsize=12)
fig.savefig(f"{W}/figures/{PREFIX}_biggrain_rank{RANK}.png", dpi=150)
print(f"\nsaved {PREFIX}_biggrain_rank{RANK}.png")
