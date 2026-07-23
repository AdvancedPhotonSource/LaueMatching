"""Do the spatial lobes of the largest cluster correspond to its misorientation modes?

big_grain_diagnostic showed the largest validated alpha cluster is (a) spatially split
into two large disconnected lobes and (b) bimodal in internal misorientation. If those
two facts coincide -- one lobe per mode -- the cluster is two distinct crystallites that
the 1.0 deg clustering tolerance merged, not one 1529-position grain.

Null: if the cluster were a single grain whose spread is unrelated to position, the
misorientation distributions of the two lobes would be interchangeable. We test with a
label-shuffle on the lobe assignment (difference of medians).
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

W = os.environ.get("LAUE_WORK", "/net/hpcs34/data34c/for_Hemant/lauematching_ti")
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
SQ2 = np.sqrt(2.0); NSHUF = 2000

z = np.load(f"{W}/peel_map/{PREFIX}_alpha_validated.npz", allow_pickle=True)
oms, X, Z, lab = z["oms"], z["X"].astype(float), z["Z"].astype(float), z["labels"]
counts = np.bincount(lab[lab >= 0]); cid = int(np.argmax(counts))
sel = np.where(lab == cid)[0]

Xu = np.unique(np.round(X, 4)); Zu = np.unique(np.round(Z, 4))
xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
grid = np.zeros((len(Zu), len(Xu)), bool)
for i in sel:
    grid[zi[round(Z[i], 4)], xi[round(X[i], 4)]] = True
cc, n = ndi.label(grid)
sizes = np.bincount(cc.ravel())[1:]
top2 = np.argsort(sizes)[::-1][:2] + 1
print(f"cluster #{cid}: {len(sel)} instances, {n} components; "
      f"two largest = {sizes[top2[0]-1]} and {sizes[top2[1]-1]} positions")

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

lobe = np.zeros(len(sel), int)
for k, comp in enumerate(top2, start=1):
    for j, i in enumerate(sel):
        if cc[zi[round(Z[i], 4)], xi[round(X[i], 4)]] == comp:
            lobe[j] = k

sub = oms[sel]
d = miso(sub[lobe == 1][0], sub)          # measure everything from lobe-1's orientation
d1, d2 = d[lobe == 1], d[lobe == 2]
print(f"\nmisorientation from lobe-1 reference:")
print(f"  lobe 1 (n={len(d1)}): median {np.median(d1):.3f} deg, mean {d1.mean():.3f}")
print(f"  lobe 2 (n={len(d2)}): median {np.median(d2):.3f} deg, mean {d2.mean():.3f}")
obs = abs(np.median(d2) - np.median(d1))
print(f"  |difference of medians| = {obs:.3f} deg")

rng = np.random.default_rng(1)
both = np.concatenate([d1, d2]); n1 = len(d1)
null = np.empty(NSHUF)
for s in range(NSHUF):
    p = rng.permutation(both)
    null[s] = abs(np.median(p[n1:]) - np.median(p[:n1]))
print(f"  NULL (lobe labels shuffled, {NSHUF} draws): {null.mean():.3f} +/- {null.std():.3f}, "
      f"max {null.max():.3f}")
print(f"  -> observed is {(obs-null.mean())/null.std():.0f} sigma beyond the shuffle null")
print(f"\nVERDICT: {'TWO DISTINCT CRYSTALLITES merged by the 1.0 deg tolerance' if obs > null.max() else 'lobes are orientationally interchangeable — consistent with one grain'}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
xs = (Xu - Xu.min()); zs = (Zu - Zu.min())*SQ2
ext = [xs.min()-.125, xs.max()+.125, zs.min()-.09, zs.max()+.09]
m = np.full(grid.shape, np.nan)
m[cc == top2[0]] = 1; m[cc == top2[1]] = 2
ax[0].imshow(np.where(grid, .3, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1, aspect="equal")
LOBECMAP = ListedColormap(["#b33a3a", "#4269d0"])   # lobe1 red, lobe2 blue — matches panel B
ax[0].imshow(m, origin="lower", extent=ext, cmap=LOBECMAP, vmin=1, vmax=2, aspect="equal")
ax[0].set_xlabel(r"sample-surface X ($\mu$m)"); ax[0].set_ylabel(r"sample-surface Z ($\mu$m)")
ax[0].set_title(f"A · the two lobes of the single {len(sel)}-position cluster\n"
                f"red = lobe 1 ({len(d1)} pos), blue = lobe 2 ({len(d2)} pos)", fontsize=10)
b = np.linspace(0, max(d.max(), .8), 45)
ax[1].hist(d1, bins=b, color="#b33a3a", alpha=.75, label=f"lobe 1 (n={len(d1)})")
ax[1].hist(d2, bins=b, color="#4269d0", alpha=.75, label=f"lobe 2 (n={len(d2)})")
ax[1].axvline(1.0, color="#555", ls="--", lw=1.3)
ax[1].set_xlabel("misorientation from lobe-1 reference (deg)"); ax[1].set_ylabel("instances")
ax[1].set_title(f"B · each lobe has its own orientation\n"
                f"medians {np.median(d1):.2f}$^\\circ$ vs {np.median(d2):.2f}$^\\circ$; "
                f"shuffle null {null.mean():.3f}$\\pm${null.std():.3f}", fontsize=10)
ax[1].legend(fontsize=9); ax[1].grid(alpha=.25, lw=.5)
fig.suptitle(f"{PREFIX}: the largest 'grain' is two crystallites {obs:.2f}$^\\circ$ apart, "
             f"merged by a 1.0$^\\circ$ clustering tolerance", fontsize=12)
fig.savefig(f"{W}/figures/{PREFIX}_biggrain_split.png", dpi=150)
print(f"saved {PREFIX}_biggrain_split.png")
