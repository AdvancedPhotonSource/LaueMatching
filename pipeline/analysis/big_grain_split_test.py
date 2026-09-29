"""Do the spatial lobes of the largest cluster correspond to its misorientation modes?

big_grain_diagnostic showed the largest validated alpha cluster is (a) spatially split
into two large disconnected lobes and (b) bimodal in internal misorientation. If those
two facts coincide -- one lobe per mode -- the cluster is two distinct crystallites that
the 1.0 deg clustering tolerance merged, not one 1529-position grain.

Null: if the cluster were a single grain whose spread is unrelated to which lobe a
position sits in, the lobes' median misorientations would differ only as much as a
spatially autocorrelated field differs between two fixed regions at random. The
misorientation map is toroidally shifted against the fixed lobe masks
(raster.toroidal_shift_null, invariant 10). The label shuffle this used before
ignores autocorrelation: on a known-null synthetic (one smooth grain, two lobes)
it flagged 74% at p < 0.05, the toroidal null 3%. The shuffle is still printed
for comparison, labelled as not a valid test.
"""
import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

from laue_material import Phase
from raster import structure, connectivity, toroidal_shift_null

W = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ and figures/ live under it)")
from frame_peaks import out_prefix
PREFIX = out_prefix()
# Symmetry follows the alpha phase's space group (laue_material), not a
# hard-coded hex table. misorientation() returns DEGREES.
_PH_A = Phase.load("alpha")
def miso(A, Bs):
    return _PH_A.misorientation(A, Bs)
# Grain connectivity is the shared raster convention (raster.py), not ndimage's
# 4-neighbour default.
STRUCT = structure()
print(f"connectivity: {connectivity()}-neighbour (LAUE_CONNECTIVITY)")
# Surface-frame geometry. Slow-axis stage coordinates are de-projected by the
# MOUNT angle (raster.mount_deg, required LAUE_MOUNT_DEG; this was a hard-coded
# sqrt(2), i.e. 45 deg). Steps are MEASURED from the stage coordinates in the npz
# (median spacing), not assumed: the 0.25 um literals this replaced were one scan's
# fast-axis step, and that scan's slow-axis stage step was different (0.177 um).
from raster import mount_deg
ZSCALE = 1.0 / np.cos(np.radians(mount_deg()))


def _step(u):
    """Median spacing of sorted unique stage coordinates (um); 0 if only one."""
    d = np.diff(np.asarray(u, float))
    d = d[d > 1e-9]
    return float(np.median(d)) if len(d) else 0.0
NSHUF = 2000

z = np.load(f"{W}/peel_map/{PREFIX}_alpha_validated.npz", allow_pickle=True)
oms, X, Z, lab = z["oms"], z["X"].astype(float), z["Z"].astype(float), z["labels"]
from frame_peaks import require_labels
require_labels(lab, f"{W}/peel_map/{PREFIX}_alpha_validated.npz")
counts = np.bincount(lab[lab >= 0]); cid = int(np.argmax(counts))
sel = np.where(lab == cid)[0]

Xu = np.unique(np.round(X, 4)); Zu = np.unique(np.round(Z, 4))
xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
grid = np.zeros((len(Zu), len(Xu)), bool)
for i in sel:
    grid[zi[round(Z[i], 4)], xi[round(X[i], 4)]] = True
cc, n = ndi.label(grid, structure=STRUCT)
sizes = np.bincount(cc.ravel())[1:]
if n < 2:
    print(f"cluster #{cid}: {len(sel)} instances in ONE connected component -- no lobes to test")
    sys.exit(0)
top2 = np.argsort(sizes)[::-1][:2] + 1
print(f"cluster #{cid}: {len(sel)} instances, {n} components; "
      f"two largest = {sizes[top2[0]-1]} and {sizes[top2[1]-1]} positions")


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
print(f"  [not a valid test: ignores autocorrelation] lobe-label shuffle ({NSHUF} draws): "
      f"{null.mean():.3f} +/- {null.std():.3f}, max {null.max():.3f}")

# spatial null: the misorientation MAP (NaN off the cluster) shifted against the
# fixed lobe masks
mis_map = np.full(grid.shape, np.nan)
for j, i in enumerate(sel):
    mis_map[zi[round(Z[i], 4)], xi[round(X[i], 4)]] = d[j]
L1, L2 = cc == top2[0], cc == top2[1]
def _dmed(f):
    a, b = f[L1], f[L2]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return abs(np.median(b) - np.median(a))
tor = toroidal_shift_null(_dmed, mis_map, n=NSHUF, rng=2, alternative="greater")
tnull = tor["null"]
print(f"  SPATIAL NULL (misorientation map toroidally shifted vs the lobes, {tor['n']} shifts): "
      f"{tnull.mean() if len(tnull) else float('nan'):.3f} +/- "
      f"{tnull.std() if len(tnull) else float('nan'):.3f}; p = {tor['p']:.4g}")
split = np.isfinite(tor["p"]) and tor["p"] < 0.05
print(f"\nVERDICT: {'TWO DISTINCT CRYSTALLITES merged by the clustering tolerance (spatial null p < 0.05)' if split else 'lobes are not distinguishable from one autocorrelated grain (spatial null p >= 0.05)'}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
xs = (Xu - Xu.min()); zs = (Zu - Zu.min())*ZSCALE
hx, hz = _step(Xu) / 2, _step(Zu) * ZSCALE / 2
ext = [xs.min()-hx, xs.max()+hx, zs.min()-hz, zs.max()+hz]
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
                f"spatial-null p = {tor['p']:.3g}", fontsize=10)
ax[1].legend(fontsize=9); ax[1].grid(alpha=.25, lw=.5)
fig.suptitle(f"{PREFIX}: largest cluster's lobes differ by {obs:.2f}$^\\circ$ (median); "
             + ("two crystallites merged by the clustering tolerance" if split
                else "not distinguishable from one autocorrelated grain")
             + f" (spatial null p = {tor['p']:.3g})", fontsize=12)
os.makedirs(f"{W}/figures", exist_ok=True)
fig.savefig(f"{W}/figures/{PREFIX}_biggrain_split.png", dpi=150)
print(f"saved {PREFIX}_biggrain_split.png")
