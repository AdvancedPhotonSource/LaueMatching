"""Quantify the spatial coherence of the Burgers variant map, and redraw it honestly.

The parent-beta inference uses ORIENTATIONS ONLY -- no beam positions enter it. So if
the resulting per-position variant assignment forms contiguous domains, that is
independent confirmation: nothing in the calculation could have produced spatial
structure by construction.

Statistic: majority variant per beam position, then the fraction of neighbouring
position pairs sharing it (neighbours per raster.connectivity(): 8 by default,
LAUE_CONNECTIVITY=4 for the edge-only pairs this script used before 2026-09). Null (2026-09): the same statistic after reassigning the variants among the ALPHA
ORIENTATION CLUSTERS, keeping each cluster's footprint (raster.permute_cluster_labels).
Every instance of one cluster gets one variant, so a contiguous cluster is
"coherent" whatever its variant; the old null shuffled labels across positions,
which breaks footprints and flags any contiguous clustering (invariant 10).

The original figure foregrounded the 'retained-beta anchor', which anchor_null.py
showed is a chance match (9.2% for 1.74 deg against 2537 candidate clusters), so the
anchor is NOT drawn as corroboration here.

usage: variant_coherence.py
"""
import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raster import (connectivity, majority_map, neighbour_agreement, neighbour_offsets,
                    permute_cluster_labels)
from frame_peaks import require_labels

W = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ and figures/ live under it)")
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
NSHUF = 500
from frame_peaks import out_prefix
PREFIX = out_prefix()

z = np.load(f"{W}/peel_map/{PREFIX}_reconstruction.npz", allow_pickle=True)
inst_var = z["inst_var"]; aX = z["aX"].astype(float); aZ = z["aZ"].astype(float)
npar = len(z["parents_nv"])
if npar == 0:
    # parentbeta_reconstruct.py found no parent that cleared its nulls and wrote an
    # empty reconstruction: there is no variant map to test. Not an error.
    print(f"{PREFIX}: no parent in the reconstruction (0 parents cleared the nulls); "
          f"variant coherence not computed")
    sys.exit(0)
nv = int(z["parents_nv"][0])

Xu = np.unique(np.round(aX, 4)); Zu = np.unique(np.round(aZ, 4))
xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
nx, nz = len(Xu), len(Zu)
gi = np.array([zi[round(v, 4)] for v in aZ], int); gj = np.array([xi[round(v, 4)] for v in aX], int)

# The alpha orientation cluster of every instance (same order as the reconstruction's
# instances: both come from <prefix>_alpha_validated.npz). Every instance of one
# cluster carries ONE variant, so cluster footprints are what the null must keep.
_asrc = f"{W}/peel_map/{PREFIX}_alpha_validated.npz"
a_lab = require_labels(np.load(_asrc, allow_pickle=True)["labels"], _asrc)
if len(a_lab) != len(inst_var):
    sys.exit(f"{_asrc} has {len(a_lab)} instances but the reconstruction has {len(inst_var)}; "
             f"re-run parentbeta_reconstruct.py on the current validated npz")

# majority parent-1 variant per position (-1 where no instance carries one)
maj = majority_map(inst_var, gi, gj, (nz, nx), 12)
tot = (maj >= 0).astype(int)

# Neighbour pairs follow the shared raster connectivity (raster.py): the two
# edge offsets for 4-neighbour, plus the two diagonals for 8 (the default).
# BEHAVIOUR CHANGE 2026-09: this used 4-neighbour pairs only; LAUE_CONNECTIVITY=4
# reproduces the earlier statistic.
CONN = connectivity()
OFFSETS = neighbour_offsets()
print(f"neighbour pairs: {CONN}-connectivity (LAUE_CONNECTIVITY)")

def coherence(m):
    return neighbour_agreement(m, OFFSETS)

obs, npairs = coherence(maj)
rng = np.random.default_rng(11)
# NULL (2026-09): permute the variant among ALPHA CLUSTERS, keeping every cluster's
# footprint. The old null shuffled labels across POSITIONS, which breaks the
# footprints: every instance of a contiguous cluster shares a variant by
# construction, so position-shuffling calls any clustering "coherent" (on a
# known-null synthetic -- contiguous clusters, random variants -- it gave z ~ 20+;
# this null z ~ 0). The position shuffle is still printed, labelled invalid.
null = np.array([coherence(majority_map(permute_cluster_labels(inst_var, a_lab, rng),
                                        gi, gj, (nz, nx), 12))[0] for _ in range(NSHUF)])
vals = maj[tot > 0]
old = []
for _ in range(NSHUF):
    sh = maj.copy(); sh[tot > 0] = rng.permutation(vals); old.append(coherence(sh)[0])
old = np.array(old)
z_new = (obs - null.mean()) / max(null.std(), 1e-12)
z_old = (obs - old.mean()) / max(old.std(), 1e-12)
p_new = float(((null >= obs).sum() + 1) / (len(null) + 1))
print(f"positions with a parent-1 variant: {int((tot>0).sum()):,} of {nz*nx:,}")
print(f"neighbour pairs compared: {npairs:,}")
print(f"OBSERVED same-variant fraction: {obs:.3f}")
print(f"CLUSTER-PERMUTATION null (variants reassigned among alpha clusters, footprints kept): "
      f"mean {null.mean():.3f}  sd {null.std():.4f}  max {null.max():.3f}")
print(f"z = {z_new:.1f}   p = {p_new:.4g}")
print(f"[not a valid test: breaks cluster footprints] position-shuffle null: mean {old.mean():.3f} "
      f"sd {old.std():.4f}, z = {z_old:.1f}")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True,
                       gridspec_kw={"width_ratios": [1.15, 1]})
hx, hz = _step(Xu) / 2, _step(Zu) * ZSCALE / 2
xs = np.concatenate([[Xu[0]-hx], (Xu[:-1]+Xu[1:])/2, [Xu[-1]+hx]]) - Xu.min()
zc = (Zu - Zu.min())*ZSCALE
zs = np.concatenate([[zc[0]-hz], (zc[:-1]+zc[1:])/2, [zc[-1]+hz]])
cmap = plt.get_cmap("tab20", 12)
m = np.ma.masked_where(maj < 0, maj)
pc = ax[0].pcolormesh(xs, zs, m, cmap=cmap, vmin=-.5, vmax=11.5, shading="flat")
cb = fig.colorbar(pc, ax=ax[0], fraction=0.046, ticks=range(12))
cb.set_label(r"Burgers $\alpha$ variant of parent #1", fontsize=9)
ax[0].set_aspect("equal")
ax[0].set_xlabel(r"sample-surface X ($\mu$m)", fontsize=9)
ax[0].set_ylabel(r"sample-surface Z ($\mu$m)", fontsize=9)
ax[0].set_title("A $\\cdot$ Burgers variant assigned from ORIENTATION ALONE\n"
                "white = no parent-#1 variant at this position", fontsize=10)

ax[1].hist(null, bins=30, color="#9aa7b1", edgecolor="white", lw=.4)
ax[1].axvline(obs, color="#4269d0", lw=2.5)
ax[1].annotate(f"observed {obs:.3f}", xy=(obs, ax[1].get_ylim()[1]*.75),
               xytext=(-12, 0), textcoords="offset points", ha="right",
               fontsize=11, color="#4269d0", fontweight="bold")
ax[1].set_xlabel("fraction of neighbouring positions sharing a variant", fontsize=9)
ax[1].set_ylabel(f"shuffles (of {NSHUF})", fontsize=9)
ax[1].set_title(f"B $\\cdot$ spatial coherence vs cluster-permutation null\n"
                f"null {null.mean():.3f} $\\pm$ {null.std():.4f}, "
                f"observed is {z_new:.1f}$\\sigma$ away (p = {p_new:.3g})", fontsize=10)
ax[1].grid(alpha=.25, lw=.5)

fig.suptitle(f"Prior-$\\beta$ reconstruction: {npar} parents; parent #1 = {nv}/12 Burgers variants. "
             "Positions were never used in the inference — the domains are emergent.", fontsize=12)
fig.savefig(f"{W}/figures/{PREFIX}_variant_coherence.png", dpi=150)
print(f"saved {PREFIX}_variant_coherence.png")
