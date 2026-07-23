"""Quantify the spatial coherence of the Burgers variant map, and redraw it honestly.

The parent-beta inference uses ORIENTATIONS ONLY -- no beam positions enter it. So if
the resulting per-position variant assignment forms contiguous domains, that is
independent confirmation: nothing in the calculation could have produced spatial
structure by construction.

Statistic: majority variant per beam position, then the fraction of 4-neighbour
position pairs sharing it. Null: the same statistic after shuffling the majority-variant
labels across positions (destroys spatial arrangement, preserves variant proportions).

The original figure foregrounded the 'retained-beta anchor', which anchor_null.py
showed is a chance match (9.2% for 1.74 deg against 2537 candidate clusters), so the
anchor is NOT drawn as corroboration here.

usage: variant_coherence.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
SQ2 = np.sqrt(2.0)
NSHUF = 500
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")

z = np.load(f"{W}/peel_map/{PREFIX}_reconstruction.npz", allow_pickle=True)
inst_var = z["inst_var"]; aX = z["aX"].astype(float); aZ = z["aZ"].astype(float)
nv = int(z["parents_nv"][0]); npar = len(z["parents_nv"])

Xu = np.unique(np.round(aX, 4)); Zu = np.unique(np.round(aZ, 4))
xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
nx, nz = len(Xu), len(Zu)

# majority parent-1 variant per position (-1 where no instance carries one)
votes = np.full((nz, nx, 12), 0, int)
for v, x, zc in zip(inst_var, aX, aZ):
    if v >= 0:
        votes[zi[round(zc, 4)], xi[round(x, 4)], v] += 1
tot = votes.sum(axis=2)
maj = np.where(tot > 0, votes.argmax(axis=2), -1)

def coherence(m):
    ok = tot > 0
    same = 0; n = 0
    for di, dj in ((0, 1), (1, 0)):
        a = m[:nz-di, :nx-dj]; b = m[di:, dj:]
        va = ok[:nz-di, :nx-dj] & ok[di:, dj:]
        same += int(((a == b) & va).sum()); n += int(va.sum())
    return same / max(n, 1), n

obs, npairs = coherence(maj)
vals = maj[tot > 0]
rng = np.random.default_rng(11)
null = []
for _ in range(NSHUF):
    sh = maj.copy()
    perm = rng.permutation(vals)
    sh[tot > 0] = perm
    null.append(coherence(sh)[0])
null = np.array(null)
print(f"positions with a parent-1 variant: {int((tot>0).sum()):,} of {nz*nx:,}")
print(f"neighbour pairs compared: {npairs:,}")
print(f"OBSERVED same-variant fraction: {obs:.3f}")
print(f"SHUFFLED null: mean {null.mean():.3f}  sd {null.std():.4f}  max {null.max():.3f}")
print(f"z = {(obs-null.mean())/null.std():.1f}")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True,
                       gridspec_kw={"width_ratios": [1.15, 1]})
xs = np.concatenate([[Xu[0]-0.125], (Xu[:-1]+Xu[1:])/2, [Xu[-1]+0.125]]) - Xu.min()
zc = (Zu - Zu.min())*SQ2
zs = np.concatenate([[zc[0]-0.125], (zc[:-1]+zc[1:])/2, [zc[-1]+0.125]])
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
ax[1].set_title(f"B $\\cdot$ spatial coherence vs label-shuffle null\n"
                f"null {null.mean():.3f} $\\pm$ {null.std():.4f}, "
                f"observed is {(obs-null.mean())/null.std():.0f}$\\sigma$ away", fontsize=10)
ax[1].grid(alpha=.25, lw=.5)

fig.suptitle(f"Prior-$\\beta$ reconstruction: {npar} parents; parent #1 = {nv}/12 Burgers variants. "
             "Positions were never used in the inference — the domains are emergent.", fontsize=12)
fig.savefig(f"{W}/figures/{PREFIX}_variant_coherence.png", dpi=150)
print(f"saved {PREFIX}_variant_coherence.png")
