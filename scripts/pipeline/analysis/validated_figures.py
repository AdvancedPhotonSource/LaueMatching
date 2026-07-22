"""Validated-grain plate for the ID6 fine scan: the counterpart to
id6_report_catalog.png, built from the Poisson-validated instances so the reader can
compare the raw catalog against what survives a random-orientation null.

Panels
  A, B  validated alpha / beta grains per beam position (sample-surface frame)
  C     recurrence spectrum of the VALIDATED clusters, both phases, log-log
  D     observed spot-hit counts vs the Poisson null expectation -- the evidence
        that the surviving instances are not chance fits

usage: validated_figures.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = os.environ.get("LAUE_WORK", "/net/hpcs34/data34c/for_Hemant/lauematching_ti")
SQ2 = np.sqrt(2.0)
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
COL = {"alpha": "#4269d0", "beta": "#e8843c"}
GK = {"alpha": r"\alpha", "beta": r"\beta"}
# Random-orientation null MEASURED ON THIS SCAN by scripts/null_model.py
# (120 frames x 150 draws = 18,000 draws per phase). Do NOT substitute the
# SmallArea scan's numbers -- lambda differs with peak and reflection counts.
NULL = {"alpha": dict(mean=2.95, p999=12, mx=16, lam=3.08),
        "beta":  dict(mean=1.81, p999=10, mx=15, lam=1.91)}

def centers_to_edges(c):
    c = np.asarray(c, float); d = np.diff(c).mean()
    return np.concatenate([[c[0]-d/2], (c[:-1]+c[1:])/2, [c[-1]+d/2]])

def load(phase):
    z = np.load(f"{W}/peel_map/{PREFIX}_{phase}_validated.npz", allow_pickle=True)
    return z["X"].astype(float), z["Z"].astype(float), z["labels"], z["nhit"].astype(int)

data = {}
for ph in ("alpha", "beta"):
    try:
        data[ph] = load(ph)
    except Exception as e:
        print(f"[{ph}] not available yet: {e}")
phases = [p for p in ("alpha", "beta") if p in data]
if not phases:
    raise SystemExit("no validated npz yet")

fig = plt.figure(figsize=(17, 9.6), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

for k, ph in enumerate(phases):
    X, Z, labels, nhit = data[ph]
    Xu = np.unique(np.round(X, 4)); Zu = np.unique(np.round(Z, 4))
    xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
    grid = np.zeros((len(Zu), len(Xu)))
    for x, z in zip(X, Z):
        grid[zi[round(z, 4)], xi[round(x, 4)]] += 1
    xs = centers_to_edges(Xu - Xu.min())
    zs = centers_to_edges((Zu - Zu.min()) * SQ2)
    ax = fig.add_subplot(gs[0, k])
    m = ax.pcolormesh(xs, zs, grid, cmap="viridis", shading="flat", rasterized=True)
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(rf"validated ${GK[ph]}$ grains", fontsize=9); cb.ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"sample-surface X ($\mu$m)", fontsize=9)
    ax.set_ylabel(r"sample-surface Z ($\mu$m)", fontsize=9)
    ax.set_title(rf"{'AB'[k]} $\cdot$ VALIDATED ${GK[ph]}$ grains per position ($p<10^{{-4}}$)"
                 "\n"
                 rf"{len(X):,} instances, mean {grid.mean():.1f}, up to {int(grid.max())}",
                 fontsize=10)
    ax.tick_params(labelsize=8)

# --- C: recurrence of validated clusters ---
ax = fig.add_subplot(gs[1, 0])
for j, ph in enumerate(phases):
    labels = data[ph][2]
    if labels.max() < 0:
        continue
    counts = np.bincount(labels); counts = counts[counts > 0]
    bins = np.unique(np.round(np.logspace(0, np.log10(counts.max() + 1), 40)).astype(int))
    h, e = np.histogram(counts, bins=np.append(bins, bins[-1] + 1))
    ctr = 0.5 * (e[:-1] + e[1:]); keep = h > 0
    ax.plot(ctr[keep], h[keep], "-o", color=COL[ph], lw=2, ms=4.5)
    ax.text(0.97, 0.94 - 0.11 * j,
            rf"${GK[ph]}$: {len(counts):,} validated grains, max {counts.max():,} positions",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=COL[ph], fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("beam positions where the same validated grain recurs", fontsize=9)
ax.set_ylabel("number of validated grains", fontsize=9)
ax.set_title(r"C $\cdot$ recurrence spectrum, validated grains only", fontsize=10)
ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8)

# --- D: observed hits vs the Poisson null ---
ax = fig.add_subplot(gs[1, 1])
nullmax = max(NULL[ph]["mx"] for ph in phases)
ax.axvspan(0, nullmax, color="#999", alpha=0.16, lw=0)
for j, ph in enumerate(phases):
    nhit = data[ph][3]
    b = np.arange(0, max(nhit.max(), 2) + 2)
    ax.hist(nhit, bins=b, histtype="step", lw=2, color=COL[ph], density=True)
    ax.axvline(NULL[ph]["mean"], color=COL[ph], ls=":", lw=1.6)
    ax.text(0.97, 0.94 - 0.10 * j,
            rf"${GK[ph]}$: median {int(np.median(nhit))} hits "
            rf"(null mean {NULL[ph]['mean']}, max {NULL[ph]['mx']})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=COL[ph], fontweight="bold")
ax.text(nullmax + 0.8, ax.get_ylim()[1]*0.92,
        f"shaded: entire measured null,\n18,000 random draws/phase (max {nullmax})",
        fontsize=8.5, color="#555", va="top")
ax.set_xlabel("predicted reflections landing on a real peak", fontsize=9)
ax.set_ylabel("fraction of validated instances", fontsize=9)
ax.set_title(r"D $\cdot$ evidence per validated instance vs the null", fontsize=10)
ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8)

fig.suptitle(r"Ti-6Al-4V ID26 fine scan — grains surviving the per-frame Poisson spot test "
             r"($p<10^{-4}$ against a random-orientation null)", fontsize=12)
fig.savefig(f"{W}/figures/{PREFIX}_report_validated.png", dpi=150)
print(f"saved {PREFIX}_report_validated.png")
