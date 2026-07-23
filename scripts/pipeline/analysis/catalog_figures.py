"""Report-quality plate for the ID6 fine scan, rebuilt from the saved npz
(no re-read of the 6561 frames).

Fixes four defects in the pipeline's own quick-look figure:
  1. title said "10x10um" -- the scan is 20x20um in the SAMPLE frame (45 deg mount)
  2. fixed-size scatter markers left white gaps -> use pcolormesh on the true 81x81 grid
  3. recurrence spectrum was unreadable (unit bins out to 1938) -> log-spaced bins, log-log
  4. "indexed grains" overclaimed -> these are per-frame indexed ORIENTATIONS, not
     verified grains; "grain" is reserved for what survives validation.

usage: catalog_figures.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = os.environ.get("LAUE_WORK", "/net/hpcs34/data34c/for_Hemant/lauematching_ti")
SQ2 = np.sqrt(2.0)
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")                      # 45 deg mount: sample-frame Z = lab Z * sqrt(2)
COL = {"alpha": "#4269d0", "beta": "#e8843c"}   # CVD-validated pair (dE 28.1 protan)
GK = {"alpha": r"\alpha", "beta": r"\beta"}

def centers_to_edges(c):
    c = np.asarray(c, float); d = np.diff(c).mean()
    return np.concatenate([[c[0]-d/2], (c[:-1]+c[1:])/2, [c[-1]+d/2]])

def load(phase):
    raw = np.load(f"{W}/peel_map/{PREFIX}_{phase}_raw.npz")
    clu = np.load(f"{W}/peel_map/{PREFIX}_{phase}.npz")
    return raw["pos"], raw["poscount"], clu["labels"]

data = {ph: load(ph) for ph in ("alpha", "beta")}

fig = plt.figure(figsize=(17, 5.8), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])

for k, ph in enumerate(("alpha", "beta")):
    pos, pc, labels = data[ph]
    Xu = np.unique(np.round(pos[:, 0], 4)); Zu = np.unique(np.round(pos[:, 1], 4))
    xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
    grid = np.full((len(Zu), len(Xu)), np.nan)
    for (x, z), n in zip(pos, pc):
        grid[zi[round(z, 4)], xi[round(x, 4)]] = n
    # sample-surface frame: X unchanged, Z de-projected by sqrt(2)
    xs = centers_to_edges(Xu - Xu.min())
    zs = centers_to_edges((Zu - Zu.min()) * SQ2)
    ax = fig.add_subplot(gs[0, k])
    m = ax.pcolormesh(xs, zs, grid, cmap="viridis", shading="flat", rasterized=True)
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(rf"indexed ${GK[ph]}$ orientations at this position", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"sample-surface X ($\mu$m)", fontsize=9)
    ax.set_ylabel(r"sample-surface Z ($\mu$m)", fontsize=9)
    nmiss = int(np.isnan(grid).sum())
    ax.set_title(rf"{'AB'[k]} $\cdot$ indexed ${GK[ph]}$ orientations per position"
                 "\n"
                 rf"mean {np.nanmean(grid):.1f}, up to {int(np.nanmax(grid))}"
                 + (f" ({nmiss} without coordinates)" if nmiss else ""),
                 fontsize=10)
    ax.tick_params(labelsize=8)

# --- panel C: recurrence spectra, both phases, log-log ---
# Labels are pinned to fixed axes-fraction slots, NOT to the curve ends: the two
# curves terminate close together and offset-point labels collided illegibly.
ax = fig.add_subplot(gs[0, 2])
for j, ph in enumerate(("alpha", "beta")):
    labels = data[ph][2]
    counts = np.bincount(labels); counts = counts[counts > 0]
    bins = np.unique(np.round(np.logspace(0, np.log10(counts.max() + 1), 40)).astype(int))
    h, e = np.histogram(counts, bins=np.append(bins, bins[-1] + 1))
    ctr = 0.5 * (e[:-1] + e[1:]); keep = h > 0
    ax.plot(ctr[keep], h[keep], "-o", color=COL[ph], lw=2, ms=4.5)
    ax.text(0.97, 0.94 - 0.11 * j,
            rf"${GK[ph]}$: {len(counts):,} distinct, max {counts.max():,} positions",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=COL[ph], fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("beam positions where the same orientation recurs", fontsize=9)
ax.set_ylabel("number of distinct orientations", fontsize=9)
ax.set_title(r"C $\cdot$ recurrence spectrum (unvalidated catalog)", fontsize=10)
ax.grid(alpha=0.25, lw=0.5)
ax.tick_params(labelsize=8)

fig.suptitle(r"Ti-6Al-4V ID26 fine scan — 81$\times$81 raster, 0.25 $\mu$m step, "
             r"20$\times$20 $\mu$m in the sample frame "
             r"(45$^\circ$ mount; 20$\times$14.14 $\mu$m projected in the lab)",
             fontsize=12)
fig.savefig(f"{W}/figures/{PREFIX}_report_catalog.png", dpi=150)
print(f"saved {PREFIX}_report_catalog.png")
