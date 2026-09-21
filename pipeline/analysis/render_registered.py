"""Render the pedestal (deposit proxy) and grain-size maps in the optical frame.

The optical imager is vertically flipped relative to the scan frame, so flip the
45-deg (row) axis. Output both the deposit proxy (fluorescence pedestal; high =
more Zn = should be the BLACK plated regions) and the grain-footprint map
(bright = large contiguous grain = substrate; dark = fine deposit).

If high pedestal / fine grains land on the black deposit islands of the
micrograph, the pedestal is validated as a deposit map.

Environment: LAUE_WORK, LAUE_NR (columns), LAUE_NROWS (rows), LAUE_STEP_UM (raster
step, um) and LAUE_OPTICAL_FLIP_Y (+1 if the micrograph is vertically flipped vs the
scan, -1 if not), all required. Positions are LAUE_STEP_UM apart about the centre.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raster import (centred_extent, optical_flip_y, raster_positions, raster_shape,
                    ranking_counts, step_um, winner_per_position)

W = os.environ.get("LAUE_WORK")
if not W:
    sys.exit("LAUE_WORK is not set (work directory holding peel_map/ and analysis_out/)")
NROWS, NR = raster_shape()                # LAUE_NROWS, LAUE_NR -- required
EXT = centred_extent(NROWS, NR, step_um())   # [-100, 100, -100, 100] for 201 x 201 at 1 um
HX, HY = EXT[1], EXT[3]
FLIP_Y = optical_flip_y()                 # LAUE_OPTICAL_FLIP_Y -- required

flat = np.load(f"{W}/analysis_out/full_pedestal.npz")["flat"]        # (NROWS, NR)

# grain footprint per position: size of the cluster of the ONE top-ranked
# orientation there (winner-take-all, by distinct peaks matched)
z = np.load(f"{W}/peel_map/full_zn_clustered.npz", allow_pickle=True)
lab, fr = z["labels"], z["frames"]
row, col = raster_positions(fr, Z=z["Z"] if "Z" in z.files else None, shape=(NROWS, NR))
cnt = np.bincount(lab)
prim, sec, _ = ranking_counts(z)
best = winner_per_position(row, col, prim, sec, tiebreak=z["oms"].reshape(len(lab), -1))
foot = np.full((NROWS, NR), np.nan)
for (r, c), i in best.items():
    foot[r, c] = cnt[lab[i]]

fped = flat[::-1, :] if FLIP_Y > 0 else flat        # -> optical frame
ffoot = foot[::-1, :] if FLIP_Y > 0 else foot
FLIPTXT = "vertical flip" if FLIP_Y > 0 else "no flip"

fig, ax = plt.subplots(2, 2, figsize=(12, 11))
for a, m, t, kw in [
    (ax[0, 0], flat, "pedestal (deposit proxy) — SCAN frame", dict(cmap="inferno")),
    (ax[0, 1], fped, f"pedestal -- OPTICAL frame ({FLIPTXT})\nbright = more Zn → should be BLACK deposit", dict(cmap="inferno")),
    (ax[1, 0], np.ma.masked_invalid(foot), "grain footprint — SCAN frame",
     dict(cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.nanmax(foot)))),
    (ax[1, 1], np.ma.masked_invalid(ffoot), f"grain footprint -- OPTICAL frame ({FLIPTXT})\nbright = large grain = SUBSTRATE (gold)",
     dict(cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.nanmax(foot)))),
]:
    im = a.imshow(m, origin="lower", extent=EXT, interpolation="nearest", **kw)
    a.set_title(t, fontsize=10)
    a.set_xlabel("X (µm)"); a.set_ylabel("45° axis (µm)")
    a.plot(0, 0, "o", mfc="none", mec="cyan", ms=14, mew=2)   # scan centre (the circles)
    plt.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(f"Zn maps registered to the optical frame ({2 * HX:.0f} × {2 * HY:.0f} µm, "
             "centre = the circles)", y=1.0)
fig.tight_layout()
fig.savefig(f"{W}/analysis_out/registered_maps.png", dpi=120, bbox_inches="tight", pad_inches=0.3)
print("wrote registered_maps.png")

# correlation between the two proxies, for the record
m = np.isfinite(foot) & np.isfinite(flat)
print(f"corr(pedestal, log grain footprint) over the map = "
      f"{np.corrcoef(flat[m], np.log10(foot[m]))[0,1]:+.3f}")
print("REGISTER_DONE")
