"""Render the pedestal (deposit proxy) and grain-size maps in the optical frame.

The optical imager is vertically flipped relative to the scan frame, so flip the
45-deg (row) axis. Output both the deposit proxy (fluorescence pedestal; high =
more Zn = should be the BLACK plated regions) and the grain-footprint map
(bright = large contiguous grain = substrate; dark = fine deposit).

If high pedestal / fine grains land on the black deposit islands of the
micrograph, the pedestal is validated as a deposit map.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "$LAUE_WORK"
NR = 201
EXT = [-100, 100, -100, 100]

flat = np.load(f"{W}/analysis_out/full_pedestal.npz")["flat"]        # (201,201)

# grain footprint per position (size of the dominant grain's cluster)
z = np.load(f"{W}/peel_map/full_zn_clustered.npz", allow_pickle=True)
lab, fr, nh = z["labels"], z["frames"], z["nhit"].astype(int)
n = np.array([int(str(f).split("_")[-1].split(".")[0]) for f in fr])
row, col = (n - 1) // NR, (n - 1) % NR
cnt = np.bincount(lab)
best = {}
for i in range(len(lab)):
    k = (row[i], col[i])
    if k not in best or nh[i] > best[k][1]:
        best[k] = (i, nh[i])
foot = np.full((NR, NR), np.nan)
for (r, c), (i, _) in best.items():
    foot[r, c] = cnt[lab[i]]

fped = flat[::-1, :]                    # vertical flip -> optical frame
ffoot = foot[::-1, :]

fig, ax = plt.subplots(2, 2, figsize=(12, 11))
for a, m, t, kw in [
    (ax[0, 0], flat, "pedestal (deposit proxy) — SCAN frame", dict(cmap="inferno")),
    (ax[0, 1], fped, "pedestal — OPTICAL frame (vertical flip)\nbright = more Zn → should be BLACK deposit", dict(cmap="inferno")),
    (ax[1, 0], np.ma.masked_invalid(foot), "grain footprint — SCAN frame",
     dict(cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.nanmax(foot)))),
    (ax[1, 1], np.ma.masked_invalid(ffoot), "grain footprint — OPTICAL frame (flip)\nbright = large grain = SUBSTRATE (gold)",
     dict(cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.nanmax(foot)))),
]:
    im = a.imshow(m, origin="lower", extent=EXT, interpolation="nearest", **kw)
    a.set_title(t, fontsize=10)
    a.set_xlabel("X (µm)"); a.set_ylabel("45° axis (µm)")
    a.plot(0, 0, "o", mfc="none", mec="cyan", ms=14, mew=2)   # scan centre (the circles)
    plt.colorbar(im, ax=a, fraction=0.046)
fig.suptitle("Zn maps registered to the optical frame (200 × 200 µm, centre = the circles)", y=1.0)
fig.tight_layout()
fig.savefig(f"{W}/analysis_out/registered_maps.png", dpi=120, bbox_inches="tight", pad_inches=0.3)
print("wrote registered_maps.png")

# correlation between the two proxies, for the record
m = np.isfinite(foot) & np.isfinite(flat)
print(f"corr(pedestal, log grain footprint) over the map = "
      f"{np.corrcoef(flat[m], np.log10(foot[m]))[0,1]:+.3f}")
print("REGISTER_DONE")
