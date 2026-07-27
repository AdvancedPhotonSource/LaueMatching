"""IPF-colored grain map + grain-size distribution for the Zn scan.

Replaces the cluster-ID-colored map, which was misleading: connected-components
labelling numbers clusters in raster order, so colouring by label id produced a
smooth top-to-bottom gradient that looked like structure but was just the
numbering. Here each position is coloured by the crystal orientation itself
(the c-axis direction in the sample frame -> RGB), so a contiguous region of one
grain is a uniform colour and fine deposit is speckle -- real structure, not an
artefact of label order.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "$LAUE_WORK"
NR = 201

z = np.load(f"{W}/peel_map/full_zn_clustered.npz", allow_pickle=True)
oms, lab, X, Z, nh, fr = (z["oms"], z["labels"], z["X"].astype(float),
                          z["Z"].astype(float), z["nhit"].astype(int), z["frames"])

# raster indices from the source filename (robust; X/Z are floats)
n = np.array([int(str(f).split("_")[-1].split(".")[0]) for f in fr])
row = (n - 1) // NR
col = (n - 1) % NR

# one orientation per position: the highest-nhit validated instance there
best = {}
for i in range(len(oms)):
    key = (row[i], col[i])
    if key not in best or nh[i] > best[key][1]:
        best[key] = (i, nh[i])

# IPF-ish RGB from the c-axis direction in the sample frame
def caxis_rgb(OM):
    c = OM[:, 2]                       # crystal c in sample frame
    c = c / np.linalg.norm(c)
    c = np.abs(c)                      # fold to one octant (hex Laue symmetry)
    return c[[0, 1, 2]]                # (|x|,|y|,|z|) -> RGB

rgb = np.zeros((NR, NR, 3))
alpha = np.zeros((NR, NR))
csize = np.zeros((NR, NR))
cnt = np.bincount(lab)
for (r, c), (i, _) in best.items():
    rgb[r, c] = caxis_rgb(oms[i])
    alpha[r, c] = 1.0
    csize[r, c] = cnt[lab[i]]

img = np.dstack([rgb, alpha])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(img, origin="lower", extent=[-100, 100, -100, 100], interpolation="nearest")
ax[0].set_title("orientation map — RGB = c-axis direction in sample frame\n"
                "(uniform colour = one grain; speckle = fine deposit)")
ax[0].set_xlabel("X (µm)"); ax[0].set_ylabel("45° axis (µm)")

# footprint of the grain occupying each position, log scale — substrate vs deposit
lm = np.ma.masked_where(csize == 0, csize)
im = ax[1].imshow(lm, origin="lower", extent=[-100, 100, -100, 100],
                  cmap="viridis", norm=matplotlib.colors.LogNorm(vmin=1, vmax=cnt.max()),
                  interpolation="nearest")
ax[1].set_title("footprint of the grain at each position (instances in its cluster)\n"
                "bright = large contiguous grain (substrate); dark = fine (deposit)")
ax[1].set_xlabel("X (µm)"); ax[1].set_ylabel("45° axis (µm)")
plt.colorbar(im, ax=ax[1], label="cluster size")
fig.tight_layout()
fig.savefig(f"{W}/analysis_out/figures/plate_grainmap_ipf.png", dpi=130, bbox_inches="tight", pad_inches=0.35)
print("wrote plate_grainmap_ipf.png")

# quantify the two populations
occupied = csize[csize > 0]
print(f"positions with a validated orientation: {(csize>0).sum()} of {NR*NR} "
      f"({(csize>0).mean()*100:.1f}%)")
print(f"positions in a grain of >=100 instances (substrate-like): "
      f"{(csize>=100).sum()} ({(occupied>=100).mean()*100:.1f}% of occupied)")
print(f"positions in a grain of <10 instances (deposit-like): "
      f"{((csize>0)&(csize<10)).sum()} ({((occupied>0)&(occupied<10)).mean()*100:.1f}% of occupied)")
print("IPF_DONE")
