"""IPF-colored grain map + grain-size distribution.

Replaces the cluster-ID-colored map, which was misleading: connected-components
labelling numbers clusters in raster order, so colouring by label id produced a
smooth top-to-bottom gradient that looked like structure but was just the
numbering. Here each position is coloured by the crystal orientation itself
(the c-axis direction in the LAB frame -> RGB), so a contiguous region of one
grain is a uniform colour and fine material is speckle -- real structure, not an
artefact of label order.

FRAME. The orientation matrices are in the LAB frame and lab Z is the incident beam,
so the RGB channels here are (|x|,|y|,|z|) of the c-axis against BEAM-relative axes,
not sample-relative ones. For a texture statement, convert to the specimen surface
normal first -- derive it from the measured stage motion (both raster axes lie in the
surface, so their cross product is the normal). See Phase 1 of the runbook.

Configuration is by environment, matching the rest of the chain:

    LAUE_WORK        work directory                       (required)
    LAUE_OUT_PREFIX  prefix of peel_map/<prefix>_<phase>_clustered.npz   (default "full")
    LAUE_PHASE       phase name                           (default "zn")
    LAUE_NR          frames per raster row                (default 201)
    LAUE_IN_NPZ      explicit input npz, overrides the prefix/phase construction

The number of rows is derived from the data rather than assumed square: the earlier
hardcoded 201x201 silently mis-shaped any scan that was not (sampleH is 201x101).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = os.environ.get("LAUE_WORK")
if not W:
    sys.exit("LAUE_WORK is not set (work directory holding peel_map/ and analysis_out/)")
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "full")
PHASE = os.environ.get("LAUE_PHASE", "zn")
NR = int(os.environ.get("LAUE_NR", "201"))
SRC = os.environ.get("LAUE_IN_NPZ") or f"{W}/peel_map/{PREFIX}_{PHASE}_clustered.npz"

z = np.load(SRC, allow_pickle=True)
oms, lab, X, Z, nh, fr = (z["oms"], z["labels"], z["X"].astype(float),
                          z["Z"].astype(float), z["nhit"].astype(int), z["frames"])
print(f"{len(oms)} instances from {SRC}")

# raster indices from the source filename (robust; X/Z are floats)
n = np.array([int(str(f).split("_")[-1].split(".")[0]) for f in fr])
row = (n - 1) // NR
col = (n - 1) % NR
NROW = int(row.max()) + 1
print(f"raster {NR} x {NROW} (rows derived from the data, not assumed square)")

# one orientation per position: the highest-nhit validated instance there
best = {}
for i in range(len(oms)):
    key = (row[i], col[i])
    if key not in best or nh[i] > best[key][1]:
        best[key] = (i, nh[i])


def caxis_rgb(OM):
    """RGB from the c-axis direction, folded to one octant (hex Laue symmetry)."""
    c = OM[:, 2]
    c = c / np.linalg.norm(c)
    return np.abs(c)


rgb = np.zeros((NROW, NR, 3))
alpha = np.zeros((NROW, NR))
csize = np.zeros((NROW, NR))
cnt = np.bincount(lab[lab >= 0])
for (r, c), (i, _) in best.items():
    rgb[r, c] = caxis_rgb(oms[i])
    alpha[r, c] = 1.0
    csize[r, c] = cnt[lab[i]]

EXT = [0, NR, 0, NROW]        # micrometres at 1 um steps
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(np.dstack([rgb, alpha]), origin="lower", extent=EXT, interpolation="nearest")
ax[0].set_title("orientation map — RGB = c-axis direction (lab frame)\n"
                "uniform colour = one grain; speckle = fine material")
ax[0].set_xlabel("X (µm)"); ax[0].set_ylabel("45° axis (µm)")

lm = np.ma.masked_where(csize == 0, csize)
im = ax[1].imshow(lm, origin="lower", extent=EXT, cmap="viridis",
                  norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(cnt.max(), 2)),
                  interpolation="nearest")
ax[1].set_title("size of the grain occupying each position\n"
                "(instances in its cluster; interpret against THIS sample's morphology)")
ax[1].set_xlabel("X (µm)"); ax[1].set_ylabel("45° axis (µm)")
plt.colorbar(im, ax=ax[1], label="cluster size")
fig.tight_layout()
outdir = f"{W}/analysis_out/figures"
os.makedirs(outdir, exist_ok=True)
out = f"{outdir}/plate_grainmap_ipf_{PREFIX}.png"
fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=0.35)
print(f"wrote {out}")

occupied = csize[csize > 0]
print(f"positions with a validated orientation: {(csize>0).sum()} of {NR*NROW} "
      f"({(csize>0).mean()*100:.1f}%)")
print(f"positions in a grain of >=100 instances: "
      f"{(csize>=100).sum()} ({(occupied>=100).mean()*100:.1f}% of occupied)")
print(f"positions in a grain of <10 instances: "
      f"{((csize>0)&(csize<10)).sum()} ({(occupied<10).mean()*100:.1f}% of occupied)")
print("IPF_DONE")
