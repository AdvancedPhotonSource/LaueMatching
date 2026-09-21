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
    LAUE_OUT_PREFIX  prefix of peel_map/<prefix>_<phase>_clustered.npz   (frame_peaks.out_prefix)
    LAUE_PHASE       phase name  (required unless LAUE_PHASES lists exactly one)
    LAUE_NR          frames per raster row = columns      (required, no default)
    LAUE_NROWS       number of raster rows                (required, no default)
    LAUE_STEP_UM     raster step in micrometres           (required, no default)
    LAUE_IN_NPZ      explicit input npz, overrides the prefix/phase construction

The raster shape is given explicitly rather than assumed: the earlier hardcoded
201x201 silently mis-shaped any scan that was not (sampleH is 201x101), and the later
``LAUE_NR`` default of 201 mis-placed any scan of another width. Position and the
choice of one orientation per position come from ``raster.py``.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raster import raster_positions, raster_shape, ranking_counts, step_um, winner_per_position

W = os.environ.get("LAUE_WORK")
if not W:
    sys.exit("LAUE_WORK is not set (work directory holding peel_map/ and analysis_out/)")
from frame_peaks import out_prefix
PREFIX = out_prefix()      # one default for every script (was "full" here only)
from laue_material import phase_name
PHASE = phase_name()
NROW, NR = raster_shape()          # LAUE_NROWS, LAUE_NR -- required
STEP = step_um()                   # LAUE_STEP_UM -- required
SRC = os.environ.get("LAUE_IN_NPZ") or f"{W}/peel_map/{PREFIX}_{PHASE}_clustered.npz"

z = np.load(SRC, allow_pickle=True)
oms, lab, X, Z, fr = (z["oms"], z["labels"], z["X"].astype(float),
                      z["Z"].astype(float), z["frames"])
print(f"{len(oms)} instances from {SRC}")

# raster indices from the frame number and the explicit raster shape, checked
# against the stage coordinates (see raster.raster_positions)
row, col = raster_positions(fr, X=X, Z=Z, shape=(NROW, NR))
print(f"raster {NR} x {NROW} (columns x rows, from LAUE_NR / LAUE_NROWS)")

# one orientation per position (winner-take-all): ranked by distinct peaks matched,
# ties broken by nhit and then by the orientation itself (raster.winner_per_position)
prim, sec, rank_name = ranking_counts(z)
print(f"one orientation per position: top-ranked by {rank_name}")
best = winner_per_position(row, col, prim, sec, tiebreak=oms.reshape(len(oms), -1))


def caxis_rgb(OM):
    """RGB from the c-axis direction, folded to one octant (hex Laue symmetry)."""
    c = OM[:, 2]
    c = c / np.linalg.norm(c)
    return np.abs(c)


rgb = np.zeros((NROW, NR, 3))
alpha = np.zeros((NROW, NR))
csize = np.zeros((NROW, NR))
cnt = np.bincount(lab[lab >= 0])
for (r, c), i in best.items():
    rgb[r, c] = caxis_rgb(oms[i])
    alpha[r, c] = 1.0
    csize[r, c] = cnt[lab[i]]

EXT = [0, NR * STEP, 0, NROW * STEP]        # micrometres
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
