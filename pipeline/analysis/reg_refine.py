"""Registration sensitivity: is the grain-size<->optical-deposit agreement robust,
or an artefact of a slightly-off registration? Scan small shift/rotation/scale
around the measured values (scale bar + markers + vertical flip) and report where
the agreement peaks. If the measured point is already near-optimal, the result is
solid; the location of the optimum also refines the registration honestly.

Environment: LAUE_WORK, LAUE_NR (columns), LAUE_NROWS (rows), LAUE_STEP_UM (raster
step, um) and the measured registration LAUE_OPTICAL_CX/_CY/_PX_PER_UM/_FLIP_Y
(raster.optical_registration), all required. The search is a fixed neighbourhood of
the measured registration: centre +-10 px in 4 px steps, scale +-2 steps of 1/15 of
the measured value, rotation +-12 deg (for 0.6 px/um this is the original grid). The grain footprint takes one orientation
per position, top-ranked by distinct peaks matched (raster.winner_per_position).
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from raster import (optical_registration, raster_positions, raster_shape, ranking_counts,
                    step_um, winner_per_position)

W = os.environ.get("LAUE_WORK")
if not W:
    sys.exit("LAUE_WORK is not set (work directory holding optical.png, peel_map/ and analysis_out/)")
NROWS, NR = raster_shape()                # LAUE_NROWS, LAUE_NR -- required
STEP = step_um()                          # LAUE_STEP_UM -- required
CX0, CY0, PPU0, FLIP_Y = optical_registration()      # LAUE_OPTICAL_* -- required

im = np.array(Image.open(f"{W}/optical.png").convert("RGB")).astype(float)
Rr, Gg, Bb = im[:, :, 0], im[:, :, 1], im[:, :, 2]
lum = 0.30 * Rr + 0.59 * Gg + 0.11 * Bb
overlay = ((Gg > 110) & (Rr < 120) & (Bb < 120)) | ((Rr > 210) & (Gg > 210) & (Bb > 210)) \
          | ((Rr > 150) & (Gg < 90) & (Bb < 90)) | ((Bb > 140) & (Rr < 110))
lc = lum.copy(); lc[overlay] = np.nan
lc = lc[tuple(ndi.distance_transform_edt(np.isnan(lc), return_distances=False, return_indices=True))]
lum_s = ndi.median_filter(lc, 5)
h, e = np.histogram(lum_s[~overlay], 256, (0, 256)); ctr = 0.5 * (e[:-1] + e[1:])
band = (ctr > 40) & (ctr < 190)
thr = ctr[band][np.argmin(ndi.gaussian_filter1d(h.astype(float), 3)[band])]
black = (lum_s < thr).astype(float)

# grain footprint map
z = np.load(f"{W}/peel_map/full_zn_clustered.npz", allow_pickle=True)
lab, fr = z["labels"], z["frames"]
gr, gc = raster_positions(fr, Z=z["Z"] if "Z" in z.files else None, shape=(NROWS, NR))
cnt = np.bincount(lab); foot = np.full((NROWS, NR), np.nan)
prim, sec, _ = ranking_counts(z)
best = winner_per_position(gr, gc, prim, sec, tiebreak=z["oms"].reshape(len(lab), -1))
for (rr, cc2), i in best.items():
    foot[rr, cc2] = cnt[lab[i]]
lf = np.log10(foot)
okmap = np.isfinite(lf)

rr, cc = np.meshgrid(np.arange(NROWS), np.arange(NR), indexing="ij")   # rr=45deg, cc=X
Xum = (cc - (NR - 1) / 2) * STEP; Yum = (rr - (NROWS - 1) / 2) * STEP


def corr_at(cx, cy, ppu, rot_deg, flipy=FLIP_Y):
    th = np.radians(rot_deg)
    xr = Xum * np.cos(th) - Yum * np.sin(th)
    yr = Xum * np.sin(th) + Yum * np.cos(th)
    px = cx + xr * ppu
    py = cy + flipy * yr * ppu
    pxi = np.clip(np.round(px).astype(int), 0, im.shape[1] - 1)
    pyi = np.clip(np.round(py).astype(int), 0, im.shape[0] - 1)
    bl = black[pyi, pxi]
    a, b = lf[okmap], bl[okmap]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


base = corr_at(CX0, CY0, PPU0, 0)
print(f"measured registration: corr(log footprint, black) = {base:+.3f}")
best_r = (base, CX0, CY0, PPU0, 0)
for cx in CX0 + np.arange(-10, 11, 4):
    for cy in CY0 + np.arange(-10, 11, 4):
        for ppu in PPU0 * (1 + np.arange(-2, 3) / 15):
            for rot in (-12, -8, -4, 0, 4, 8, 12):
                r = corr_at(cx, cy, ppu, rot)
                if r < best_r[0]:                    # most negative = best agreement
                    best_r = (r, cx, cy, ppu, rot)
print(f"best in scan: corr = {best_r[0]:+.3f} at cx={best_r[1]} cy={best_r[2]} "
      f"px/um={best_r[3]} rot={best_r[4]} deg")
print(f"  (measured {base:+.3f} vs best {best_r[0]:+.3f}: "
      f"{'measured already near-optimal' if abs(best_r[0]-base)<0.06 else 'refinement helps'})")
np.savez(f"{W}/analysis_out/reg_refine.npz", base=base, best=np.array(best_r))
print("REFINE_DONE")
