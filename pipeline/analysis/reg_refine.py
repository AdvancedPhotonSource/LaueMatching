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
The clustered npz is peel_map/<LAUE_OUT_PREFIX>_*_clustered.npz (or LAUE_CLUSTERED_NPZ);
it used to be one campaign's hard-coded file name.

THE BEST OF 1,260 REGISTRATIONS NEEDS A NULL. The best correlation over the grid is a
maximum over many tries on autocorrelated maps; before 2026-09 it was reported bare.
Its null (invariant 10): toroidally shift the footprint map and redo the WHOLE grid
search, LAUE_NULL_REPS times (default 200); p(best) is the fraction of shifted maps
whose best-of-grid is at least as negative. The measured registration's single
correlation gets the same shift null. On a known-null synthetic a naive Pearson test
of the best-of-grid value flagged 76% at p < 0.05; this null 5%.
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from raster import (optical_registration, positions_per_label, raster_positions,
                    raster_shape, ranking_counts, step_um, toroidal_shift_null,
                    winner_per_position)
from frame_peaks import clustered_npz

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
z = np.load(clustered_npz(W), allow_pickle=True)
lab, fr = z["labels"], z["frames"]
gr, gc = raster_positions(fr, Z=z["Z"] if "Z" in z.files else None, shape=(NROWS, NR))
cnt = positions_per_label(lab, gr, gc)       # footprint in POSITIONS, not instances
foot = np.full((NROWS, NR), np.nan)
prim, sec, _ = ranking_counts(z)
best = winner_per_position(gr, gc, prim, sec, tiebreak=z["oms"].reshape(len(lab), -1))
for (rr, cc2), i in best.items():
    foot[rr, cc2] = cnt[lab[i]]
lf = np.log10(foot)
okmap = np.isfinite(lf)

rr, cc = np.meshgrid(np.arange(NROWS), np.arange(NR), indexing="ij")   # rr=45deg, cc=X
Xum = (cc - (NR - 1) / 2) * STEP; Yum = (rr - (NROWS - 1) / 2) * STEP


def black_at(cx, cy, ppu, rot_deg, flipy=FLIP_Y):
    """The registered optical deposit mask on the scan grid, for one registration."""
    th = np.radians(rot_deg)
    xr = Xum * np.cos(th) - Yum * np.sin(th)
    yr = Xum * np.sin(th) + Yum * np.cos(th)
    px = cx + xr * ppu
    py = cy + flipy * yr * ppu
    pxi = np.clip(np.round(px).astype(int), 0, im.shape[1] - 1)
    pyi = np.clip(np.round(py).astype(int), 0, im.shape[0] - 1)
    return black[pyi, pxi]


GRID = [(CX0, CY0, PPU0, 0)] + [
    (cx, cy, ppu, rot)
    for cx in CX0 + np.arange(-10, 11, 4)
    for cy in CY0 + np.arange(-10, 11, 4)
    for ppu in PPU0 * (1 + np.arange(-2, 3) / 15)
    for rot in (-12, -8, -4, 0, 4, 8, 12)]
# every registration's mask, once (bool, (n_grid, NROWS*NR))
BL = np.stack([black_at(*g).ravel() > 0.5 for g in GRID])


def grid_corrs(f):
    """corr(f, mask_k) for every registration k, over the finite entries of f."""
    v = np.asarray(f, float).ravel()
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full(len(GRID), np.nan)
    a = v[ok] - v[ok].mean()
    B = BL[:, ok].astype(float)
    B -= B.mean(axis=1, keepdims=True)
    d = np.sqrt((a @ a) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (B @ a) / d, 0.0)


rs = grid_corrs(lf)
base = float(rs[0])
k = int(np.nanargmin(rs))                    # most negative = best agreement
best_r = (float(rs[k]),) + tuple(float(x) for x in GRID[k])
print(f"measured registration: corr(log footprint, black) = {base:+.3f}")
print(f"best in scan ({len(GRID)} registrations): corr = {best_r[0]:+.3f} at cx={best_r[1]} "
      f"cy={best_r[2]} px/um={best_r[3]} rot={best_r[4]} deg")
NREPS = int(os.environ.get("LAUE_NULL_REPS", "200"))
t_base = toroidal_shift_null(lambda f: float(grid_corrs(f)[0]), lf, n=NREPS, rng=0,
                             alternative="less")
t_best = toroidal_shift_null(lambda f: float(np.nanmin(grid_corrs(f))), lf, n=NREPS, rng=1,
                             alternative="less")
print(f"  spatial null (footprint map toroidally shifted, {t_best['n']} shifts, whole grid "
      f"re-searched each time): p(measured) = {t_base['p']:.4g}, p(best of grid) = "
      f"{t_best['p']:.4g}")
print(f"  (measured {base:+.3f} vs best {best_r[0]:+.3f}: "
      f"{'measured already near-optimal' if abs(best_r[0]-base)<0.06 else 'refinement helps'}"
      f"; the best-of-grid value is only evidence if p(best of grid) is small)")
np.savez(f"{W}/analysis_out/reg_refine.npz", base=base, best=np.array(best_r),
         p_base_toroidal=t_base["p"], p_best_toroidal=t_best["p"])
print("REFINE_DONE")
