"""Is the background map sample structure, or detector/beam drift over scan time?

This is the control that decides whether ANY of the background story survives.
Scan order is a raster: frame index = (row-1)*201 + col. A temporal drift is
therefore a function of (row, col) that is monotone in row and repeats within
each row -- i.e. horizontal banding. Sample structure is not obliged to align
with either axis, and the observed map is a lobed diagonal ridge.

Three tests:
  1. correlation of background with frame index (scan time) over the whole map;
  2. how much variance a smooth function of scan time can explain, and whether
     the residual still contains the ridge;
  3. an axis-alignment statistic -- the gradient orientation of the background
     field. Drift gives gradients along the slow axis; the ridge does not.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "$LAUE_WORK"
z = np.load(f"{W}/survey/bgmap_stride4.npz")
rows, cols = z["rows"], z["cols"]
bg, i0, npk = z["bg25"], z["i0"], z["npk"]

R, C = np.meshgrid(rows, cols, indexing="ij")
frame_idx = (R - 1) * 201 + C           # true acquisition order
ok = np.isfinite(bg) & np.isfinite(i0) & (i0 > 1000)   # drop beam-dropout frames
print(f"grid {bg.shape}, {ok.sum()} usable of {bg.size} "
      f"({(~ok).sum()} dropped: beam dropouts / read failures)\n")

b = bg[ok]; t = frame_idx[ok].astype(float); ii = i0[ok]; pk = npk[ok]
rr = R[ok].astype(float); cc = C[ok].astype(float)


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


print("=== 1. correlations over the whole map ===")
print(f"  corr(background, frame index / scan time) = {pearson(b, t):+.4f}")
print(f"  corr(background, I0)                      = {pearson(b, ii):+.4f}")
print(f"  corr(background, row  [slow axis])        = {pearson(b, rr):+.4f}")
print(f"  corr(background, col  [fast axis, X])     = {pearson(b, cc):+.4f}")
print(f"  corr(background, peak count)              = {pearson(b, pk):+.4f}")

print("\n=== 2. variance explained ===")
def r2(design):
    A = np.column_stack([np.ones(len(b))] + design)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ coef
    return 1 - ((b - pred) ** 2).sum() / ((b - b.mean()) ** 2).sum(), pred

# a smooth (cubic) function of scan time -- the most generous drift model
tn = (t - t.mean()) / t.std()
r2_time, pred_time = r2([tn, tn ** 2, tn ** 3])
print(f"  cubic in scan time                  R^2 = {r2_time:.4f}")
r2_i0, _ = r2([(ii - ii.mean()) / ii.std()])
print(f"  I0                                  R^2 = {r2_i0:.4f}")
# a smooth function of POSITION (2-D quadratic) -- a generic geometric trend
xn = (cc - cc.mean()) / cc.std(); yn = (rr - rr.mean()) / rr.std()
r2_pos, _ = r2([xn, yn, xn ** 2, yn ** 2, xn * yn])
print(f"  2-D quadratic in position           R^2 = {r2_pos:.4f}")

resid = b - pred_time
print(f"\n  residual after removing the time model: sd {resid.std():.2f} ADU "
      f"(raw sd {b.std():.2f} ADU) -> {100*resid.std()/b.std():.1f}% of the structure survives")

print("\n=== 3. is the structure axis-aligned (drift) or not? ===")
G = np.where(np.isfinite(bg), bg, np.nan)
gy, gx = np.gradient(np.nan_to_num(G, nan=np.nanmean(G)))
m = np.isfinite(bg)
ang = np.arctan2(gy[m], gx[m])
mag = np.hypot(gx[m], gy[m])
w = mag > np.percentile(mag, 75)          # strongest gradients only
a2 = 2 * ang[w]                            # axial (mod 180 deg) statistic
Rlen = np.hypot(np.cos(a2).mean(), np.sin(a2).mean())
mean_dir = np.degrees(0.5 * np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()))
print(f"  dominant gradient direction: {mean_dir:+.1f} deg from the X axis")
print(f"  axial concentration R = {Rlen:.3f}  (0 = isotropic, 1 = a single direction)")
print("  a pure scan-time drift would put this at ~90 deg (gradient along the slow axis)")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
im = ax[0].imshow(bg, origin="lower", extent=[-100, 100, -100, 100]); ax[0].set_title("background (bg25)")
plt.colorbar(im, ax=ax[0])
P = np.full(bg.shape, np.nan); P[ok] = pred_time
im = ax[1].imshow(P, origin="lower", extent=[-100, 100, -100, 100],
                  vmin=np.nanmin(bg), vmax=np.nanmax(bg))
ax[1].set_title(f"best cubic-in-scan-time model\n$R^2$={r2_time:.3f}"); plt.colorbar(im, ax=ax[1])
Rz = np.full(bg.shape, np.nan); Rz[ok] = resid
im = ax[2].imshow(Rz, origin="lower", extent=[-100, 100, -100, 100], cmap="coolwarm")
ax[2].set_title("residual (structure a drift cannot explain)"); plt.colorbar(im, ax=ax[2])
for a in ax: a.set_xlabel("X (um)"); a.set_ylabel("45-deg axis (um)")
fig.tight_layout(); fig.savefig(f"{W}/analysis_out/drift_control.png", dpi=115, bbox_inches="tight", pad_inches=0.35)
print(f"\nwrote {W}/analysis_out/drift_control.png")
print("DRIFT_CONTROL_DONE")
