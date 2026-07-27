"""Register the scan maps to the optical micrograph and test the deposit proxy.

Registration (measured from optical.png):
  scale     60 px = 100 um  -> 0.600 px/um
  centre    the red/blue scan-centre markers at pixel (398, 284)
  flip      the optical imager is vertically flipped vs the scan frame
  no rotation, no x-flip (user: "vertically flipped, that's all")

For each of the 201x201 scan positions (X = col, 45deg-axis = row) map to an
optical pixel, sample the black/gold classification, and test whether the
fluorescence pedestal (the deposit proxy) predicts the BLACK (grown-Zn) regions.
"""
import numpy as np
from PIL import Image

W = "$LAUE_WORK"
NR = 201
PX_PER_UM = 60.0 / 100.0                  # scale bar: 60 px = 100 um
CX, CY = 398.0, 284.0                     # scan centre in optical pixels
FLIP_Y = +1                               # +1 => vertical flip applied (imager is flipped)

# --- optical image -> black(deposit)/gold(substrate) classification ---------
im = np.array(Image.open(f"{W}/optical.png").convert("RGB")).astype(float)
Rr, Gg, Bb = im[:, :, 0], im[:, :, 1], im[:, :, 2]
lum = 0.30 * Rr + 0.59 * Gg + 0.11 * Bb
from scipy import ndimage as ndi
# overlay pixels that are NOT sample: green grid, and bright white text/bar/markers
overlay = ((Gg > 110) & (Rr < 120) & (Bb < 120)) | ((Rr > 210) & (Gg > 210) & (Bb > 210)) \
          | ((Rr > 150) & (Gg < 90) & (Bb < 90)) | ((Bb > 140) & (Rr < 110))
# replace overlay pixels with a large-radius median so they do not create fake black/gold
lum_clean = lum.copy()
lum_clean[overlay] = np.nan
# fill NaNs by nearest-neighbour, then smooth
idx = ndi.distance_transform_edt(np.isnan(lum_clean), return_distances=False, return_indices=True)
lum_clean = lum_clean[tuple(idx)]
lum_s = ndi.median_filter(lum_clean, 5)
# threshold black(deposit) vs gold(substrate): valley between the two peaks (search 40..190)
h, edges = np.histogram(lum_s[~overlay], 256, (0, 256))
ctr = 0.5 * (edges[:-1] + edges[1:])
band = (ctr > 40) & (ctr < 190)
hs = ndi.gaussian_filter1d(h.astype(float), 3)
valley = ctr[band][np.argmin(hs[band])]
thr = float(valley)
black = lum_s < thr
print(f"optical: {im.shape[1]}x{im.shape[0]}, overlay px masked = {overlay.sum()}")
print(f"black/gold valley threshold = {thr:.0f}")
print(f"black (deposit) area fraction (sample pixels): {(black & ~overlay).mean()/(~overlay).mean()*100:.1f}%")

# --- map each scan position to an optical pixel, sample black/gold ----------
r = np.arange(NR); c = np.arange(NR)
CC, RR = np.meshgrid(c, r)                  # RR = 45deg axis (row), CC = X (col)
Xum = (CC - (NR - 1) / 2)                    # -100..100 um
Yum = (RR - (NR - 1) / 2)
px = CX + Xum * PX_PER_UM
py = CY + FLIP_Y * Yum * PX_PER_UM           # vertical flip
pxi = np.clip(np.round(px).astype(int), 0, im.shape[1] - 1)
pyi = np.clip(np.round(py).astype(int), 0, im.shape[0] - 1)
is_black = black[pyi, pxi].astype(float)     # (NR,NR) registered deposit mask
print(f"scan footprint on the optical image: x {pxi.min()}-{pxi.max()}, y {pyi.min()}-{pyi.max()} px")
print(f"black fraction WITHIN the scan box: {is_black.mean()*100:.1f}%")

# --- the maps ---------------------------------------------------------------
flat = np.load(f"{W}/analysis_out/full_pedestal.npz")["flat"]        # (row=45deg, col=X)
z = np.load(f"{W}/peel_map/full_zn_clustered.npz", allow_pickle=True)
lab, fr, nh = z["labels"], z["frames"], z["nhit"].astype(int)
n = np.array([int(str(f).split("_")[-1].split(".")[0]) for f in fr])
gr, gc = (n - 1) // NR, (n - 1) % NR
cnt = np.bincount(lab)
foot = np.full((NR, NR), np.nan); best = {}
for i in range(len(lab)):
    k = (gr[i], gc[i])
    if k not in best or nh[i] > nh[best[k]]:
        best[k] = i
for (rr, cc2), i in best.items():
    foot[rr, cc2] = cnt[lab[i]]

# --- THE TEST: does the pedestal predict the black (deposit) regions? -------
m = np.isfinite(flat)
pv = flat[m]; bv = is_black[m]
def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))
r_pb = pear(pv, bv)
print("\n=== PEDESTAL vs OPTICAL DEPOSIT (black) ===")
print(f"  corr(pedestal, is_black) = {r_pb:+.3f}")
print(f"  pedestal on BLACK positions : {pv[bv > 0.5].mean():.1f} ADU (n={int((bv>0.5).sum())})")
print(f"  pedestal on GOLD positions  : {pv[bv < 0.5].mean():.1f} ADU (n={int((bv<0.5).sum())})")
print(f"  difference: {pv[bv>0.5].mean() - pv[bv<0.5].mean():+.1f} ADU")
# AUC: how well pedestal ranks black vs gold
from scipy.stats import rankdata
rk = rankdata(pv); npos = (bv > 0.5).sum(); nneg = (bv < 0.5).sum()
auc = (rk[bv > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg)
print(f"  AUC (pedestal ranks black>gold): {auc:.3f}  (0.5 = chance)")

fo = foot[m]; ok = np.isfinite(fo)
print(f"\n=== GRAIN SIZE vs OPTICAL DEPOSIT ===")
print(f"  corr(log footprint, is_black) = {pear(np.log10(fo[ok]), bv[ok]):+.3f}")
print(f"  median grain footprint on BLACK: {np.median(fo[ok & (bv>0.5)]):.0f}  "
      f"on GOLD: {np.median(fo[ok & (bv<0.5)]):.0f}")

# --- overlay figure ---------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
# 1: optical crop with scan box + centre
x0, x1 = int(CX - 100 * PX_PER_UM), int(CX + 100 * PX_PER_UM)
y0, y1 = int(CY - 100 * PX_PER_UM), int(CY + 100 * PX_PER_UM)
ax[0].imshow(im.astype(np.uint8))
ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=2))
ax[0].plot(CX, CY, "+", color="red", ms=12, mew=2)
ax[0].set_title("optical: scan box (cyan) + centre"); ax[0].axis("off")
# 2: registered deposit mask
ax[1].imshow(is_black, origin="lower", extent=[-100, 100, -100, 100], cmap="gray_r")
ax[1].set_title("optical deposit (black) registered to scan"); ax[1].set_xlabel("X µm")
# 3: pedestal, same frame
im3 = ax[2].imshow(flat, origin="lower", extent=[-100, 100, -100, 100], cmap="inferno")
ax[2].contour(np.linspace(-100, 100, NR), np.linspace(-100, 100, NR), is_black,
              levels=[0.5], colors="cyan", linewidths=1.2)
ax[2].set_title("pedestal (deposit proxy) + optical-deposit outline"); ax[2].set_xlabel("X µm")
plt.colorbar(im3, ax=ax[2], fraction=0.046)
fig.suptitle(f"Optical registration: corr(pedestal, black) = {r_pb:+.2f}, AUC = {auc:.2f}", y=1.02)
fig.tight_layout(); fig.savefig(f"{W}/analysis_out/optical_overlay.png", dpi=120,
                                bbox_inches="tight", pad_inches=0.3)
np.savez(f"{W}/analysis_out/optical_registration.npz",
         is_black=is_black, px_per_um=PX_PER_UM, cx=CX, cy=CY, thr=thr, auc=auc, r_pb=r_pb)
print("\nwrote optical_overlay.png")
print("OPTICAL_DONE")
