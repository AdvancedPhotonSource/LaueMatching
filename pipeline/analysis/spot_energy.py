"""Per-spot energy from the indexer output, and the absorption-hardening test.

The indexer stores, per assigned spot, its (h,k,l) [cols 3,4,5], detector pixel
[6,7] and intensity [11], and per orientation the 3x3 matrix [cols 23:32]. So the
photon energy of every assigned reflection is exactly computable -- no matching
heuristic:

    q = OM @ B @ hkl ;  E = hc |q| / (4 pi sin(theta))

Why it matters here: substrate and deposit are the SAME phase, so nothing
crystallographic separates them. But the 1/e sampled depth in Zn runs from ~3 um
at 12 keV to ~41 um at 30 keV (midas_hkls, 45 deg, in+out). A thick Zn overlayer
therefore suppresses LOW-energy reflections preferentially. Prediction: positions
with a high fluorescence pedestal (more Zn in the path) show a spot population
shifted to higher energy.

Stage 1 here is a self-consistency check: the computed (px,py) for each stored
hkl must land on the stored pixel position. If that fails, the energies are
meaningless and nothing downstream is trustworthy.
"""
import sys, os, glob, json
import numpy as np, h5py
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "$LAUE_WORK/analysis")
from laue_material import Phase

W = "$LAUE_WORK"
RESDIR = sys.argv[1]
OUT = sys.argv[2]
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 4

PH = Phase(f"{W}/params/params_Zn_p99.8.txt", "zn")
HC = 1.2398419739
# stream layout (35 cols): om_start=23, grain=1, n_matches=6
OM_START, GRAIN_COL, NMATCH_COL = 23, 1, 6
S_GRAIN, S_H, S_K, S_L, S_X, S_Y, S_I = 1, 3, 4, 5, 6, 7, 11


def spot_energies(OM, hkls):
    """(E_keV, px, py) for reflections hkls under orientation OM."""
    q = (OM @ PH.B @ hkls.T).T
    ql = np.linalg.norm(q, axis=1)
    ok = ql > 1e-9
    E = np.full(len(hkls), np.nan); px = np.full(len(hkls), np.nan); py = np.full(len(hkls), np.nan)
    if not ok.any():
        return E, px, py
    qh = q[ok] / ql[ok, None]
    st = -qh[:, 2]
    kf = PH.ki - 2 * qh[:, 2:3] * qh
    xd = (PH.roti @ kf.T).T
    good = (xd[:, 2] > 0) & (st > 1e-9)
    xs = np.full_like(xd, np.nan)
    xs[good] = xd[good] * PH.P[2] / xd[good, 2:3]
    pxo = (xs[:, 0] - PH.P[0]) / PH.dx + 0.5 * (PH.npx_x - 1)
    pyo = (xs[:, 1] - PH.P[1]) / PH.dy + 0.5 * (PH.npx_y - 1)
    Eo = np.where(good, HC * ql[ok] / np.where(st > 1e-9, st, np.nan) / (4 * np.pi), np.nan)
    E[ok], px[ok], py[ok] = Eo, pxo, pyo
    return E, px, py


def one(f):
    try:
        with h5py.File(f, "r") as h:
            g = h["entry/results"]
            ori = g["filtered_orientations"][()]
            sp = g["filtered_spots"][()]
            src = g.attrs.get("source_file", "")
    except Exception:
        return None
    if not len(ori) or not len(sp):
        return None
    oms = {int(r[GRAIN_COL]): r[OM_START:OM_START + 9].reshape(3, 3) for r in ori}
    rec = []
    for gn in np.unique(sp[:, S_GRAIN]).astype(int):
        if gn not in oms:
            continue
        m = sp[:, S_GRAIN].astype(int) == gn
        hkl = sp[m][:, [S_H, S_K, S_L]]
        E, pxc, pyc = spot_energies(oms[gn], hkl)
        dx = pxc - sp[m][:, S_X]
        dy = pyc - sp[m][:, S_Y]
        rec.append(np.c_[np.full(m.sum(), gn), E, sp[m][:, S_X], sp[m][:, S_Y],
                         sp[m][:, S_I], np.hypot(dx, dy)])
    if not rec:
        return None
    a = np.vstack(rec)
    return (os.path.basename(f), str(src), a)


files = sorted(glob.glob(f"{RESDIR}/results/image_*.output.h5"))
print(f"{len(files)} output files", flush=True)
rows = []
with ProcessPoolExecutor(max_workers=NW) as ex:
    for r in ex.map(one, files, chunksize=8):
        if r: rows.append(r)
print(f"{len(rows)} with spots", flush=True)

allsp = np.vstack([r[2] for r in rows])
resid = allsp[:, 5]
finite = np.isfinite(resid)
print("\n=== SELF-CONSISTENCY: predicted pixel vs stored pixel ===")
print(f"  {finite.sum()} assigned spots")
print(f"  residual |dpix|: median {np.median(resid[finite]):.3f}  "
      f"p90 {np.percentile(resid[finite],90):.3f}  p99 {np.percentile(resid[finite],99):.3f}  "
      f"max {resid[finite].max():.3f}")
frac_ok = float((resid[finite] < 5).mean())
print(f"  fraction within 5 px: {frac_ok*100:.2f}%")
if frac_ok < 0.9:
    print("  *** FAILED: the stored hkl and orientation do not reproduce the stored pixel.")
    print("  *** Energies below are NOT trustworthy. Stopping.")
    sys.exit(2)

E = allsp[:, 1][finite & (resid < 5)]
print("\n=== ASSIGNED-SPOT ENERGY DISTRIBUTION ===")
for q in (1, 5, 25, 50, 75, 95, 99):
    print(f"  p{q:<3d} {np.percentile(E,q):7.2f} keV")
print(f"  fraction below 15 keV: {(E<15).mean()*100:.1f}%   below 20: {(E<20).mean()*100:.1f}%")

np.savez(OUT,
         spots=allsp,
         files=np.array([r[0] for r in rows]),
         sources=np.array([r[1] for r in rows]),
         counts=np.array([len(r[2]) for r in rows]))
print(f"\nwrote {OUT}", flush=True)
print("SPOT_ENERGY_DONE", flush=True)
