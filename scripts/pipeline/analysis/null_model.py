"""Measure the random-orientation null FOR THIS SCAN, rather than reusing the
number quoted for the SmallArea scan (peak counts and predicted-reflection counts
differ, so lambda differs).

For a sample of frames spread across the raster: detect SNR>8 peaks with the same
fast downsampled-median background the validator uses, then draw random orientations,
project their full Laue pattern, and count how many predicted reflections land within
TOL px of a real peak. Reports mean / 99th pct / max hits and the analytic Poisson
lambda, per phase.

usage: null_model.py [nframes] [ndraws] [nworkers]
"""
import os
import numpy as np, h5py, glob, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
TESTSCANS = os.environ.get("LAUE_TESTSCANS", "$LAUE_DATA-2/Thompson_202607/Initial_Indexing_TestScans")
DATA = os.environ.get("LAUE_SCAN_DATA", f"{TESTSCANS}/ID26-10x10um_0p25umStepSize_TestingIndexing")
RESDIR = lambda ph: os.environ.get(f"LAUE_SCAN_{ph.upper()}", f"{TESTSCANS}/laue_Matching_Results/results/{ph}_20260717_161826")
H5LOC = "/entry1/data/data"; HC = 1.2398419739; TOL = 8.0; NPX = 2048
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
P = np.array([0.028834, 0.002715, 0.513399]); Rrod = np.array([-1.20334591, -1.2137853, -1.21669634])
dx = dy = 0.0002; Elo, Ehi = 5., 30.
angr = np.linalg.norm(Rrod); v = Rrod/angr; c_, s_ = np.cos(angr), np.sin(angr)
rot = np.array([[c_+(1-c_)*v[0]**2, (1-c_)*v[0]*v[1]-s_*v[2], (1-c_)*v[0]*v[2]+s_*v[1]],
                [(1-c_)*v[1]*v[0]+s_*v[2], c_+(1-c_)*v[1]**2, (1-c_)*v[1]*v[2]-s_*v[0]],
                [(1-c_)*v[2]*v[0]-s_*v[1], (1-c_)*v[2]*v[1]+s_*v[0], c_+(1-c_)*v[2]**2]])
roti = np.linalg.inv(rot); ki = np.array([0, 0, 1.0])

NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NDR = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NW  = int(sys.argv[3]) if len(sys.argv) > 3 else 4

def hexB():
    a, b, c = 0.2921, 0.2921, 0.4665; cg, sg = cos(120*pi/180), sin(120*pi/180); pv = 2*pi/(a*b*c*sg)
    a0, a1, a2 = a, 0, 0; b0, b1, b2 = b*cg, b*sg, 0; c0, c1, c2 = 0, 0, c
    return np.array([[(b1*c2-b2*c1), (c1*a2-c2*a1), (a1*b2-a2*b1)],
                     [(b2*c0-b0*c2), (c2*a0-c0*a2), (a2*b0-a0*b2)],
                     [(b0*c1-b1*c0), (c0*a1-c1*a0), (a0*b1-a1*b0)]])*pv

BS = {"alpha": (hexB(), np.loadtxt(f"{W}/params/valid_hkls_Ti_alpha.csv")[:, :3]),
      "beta":  (np.eye(3)*2*pi/0.33065, np.loadtxt(f"{W}/params/valid_hkls_Ti_beta.csv")[:, :3])}

def project(OM, B, HKLS):
    q = (OM@B@HKLS.T).T; ql = np.linalg.norm(q, axis=1); m = ql > 1e-9
    q, ql = q[m], ql[m]; qh = q/ql[:, None]
    kf = ki - 2*qh[:, 2:3]*qh; xd = (roti@kf.T).T; m = xd[:, 2] > 0
    xd, ql, qh = xd[m], ql[m], qh[m]; xs = xd*P[2]/xd[:, 2:3]
    px = (xs[:, 0]-P[0])/dx + 0.5*(NPX-1); py = (xs[:, 1]-P[1])/dy + 0.5*(NPX-1); st = -qh[:, 2]
    mk = (px >= 0) & (px < NPX-1) & (py >= 0) & (py < NPX-1) & (st > 1e-9)
    E = HC*ql[mk]/st[mk]/(4*pi); me = (E > Elo) & (E < Ehi)
    return np.c_[px[mk][me], py[mk][me]]

def rand_om(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

mapping = json.load(open(f"{RESDIR('alpha')}/frame_mapping.json"))
img2file = {int(k): vv["file"] for k, vv in mapping.items() if isinstance(vv, dict) and "file" in vv}
allimg = sorted(img2file)
sel = allimg[::max(1, len(allimg)//NFR)][:NFR]

def job(inum):
    fn = img2file[inum]
    try:
        with h5py.File(f"{DATA}/{fn}", "r") as f:
            raw = f[H5LOC][()].astype(float)
    except Exception:
        return None
    med = np.median(raw); mad = 1.4826*np.median(np.abs(raw-med))
    bg4 = ndi.median_filter(raw[::4, ::4], 25); bg = np.kron(bg4, np.ones((4, 4)))[:NPX, :NPX]
    sub = raw - bg
    pk = (sub == ndi.maximum_filter(sub, 9)) & (sub > 8*mad); ys, xs = np.where(pk)
    if len(xs) < 5:
        return None
    tree = cKDTree(np.c_[xs, ys]); npeaks = len(xs)
    rng = np.random.default_rng(inum)
    out = {}
    for ph, (B, HKL) in BS.items():
        hits = []; npred = []
        for _ in range(NDR):
            pr = project(rand_om(rng), B, HKL)
            if not len(pr):
                hits.append(0); npred.append(0); continue
            d, _ = tree.query(pr); hits.append(int((d < TOL).sum())); npred.append(len(pr))
        out[ph] = (np.array(hits), float(np.mean(npred)), npeaks)
    return out

acc = {ph: [] for ph in BS}; lam = {ph: [] for ph in BS}; npk = []
with ProcessPoolExecutor(max_workers=NW) as ex:
    for r in ex.map(job, sel):
        if r is None:
            continue
        for ph, (h, mp, npeaks) in r.items():
            acc[ph].append(h)
            lam[ph].append(mp*npeaks*pi*TOL*TOL/(NPX*NPX))
        npk.append(list(r.values())[0][2])

print(f"frames sampled: {len(npk)}, draws/frame/phase: {NDR}")
print(f"median SNR>8 peaks per frame: {int(np.median(npk))}")
for ph in BS:
    h = np.concatenate(acc[ph])
    print(f"\n[{ph}] RANDOM-ORIENTATION NULL over {len(h):,} draws")
    print(f"   mean hits {h.mean():.2f}   median {int(np.median(h))}   "
          f"99th pct {np.percentile(h,99):.0f}   99.9th {np.percentile(h,99.9):.0f}   max {h.max()}")
    print(f"   analytic Poisson lambda (mean over frames): {np.mean(lam[ph]):.2f}")
