"""Empirical null for the ALPHA-EXCLUSION test.

The census scores each beta grain by how many of its predicted reflections land on
peaks that no validated alpha grain of that frame explains, and assigns significance
with an analytic Poisson model. That model assumes uniformly scattered peaks; real
peak fields are clustered, and we already showed the analytic gate is optimistic
(measured null reached 15 hits where Poisson(1.91) forbids it).

So measure the same statistic under the null: on each sampled frame, build the
alpha-claimed mask exactly as the census does, then draw RANDOM beta orientations and
count how many of their reflections land on alpha-unclaimed peaks.

usage: exclusion_null.py [nframes] [ndraws] [nworkers]
"""
import os
import numpy as np, h5py, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
TESTSCANS = os.environ.get("LAUE_TESTSCANS", "$LAUE_DATA-2/Thompson_202607/Initial_Indexing_TestScans")
DATA = os.environ.get("LAUE_SCAN_DATA", f"{TESTSCANS}/ID26-10x10um_0p25umStepSize_TestingIndexing")
H5LOC = "/entry1/data/data"; HC = 1.2398419739; TOL = 8.0; NPX = 2048
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
P = np.array([0.028834, 0.002715, 0.513399]); Rrod = np.array([-1.20334591, -1.2137853, -1.21669634])
dx = dy = 0.0002; Elo, Ehi = 5., 30.
angr = np.linalg.norm(Rrod); v = Rrod/angr; c_, s_ = np.cos(angr), np.sin(angr)
rot = np.array([[c_+(1-c_)*v[0]**2, (1-c_)*v[0]*v[1]-s_*v[2], (1-c_)*v[0]*v[2]+s_*v[1]],
                [(1-c_)*v[1]*v[0]+s_*v[2], c_+(1-c_)*v[1]**2, (1-c_)*v[1]*v[2]-s_*v[0]],
                [(1-c_)*v[2]*v[0]-s_*v[1], (1-c_)*v[2]*v[1]+s_*v[0], c_+(1-c_)*v[2]**2]])
roti = np.linalg.inv(rot); ki = np.array([0, 0, 1.0])

NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 120
NDR = int(sys.argv[2]) if len(sys.argv) > 2 else 150
NW  = int(sys.argv[3]) if len(sys.argv) > 3 else 12

def hexB():
    a, b, c = 0.2921, 0.2921, 0.4665; cg, sg = cos(120*pi/180), sin(120*pi/180); pv = 2*pi/(a*b*c*sg)
    a0, a1, a2 = a, 0, 0; b0, b1, b2 = b*cg, b*sg, 0; c0, c1, c2 = 0, 0, c
    return np.array([[(b1*c2-b2*c1), (c1*a2-c2*a1), (a1*b2-a2*b1)],
                     [(b2*c0-b0*c2), (c2*a0-c0*a2), (a2*b0-a0*b2)],
                     [(b0*c1-b1*c0), (c0*a1-c1*a0), (a0*b1-a1*b0)]])*pv

B_ALP = hexB(); HKL_A = np.loadtxt(f"{W}/params/valid_hkls_Ti_alpha.csv")[:, :3]
B_BET = np.eye(3)*2*pi/0.33065; HKL_B = np.loadtxt(f"{W}/params/valid_hkls_Ti_beta.csv")[:, :3]

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

za = np.load(f"{W}/peel_map/{PREFIX}_alpha_validated.npz", allow_pickle=True)
a_oms = za["oms"]; a_fr = np.asarray([str(f) for f in za["frames"]])
alpha_by_frame = {}
for OM, fn in zip(a_oms, a_fr):
    alpha_by_frame.setdefault(fn, []).append(OM)

zb = np.load(f"{W}/peel_map/{PREFIX}_beta_validated.npz", allow_pickle=True)
b_fr = np.asarray([str(f) for f in zb["frames"]])
frames = sorted(set(b_fr) & set(alpha_by_frame))
sel = frames[::max(1, len(frames)//NFR)][:NFR]
print(f"frames with both phases: {len(frames)}, sampling {len(sel)}", flush=True)

def job(fn):
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
    claimed = np.zeros(npeaks, bool)
    for OMa in alpha_by_frame.get(fn, []):
        pra = project(OMa, B_ALP, HKL_A)
        if len(pra):
            da, ja = tree.query(pra); claimed[ja[da < TOL]] = True
    rng = np.random.default_rng(abs(hash(fn)) % (2**32))
    uniq = []
    for _ in range(NDR):
        pr = project(rand_om(rng), B_BET, HKL_B)
        if not len(pr):
            uniq.append(0); continue
        d, j = tree.query(pr); hit = d < TOL
        uniq.append(int((hit & ~claimed[j]).sum()))
    return np.array(uniq), int(claimed.sum()), npeaks

acc = []; frac_claimed = []
with ProcessPoolExecutor(max_workers=NW) as ex:
    for r in ex.map(job, sel):
        if r is None:
            continue
        u, ncl, npk = r
        acc.append(u); frac_claimed.append(ncl/npk)

u = np.concatenate(acc)
print(f"\nalpha claims {100*np.mean(frac_claimed):.1f}% of peaks on average")
print(f"\nRANDOM-BETA NULL on ALPHA-UNCLAIMED peaks, {len(u):,} draws")
print(f"  mean {u.mean():.2f}  median {int(np.median(u))}  "
      f"99th {np.percentile(u,99):.0f}  99.9th {np.percentile(u,99.9):.0f}  max {u.max()}")
print(f"\nCompare: census reports median 10 alpha-unclaimed unique hits per real beta grain.")
