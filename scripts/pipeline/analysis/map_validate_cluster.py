"""Validate all batch-peeled orientations per frame (Poisson spot-test), then
cluster validated instances map-wide (hex miso) into confidence-tiered grains.

Validator (analytic version of the test-frame SNR check): for each orientation,
h = predicted reflections landing within TOL px of a real peak (SNR>8 local
maxima of that frame). Null: Poisson lambda = n_pred * N_peaks * pi*TOL^2/Npx^2.
VERIFIED if P(X >= h) < 1e-4.
"""
import numpy as np, h5py, glob, os
from math import cos, sin, pi
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import poisson
from concurrent.futures import ProcessPoolExecutor

WORK = "$LAUE_WORK"
DATA = "$LAUE_DATA-2/Thompson_202607/ID6_950C_HIP/SmallAreaTest1"
H5LOC = "/entry1/data/data"
HC = 1.2398419739; TOL = 8.0; NPX = 2048

P = np.array([0.028834, 0.002715, 0.513399]); Rrod = np.array([-1.20334591, -1.2137853, -1.21669634])
dx = dy = 0.0002; Elo, Ehi = 5., 30.
angr = np.linalg.norm(Rrod); v = Rrod/angr; c_, s_ = np.cos(angr), np.sin(angr)
rot = np.array([[c_+(1-c_)*v[0]**2,(1-c_)*v[0]*v[1]-s_*v[2],(1-c_)*v[0]*v[2]+s_*v[1]],
                [(1-c_)*v[1]*v[0]+s_*v[2],c_+(1-c_)*v[1]**2,(1-c_)*v[1]*v[2]-s_*v[0]],
                [(1-c_)*v[2]*v[0]-s_*v[1],(1-c_)*v[2]*v[1]+s_*v[0],c_+(1-c_)*v[2]**2]])
roti = np.linalg.inv(rot); ki = np.array([0, 0, 1.0])
a, b, c = 0.2921, 0.2921, 0.4665; cg, sg = cos(120*pi/180), sin(120*pi/180)
pv = 2*pi/(a*b*c*sg)
a0,a1,a2=a,0,0; b0,b1,b2=b*cg,b*sg,0; c0,c1,c2=0,0,c
B = np.array([[(b1*c2-b2*c1),(c1*a2-c2*a1),(a1*b2-a2*b1)],
              [(b2*c0-b0*c2),(c2*a0-c0*a2),(a2*b0-a0*b2)],
              [(b0*c1-b1*c0),(c0*a1-c1*a0),(a0*b1-a1*b0)]])*pv
HKLS = np.loadtxt(f"{WORK}/params/valid_hkls_Ti_alpha.csv")[:, :3]

def project(OM):
    q = (OM@B@HKLS.T).T; ql = np.linalg.norm(q, axis=1); m = ql > 1e-9
    q, ql = q[m], ql[m]; qh = q/ql[:, None]
    kf = ki - 2*qh[:, 2:3]*qh; xd = (roti@kf.T).T; m = xd[:, 2] > 0
    xd, ql, qh = xd[m], ql[m], qh[m]; xs = xd*P[2]/xd[:, 2:3]
    px = (xs[:, 0]-P[0])/dx + 0.5*(NPX-1); py = (xs[:, 1]-P[1])/dy + 0.5*(NPX-1); st = -qh[:, 2]
    mk = (px >= 0) & (px < NPX-1) & (py >= 0) & (py < NPX-1) & (st > 1e-9)
    E = HC*ql[mk]/st[mk]/(4*pi); me = (E > Elo) & (E < Ehi)
    return np.c_[px[mk][me], py[mk][me]]

def validate_frame(item):
    fn, oms_flat = item
    oms = oms_flat.reshape(-1, 3, 3)
    with h5py.File(f"{DATA}/{fn}", "r") as f:
        raw = f[H5LOC][()].astype(float)
    med = np.median(raw); mad = 1.4826*np.median(np.abs(raw-med))
    sub = raw - ndi.median_filter(raw, 25)
    pk = (sub == ndi.maximum_filter(sub, 9)) & (sub > 8*mad)
    ys, xs = np.where(pk)
    if len(xs) < 5:
        return fn, np.zeros(len(oms), bool), np.zeros(len(oms)), 0
    tree = cKDTree(np.c_[xs, ys]); npeaks = len(xs)
    ok = np.zeros(len(oms), bool); hits = np.zeros(len(oms))
    for i, OM in enumerate(oms):
        pr = project(OM)
        if len(pr) == 0: continue
        d, _ = tree.query(pr)
        h = int((d < TOL).sum())
        lam = len(pr) * npeaks * pi * TOL * TOL / (NPX*NPX)
        ok[i] = poisson.sf(h - 1, lam) < 1e-4
        hits[i] = h
    return fn, ok, hits, npeaks

if __name__ == "__main__":
    with np.load(f"{WORK}/peel_map/accepted_per_frame.npz") as z:
        ACC = {fn: z[fn].copy() for fn in z.files}   # eager load: no shared zip handle across forks
    results = {}
    with ProcessPoolExecutor(max_workers=32) as ex:
        for fn, ok, hits, npeaks in ex.map(validate_frame, list(ACC.items())):
            results[fn] = (ok, hits, npeaks)
    tot = sum(len(v[0]) for v in results.values())
    ver = sum(int(v[0].sum()) for v in results.values())
    print(f"instances: {tot}, VERIFIED (Poisson p<1e-4): {ver} ({100*ver/tot:.0f}%)", flush=True)

    # map-wide clustering of VERIFIED instances
    def rmat(ax, deg):
        u = np.asarray(ax, float); u /= np.linalg.norm(u); t = np.radians(deg)
        K = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
        return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)
    OPS = [rmat([0,0,1],60*k) for k in range(6)] + \
          [rmat([np.cos(np.radians(x)),np.sin(np.radians(x)),0],180) for x in (0,30,60,90,120,150)]
    def miso_min(A, Bs):
        best = np.full(len(Bs), 999.)
        for S in OPS:
            tr = np.einsum('ij,kj,mki->m', S, A, Bs)
            best = np.minimum(best, np.degrees(np.arccos(np.clip((tr-1)/2, -1, 1))))
        return best
    oms_v, fns_v = [], []
    for fn, (ok, hits, npeaks) in results.items():
        oms = ACC[fn].reshape(-1, 3, 3)
        for OM, o in zip(oms, ok):
            if o: oms_v.append(OM); fns_v.append(fn)
    oms_v = np.array(oms_v)
    print(f"clustering {len(oms_v)} verified instances...", flush=True)
    labels = np.full(len(oms_v), -1); cid = 0
    for i in range(len(oms_v)):
        if labels[i] >= 0: continue
        un = np.where(labels < 0)[0]
        d = miso_min(oms_v[i], oms_v[un]); labels[un[d < 0.7]] = cid; cid += 1
    counts = np.bincount(labels)
    print(f"verified grains (clusters): {cid}")
    for k in (1, 2, 3, 5, 10):
        print(f"  in >= {k} frames: {(counts >= k).sum()}")
    per_frame = ver / max(1, len(results))
    print(f"mean VERIFIED grains per frame: {per_frame:.1f}")
    np.savez(f"{WORK}/peel_map/verified_clusters.npz",
             oms=oms_v, labels=labels, frames=np.array(fns_v))
    print("saved verified_clusters.npz")
