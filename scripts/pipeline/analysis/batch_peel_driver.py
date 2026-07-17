"""Batch iterative peel over the full SmallAreaTest1 map.

Each PASS = one orchestrator run over ALL frames (daemon loads once), then
per-frame masking of every accepted grain's full projected pattern, writing
the residual frames for the next pass. Sigma-cap schedule widens per pass.
Outputs per-frame accepted orientation stacks + a status log.
"""
import numpy as np, h5py, subprocess, os, sys, shutil, glob, json, time
from math import cos, sin, pi

WORK = "/net/hpcs34/data34c/for_Hemant/lauematching_ti"
LM = "/home/beams/EPIX34ID/opt/LaueMatching"
PY = sys.executable
DATA = "/net/hpcs34/data34c/Run2026-2/Thompson_202607/ID6_950C_HIP/SmallAreaTest1"
H5LOC = "/entry1/data/data"
MAX_PASS = 6
MASK_R = 10
SIGCAPS = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
HC = 1.2398419739
ST = open(f"{WORK}/batch_peel_status.txt", "w", buffering=1)
def log(m): ST.write(m + "\n"); print(m, flush=True)

P = np.array([0.028834, 0.002715, 0.513399]); Rrod = np.array([-1.20334591, -1.2137853, -1.21669634])
dx = dy = 0.0002; nPx = 2048; Elo, Ehi = 5., 30.
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
    px = (xs[:, 0]-P[0])/dx + 0.5*(nPx-1); py = (xs[:, 1]-P[1])/dy + 0.5*(nPx-1); st = -qh[:, 2]
    mk = (px >= 0) & (px < nPx-1) & (py >= 0) & (py < nPx-1) & (st > 1e-9)
    E = HC*ql[mk]/st[mk]/(4*pi); me = (E > Elo) & (E < Ehi)
    return np.c_[px[mk][me], py[mk][me]]

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

frames = sorted(os.path.basename(f) for f in glob.glob(f"{DATA}/*.h5"))
log(f"BATCH PEEL: {len(frames)} frames, {MAX_PASS} passes max")
accepted = {fn: [] for fn in frames}
os.makedirs(f"{WORK}/peel_map", exist_ok=True)
cur_folder = DATA

for p_i in range(1, MAX_PASS+1):
    t0 = time.time()
    cfg = f"{WORK}/params/params_batchpeel_p{p_i}.txt"
    base = open(f"{WORK}/params/params_Ti_alpha.txt").read()
    lines = []
    for ln in base.splitlines():
        k = ln.split()[0] if ln.split() else ""
        if k == "ThresholdPercentile": ln = "ThresholdPercentile 99.8"
        if k == "MinNrSpots": ln = "MinNrSpots 8"
        if k == "MinIntensity": ln = "MinIntensity 50"
        if k == "BackgroundFile": ln = f"BackgroundFile {WORK}/peel_map/bg_p{p_i}.bin"
        lines.append(ln)
    lines.append(f"GaussSigmaMax {SIGCAPS[p_i-1]}")
    open(cfg, "w").write("\n".join(lines) + "\n")
    out = f"{WORK}/results/batchpeel_pass{p_i}"
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([PY, f"{LM}/scripts/laue_orchestrator.py",
                        "--config", cfg, "--folder", cur_folder, "--h5-location", H5LOC,
                        "--ncpus", "32", "--port", "60517", "--flush-time", "120",
                        "--output-dir", out],
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    mapping = {}
    try:
        mp = json.load(open(f"{out}/frame_mapping.json"))
        mapping = {int(k): vv["file"] for k, vv in mp.items() if isinstance(vv, dict) and "file" in vv}
    except Exception as e:
        log(f"pass {p_i}: mapping load failed: {e}")
    new_total = 0
    for h5 in glob.glob(f"{out}/results/image_*.output.h5"):
        inum = int(h5.split("image_")[1][:5])
        fn = mapping.get(inum)
        if fn is None: continue
        # residual frames are named identically to originals
        fn = os.path.basename(fn)
        try:
            with h5py.File(h5, "r") as f:
                filt = f["entry/results/filtered_orientations"][()]
        except Exception:
            continue
        for row in filt:
            OM = row[23:32].reshape(3, 3)
            if accepted[fn] and miso_min(OM, np.array(accepted[fn])).min() < 0.7:
                continue
            accepted[fn].append(OM); new_total += 1
    tot = sum(len(x) for x in accepted.values())
    log(f"pass {p_i}: +{new_total} new (total {tot}) in {time.time()-t0:.0f}s")
    if new_total < max(5, 0.01*tot):
        log("converged."); break
    if p_i == MAX_PASS: break
    # build residuals
    t1 = time.time()
    nxt = f"{WORK}/peel_map/pass{p_i+1}"
    shutil.rmtree(nxt, ignore_errors=True); os.makedirs(nxt)
    yy, xx = np.mgrid[-MASK_R:MASK_R+1, -MASK_R:MASK_R+1]
    disk = (xx*xx + yy*yy) <= MASK_R*MASK_R
    for fn in frames:
        src = f"{DATA}/{fn}"
        with h5py.File(src, "r") as f:
            img = f[H5LOC][()]
        med = np.median(img)
        if accepted[fn]:
            for x, y in np.vstack([project(OM) for OM in accepted[fn]]):
                xi, yi = int(round(x)), int(round(y))
                x0, x1 = max(0, xi-MASK_R), min(nPx, xi+MASK_R+1)
                y0, y1 = max(0, yi-MASK_R), min(nPx, yi+MASK_R+1)
                img[y0:y1, x0:x1][disk[(y0-yi+MASK_R):(y1-yi+MASK_R), (x0-xi+MASK_R):(x1-xi+MASK_R)]] = med
        with h5py.File(f"{nxt}/{fn}", "w") as f:
            f.create_dataset(H5LOC, data=img)
    log(f"pass {p_i}: residuals written in {time.time()-t1:.0f}s")
    cur_folder = nxt

# save accepted stacks
np.savez(f"{WORK}/peel_map/accepted_per_frame.npz",
         **{fn: np.array(oms) for fn, oms in accepted.items() if oms})
log(f"BATCH PEEL DONE: {sum(len(x) for x in accepted.values())} orientation instances "
    f"across {sum(1 for x in accepted.values() if x)} frames")
