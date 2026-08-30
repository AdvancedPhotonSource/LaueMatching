"""Honest beta census under the ALPHA-EXCLUSION null.

For every one of the 767 verified beta clusters:
  - representative (cluster-mean-instance) BCC pattern;
  - on each frame the cluster was verified in, count predicted reflections that
    land on an SNR>8 peak NOT claimed by ANY verified alpha grain of that frame
    (TOL px), and the Poisson p on the alpha-UNCLAIMED peak subset only;
  - single-frame clusters: keep the frame's p_unique;
  - multi-frame clusters: Fisher-combine p_unique across frames AND require
    spatial recurrence (already implied by clustering).
Report how many beta grains survive gates 1e-2 / 1e-3 / 1e-4 on the hard
alpha-exclusion null, split by recurrence, plus the unique-hit distribution.
Writes peel_map/beta_census.npz for the story.
"""
import os
import numpy as np, h5py, sys
from math import pi
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import poisson, combine_pvalues
from concurrent.futures import ProcessPoolExecutor

WORK = os.environ.get("LAUE_WORK", "$LAUE_WORK")
TESTSCANS = os.environ.get("LAUE_TESTSCANS", "$LAUE_DATA-2/Thompson_202607/Initial_Indexing_TestScans")
# argv[1]=scan prefix, argv[2]=nworkers. "smallarea" reproduces the original hardcoded run;
# alpha/beta instances are read from <prefix>_{alpha,beta}_validated.npz (or the original
# verified_clusters/beta_verified pair for smallarea).
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "smallarea"
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SCANS = {
 "smallarea": dict(data="$LAUE_DATA-2/Thompson_202607/ID6_950C_HIP/SmallAreaTest1",
                   alpha_npz="verified_clusters.npz", beta_npz="beta_verified.npz",
                   out="beta_census"),
 "env": dict(data=os.environ.get("LAUE_SCAN_DATA", ""),
                   alpha_npz=f'{os.environ.get("LAUE_OUT_PREFIX", "env")}_alpha_validated.npz',
                   beta_npz=f'{os.environ.get("LAUE_OUT_PREFIX", "env")}_beta_validated.npz',
                   out=f'{os.environ.get("LAUE_OUT_PREFIX", "env")}_census'),
 # NB legacy key: this is specimen ID26 and the scan is 20x20 um in the sample
 # frame, not 10x10 -- the folder name is wrong. Kept so existing *_id6_10x10_*
 # outputs stay resolvable; prefer the "env" entry for new work.
 "id6_10x10": dict(data=f"{TESTSCANS}/ID26-10x10um_0p25umStepSize_TestingIndexing",
                   alpha_npz="id6_10x10_alpha_validated.npz",
                   beta_npz="id6_10x10_beta_validated.npz",
                   out="id6_10x10_census"),
}
if PREFIX not in SCANS:
    sys.exit(f"unknown scan {PREFIX!r}; choose from {sorted(SCANS)}")
CFG = SCANS[PREFIX]; DATA = CFG["data"]
H5LOC = "/entry1/data/data"; HC = 1.2398419739; TOL = 8.0; NPX = _PH_A.npx_x
from laue_material import Phase
_PH_A = Phase.load("alpha")
_PH_B = Phase.load("beta")
# Detector geometry from the parameter file the indexer used (laue_material).
P = _PH_A.P; Rrod = _PH_A.Rrod
dx = _PH_A.dx; dy = _PH_A.dy; Elo, Ehi = _PH_A.Elo, _PH_A.Ehi
rot = _PH_A.rot; roti = _PH_A.roti; ki = _PH_A.ki
B_beta = _PH_B.B
HKL_b = _PH_B.hkls

HKL_a = _PH_A.hkls

def project(OM, B, HKLS):
    q = (OM@B@HKLS.T).T; ql = np.linalg.norm(q, axis=1); m = ql > 1e-9
    q, ql = q[m], ql[m]; qh = q/ql[:, None]
    kf = ki - 2*qh[:, 2:3]*qh; xd = (roti@kf.T).T; m = xd[:, 2] > 0
    xd, ql, qh = xd[m], ql[m], qh[m]; xs = xd*P[2]/xd[:, 2:3]
    px = (xs[:, 0]-P[0])/dx + 0.5*(NPX-1); py = (xs[:, 1]-P[1])/dy + 0.5*(NPX-1); st = -qh[:, 2]
    mk = (px >= 0) & (px < NPX-1) & (py >= 0) & (py < NPX-1) & (st > 1e-9)
    E = HC*ql[mk]/st[mk]/(4*pi); me = (E > Elo) & (E < Ehi)
    return np.c_[px[mk][me], py[mk][me]]

# ---- load all verified instances ------------------------------------------
zb = np.load(f"{WORK}/peel_map/{CFG['beta_npz']}", allow_pickle=True)
b_oms, b_lab, b_fr = zb["oms"], zb["labels"], np.asarray([str(f) for f in zb["frames"]])
za = np.load(f"{WORK}/peel_map/{CFG['alpha_npz']}", allow_pickle=True)
a_oms, a_fr = za["oms"], np.asarray([str(f) for f in za["frames"]])
print(f"[{PREFIX}] alpha instances {len(a_oms)}, beta instances {len(b_oms)}, "
      f"beta clusters {b_lab.max()+1}", flush=True)
alpha_by_frame = {}
for OM, fn in zip(a_oms, a_fr):
    alpha_by_frame.setdefault(fn, []).append(OM)

# per-frame job: for each beta instance ON this frame return (gidx, hit, unique, p_all, p_unq)
inst_frame = list(b_fr)
frames = sorted(set(inst_frame))
def frame_job(fn):
    idx = np.where(b_fr == fn)[0]
    with h5py.File(f"{DATA}/{fn}", "r") as f: raw = f[H5LOC][()].astype(float)
    med = np.median(raw); mad = 1.4826*np.median(np.abs(raw-med))
    # downsampled-median background: ~16x faster than full median_filter(25), same peaks.
    # The full filter is ~20 s/frame, which is untenable over a 6561-frame scan.
    bg4 = ndi.median_filter(raw[::4, ::4], 25); bg = np.kron(bg4, np.ones((4, 4)))[:NPX, :NPX]
    sub = raw - bg
    pk = (sub == ndi.maximum_filter(sub, 9)) & (sub > 8*mad)
    ys, xs = np.where(pk); npeaks = len(xs)
    out = []
    if npeaks < 5:
        for i in idx: out.append((int(i), 0, 0, 1.0, 1.0))
        return out
    tree = cKDTree(np.c_[xs, ys])
    claimed = np.zeros(npeaks, bool)
    for OMa in alpha_by_frame.get(fn, []):
        pra = project(OMa, B_alp, HKL_a)
        if len(pra):
            da, ja = tree.query(pra); claimed[ja[da < TOL]] = True
    n_uncl = int((~claimed).sum())
    for i in idx:
        pr = project(b_oms[i], B_beta, HKL_b)
        if not len(pr): out.append((int(i), 0, 0, 1.0, 1.0)); continue
        d, j = tree.query(pr); hit = d < TOL
        unique = hit & ~claimed[j]
        lam = len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
        lam_u = len(pr)*max(n_uncl,1)*pi*TOL*TOL/(NPX*NPX)
        p_all = poisson.sf(int(hit.sum())-1, lam)
        p_unq = poisson.sf(int(unique.sum())-1, lam_u)
        out.append((int(i), int(hit.sum()), int(unique.sum()), float(p_all), float(p_unq)))
    return out

inst_pall = np.ones(len(b_oms)); inst_punq = np.ones(len(b_oms))
inst_hit = np.zeros(len(b_oms), int); inst_unq = np.zeros(len(b_oms), int)
with ProcessPoolExecutor(max_workers=NW) as ex:
    for res in ex.map(frame_job, frames):
        for i, h, u, pa, pu in res:
            inst_pall[i] = pa; inst_punq[i] = pu; inst_hit[i] = h; inst_unq[i] = u

# ---- aggregate per cluster -------------------------------------------------
ncl = b_lab.max()+1
counts = np.bincount(b_lab, minlength=ncl)
rows = []
for cid in range(ncl):
    ii = np.where(b_lab == cid)[0]
    nfr = len(ii)
    best_punq = float(inst_punq[ii].min())
    # multi-frame: Fisher combine per-frame p_unique
    comb = combine_pvalues(inst_punq[ii], method="fisher")[1] if nfr >= 2 else best_punq
    rows.append((cid, nfr, int(inst_unq[ii].sum()), int(inst_unq[ii].max()),
                 best_punq, float(comb)))
rows = np.array(rows)
nfr_c = rows[:, 1]; tot_unq = rows[:, 2]; best_p = rows[:, 4]; comb_p = rows[:, 5]

def report(gate):
    # single-frame grains: judged by their one frame's p_unique
    single = (nfr_c == 1)
    multi = (nfr_c >= 2)
    s_pass = int(((best_p < gate) & single).sum())
    m_pass = int(((comb_p < gate) & multi).sum())
    return s_pass, m_pass, int(single.sum()), int(multi.sum())

print(f"TOTAL verified beta clusters: {ncl}")
print(f"  recurrent (>=2 frames): {(nfr_c>=2).sum()}, single-frame: {(nfr_c==1).sum()}")
print(f"  total beta instances: {len(b_oms)}")
print("\nALPHA-EXCLUSION SURVIVORS (unique-hit Poisson null on alpha-unclaimed peaks):")
print(f"{'gate':>8} {'single':>10} {'multi(Fisher)':>14} {'TOTAL beta grains':>18}")
for gate in (1e-2, 1e-3, 1e-4):
    s, m, ns, nm = report(gate)
    print(f"{gate:>8.0e} {s:>4}/{ns:<5} {m:>4}/{nm:<8} {s+m:>10}")

# grains with >=1 alpha-unique hit at all
print(f"\nclusters with >=1 alpha-unclaimed unique hit summed over frames: "
      f"{int((tot_unq>=1).sum())}")
print(f"clusters with >=3 unique hits (summed): {int((tot_unq>=3).sum())}")
print(f"unique-hit total distribution (summed per grain): "
      f"median {int(np.median(tot_unq))}, 90pct {int(np.percentile(tot_unq,90))}, max {int(tot_unq.max())}")

np.savez(f"{WORK}/peel_map/{CFG['out']}.npz",
         cid=rows[:,0].astype(int), nfr=nfr_c.astype(int), tot_unq=tot_unq.astype(int),
         best_punq=best_p, comb_punq=comb_p, counts=counts)
print(f"\nsaved {CFG['out']}.npz")
