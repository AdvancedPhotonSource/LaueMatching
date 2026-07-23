"""Stage A of the parent-beta reconstruction: per-frame Poisson validation of
BOTH phases on the 10,201-frame 100x100um_TestScan_About1parentbeta scan, with
stage coordinates captured for the position-resolved analysis.

For each phase (alpha hex, beta bcc): read filtered_orientations (cols 23:32)
from every image_*.output.h5, validate each against SNR>8 peaks of its frame
(full predicted pattern; Poisson p<1e-4 on the whole peak list), and save the
survivors with their (sampleX, sampleZ). Light greedy clustering afterwards.

Writes peel_map/parentbeta_{alpha,beta}_validated.npz  (oms, frames, X, Z,
nhit, labels).
"""
import os
import numpy as np, h5py, glob, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import poisson
from concurrent.futures import ProcessPoolExecutor

WORK = os.environ.get("LAUE_WORK", "/net/hpcs34/data34c/for_Hemant/lauematching_ti")
TESTSCANS = os.environ.get("LAUE_TESTSCANS", "/net/hpcs34/data34c/Run2026-2/Thompson_202607/Initial_Indexing_TestScans")
# Scan registry: argv[3] selects. "parentbeta" reproduces the original hardcoded 100x100um run.
# resdir/{phase} is where the indexer's output.h5 + frame_mapping.json live; out/{phase} names the npz.
SCANS={
 "parentbeta": dict(data=f"{TESTSCANS}/ID6-100x100um_TestScan_About1parentbeta",
                    resdir=lambda ph: f"{WORK}/results/parentbeta_{ph}",
                    out=lambda ph: f"parentbeta_{ph}"),
 # ID6 10x10um 0.25um-step fine scan (6561 frames); raw data was renamed ID6 -> ID26
 "env": dict(data=os.environ.get("LAUE_SCAN_DATA", ""),
                    resdir=lambda ph: os.environ.get(f"LAUE_SCAN_{ph.upper()}", ""),
                    out=lambda ph: f'{os.environ.get("LAUE_OUT_PREFIX", "env")}_{ph}'),
 # NB legacy key: this is specimen ID26 and the scan is 20x20 um in the sample
 # frame, not 10x10 -- the folder name is wrong. Kept so existing *_id6_10x10_*
 # outputs stay resolvable; prefer the "env" entry for new work.
 "id6_10x10":  dict(data=f"{TESTSCANS}/ID26-10x10um_0p25umStepSize_TestingIndexing",
                    resdir=lambda ph: f"{TESTSCANS}/laue_Matching_Results/results/{ph}_20260717_161826",
                    out=lambda ph: f"id6_10x10_{ph}"),
}
H5LOC="/entry1/data/data"; HC=1.2398419739; TOL=8.0; NPX=2048
P=np.array([0.028834,0.002715,0.513399]); Rrod=np.array([-1.20334591,-1.2137853,-1.21669634])
dx=dy=0.0002; Elo,Ehi=5.,30.
angr=np.linalg.norm(Rrod); v=Rrod/angr; c_,s_=np.cos(angr),np.sin(angr)
rot=np.array([[c_+(1-c_)*v[0]**2,(1-c_)*v[0]*v[1]-s_*v[2],(1-c_)*v[0]*v[2]+s_*v[1]],
              [(1-c_)*v[1]*v[0]+s_*v[2],c_+(1-c_)*v[1]**2,(1-c_)*v[1]*v[2]-s_*v[0]],
              [(1-c_)*v[2]*v[0]-s_*v[1],(1-c_)*v[2]*v[1]+s_*v[0],c_+(1-c_)*v[2]**2]])
roti=np.linalg.inv(rot); ki=np.array([0,0,1.0])

def hexB():
    a,b,c=0.2921,0.2921,0.4665; cg,sg=cos(120*pi/180),sin(120*pi/180); pv=2*pi/(a*b*c*sg)
    a0,a1,a2=a,0,0; b0,b1,b2=b*cg,b*sg,0; c0,c1,c2=0,0,c
    return np.array([[(b1*c2-b2*c1),(c1*a2-c2*a1),(a1*b2-a2*b1)],
                     [(b2*c0-b0*c2),(c2*a0-c0*a2),(a2*b0-a0*b2)],
                     [(b0*c1-b1*c0),(c0*a1-c1*a0),(a0*b1-a1*b0)]])*pv

PHASE=sys.argv[1] if len(sys.argv)>1 else "beta"
NW=int(sys.argv[2]) if len(sys.argv)>2 else 36
SCAN=sys.argv[3] if len(sys.argv)>3 else "parentbeta"
if SCAN not in SCANS:
    sys.exit(f"unknown scan {SCAN!r}; choose from {sorted(SCANS)}")
CFG=SCANS[SCAN]; DATA=CFG["data"]; RESDIR=CFG["resdir"](PHASE); RES=CFG["out"](PHASE)
if PHASE=="alpha":
    B=hexB(); HKL=np.loadtxt(f"{WORK}/params/valid_hkls_Ti_alpha.csv")[:,:3]
else:
    B=np.eye(3)*2*pi/0.33065; HKL=np.loadtxt(f"{WORK}/params/valid_hkls_Ti_beta.csv")[:,:3]
print(f"[{PHASE}] scan={SCAN} resdir={RESDIR}",flush=True)

def project(OM):
    q=(OM@B@HKL.T).T; ql=np.linalg.norm(q,axis=1); m=ql>1e-9
    q,ql=q[m],ql[m]; qh=q/ql[:,None]
    kf=ki-2*qh[:,2:3]*qh; xd=(roti@kf.T).T; m=xd[:,2]>0
    xd,ql,qh=xd[m],ql[m],qh[m]; xs=xd*P[2]/xd[:,2:3]
    px=(xs[:,0]-P[0])/dx+0.5*(NPX-1); py=(xs[:,1]-P[1])/dy+0.5*(NPX-1); st=-qh[:,2]
    mk=(px>=0)&(px<NPX-1)&(py>=0)&(py<NPX-1)&(st>1e-9)
    E=HC*ql[mk]/st[mk]/(4*pi); me=(E>Elo)&(E<Ehi)
    return np.c_[px[mk][me],py[mk][me]]

mapping=json.load(open(f"{RESDIR}/frame_mapping.json"))
img2file={int(k):vv["file"] for k,vv in mapping.items() if isinstance(vv,dict) and "file" in vv}
h5s=sorted(glob.glob(f"{RESDIR}/results/image_*.output.h5"))
print(f"[{PHASE}] {len(h5s)} output.h5 files",flush=True)

def validate(h5):
    inum=int(h5.split("image_")[1][:5]); fn=img2file.get(inum)
    if fn is None: return None
    try:
        with h5py.File(h5,"r") as f:
            filt=f["entry/results/filtered_orientations"][()]
    except Exception: return None
    if not len(filt): return None
    oms=filt[:,23:32].reshape(-1,3,3)
    try:
        with h5py.File(f"{DATA}/{fn}","r") as f:
            raw=f[H5LOC][()].astype(float)
            X=float(f["entry1/sample/sampleX"][()].ravel()[0]); Z=float(f["entry1/sample/sampleZ"][()].ravel()[0])
    except Exception: return None
    med=np.median(raw); mad=1.4826*np.median(np.abs(raw-med))
    # downsampled-median background: ~16x faster than full median_filter(25), same peaks
    bg4=ndi.median_filter(raw[::4,::4],25); bg=np.kron(bg4,np.ones((4,4)))[:NPX,:NPX]
    sub=raw-bg
    pk=(sub==ndi.maximum_filter(sub,9))&(sub>8*mad); ys,xs=np.where(pk)
    out=[]
    if len(xs)>=5:
        tree=cKDTree(np.c_[xs,ys]); npeaks=len(xs)
        for i,OM in enumerate(oms):
            pr=project(OM)
            if not len(pr): continue
            d,_=tree.query(pr); h=int((d<TOL).sum())
            lam=len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
            if poisson.sf(h-1,lam)<1e-4:
                out.append((OM,fn,X,Z,h))
    return out

oms_v=[]; fr_v=[]; X_v=[]; Z_v=[]; nh_v=[]; done=0; tot_inst=0
with ProcessPoolExecutor(max_workers=NW) as ex:
    for res in ex.map(validate, h5s, chunksize=8):
        done+=1
        if res:
            for OM,fn,X,Z,h in res:
                oms_v.append(OM); fr_v.append(fn); X_v.append(X); Z_v.append(Z); nh_v.append(h)
        if done%1000==0: print(f"[{PHASE}] {done}/{len(h5s)} frames, {len(oms_v)} validated so far",flush=True)
oms_v=np.array(oms_v)
print(f"[{PHASE}] VALIDATED instances: {len(oms_v)}",flush=True)
# SAVE validated instances immediately (protect the expensive I/O before clustering)
np.savez(f"{WORK}/peel_map/{RES}_validated.npz",
         oms=oms_v, frames=np.array(fr_v), X=np.array(X_v), Z=np.array(Z_v),
         nhit=np.array(nh_v), labels=np.full(len(oms_v),-1))
print(f"[{PHASE}] saved validated (pre-cluster)",flush=True)

# light greedy clustering (hex 12 / cubic 24 ops)
def rmat(ax,deg):
    u=np.asarray(ax,float); u/=np.linalg.norm(u); t=np.radians(deg)
    K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)
if PHASE=="alpha":
    OPS=[rmat([0,0,1],60*k) for k in range(6)]+[rmat([np.cos(np.radians(x)),np.sin(np.radians(x)),0],180) for x in (0,30,60,90,120,150)]
else:
    OPS=[np.eye(3)]+[rmat(ax,d) for ax,d in [([1,0,0],90),([1,0,0],180),([1,0,0],270),([0,1,0],90),([0,1,0],180),([0,1,0],270),
        ([0,0,1],90),([0,0,1],180),([0,0,1],270),([1,1,0],180),([1,-1,0],180),([1,0,1],180),([-1,0,1],180),([0,1,1],180),
        ([0,1,-1],180),([1,1,1],120),([1,1,1],240),([1,-1,1],120),([1,-1,1],240),([-1,1,1],120),([-1,1,1],240),([1,1,-1],120),([1,1,-1],240)]]
OPS=np.array(OPS)
def miso_min(A,Bs):
    best=np.full(len(Bs),999.)
    for S in OPS:
        tr=np.einsum('ij,kj,mki->m',S,A,Bs); best=np.minimum(best,np.degrees(np.arccos(np.clip((tr-1)/2,-1,1))))
    return best
labels=np.full(len(oms_v),-1); cid=0
for i in range(len(oms_v)):
    if labels[i]>=0: continue
    un=np.where(labels<0)[0]; d=miso_min(oms_v[i],oms_v[un]); labels[un[d<1.0]]=cid; cid+=1
counts=np.bincount(labels) if len(oms_v) else np.array([])
print(f"[{PHASE}] clusters (<1.0 deg): {cid}; top sizes {sorted(counts,reverse=True)[:12]}",flush=True)
np.savez(f"{WORK}/peel_map/{RES}_validated.npz",
         oms=oms_v, frames=np.array(fr_v), X=np.array(X_v), Z=np.array(Z_v),
         nhit=np.array(nh_v), labels=labels)
print(f"[{PHASE}] saved {RES}_validated.npz",flush=True)
