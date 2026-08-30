"""Validate the full-map beta orientations (Poisson spot-test, BCC) + cluster."""
import numpy as np, h5py, glob, json
from math import pi
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import poisson
WORK="$LAUE_WORK"
DATA="$LAUE_DATA-2/Thompson_202607/ID6_950C_HIP/SmallAreaTest1"
H5LOC="/entry1/data/data"; HC=1.2398419739; TOL=8.0; NPX = _PH_B.npx_x
from laue_material import Phase
_PH_B = Phase.load("beta")
# Detector geometry from the parameter file the indexer used (laue_material).
P = _PH_B.P; Rrod = _PH_B.Rrod
dx = _PH_B.dx; dy = _PH_B.dy; Elo, Ehi = _PH_B.Elo, _PH_B.Ehi
rot = _PH_B.rot; roti = _PH_B.roti; ki = _PH_B.ki
B=_PH_B.B
HKLS=_PH_B.hkls
def project(OM):
    q=(OM@B@HKLS.T).T; ql=np.linalg.norm(q,axis=1); m=ql>1e-9
    q,ql=q[m],ql[m]; qh=q/ql[:,None]
    kf=ki-2*qh[:,2:3]*qh; xd=(roti@kf.T).T; m=xd[:,2]>0
    xd,ql,qh=xd[m],ql[m],qh[m]; xs=xd*P[2]/xd[:,2:3]
    px=(xs[:,0]-P[0])/dx+0.5*(NPX-1); py=(xs[:,1]-P[1])/dy+0.5*(NPX-1); st=-qh[:,2]
    mk=(px>=0)&(px<NPX-1)&(py>=0)&(py<NPX-1)&(st>1e-9)
    E=HC*ql[mk]/st[mk]/(4*pi); me=(E>Elo)&(E<Ehi)
    return np.c_[px[mk][me],py[mk][me]]
def validate_beta_frame(item):
    fn, oms_flat = item
    oms = oms_flat.reshape(-1,3,3)
    with h5py.File(f"{DATA}/{fn}","r") as f: raw=f[H5LOC][()].astype(float)
    med=np.median(raw); mad=1.4826*np.median(np.abs(raw-med))
    sub=raw-ndi.median_filter(raw,25)
    pk=(sub==ndi.maximum_filter(sub,9))&(sub>8*mad)
    ys,xs=np.where(pk)
    keep=np.zeros(len(oms),bool)
    if len(xs)>=5:
        tree=cKDTree(np.c_[xs,ys]); npeaks=len(xs)
        for i,OM in enumerate(oms):
            pr=project(OM)
            if not len(pr): continue
            d,_=tree.query(pr); h=int((d<TOL).sum())
            lam=len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
            keep[i]=poisson.sf(h-1,lam)<1e-4
    return fn, oms, keep

from concurrent.futures import ProcessPoolExecutor
mapping=json.load(open(f"{WORK}/results/full_scan1_beta/frame_mapping.json"))
img2file={int(k):vv["file"] for k,vv in mapping.items() if isinstance(vv,dict) and "file" in vv}
per_frame={}
for h5 in sorted(glob.glob(f"{WORK}/results/full_scan1_beta/results/image_*.output.h5")):
    inum=int(h5.split("image_")[1][:5]); fn=img2file.get(inum)
    if fn is None: continue
    try:
        with h5py.File(h5,"r") as f: filt=f["entry/results/filtered_orientations"][()]
    except Exception: continue
    if len(filt): per_frame[fn]=filt[:,23:32]
tot=sum(len(v) for v in per_frame.values()); ver=0; oms_v=[]; fr_v=[]
with ProcessPoolExecutor(max_workers=32) as ex:
    for fn,oms,keep in ex.map(validate_beta_frame, list(per_frame.items())):
        for OM,k in zip(oms,keep):
            if k: ver+=1; oms_v.append(OM); fr_v.append(fn)
print(f"beta instances {tot}, VERIFIED {ver}", flush=True)
# cluster (cubic ops)
def rmat(ax,deg):
    u=np.asarray(ax,float);u/=np.linalg.norm(u);t=np.radians(deg)
    K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)
OPS=[np.eye(3)]
for ax,d in [([1,0,0],90),([1,0,0],180),([1,0,0],270),([0,1,0],90),([0,1,0],180),([0,1,0],270),
             ([0,0,1],90),([0,0,1],180),([0,0,1],270),([1,1,0],180),([1,-1,0],180),([1,0,1],180),
             ([-1,0,1],180),([0,1,1],180),([0,1,-1],180),([1,1,1],120),([1,1,1],240),([1,-1,1],120),
             ([1,-1,1],240),([-1,1,1],120),([-1,1,1],240),([1,1,-1],120),([1,1,-1],240)]:
    OPS.append(rmat(ax,d))
def miso_min(A,Bs):
    best=np.full(len(Bs),999.)
    for S in OPS:
        tr=np.einsum('ij,kj,mki->m',S,A,Bs)
        best=np.minimum(best,np.degrees(np.arccos(np.clip((tr-1)/2,-1,1))))
    return best
oms_v=np.array(oms_v)
if len(oms_v):
    labels=np.full(len(oms_v),-1); cid=0
    for i in range(len(oms_v)):
        if labels[i]>=0: continue
        un=np.where(labels<0)[0]
        d=miso_min(oms_v[i],oms_v[un]); labels[un[d<0.7]]=cid; cid+=1
    counts=np.bincount(labels)
    print(f"verified BETA grains: {cid}; at >=2 frames: {(counts>=2).sum()}; >=5: {(counts>=5).sum()}")
    np.savez(f"{WORK}/peel_map/beta_verified.npz", oms=oms_v, labels=labels, frames=np.array(fr_v))
