"""Cross-frame backfill + grain-extent (shape) map on the 10,201-frame
100x100um_TestScan_About1parentbeta scan, for either phase.

Master list = the phase's VALIDATED grain orientations (from
parentbeta_<phase>_validated.npz). Each is forward-projected into EVERY frame
and tested for presence (Poisson p<1e-5, multiple-testing-aware). Present-but-
missed detections are added -> each grain's full spatial extent -> shape map.

Fast downsampled-median background (~16x faster) makes 10,201 frames tractable.
usage: parentbeta_backfill.py {alpha|beta}
Outputs: peel_map/parentbeta_<phase>_extent.npz, figures/parentbeta_<phase>_extent.png
"""
import numpy as np, h5py, glob, json, sys
from math import pi, cos, sin
from scipy.spatial import cKDTree, ConvexHull
from scipy import ndimage as ndi
from scipy.stats import poisson
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WORK="/net/hpcs34/data34c/for_Hemant/lauematching_ti"
DATA="/net/hpcs34/data34c/Run2026-2/Thompson_202607/Initial_Indexing_TestScans/ID6-100x100um_TestScan_About1parentbeta"
H5LOC="/entry1/data/data"; HC=1.2398419739; TOL=8.0; NPX=2048; PGATE=1e-5
P=np.array([0.028834,0.002715,0.513399]); Rrod=np.array([-1.20334591,-1.2137853,-1.21669634])
dx=dy=0.0002; Elo,Ehi=5.,30.
angr=np.linalg.norm(Rrod); v=Rrod/angr; c_,s_=np.cos(angr),np.sin(angr)
rot=np.array([[c_+(1-c_)*v[0]**2,(1-c_)*v[0]*v[1]-s_*v[2],(1-c_)*v[0]*v[2]+s_*v[1]],
              [(1-c_)*v[1]*v[0]+s_*v[2],c_+(1-c_)*v[1]**2,(1-c_)*v[1]*v[2]-s_*v[0]],
              [(1-c_)*v[2]*v[0]-s_*v[1],(1-c_)*v[2]*v[1]+s_*v[0],c_+(1-c_)*v[2]**2]])
roti=np.linalg.inv(rot); ki=np.array([0,0,1.0])
PHASE=sys.argv[1] if len(sys.argv)>1 else "beta"
if PHASE=="alpha":
    a,b,c=0.2921,0.2921,0.4665; cg,sg=cos(120*pi/180),sin(120*pi/180); pv=2*pi/(a*b*c*sg)
    a0,a1,a2=a,0,0; b0,b1,b2=b*cg,b*sg,0; c0,c1,c2=0,0,c
    B=np.array([[(b1*c2-b2*c1),(c1*a2-c2*a1),(a1*b2-a2*b1)],
                [(b2*c0-b0*c2),(c2*a0-c0*a2),(a2*b0-a0*b2)],
                [(b0*c1-b1*c0),(c0*a1-c1*a0),(a0*b1-a1*b0)]])*pv
    HKL=np.loadtxt(f"{WORK}/params/valid_hkls_Ti_alpha.csv")[:,:3]
else:
    B=np.eye(3)*2*pi/0.33065; HKL=np.loadtxt(f"{WORK}/params/valid_hkls_Ti_beta.csv")[:,:3]

def project(OM):
    q=(OM@B@HKL.T).T; ql=np.linalg.norm(q,axis=1); m=ql>1e-9
    q,ql=q[m],ql[m]; qh=q/ql[:,None]
    kf=ki-2*qh[:,2:3]*qh; xd=(roti@kf.T).T; m=xd[:,2]>0
    xd,ql,qh=xd[m],ql[m],qh[m]; xs=xd*P[2]/xd[:,2:3]
    px=(xs[:,0]-P[0])/dx+0.5*(NPX-1); py=(xs[:,1]-P[1])/dy+0.5*(NPX-1); st=-qh[:,2]
    mk=(px>=0)&(px<NPX-1)&(py>=0)&(py<NPX-1)&(st>1e-9)
    E=HC*ql[mk]/st[mk]/(4*pi); me=(E>Elo)&(E<Ehi)
    return np.c_[px[mk][me],py[mk][me]]

# master list: validated grain reps (one OM per cluster label)
z=np.load(f"{WORK}/peel_map/parentbeta_{PHASE}_validated.npz",allow_pickle=True)
oms=z["oms"]; labels=z["labels"]; vfr=np.array([str(f) for f in z["frames"]])
if labels.max()<0 or not (labels>=0).all():
    # not clustered -> dedupe here by 1 deg (cubic/hex)
    print("clustering master list on the fly...",flush=True)
    def rmat(ax,deg):
        u=np.asarray(ax,float);u/=np.linalg.norm(u);t=np.radians(deg)
        K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
        return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)
    if PHASE=="alpha":
        OPS=[rmat([0,0,1],60*k) for k in range(6)]+[rmat([np.cos(np.radians(x)),np.sin(np.radians(x)),0],180) for x in (0,30,60,90,120,150)]
    else:
        OPS=[np.eye(3)]+[rmat(ax,d) for ax,d in [([1,0,0],90),([1,0,0],180),([1,0,0],270),([0,1,0],90),([0,1,0],180),([0,1,0],270),([0,0,1],90),([0,0,1],180),([0,0,1],270),([1,1,0],180),([1,-1,0],180),([1,0,1],180),([-1,0,1],180),([0,1,1],180),([0,1,-1],180),([1,1,1],120),([1,1,1],240),([1,-1,1],120),([1,-1,1],240),([-1,1,1],120),([-1,1,1],240),([1,1,-1],120),([1,1,-1],240)]]
    OPS=np.array(OPS)
    def mm(A,Bs):
        best=np.full(len(Bs),999.)
        for S in OPS: best=np.minimum(best,np.degrees(np.arccos(np.clip((np.einsum('ij,kj,mki->m',S,A,Bs)-1)/2,-1,1))))
        return best
    labels=np.full(len(oms),-1); cid=0
    for i in range(len(oms)):
        if labels[i]>=0: continue
        un=np.where(labels<0)[0]; labels[un[mm(oms[i],oms[un])<1.0]]=cid; cid+=1
# master = only REAL recurring grains (clusters seen at >= MINSZ frames), not
# the thousands of spurious singletons -> keeps the backfill fast + meaningful.
MINSZ=int(sys.argv[2]) if len(sys.argv)>2 else 5
sizes=np.bincount(labels); keep=np.where(sizes>=MINSZ)[0]
reps=np.array([oms[np.where(labels==k)[0][0]] for k in keep])
orig=[set(vfr[labels==k]) for k in keep]
ngr=len(reps)
PRED=[project(R) for R in reps]
print(f"[{PHASE}] {len(sizes)} clusters total; {ngr} master grains at >= {MINSZ} frames; "
      f"mean predicted spots {np.mean([len(p) for p in PRED]):.0f}",flush=True)

mapping=json.load(open(f"{WORK}/results/parentbeta_{PHASE}/frame_mapping.json"))
img2file={int(k):vv["file"] for k,vv in mapping.items() if isinstance(vv,dict) and "file" in vv}
frames=sorted(set(img2file.values()))
print(f"[{PHASE}] frames to scan: {len(frames)}",flush=True)

def scan(fn):
    try:
        with h5py.File(f"{DATA}/{fn}","r") as f:
            raw=f[H5LOC][()].astype(float); X=float(f["entry1/sample/sampleX"][()][0]); Z=float(f["entry1/sample/sampleZ"][()][0])
    except Exception: return None
    med=np.median(raw); mad=1.4826*np.median(np.abs(raw-med))
    bg4=ndi.median_filter(raw[::4,::4],25); bg=np.kron(bg4,np.ones((4,4)))[:NPX,:NPX]
    sub=raw-bg; pk=(sub==ndi.maximum_filter(sub,9))&(sub>8*mad); ys,xs=np.where(pk)
    present=np.zeros(ngr,bool)
    if len(xs)>=5:
        tree=cKDTree(np.c_[xs,ys]); npeaks=len(xs)
        for g in range(ngr):
            pr=PRED[g]
            if not len(pr): continue
            d,_=tree.query(pr); h=int((d<TOL).sum())
            lam=len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
            if poisson.sf(h-1,lam)<PGATE: present[g]=True
    return fn,X,Z,present

PRESENT={}; FRPOS={}; done=0
with ProcessPoolExecutor(max_workers=36) as ex:
    for r in ex.map(scan, frames, chunksize=8):
        done+=1
        if r:
            fn,X,Z,present=r; PRESENT[fn]=present; FRPOS[fn]=(X,Z)
        if done%2000==0: print(f"[{PHASE}] {done}/{len(frames)} frames scanned",flush=True)
# extent = UNION of original confirmed frames and backfilled present frames
ext=[[] for _ in range(ngr)]; extfr=[set() for _ in range(ngr)]
for g in range(ngr):
    for fn in orig[g]:
        if fn in FRPOS: ext[g].append(FRPOS[fn]); extfr[g].add(fn)
added=0
for fn,present in PRESENT.items():
    for g in np.where(present)[0]:
        if fn not in extfr[g]: ext[g].append(FRPOS[fn]); extfr[g].add(fn); added+=1
extn=np.array([len(e) for e in ext]); orig_tot=sum(len(o) for o in orig)
print(f"[{PHASE}] original detections {orig_tot}; after backfill {int(extn.sum())}; ADDED {added} "
      f"(+{100*added/max(orig_tot,1):.0f}%)",flush=True)
print(f"[{PHASE}] grains extent>=1 {(extn>=1).sum()}; >=5 {(extn>=5).sum()}; "
      f">=20 {(extn>=20).sum()}; max extent {extn.max()}",flush=True)
print(f"[{PHASE}] multiple-testing: {ngr}x{len(frames)} tests at p<{PGATE:g} -> ~{ngr*len(frames)*PGATE:.0f} expected false",flush=True)

# grain-extent (shape) map
fig,ax=plt.subplots(figsize=(11,10))
order=np.argsort(-extn); cmap=plt.get_cmap("tab20"); drawn=0
for k in order:
    if extn[k]<5: continue
    E=np.array(ext[k]); col=cmap(drawn%20)
    if len(E)>=3:
        try:
            h=ConvexHull(E); poly=np.vstack([E[h.vertices],E[h.vertices][:1]])
            ax.fill(poly[:,0],poly[:,1],color=col,alpha=0.16,lw=0)
            ax.plot(poly[:,0],poly[:,1],color=col,lw=1.0,alpha=0.5)
        except Exception: pass
    else:
        ax.scatter(E[:,0],E[:,1],s=20,color=col)
    drawn+=1
    if drawn>=80: break
ax.set_xlabel("sampleX (µm)"); ax.set_ylabel("sampleZ (µm)"); ax.set_aspect("equal")
ttl = ("prior-beta grain shape (retained beta ~one orientation across the field)" if PHASE=="beta"
       else "alpha grain shapes / Burgers-variant colonies of the prior-beta grain")
ax.set_title(f"parent-beta scan ({PHASE}): grain-extent map by backfill — {drawn} grains\n{ttl}\n"
             f"each confirmed grain forward-projected into all {len(frames)} frames (overlap expected)",fontsize=11)
fig.tight_layout(); fig.savefig(f"{WORK}/figures/parentbeta_{PHASE}_extent.png",dpi=125)
print(f"[{PHASE}] saved parentbeta_{PHASE}_extent.png",flush=True)
np.savez(f"{WORK}/peel_map/parentbeta_{PHASE}_extent.npz",
         extn=extn, reps=reps, ext=np.array(ext,dtype=object))
print(f"[{PHASE}] saved parentbeta_{PHASE}_extent.npz",flush=True)
