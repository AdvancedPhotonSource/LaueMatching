"""Quick-look grain map of an indexed scan.

Reads the indexer's filtered_orientations plus the real stage coordinates from the raw
frames, clusters into grains, and maps them. This is the UNVALIDATED catalog -- run the
validation chain before treating any count here as a grain count.

All paths come from the environment so this runs against any scan:
    LAUE_SCAN_DATA    raw frames
    LAUE_SCAN_ALPHA / LAUE_SCAN_BETA   indexing-run directory for that phase
    LAUE_WORK         output root (peel_map/ and figures/ are written under it)
    LAUE_OUT_PREFIX   basename prefix for the npz/png

usage: scan_map.py {alpha|beta} [nworkers]
"""
import os
import numpy as np, h5py, glob, json, sys
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PHASE=sys.argv[1] if len(sys.argv)>1 else "alpha"
NWORK=int(sys.argv[2]) if len(sys.argv)>2 else 12
W=os.environ.get("LAUE_WORK", ".")
DATA=os.environ["LAUE_SCAN_DATA"]
RUNDIR=os.environ[f"LAUE_SCAN_{PHASE.upper()}"]
PREFIX=os.environ.get("LAUE_OUT_PREFIX", "scan")
OUT=f"{W}/peel_map"; FIG=f"{W}/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

mapping=json.load(open(f"{RUNDIR}/frame_mapping.json"))
img2file={int(k):v["file"] for k,v in mapping.items() if isinstance(v,dict) and "file" in v}

def read_one(h5):
    inum=int(h5.split("image_")[1][:5]); fn=img2file.get(inum)
    if fn is None: return None
    try:
        with h5py.File(h5,"r") as f: filt=f["entry/results/filtered_orientations"][()]
    except Exception: return None
    if not len(filt): return (fn,None,None,None)
    oms=filt[:,23:32].reshape(-1,3,3)
    try:
        with h5py.File(f"{DATA}/{fn}","r") as f:
            X=float(f["entry1/sample/sampleX"][()].ravel()[0]); Z=float(f["entry1/sample/sampleZ"][()].ravel()[0])
    except Exception: return (fn,oms,None,None)
    return (fn,oms,X,Z)

h5s=sorted(glob.glob(f"{RUNDIR}/results/image_*.output.h5"))
print(f"[{PHASE}] {len(h5s)} output.h5",flush=True)
oms_all=[]; X_all=[]; Z_all=[]; fr_all=[]; per_pos={}
with ProcessPoolExecutor(max_workers=NWORK) as ex:
    for r in ex.map(read_one, h5s, chunksize=16):
        if r is None: continue
        fn,oms,X,Z=r
        if oms is None or X is None:
            if X is not None: per_pos[(X,Z)]=0
            continue
        per_pos[(X,Z)]=len(oms)
        for OM in oms: oms_all.append(OM); X_all.append(X); Z_all.append(Z); fr_all.append(fn)
oms_all=np.array(oms_all); X_all=np.array(X_all); Z_all=np.array(Z_all)
print(f"[{PHASE}] indexed grains {len(oms_all)}; frames with position {len(per_pos)}",flush=True)

# pre-cluster save: clustering is O(N^2) and has died before, losing the whole read pass
np.savez(f"{OUT}/{PREFIX}_{PHASE}_raw.npz", oms=oms_all, X=X_all, Z=Z_all,
         pos=np.array(list(per_pos.keys())), poscount=np.array(list(per_pos.values())))
print(f"[{PHASE}] saved pre-cluster raw npz",flush=True)

# cluster into grains (subsample-safe greedy)
def rmat(ax,deg):
    u=np.asarray(ax,float);u/=np.linalg.norm(u);t=np.radians(deg)
    K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)
if PHASE=="alpha":
    OPS=[rmat([0,0,1],60*k) for k in range(6)]+[rmat([np.cos(np.radians(x)),np.sin(np.radians(x)),0],180) for x in (0,30,60,90,120,150)]
else:
    OPS=[np.eye(3)]+[rmat(ax,d) for ax,d in [([1,0,0],90),([1,0,0],180),([1,0,0],270),([0,1,0],90),([0,1,0],180),([0,1,0],270),([0,0,1],90),([0,0,1],180),([0,0,1],270),([1,1,0],180),([1,-1,0],180),([1,0,1],180),([-1,0,1],180),([0,1,1],180),([0,1,-1],180),([1,1,1],120),([1,1,1],240),([1,-1,1],120),([1,-1,1],240),([-1,1,1],120),([-1,1,1],240),([1,1,-1],120),([1,1,-1],240)]]
OPS=np.array(OPS)
def miso(A,Bs):
    best=np.full(len(Bs),999.)
    for S in OPS: best=np.minimum(best,np.degrees(np.arccos(np.clip((np.einsum('ij,kj,mki->m',S,A,Bs)-1)/2,-1,1))))
    return best
labels=np.full(len(oms_all),-1); cid=0
for i in range(len(oms_all)):
    if labels[i]>=0: continue
    un=np.where(labels<0)[0]; labels[un[miso(oms_all[i],oms_all[un])<1.5]]=cid; cid+=1
counts=np.bincount(labels)
print(f"[{PHASE}] grains (clusters<1.5deg): {cid}; recurring>=2: {(counts>=2).sum()}; >=5: {(counts>=5).sum()}; >=10: {(counts>=10).sum()}; max {counts.max()}",flush=True)
np.savez(f"{OUT}/{PREFIX}_{PHASE}.npz", oms=oms_all, X=X_all, Z=Z_all, labels=labels)

# --- figure: grains-per-position heatmap + recurrence spectrum ---
pos=np.array(list(per_pos.keys())); pc=np.array(list(per_pos.values()))
fig,(axA,axB)=plt.subplots(1,2,figsize=(17,7))
sc=axA.scatter(pos[:,0],pos[:,1],c=pc,cmap="viridis",s=10,marker="s")
fig.colorbar(sc,ax=axA,fraction=0.046).set_label("indexed grains at this position")
axA.set_xlabel("sampleX (µm)"); axA.set_ylabel("sampleZ (µm)"); axA.set_aspect("equal")
axA.set_title(f"A · Indexed {PHASE} grains per position\n{len(per_pos)} positions, mean {pc.mean():.1f}, up to {pc.max()}")
pp=counts[counts>0]
axB.hist(pp,bins=np.arange(1,pp.max()+2),color="#2e5e73",edgecolor="white",lw=.3); axB.set_yscale("log")
axB.set_xlabel("beam positions where the same grain appears"); axB.set_ylabel("# grains (log)")
axB.set_title(f"B · Recurrence spectrum\n{cid} grains; {(pp>=2).sum()} at ≥2 positions, up to {pp.max()}")
fig.suptitle(f"ID6 10×10µm 0.25µm-step scan ({PHASE}): {len(oms_all)} indexed grains → {cid} distinct, "
             f"{(counts>=2).sum()} recurring",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{FIG}/{PREFIX}_{PHASE}_map.png",dpi=125)
print(f"[{PHASE}] saved {PREFIX}_{PHASE}_map.png",flush=True)
