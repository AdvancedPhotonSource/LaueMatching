"""Cross-frame backfill + grain-extent (shape) map on the 10,201-frame
100x100um_TestScan_About1parentbeta scan, for either phase.

Master list = the phase's VALIDATED grain orientations (from
parentbeta_<phase>_validated.npz). Each is forward-projected into EVERY frame
and tested for presence (Poisson p<1e-5, multiple-testing-aware). Present-but-
missed detections are added -> each grain's full spatial extent -> shape map.

Fast downsampled-median background (~16x faster) makes 10,201 frames tractable.
usage: parentbeta_backfill.py {alpha|beta} [min_cluster_size]
env:   LAUE_WORK, LAUE_SCAN_DATA (raw frames), LAUE_PARAMS_<PHASE>;
       LAUE_SCAN_<PHASE> (indexing run; default $LAUE_WORK/results/parentbeta_<phase>)
Outputs: peel_map/parentbeta_<phase>_extent.npz, figures/parentbeta_<phase>_extent.png
"""
import os
import numpy as np, h5py, glob, json, sys
from math import pi
from scipy.spatial import cKDTree, ConvexHull
from scipy.stats import poisson
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ and figures/ live under it)")
DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (folder of the raw frames)")
H5LOC="/entry1/data/data"; TOL=8.0; PGATE=1e-5
from laue_material import phase_name
PHASE=sys.argv[1] if len(sys.argv)>1 else phase_name()   # argv, else LAUE_PHASE / the single LAUE_PHASES entry
# indexing run for this phase (frame_mapping.json); defaults to the legacy location
RUN=os.environ.get(f"LAUE_SCAN_{PHASE.upper()}", f"{WORK}/results/parentbeta_{PHASE}")
# Lattice, reflections, geometry AND symmetry from the parameter file the indexer used.
from laue_material import Phase
from frame_peaks import detect_peaks, count_matched_peaks, analytic_gate_note
_PH=Phase.load(PHASE); NPX=_PH.npx_x

def project(OM):
    return _PH.project(OM)

# master list: validated grain reps (one OM per cluster label)
z=np.load(f"{WORK}/peel_map/parentbeta_{PHASE}_validated.npz",allow_pickle=True)
oms=z["oms"]; labels=z["labels"]; vfr=np.array([str(f) for f in z["frames"]])
if labels.max()<0 or not (labels>=0).all():
    # not clustered -> dedupe here by 1 deg. Operators follow the phase's space
    # group: the old "alpha -> hex, anything else -> cubic" rule gave a hex phase
    # with any other name 24 cubic operators.
    print(f"clustering master list on the fly ({len(_PH.sym_ops)} operators, SG {_PH.sgnum})...",flush=True)
    labels=np.full(len(oms),-1); cid=0
    for i in range(len(oms)):
        if labels[i]>=0: continue
        un=np.where(labels<0)[0]; labels[un[_PH.misorientation(oms[i],oms[un])<1.0]]=cid; cid+=1
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

mapping=json.load(open(f"{RUN}/frame_mapping.json"))
img2file={int(k):vv["file"] for k,vv in mapping.items() if isinstance(vv,dict) and "file" in vv}
frames=sorted(set(img2file.values()))
print(f"[{PHASE}] frames to scan: {len(frames)}",flush=True)

def scan(fn):
    try:
        with h5py.File(f"{DATA}/{fn}","r") as f:
            raw=f[H5LOC][()].astype(float); X=float(f["entry1/sample/sampleX"][()][0]); Z=float(f["entry1/sample/sampleZ"][()][0])
    except Exception: return None
    # shared detector (frame_peaks): the same downsampled-median background, 9 px
    # maximum filter and SNR 8 used here before, plus plateau/halo/bloom handling
    xs,ys,_=detect_peaks(raw)
    present=np.zeros(ngr,bool)
    if len(xs)>=5:
        tree=cKDTree(np.c_[xs,ys]); npeaks=len(xs)
        for g in range(ngr):
            pr=PRED[g]
            if not len(pr): continue
            # presence gate on nhit, the statistic the Poisson lambda describes
            _,h=count_matched_peaks(tree,pr,TOL)
            lam=len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
            if poisson.sf(h-1,lam)<PGATE: present[g]=True
    return fn,X,Z,present

# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    analytic_gate_note("parentbeta_backfill")
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
