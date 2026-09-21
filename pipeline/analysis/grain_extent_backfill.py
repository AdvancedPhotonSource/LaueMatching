"""Grain-extent completion by cross-frame backfill.

Idea (user's): LaueMatching reports grains per frame, but a grain genuinely
present at a frame can be missed there (peel order, threshold). So:
  1. master list = every VERIFIED alpha grain (1782 cluster orientations);
  2. forward-project each into EVERY frame and test presence against that
     frame's SNR>8 peaks (Poisson p<1e-5 -- stricter than per-frame validation
     because this is a 1782x367 multiple-testing scan);
  3. wherever a grain is present but was not originally reported there, ADD it;
  4. the set of frames where a grain is present = its spatial extent -> grain
     SHAPE map (overlap is fine; there is no depth resolution).

Outputs: peel_map/grain_extent.npz, figures/grain_extent_map.png
"""
import os, sys
import numpy as np, h5py
from math import pi
from scipy.spatial import cKDTree, ConvexHull
from scipy.stats import poisson
from concurrent.futures import ProcessPoolExecutor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ and figures/ live under it)")
DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (folder of the raw frames)")
H5LOC="/entry1/data/data"; TOL=8.0; PGATE=1e-5
from laue_material import Phase, phase_name
from frame_peaks import detect_peaks, count_matched_peaks, analytic_gate_note
# The master list is verified_clusters.npz from map_validate_cluster.py, whose
# phase is LAUE_PHASE; project with the same phase.
_PH = Phase.load(phase_name())
NPX = _PH.npx_x

def project(OM):
    return _PH.project(OM)

# master list: verified alpha grain representatives (one OM per cluster)
z=np.load(f"{WORK}/peel_map/verified_clusters.npz",allow_pickle=True)
oms=z["oms"]; labels=z["labels"]; frames=np.array([str(f) for f in z["frames"]])
ngr=labels.max()+1
reps=np.array([oms[np.where(labels==k)[0][0]] for k in range(ngr)])
# precompute each grain's predicted pattern ONCE
PRED=[project(R) for R in reps]
npred=np.array([len(p) for p in PRED])
print(f"master grains: {ngr}; mean predicted spots {npred.mean():.0f}",flush=True)

uniq_fr=sorted(set(frames))
pos={}
def getpos(fn):
    with h5py.File(f"{DATA}/{fn}","r") as f:
        return (float(f["entry1/sample/sampleX"][()][0]),float(f["entry1/sample/sampleZ"][()][0]))
# original extent: frames where each grain was reported
orig=[set() for _ in range(ngr)]
for lab,fn in zip(labels,frames): orig[lab].add(fn)

def scan_frame(fn):
    with h5py.File(f"{DATA}/{fn}","r") as f:
        raw=f[H5LOC][()].astype(float); X=float(f["entry1/sample/sampleX"][()][0]); Z=float(f["entry1/sample/sampleZ"][()][0])
    # shared detector (frame_peaks). BEHAVIOUR CHANGE: replaces the full-frame
    # median_filter(25) background with the detector the validators and null use.
    xs,ys,_=detect_peaks(raw)
    present=np.zeros(ngr,bool)
    if len(xs)>=5:
        tree=cKDTree(np.c_[xs,ys]); npeaks=len(xs)
        for g in range(ngr):
            pr=PRED[g]
            if not len(pr): continue
            # presence gate on nhit (predicted reflections on a peak) -- the
            # statistic the analytic Poisson lambda describes
            _,h=count_matched_peaks(tree,pr,TOL)
            lam=len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
            if poisson.sf(h-1,lam)<PGATE: present[g]=True
    return fn,X,Z,present

# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    analytic_gate_note("grain_extent_backfill")
    fr_pos={}; PRESENT={}
    with ProcessPoolExecutor(max_workers=32) as ex:
        for fn,X,Z,present in ex.map(scan_frame, uniq_fr):
            fr_pos[fn]=(X,Z); PRESENT[fn]=present
    print("scan complete",flush=True)

    # extent per grain = UNION of original confirmed detections (p<1e-4) and
    # backfilled present frames (p<1e-5) -- backfill only ADDS, never re-gates.
    ext=[[] for _ in range(ngr)]; extfr=[set() for _ in range(ngr)]
    for g in range(ngr):                       # seed with original confirmed frames
        for fn in orig[g]:
            ext[g].append(fr_pos[fn]); extfr[g].add(fn)
    added=0
    for fn in uniq_fr:
        pr=PRESENT[fn]
        for g in np.where(pr)[0]:
            if fn not in extfr[g]:
                ext[g].append(fr_pos[fn]); extfr[g].add(fn); added+=1
    orig_tot=sum(len(o) for o in orig)
    extn=np.array([len(e) for e in ext]); orign=np.array([len(o) for o in orig])
    new_tot=int(extn.sum())
    print(f"original (grain,frame) detections: {orig_tot}; after backfill: {new_tot}; "
          f"ADDED {added} (+{100*added/orig_tot:.0f}%)",flush=True)
    print(f"grains now spanning >=2 positions: {(extn>=2).sum()} (was {(orign>=2).sum()}); "
          f">=5: {(extn>=5).sum()}; >=10: {(extn>=10).sum()}; max extent {extn.max()}",flush=True)
    exp_fp=ngr*len(uniq_fr)*PGATE
    print(f"multiple-testing control: {ngr}x{len(uniq_fr)} tests at p<{PGATE:g} -> ~{exp_fp:.0f} expected false backfills",flush=True)

    # ---- grain-extent (shape) map: a readable, spatially-diverse selection ------
    fig,ax=plt.subplots(figsize=(12,9.5))
    allpos=np.array([fr_pos[fn] for fn in uniq_fr])
    ax.scatter(allpos[:,0],allpos[:,1],s=7,c="#e9edf0",zorder=1)
    # candidates: real extent but not map-spanning (so individual shapes are visible)
    cand=[]
    for k in np.where((extn>=8)&(extn<=40))[0]:
        E=np.array(ext[k]); cen=E.mean(0); cand.append((k,cen,extn[k]))
    cand.sort(key=lambda t:-t[2])
    picked=[]                                        # greedy for spatial spread
    for k,cen,n in cand:
        if all(np.hypot(*(cen-pc))>2.5 for _,pc,_ in picked): picked.append((k,cen,n))
        if len(picked)>=12: break
    cmap=plt.get_cmap("tab10")
    for j,(k,cen,n) in enumerate(picked):
        E=np.array(ext[k]); col=cmap(j%10)
        try:
            h=ConvexHull(E); poly=np.vstack([E[h.vertices],E[h.vertices][:1]])
            ax.fill(poly[:,0],poly[:,1],color=col,alpha=0.13,zorder=2,lw=0)
            ax.plot(poly[:,0],poly[:,1],color=col,lw=2.0,alpha=0.9,zorder=3)
        except Exception: pass
        ax.scatter(E[:,0],E[:,1],s=16,color=col,edgecolors="white",linewidths=0.3,zorder=4)
    ax.set_xlabel("sampleX (µm)"); ax.set_ylabel("sampleZ (µm)"); ax.set_aspect("equal")
    ax.set_title(f"Grain shapes recovered by cross-frame backfill\n"
                 f"{len(picked)} representative grains (each confirmed grain forward-projected into every "
                 f"frame;\npresent-but-missed detections added: +{added} total, +{100*added/orig_tot:.0f}%). "
                 f"Overlap is real — no depth resolution.",fontsize=11)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/grain_extent_map.png",dpi=125)
    print("saved grain_extent_map.png",flush=True)
    np.savez(f"{WORK}/peel_map/grain_extent.npz",
             extn=extn, orign=orign, added=added, reps=reps,
             ext=np.array(ext,dtype=object), fr_pos=np.array([fr_pos[f] for f in uniq_fr]),
             frames=np.array(uniq_fr))
    print("saved grain_extent.npz",flush=True)
