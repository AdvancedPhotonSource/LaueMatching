"""Validate the full-map beta orientations (Poisson spot-test) + cluster.

Needs LAUE_WORK, LAUE_SCAN_DATA (raw frames) and LAUE_SCAN_BETA (the beta indexing
run holding frame_mapping.json and results/). Peaks and hit counts come from
frame_peaks (detect_peaks, count_matched_peaks); the Poisson gate is on nhit, and
nhit_distinct is saved alongside. Clustering symmetry follows the beta phase's
space group (laue_material), not a hard-coded cubic table.
"""
import os, sys
import numpy as np, h5py, glob, json
from math import pi
from scipy.spatial import cKDTree
from scipy.stats import poisson
WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ is written under it)")
DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (folder of the raw frames)")
RUN = os.environ.get("LAUE_SCAN_BETA") or sys.exit("LAUE_SCAN_BETA is not set (beta indexing-run directory)")
H5LOC="/entry1/data/data"; TOL=8.0
from laue_material import Phase
from frame_peaks import detect_peaks, count_matched_peaks, analytic_gate_note, image_number, orientation_block
_PH_B = Phase.load("beta")
NPX = _PH_B.npx_x
def project(OM):
    return _PH_B.project(OM)
def validate_beta_frame(item):
    fn, oms_flat = item
    oms = oms_flat.reshape(-1,3,3)
    with h5py.File(f"{DATA}/{fn}","r") as f: raw=f[H5LOC][()].astype(float)
    # shared detector. BEHAVIOUR CHANGE: the full-frame median_filter(25)
    # background this script used is replaced by frame_peaks' downsampled median
    # plus its plateau/halo/bloom handling -- the detector the null is measured with.
    xs,ys,_=detect_peaks(raw)
    keep=np.zeros(len(oms),bool); nh=np.zeros(len(oms),int); nd=np.zeros(len(oms),int)
    if len(xs)>=5:
        tree=cKDTree(np.c_[xs,ys]); npeaks=len(xs)
        for i,OM in enumerate(oms):
            pr=project(OM)
            if not len(pr): continue
            h_dist,h=count_matched_peaks(tree,pr,TOL)
            lam=len(pr)*npeaks*pi*TOL*TOL/(NPX*NPX)
            keep[i]=poisson.sf(h-1,lam)<1e-4
            nh[i]=h; nd[i]=h_dist
    return fn, oms, keep, nh, nd

from concurrent.futures import ProcessPoolExecutor
# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    analytic_gate_note("beta_map_validate")
    mapping=json.load(open(f"{RUN}/frame_mapping.json"))
    img2file={int(k):vv["file"] for k,vv in mapping.items() if isinstance(vv,dict) and "file" in vv}
    per_frame={}
    for h5 in sorted(glob.glob(f"{RUN}/results/image_*.output.h5")):
        inum=image_number(h5); fn=img2file.get(inum)
        if fn is None: continue
        try:
            with h5py.File(h5,"r") as f: filt=f["entry/results/filtered_orientations"][()]
        except Exception: continue
        if len(filt): per_frame[fn]=orientation_block(filt,h5)   # layout by column count
    tot=sum(len(v) for v in per_frame.values()); ver=0; oms_v=[]; fr_v=[]; nh_v=[]; nd_v=[]
    with ProcessPoolExecutor(max_workers=32) as ex:
        for fn,oms,keep,nh,nd in ex.map(validate_beta_frame, list(per_frame.items())):
            for OM,k,h,hd in zip(oms,keep,nh,nd):
                if k: ver+=1; oms_v.append(OM); fr_v.append(fn); nh_v.append(h); nd_v.append(hd)
    print(f"beta instances {tot}, VERIFIED {ver}", flush=True)
    if ver:
        print(f"  hits per verified instance: nhit median {int(np.median(nh_v))}, "
              f"nhit_distinct median {int(np.median(nd_v))}", flush=True)
    # cluster; operators follow the beta phase's space group
    oms_v=np.array(oms_v)
    if len(oms_v):
        labels=np.full(len(oms_v),-1); cid=0
        for i in range(len(oms_v)):
            if labels[i]>=0: continue
            un=np.where(labels<0)[0]
            d=_PH_B.misorientation(oms_v[i],oms_v[un]); labels[un[d<0.7]]=cid; cid+=1
        counts=np.bincount(labels)
        print(f"verified BETA grains: {cid}; at >=2 frames: {(counts>=2).sum()}; >=5: {(counts>=5).sum()}")
        np.savez(f"{WORK}/peel_map/beta_verified.npz", oms=oms_v, labels=labels, frames=np.array(fr_v),
                 nhit=np.array(nh_v), nhit_distinct=np.array(nd_v))
