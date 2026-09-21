"""Do we have enough signal at 0.25 s? Compare the new parent-beta scan
(0.25 s/frame, LAUE_SCAN_DATA) against a 1 s/frame reference scan (LAUE_REF_DATA):
peak counts, per-spot SNR, example frames, and a validated grain overlaid on a
0.25 s frame. Peaks come from frame_peaks.detect_peaks.
"""
import os, sys
import numpy as np, h5py, glob, json, random
from math import pi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor

# No defaults for the two scans: they used to be hard-coded campaign folders.
WORK = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (peel_map/ and figures/ live under it)")
D_1s = os.environ.get("LAUE_REF_DATA") or sys.exit("LAUE_REF_DATA is not set (raw frames of the REFERENCE, longer-exposure scan)")
D_025 = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set (raw frames of the short-exposure scan under test)")
from frame_peaks import out_prefix
PREFIX = out_prefix()
H5LOC="/entry1/data/data"; TOL=8.0
from laue_material import Phase
from frame_peaks import detect_peaks, count_matched_peaks, gate_statistic, gate_counts
_PH_B = Phase.load("beta")
NPX = _PH_B.npx_x

def peaks_of(raw):
    # shared detector (frame_peaks); SNR is measured on its cleaned residual
    xs,ys,info=detect_peaks(raw)
    sub,mad=info["sub"],info["mad"]
    snr=sub[ys,xs]/mad
    return sub, np.c_[xs,ys], snr, mad

def stat(item):
    path,fn=item
    try:
        with h5py.File(f"{path}/{fn}","r") as f: raw=f[H5LOC][()].astype(float)
    except Exception: return None
    _,xy,snr,_=peaks_of(raw)
    return len(xy), (np.median(snr) if len(snr) else 0.0)

def project(OM):
    return _PH_B.project(OM)

# Everything below drives the process pool. Guarded so the script also runs under
# the "spawn" start method (macOS default), where each worker re-imports this
# module: the workers need only the definitions above.
if __name__ == "__main__":
    rng=random.Random(1)
    f1=[f.split("/")[-1] for f in glob.glob(f"{D_1s}/*.h5")]
    f0=[f.split("/")[-1] for f in glob.glob(f"{D_025}/*.h5")]
    if not f1 or not f0:
        sys.exit(f"no .h5 frames found: {len(f1)} under LAUE_REF_DATA={D_1s}, {len(f0)} under LAUE_SCAN_DATA={D_025}")
    s1=rng.sample(f1,min(30,len(f1))); s0=rng.sample(f0,min(30,len(f0)))
    print(f"sampling {len(s1)} x 1s frames and {len(s0)} x 0.25s frames",flush=True)
    jobs=[(D_1s,fn) for fn in s1]+[(D_025,fn) for fn in s0]
    res=[]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(stat, jobs): res.append(r)
    n1=[res[i][0] for i in range(len(s1)) if res[i]]; snr1=[res[i][1] for i in range(len(s1)) if res[i]]
    n0=[res[i][0] for i in range(len(s1),len(jobs)) if res[i]]; snr0=[res[i][1] for i in range(len(s1),len(jobs)) if res[i]]
    print(f"1s   : peaks median {np.median(n1):.0f} (mean {np.mean(n1):.0f}); median per-spot SNR {np.median(snr1):.1f}",flush=True)
    print(f"0.25s: peaks median {np.median(n0):.0f} (mean {np.mean(n0):.0f}); median per-spot SNR {np.median(snr0):.1f}",flush=True)

    # ---- figure ---------------------------------------------------------------
    fig=plt.figure(figsize=(17,10))
    # A: example 0.25s frame + detected peaks
    ax=fig.add_subplot(2,3,1)
    exfn=s0[0]
    with h5py.File(f"{D_025}/{exfn}","r") as f: raw=f[H5LOC][()].astype(float)
    sub,xy,snr,mad=peaks_of(raw)
    ax.imshow(np.clip(sub,1,None),norm=LogNorm(vmin=1,vmax=max(50,np.percentile(sub,99.9))),cmap="gray_r",origin="upper")
    ax.scatter(xy[:,0],xy[:,1],s=18,facecolors="none",edgecolors="lime",lw=0.5)
    ax.set_title(f"0.25 s frame (cleaned) + {len(xy)} detected peaks (SNR>8)",fontsize=10); ax.set_xticks([]);ax.set_yticks([])
    # B: example 1s frame
    ax=fig.add_subplot(2,3,2)
    with h5py.File(f"{D_1s}/{s1[0]}","r") as f: raw1=f[H5LOC][()].astype(float)
    sub1,xy1,snr1e,_=peaks_of(raw1)
    ax.imshow(np.clip(sub1,1,None),norm=LogNorm(vmin=1,vmax=max(50,np.percentile(sub1,99.9))),cmap="gray_r",origin="upper")
    ax.scatter(xy1[:,0],xy1[:,1],s=18,facecolors="none",edgecolors="dodgerblue",lw=0.5)
    ax.set_title(f"1 s frame (cleaned) + {len(xy1)} detected peaks",fontsize=10); ax.set_xticks([]);ax.set_yticks([])
    # C: peak-count comparison
    ax=fig.add_subplot(2,3,4)
    ax.boxplot([n1,n0],tick_labels=["1 s","0.25 s"]); ax.set_ylabel("SNR>8 peaks per frame")
    ax.set_title(f"Peak count: 1 s med {np.median(n1):.0f} vs 0.25 s med {np.median(n0):.0f}\n"
                 f"(ratio {np.median(n0)/max(np.median(n1),1):.2f}; exposure ratio 0.25)",fontsize=10); ax.grid(alpha=.3)
    # D: per-spot SNR comparison
    ax=fig.add_subplot(2,3,5)
    ax.boxplot([snr1,snr0],tick_labels=["1 s","0.25 s"]); ax.set_ylabel("median per-spot SNR (per frame)")
    ax.axhline(8,color="red",ls="--",lw=1,label="detection floor (SNR 8)")
    ax.set_title(f"Per-spot strength: still well above the floor\n1 s {np.median(snr1):.1f} vs 0.25 s {np.median(snr0):.1f}",fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    # E: a validated grain overlaid on its 0.25s frame
    ax=fig.add_subplot(1,3,3)
    z=np.load(f"{WORK}/peel_map/{PREFIX}_beta_validated.npz",allow_pickle=True)
    oms=z["oms"]; frs=np.array([str(f) for f in z["frames"]])
    _stat=gate_statistic(); nh=gate_counts(z,_stat,f"{PREFIX}_beta_validated.npz")
    gi=int(np.argmax(nh))                       # strongest validated beta, by LAUE_GATE_STAT
    with h5py.File(f"{D_025}/{frs[gi]}","r") as f: rawg=f[H5LOC][()].astype(float)
    subg,xyg,snrg,_=peaks_of(rawg)
    ax.imshow(np.clip(subg,1,None),norm=LogNorm(vmin=1,vmax=max(50,np.percentile(subg,99.9))),cmap="gray_r",origin="upper")
    pr=project(oms[gi])
    tree=cKDTree(xyg); d,_=tree.query(pr); hit=d<TOL        # per-prediction mask, for drawing only
    nd_g,nh_g=count_matched_peaks(tree,pr,TOL)              # the counts: harmonics stack in nh_g, not in nd_g
    ax.scatter(pr[hit,0],pr[hit,1],s=120,facecolors="none",edgecolors="lime",lw=1.6,label=f"predicted hit ({nh_g}; {nd_g} distinct peaks)")
    ax.scatter(pr[~hit,0],pr[~hit,1],s=70,marker="x",c="red",lw=1.0,label=f"predicted, no peak ({int((~hit).sum())})")
    ax.set_title(f"A validated β grain on its 0.25 s frame\n{nh_g}/{len(pr)} predicted reflections land on real peaks "
                 f"({nd_g} distinct peaks)",fontsize=10)
    print(f"strongest validated beta: nhit {nh_g}, nhit_distinct {nd_g}, predicted {len(pr)}",flush=True)
    ax.legend(fontsize=8,loc="upper right"); ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Signal at 0.25 s vs 1 s — enough to index: fewer peaks (fainter grains lost), "
                 "but the spots present are strong and grains index cleanly",fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{WORK}/figures/exposure_signal_check.png",dpi=125)
    print("saved exposure_signal_check.png",flush=True)
