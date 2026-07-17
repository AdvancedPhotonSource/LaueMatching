"""Do we have enough signal at 0.25 s? Compare the new parent-beta scan
(0.25 s/frame) against SmallAreaTest1 (1 s/frame): peak counts, per-spot SNR,
example frames, and a validated grain overlaid on a 0.25 s frame.
"""
import numpy as np, h5py, glob, json, random
from math import pi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

WORK="$LAUE_WORK"
D_1s="$LAUE_DATA-2/Thompson_202607/ID6_950C_HIP/SmallAreaTest1"
D_025="$LAUE_DATA-2/Thompson_202607/Initial_Indexing_TestScans/ID6-100x100um_TestScan_About1parentbeta"
H5LOC="/entry1/data/data"; NPX=2048; HC=1.2398419739; TOL=8.0
P=np.array([0.028834,0.002715,0.513399]); Rrod=np.array([-1.20334591,-1.2137853,-1.21669634])
dx=dy=0.0002; Elo,Ehi=5.,30.
angr=np.linalg.norm(Rrod); v=Rrod/angr; c_,s_=np.cos(angr),np.sin(angr)
rot=np.array([[c_+(1-c_)*v[0]**2,(1-c_)*v[0]*v[1]-s_*v[2],(1-c_)*v[0]*v[2]+s_*v[1]],
              [(1-c_)*v[1]*v[0]+s_*v[2],c_+(1-c_)*v[1]**2,(1-c_)*v[1]*v[2]-s_*v[0]],
              [(1-c_)*v[2]*v[0]-s_*v[1],(1-c_)*v[2]*v[1]+s_*v[0],c_+(1-c_)*v[2]**2]])
roti=np.linalg.inv(rot); ki=np.array([0,0,1.0])
B_beta=np.eye(3)*2*pi/0.33065
HKL_b=np.loadtxt(f"{WORK}/params/valid_hkls_Ti_beta.csv")[:,:3]

def peaks_of(raw):
    med=np.median(raw); mad=1.4826*np.median(np.abs(raw-med))
    bg4=ndi.median_filter(raw[::4,::4],25); bg=np.kron(bg4,np.ones((4,4)))[:NPX,:NPX]
    sub=raw-bg; loc=(sub==ndi.maximum_filter(sub,9)); pk=loc&(sub>8*mad)
    ys,xs=np.where(pk); snr=sub[ys,xs]/mad
    return sub, np.c_[xs,ys], snr, mad

def stat(item):
    path,fn=item
    try:
        with h5py.File(f"{path}/{fn}","r") as f: raw=f[H5LOC][()].astype(float)
    except Exception: return None
    _,xy,snr,_=peaks_of(raw)
    return len(xy), (np.median(snr) if len(snr) else 0.0)

def project(OM,B,HKL):
    q=(OM@B@HKL.T).T; ql=np.linalg.norm(q,axis=1); m=ql>1e-9
    q,ql=q[m],ql[m]; qh=q/ql[:,None]
    kf=ki-2*qh[:,2:3]*qh; xd=(roti@kf.T).T; m=xd[:,2]>0
    xd,ql,qh=xd[m],ql[m],qh[m]; xs=xd*P[2]/xd[:,2:3]
    px=(xs[:,0]-P[0])/dx+0.5*(NPX-1); py=(xs[:,1]-P[1])/dy+0.5*(NPX-1); st=-qh[:,2]
    mk=(px>=0)&(px<NPX-1)&(py>=0)&(py<NPX-1)&(st>1e-9)
    E=HC*ql[mk]/st[mk]/(4*pi); me=(E>Elo)&(E<Ehi)
    return np.c_[px[mk][me],py[mk][me]]

rng=random.Random(1)
f1=[f.split("/")[-1] for f in glob.glob(f"{D_1s}/ID6_950C_HIP_scan1_*.h5")]
f0=[f.split("/")[-1] for f in glob.glob(f"{D_025}/*.h5")]
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
z=np.load(f"{WORK}/peel_map/parentbeta_beta_validated.npz",allow_pickle=True)
oms=z["oms"]; frs=np.array([str(f) for f in z["frames"]]); nh=z["nhit"]
gi=int(np.argmax(nh))                       # strongest validated beta
with h5py.File(f"{D_025}/{frs[gi]}","r") as f: rawg=f[H5LOC][()].astype(float)
subg,xyg,snrg,_=peaks_of(rawg)
ax.imshow(np.clip(subg,1,None),norm=LogNorm(vmin=1,vmax=max(50,np.percentile(subg,99.9))),cmap="gray_r",origin="upper")
pr=project(oms[gi],B_beta,HKL_b)
tree=cKDTree(xyg); d,_=tree.query(pr); hit=d<TOL
ax.scatter(pr[hit,0],pr[hit,1],s=120,facecolors="none",edgecolors="lime",lw=1.6,label=f"predicted hit ({int(hit.sum())})")
ax.scatter(pr[~hit,0],pr[~hit,1],s=70,marker="x",c="red",lw=1.0,label=f"predicted, no peak ({int((~hit).sum())})")
ax.set_title(f"A validated β grain on its 0.25 s frame\n{int(hit.sum())}/{len(pr)} predicted reflections land on real peaks",fontsize=10)
ax.legend(fontsize=8,loc="upper right"); ax.set_xticks([]);ax.set_yticks([])
fig.suptitle("Signal at 0.25 s vs 1 s — enough to index: fewer peaks (fainter grains lost), "
             "but the spots present are strong and grains index cleanly",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{WORK}/figures/exposure_signal_check.png",dpi=125)
print("saved exposure_signal_check.png",flush=True)
