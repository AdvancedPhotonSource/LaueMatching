"""Stage C: six-gate Burgers parent-beta reconstruction on the
100x100um_TestScan_About1parentbeta scan (expected ~1 prior-beta grain).

Inputs: peel_map/parentbeta_{alpha,beta}_validated.npz (Poisson-validated
instances with stage coords), from parentbeta_validate.py.

Gates (all reported, with nulls):
 1. SYNTHETIC gate: plant a beta, recover its 12 alpha variants + reject decoys.
 2. MODEL COMPARISON: parent inferred from ALPHA (variant back-projection ->
    vote) vs random-beta and Burgers-adjacent-decoy nulls; 1 vs 2 parents.
 3. VALIDATED alpha only (already Poisson p<1e-4).
 4. RETAINED-BETA ANCHOR: directly-indexed dominant beta must equal the
    alpha-inferred parent (interlath retained beta inherits parent orientation).
 5. SPATIAL COHERENCE (posterior): assign each alpha to its best Burgers variant
    of the parent; variant-ID colonies should be contiguous; map figure.
 6. COMPLETENESS: full variant table, tolerances, every number with its null.
"""
import os
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
WORK = os.environ.get("LAUE_WORK", "/net/hpcs34/data34c/for_Hemant/lauematching_ti")
TOL=2.0   # deg, orientation match tolerance (Burgers OR scatter in Ti-64 ~1-2 deg)
# argv[1]=min alpha cluster size, argv[2]=scan prefix ("parentbeta" | "id6_10x10").
# The prefix selects which *_validated.npz pair to read and names the outputs.
PREFIX=sys.argv[2] if len(sys.argv)>2 else "parentbeta"

# ---------- symmetry + Burgers ----------------------------------------------
def rmat(ax,deg):
    u=np.asarray(ax,float); u/=np.linalg.norm(u); t=np.radians(deg)
    K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*(K@K)
HEX=np.array([rmat([0,0,1],60*k) for k in range(6)]+
             [rmat([np.cos(np.radians(x)),np.sin(np.radians(x)),0],180) for x in (0,30,60,90,120,150)])
CUB=[np.eye(3)]
for ax,d in [([1,0,0],90),([1,0,0],180),([1,0,0],270),([0,1,0],90),([0,1,0],180),([0,1,0],270),
    ([0,0,1],90),([0,0,1],180),([0,0,1],270),([1,1,0],180),([1,-1,0],180),([1,0,1],180),([-1,0,1],180),
    ([0,1,1],180),([0,1,-1],180),([1,1,1],120),([1,1,1],240),([1,-1,1],120),([1,-1,1],240),([-1,1,1],120),
    ([-1,1,1],240),([1,1,-1],120),([1,1,-1],240)]:
    CUB.append(rmat(ax,d))
CUB=np.array(CUB)
def miso(A,Bs,OPS):
    best=np.full(len(Bs),999.)
    for S in OPS:
        tr=np.einsum('ij,kj,mki->m',S,A,Bs); best=np.minimum(best,np.degrees(np.arccos(np.clip((tr-1)/2,-1,1))))
    return best
def hexmiso(A,Bs): return miso(A,Bs,HEX)
def cubmiso(A,Bs): return miso(A,Bs,CUB)
def burgers_Cv():
    planes=[(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1)]; Cs=[]
    for n in planes:
        n=np.array(n,float); dirs=[]
        for s in [(1,1,1),(1,1,-1),(1,-1,1),(-1,1,1)]:
            d=np.array(s,float)
            if abs(n@d)<1e-9 and not any(np.allclose(d,x)or np.allclose(d,-x) for x in dirs): dirs.append(d)
        for d in dirs:
            a=d/np.linalg.norm(d); c=n/np.linalg.norm(n); b=np.cross(c,a)
            Cs.append(np.column_stack([a,b,c]))
    return np.array(Cs)
CV=burgers_Cv()
def pred_alpha(OMb): return np.einsum('ij,vjk->vik',OMb,CV)     # (12,3,3) alpha variants
def cand_parents(OMa): return np.einsum('ij,vkj->vik',OMa,CV)   # OMa @ CV[v].T  -> 12 candidate betas
def rand_om(rng):
    q=rng.normal(size=4); q/=np.linalg.norm(q); w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def variants_matched(OMb, alpha_reps, tol=TOL):
    pa=pred_alpha(OMb); return sum(1 for v in range(12) if hexmiso(pa[v],alpha_reps).min()<tol)

# ================= GATE 1: SYNTHETIC =================
print("="*66); print("GATE 1  SYNTHETIC"); print("="*66)
rng=np.random.default_rng(3)
OMb=rand_om(rng); planted=pred_alpha(OMb)
noisy=np.array([rmat(rng.normal(size=3),rng.uniform(0.1,0.5))@A for A in planted])
junk=np.array([rand_om(rng) for _ in range(80)])
asyn=np.vstack([noisy,junk])
print(f"true-beta variants matched: {variants_matched(OMb,asyn)}/12")
null=[variants_matched(rand_om(rng),asyn) for _ in range(300)]
print(f"random-beta null: mean {np.mean(null):.2f} max {np.max(null)}")
dec=rmat(rng.normal(size=3),8.0)@OMb
print(f"8deg decoy: {variants_matched(dec,asyn)}/12")
assert variants_matched(OMb,asyn)>=11 and max(null)<11
print("GATE 1 PASSED\n")

# ================= load validated =================
def load(phase):
    z=np.load(f"{WORK}/peel_map/{PREFIX}_{phase}_validated.npz",allow_pickle=True)
    return z["oms"],z["X"].astype(float),z["Z"].astype(float),z["labels"]
aom,aX,aZ,alab=load("alpha"); bom,bX,bZ,blab=load("beta")
print(f"validated alpha instances {len(aom)}, beta instances {len(bom)}")

# cluster reps (use saved labels if valid, else recluster with given OPS)
def reps_from(oms, labels, OPS, tol=1.0):
    if labels.max()>=0 and (labels>=0).all():
        L=labels
    else:
        L=np.full(len(oms),-1); cid=0
        for i in range(len(oms)):
            if L[i]>=0: continue
            un=np.where(L<0)[0]; d=miso(oms[i],oms[un],OPS); L[un[d<tol]]=cid; cid+=1
    reps=[]; idxs=[]
    for c in range(L.max()+1):
        ii=np.where(L==c)[0]
        if len(ii): reps.append(oms[ii[0]]); idxs.append(ii)
    return np.array(reps), idxs, L
a_reps_all,a_idx_all,alab=reps_from(aom,alab,HEX)
b_reps_all,b_idx_all,blab=reps_from(bom,blab,CUB)
a_sz_all=np.array([len(x) for x in a_idx_all]); b_sz_all=np.array([len(x) for x in b_idx_all])
print(f"alpha clusters {len(a_reps_all)} (top sizes {sorted(a_sz_all,reverse=True)[:14]})")
print(f"beta  clusters {len(b_reps_all)} (top sizes {sorted(b_sz_all,reverse=True)[:8]})")
# keep only SIGNIFICANT alpha orientations (real recurring grains) for parent inference
ASZ=int(sys.argv[1]) if len(sys.argv)>1 else 30
sel=a_sz_all>=ASZ
a_reps=a_reps_all[sel]; a_idx=[a_idx_all[i] for i in np.where(sel)[0]]; a_sz=a_sz_all[sel]
b_reps=b_reps_all; b_idx=b_idx_all; b_sz=b_sz_all
print(f"using {len(a_reps)} significant alpha orientations (>= {ASZ} instances) for parent inference")

# fair retained-beta anchor: nearest beta cluster (size>=2) to a candidate parent
def beta_anchor(B, minsize=2):
    cand=np.where(b_sz>=minsize)[0]
    if not len(cand): return 999.,-1,0
    d=cubmiso(B,b_reps[cand]); j=int(d.argmin())
    return float(d.min()), int(cand[j]), int(b_sz[cand[j]])

def infer_parent(rep_subset):
    cloud=np.array([c for OMa in rep_subset for c in cand_parents(OMa)])
    L=np.full(len(cloud),-1); cc=0
    for i in range(len(cloud)):
        if L[i]>=0: continue
        un=np.where(L<0)[0]; L[un[cubmiso(cloud[i],cloud[un])<TOL]]=cc; cc+=1
    cnt=np.bincount(L); return cloud[np.where(L==cnt.argmax())[0][0]]

# ============ GATE 2 + 2b: ITERATIVE MULTI-PARENT EXTRACTION ============
print("\n"+"="*66); print("GATE 2+2b  ITERATIVE MULTI-PARENT (each vs nulls + anchor)"); print("="*66)
rng2=np.random.default_rng(5)
assigned=np.zeros(len(a_reps),bool)         # which significant alpha clusters are explained
inst_var=np.full(len(aom),-1)               # per-instance variant id (parent 1 only, for the map)
inst_par=np.full(len(aom),-1)               # per-instance parent id
parents=[]; MAXP=4
for pidx in range(MAXP):
    un=np.where(~assigned)[0]
    if len(un)<8: print(f"only {len(un)} unassigned alpha clusters left -> stop"); break
    sub=a_reps[un]
    B=infer_parent(sub)
    nv=variants_matched(B,sub)
    rn=[variants_matched(rand_om(rng2),sub) for _ in range(400)]
    dn=[variants_matched(rmat(rng2.normal(size=3),rng2.uniform(5,10))@B,sub) for _ in range(150)]
    r99=np.percentile(rn,99); anc,bj,bsz_=beta_anchor(B)
    ok = nv>=5 and nv>max(r99,np.max(dn))
    print(f"\nparent candidate #{pidx+1}: {nv}/12 variants of {len(un)} unassigned alpha | "
          f"random null mean {np.mean(rn):.2f}/99pct {r99:.0f}/max {np.max(rn)} | "
          f"decoy null mean {np.mean(dn):.2f}/max {np.max(dn)} | retained-beta anchor {anc:.2f} deg (size {bsz_})")
    if not ok:
        print(f"  -> NOT significant (need >5 and > nulls). Stop; {pidx} parent(s) found."); break
    # accept: assign alpha clusters matching this parent's variants
    pa=pred_alpha(B); bv=np.full(len(a_reps),-1); bd=np.full(len(a_reps),999.)
    for v in range(12):
        d=hexmiso(pa[v],a_reps); m=(d<bd)&(~assigned); bd[m]=d[m]; bv[m]=v
    newly=(bd<TOL)&(~assigned)
    ninst=int(sum(a_sz[newly]))
    print(f"  -> ACCEPTED parent #{pidx+1}: {int(newly.sum())} alpha clusters, {ninst} instances, "
          f"anchor {anc:.2f} deg {'(CONSISTENT)' if anc<TOL else '(no matching retained beta)'}")
    for c in np.where(newly)[0]:
        inst_par[a_idx[c]]=pidx
        if pidx==0: inst_var[a_idx[c]]=bv[c]
    parents.append(dict(B=B,nv=int(nv),r99=float(r99),decoy_max=int(np.max(dn)),anchor=anc,
                        nclus=int(newly.sum()),ninst=ninst))
    assigned|=newly
B_inf=parents[0]["B"] if parents else infer_parent(a_reps)
tot_assigned=int(sum(a_sz[assigned])); tot_sig=int(a_sz.sum())
print(f"\n=> {len(parents)} significant prior-beta grain(s); "
      f"{tot_assigned}/{tot_sig} significant-alpha instances ({100*tot_assigned/tot_sig:.0f}%) explained")

# No parent cleared the random + decoy nulls at this min-cluster-size. Everything below
# (GATE 5/6, the figure, the summary) assumes at least one parent, so write a well-formed
# EMPTY reconstruction -- so downstream variant_coherence.py finds a file rather than
# FileNotFoundError -- and exit cleanly instead of crashing on parents[0]. Sparse scans
# (small area, coarse step) legitimately land here; retry with a lower min-cluster-size.
if not parents:
    print(f"\nNo prior-beta parent survived the nulls at min-cluster-size {ASZ}. "
          f"Writing an empty reconstruction for {PREFIX}; retry with a smaller min size "
          f"if a parent is expected.")
    np.savez(f"{WORK}/peel_map/{PREFIX}_reconstruction.npz",
             parents_B=np.empty((0, 3, 3)), parents_nv=np.empty(0, int),
             parents_anchor=np.empty(0), parents_ninst=np.empty(0, int),
             inst_var=inst_var, inst_par=inst_par, aX=aX, aZ=aZ, bX=bX, bZ=bZ)
    print(f"saved {PREFIX}_reconstruction.npz (0 parents)")
    sys.exit(0)

# ================= GATE 5: spatial coherence + GATE 6 table =================
print("\n"+"="*66); print("GATE 5  SPATIAL / GATE 6  VARIANT TABLE (parent #1)"); print("="*66)
ncol=int((inst_var>=0).sum())
print(f"alpha instances on parent #1 variants: {ncol}/{len(aom)} ({100*ncol/len(aom):.0f}%); "
      f"all parents: {int((inst_par>=0).sum())} ({100*(inst_par>=0).sum()/len(aom):.0f}%)")
p1=parents[0]; pa=pred_alpha(B_inf); vc=np.array([(inst_var==v).sum() for v in range(12)])
print(f"{'variant':>7} {'matched?':>9} {'#alpha_inst':>12} {'min_miso_deg':>13}")
for v in range(12):
    d=hexmiso(pa[v],a_reps)
    print(f"{v:>7} {'YES' if d.min()<TOL else 'no':>9} {int(vc[v]):>12} {d.min():>13.2f}")

# figure: parent-1 variant-ID map + per-parent summary
fig,ax=plt.subplots(1,2,figsize=(14,6)); cmap=plt.get_cmap("tab20")
sel=inst_var>=0
sc=ax[0].scatter(aX[sel],aZ[sel],c=inst_var[sel],cmap=cmap,vmin=0,vmax=11,s=14,marker="s")
ax[0].scatter(aX[~sel],aZ[~sel],c="lightgray",s=5,marker=".")
anc1=p1["anchor"]
if anc1<TOL:
    bj=beta_anchor(B_inf)[1]; bm=b_idx[bj]
    ax[0].scatter(bX[bm],bZ[bm],facecolors="none",edgecolors="red",s=70,lw=1.4,label=f"retained β at parent ({anc1:.2f}°)")
ax[0].set_xlabel("sampleX (µm)"); ax[0].set_ylabel("sampleZ (µm)"); ax[0].set_aspect("equal")
ax[0].set_title(f"Parent #1 Burgers variant-ID map\n{p1['nv']}/12 variants; retained-β anchor {anc1:.2f}°")
ax[0].legend(fontsize=8,loc="upper right"); fig.colorbar(sc,ax=ax[0],label="α variant (0-11)",fraction=.046)
ax[1].bar(range(12),vc,color=[cmap(v) for v in range(12)])
ax[1].set_xlabel("Burgers α variant"); ax[1].set_ylabel("# validated α instances")
ax[1].set_title("Parent #1 variant occupancy (selection)")
fig.suptitle(f"Parent-β reconstruction: {len(parents)} prior-β grain(s); "
             f"parent #1 = {p1['nv']}/12 variants (null 99pct {p1['r99']:.0f}), anchor {anc1:.2f}°",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{WORK}/figures/{PREFIX}_reconstruction.png",dpi=140)
print(f"saved {PREFIX}_reconstruction.png")

np.savez(f"{WORK}/peel_map/{PREFIX}_reconstruction.npz",
         parents_B=np.array([p["B"] for p in parents]),
         parents_nv=np.array([p["nv"] for p in parents]),
         parents_anchor=np.array([p["anchor"] for p in parents]),
         parents_ninst=np.array([p["ninst"] for p in parents]),
         inst_var=inst_var, inst_par=inst_par, aX=aX, aZ=aZ, bX=bX, bZ=bZ)
print(f"\nsaved {PREFIX}_reconstruction.npz")
print("\n"+"="*66+"\nSUMMARY")
for i,p in enumerate(parents):
    print(f"  parent #{i+1}: {p['nv']}/12 variants, {p['ninst']} α instances, "
          f"retained-β anchor {p['anchor']:.2f}° {'CONSISTENT' if p['anchor']<TOL else '(no clean anchor)'} "
          f"(random null 99pct {p['r99']:.0f}, decoy max {p['decoy_max']})")
print(f"  => {len(parents)} prior-β grain(s); {100*tot_assigned/tot_sig:.0f}% of significant α explained")
