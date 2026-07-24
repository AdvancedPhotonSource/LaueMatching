"""Re-define a grain as CONTIGUOUS + consistent orientation, and recount.

The pipeline clusters validated instances by orientation alone. That merges regions
which share an orientation but are spatially disjoint -- the largest 'grain' in ID26
turned out to be two lobes 0.44 deg apart, 1,529 positions between them.

Two separate issues, and they pull in opposite directions:

  * ORIENTATION TOLERANCE. 1.0 deg is a defensible grain criterion -- conventionally a
    boundary is >=1-2 deg, so a 0.44 deg difference is subgrain structure, not a grain
    boundary. Tightening the tolerance alone would over-split deformed grains.
  * SPATIAL CONNECTIVITY. Regardless of misorientation, two disjoint regions are not one
    grain. Orientation-only clustering has no way to express that.

So the physically meaningful count is: cluster by orientation at a conventional
tolerance, THEN split each cluster into spatially connected components. This does not
depend on choosing an aggressive tolerance, and it is what the ID26 diagnostic showed
was needed.

Reported for each tolerance:
  clusters      orientation-only (what the reports quoted)
  grains        after splitting into connected components
  grains >=5    the doubly-supported tier
  largest       biggest single grain, in beam positions

usage: regrain.py [phase] [gap]
  gap: allowed gap in positions when deciding connectivity (default 1 = strict 4-conn)
"""
import os
import sys
import numpy as np
from scipy import ndimage as ndi

W = os.environ.get("LAUE_WORK", "$LAUE_WORK")
PREFIX = os.environ.get("LAUE_OUT_PREFIX", "scan")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "alpha"
GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TOLS = [0.3, 0.5, 1.0]
# Measured null maxima MUST come from the scan being re-grained: lambda depends on that
# scan's own peak and reflection counts. Passed in via env by batch_regrain.sh, which
# parses them from the scan's own analysis log. The ID26 values are only a last-resort
# fallback and are announced loudly when used.
_NM = os.environ.get(f"LAUE_NULLMAX_{PHASE.upper()}")
if _NM is None:
    # No silent default. Inheriting one scan's null maximum for another is the
    # single error this whole chain exists to avoid, and the measured alpha
    # maximum alone ranged 14-17 across nine Ti scans -- a fallback here would
    # be wrong by more than the effect being measured.
    sys.exit(f"LAUE_NULLMAX_{PHASE.upper()} is not set. Run null_model.py on THIS scan "
             f"and pass its measured maximum; do not inherit a value from another scan.")
NULLMAX = {PHASE: int(_NM)}

z = np.load(f"{W}/peel_map/{PREFIX}_{PHASE}_validated.npz", allow_pickle=True)
oms, X, Z, lab = z["oms"], z["X"].astype(float), z["Z"].astype(float), z["labels"]
nhit = z["nhit"].astype(int)

# Symmetry follows the space group of the phase, not its name. The old
# "alpha -> hex-12, anything else -> cubic-24" rule silently gave cubic
# operators to any phase not called "alpha".
from laue_material import Phase
_ph = Phase.load(PHASE)
OPS = _ph.sym_ops
print(f"[{PHASE}] {len(OPS)} proper-rotation operators from space group {_ph.sgnum}", flush=True)

def miso(A, Bs):
    best = np.full(len(Bs), 999.)
    for S in OPS:
        tr = np.einsum('ij,kj,mki->m', S, A, Bs)
        best = np.minimum(best, np.degrees(np.arccos(np.clip((tr-1)/2, -1, 1))))
    return best

Xu = np.unique(np.round(X, 4)); Zu = np.unique(np.round(Z, 4))
xi = {v: i for i, v in enumerate(Xu)}; zi = {v: i for i, v in enumerate(Zu)}
gi = np.array([zi[round(v, 4)] for v in Z]); gj = np.array([xi[round(v, 4)] for v in X])
shape = (len(Zu), len(Xu))
struct = ndi.generate_binary_structure(2, 2)     # 8-connectivity

def split_spatial(idx):
    """Split one orientation cluster into spatially connected components."""
    m = np.zeros(shape, bool)
    m[gi[idx], gj[idx]] = True
    mm = ndi.binary_closing(m, structure=struct, iterations=GAP) if GAP > 1 else m
    cc, n = ndi.label(mm, structure=struct)
    if n <= 1:
        return [idx]
    out = []
    for c in range(1, n+1):
        keep = cc[gi[idx], gj[idx]] == c
        if keep.any():
            out.append(idx[keep])
    return out

base = [np.where(lab == c)[0] for c in range(lab.max()+1)]
print(f"{PREFIX} {PHASE}: {len(oms):,} validated instances, "
      f"{len(base):,} orientation-only clusters at 1.0 deg, gap={GAP}\n")
print(f"{'tol':>5} {'clusters':>10} {'grains':>9} {'grains>=5':>10} {'largest':>9} {'gold':>7}")

nm = NULLMAX.get(PHASE, 16)
for tol in TOLS:
    clusters = []
    for idx in base:
        if tol >= 1.0 or len(idx) == 1:
            clusters.append(idx); continue
        sub = oms[idx]; l2 = np.full(len(idx), -1); cid = 0
        for i in range(len(idx)):
            if l2[i] >= 0: continue
            un = np.where(l2 < 0)[0]
            l2[un[miso(sub[i], sub[un]) < tol]] = cid; cid += 1
        for c in range(cid):
            clusters.append(idx[l2 == c])
    grains = [g for cl in clusters for g in split_spatial(cl)]
    npos = np.array([len(set(zip(gi[g], gj[g]))) for g in grains])
    gold = sum(1 for g, p in zip(grains, npos) if p >= 5 and (nhit[g] > nm).any())
    print(f"{tol:>5.1f} {len(clusters):>10,} {len(grains):>9,} "
          f"{int((npos>=5).sum()):>10,} {int(npos.max()):>9,} {gold:>7,}")
    if abs(tol - 1.0) < 1e-9:
        np.savez(f"{W}/peel_map/{PREFIX}_{PHASE}_regrained.npz",
                 grain_of=np.concatenate([np.full(len(g), k) for k, g in enumerate(grains)]),
                 inst=np.concatenate(grains), npos=npos)
print("\ngold = grain recurs at >=5 positions AND has an instance above the measured null max")
print(f"saved {PREFIX}_{PHASE}_regrained.npz (at 1.0 deg)")
