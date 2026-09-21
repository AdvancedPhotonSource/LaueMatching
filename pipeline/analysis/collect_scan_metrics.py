"""Collect every headline metric for every analysed scan into one JSON.

Pulls from each scan's analysis log (nulls, census, Burgers, coherence) and from its
saved npz (validated instances, geometry measured from the stage coordinates, grains
under the contiguity-aware definition). Folder names are never trusted for geometry.

Grains are split into connected pieces with the shared raster connectivity
(raster.py); the above-null and gold counts use LAUE_GATE_STAT (nhit by default)
against the null measured for that statistic on each scan.

usage: LAUE_WORK=... collect_scan_metrics.py > metrics.json
"""
import glob
import json
import os
import re
import sys
import numpy as np
from scipy import ndimage as ndi

from frame_peaks import GATE_STATS, gate_statistic, null_json_path
from raster import connectivity, structure

W = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set (scans are read from $LAUE_WORK/analysis/*/)")
ANA = f"{W}/analysis"
# Slow-axis stage coordinates are de-projected to the sample surface by the MOUNT
# angle (raster.mount_deg, required LAUE_MOUNT_DEG); this was a hard-coded sqrt(2).
from raster import mount_deg
ZSCALE = 1.0 / np.cos(np.radians(mount_deg()))
# The hit statistic the above-null / gold counts use; the null is read for the SAME
# statistic. Per-scan nulls only -- a LAUE_NULLMAX_<PHASE> override is deliberately
# NOT applied here, since one value across many scans is exactly the inheritance
# the chain forbids.
STAT = gate_statistic()

def grab(txt, pat, cast=float, default=None):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else default

def scan_null(d, pref, txt, ph):
    """{mean, p999, max, source} for STAT: the scan's null json, else its analysis log.
    The log only carries the nhit statistic in the parseable line."""
    jp = null_json_path(d, pref)
    if os.path.isfile(jp):
        with open(jp) as fh:
            rec = (json.load(fh).get("phases", {}).get(ph) or {}).get(STAT)
        if rec:
            return {"mean": rec.get("mean"), "p999": rec.get("p999"), "max": rec.get("max"),
                    "statistic": STAT, "source": "null.json"}
    if STAT != "nhit":
        return {"mean": None, "p999": None, "max": None, "statistic": STAT,
                "source": "none (no null.json; the log's null line is nhit)"}
    blk = re.search(rf"\[{ph}\] RANDOM-ORIENTATION NULL.*?\n(.*?)\n", txt, re.S)
    line = blk.group(1) if blk else ""
    return {"mean": grab(line, r"mean hits ([0-9.]+)"),
            "p999": grab(line, r"99\.9th ([0-9]+)", int),
            "max": grab(line, r"max ([0-9]+)", int),
            "statistic": STAT, "source": "analysis log"}

def phase_metrics(d, pref, ph, nullmax):
    f = f"{d}/peel_map/{pref}_{ph}_validated.npz"
    if not os.path.isfile(f):
        return None
    z = np.load(f, allow_pickle=True)
    X, Z, lab = z["X"].astype(float), z["Z"].astype(float), z["labels"]
    if STAT not in z.files:
        return {"instances": int(len(X)), "error": f"no {STAT} column in {os.path.basename(f)}"}
    nhit = z[STAT].astype(int)
    out = {"instances": int(len(X)), "statistic": STAT}

    # geometry MEASURED from the stage coordinates, never from the folder name
    Xu = np.unique(np.round(X, 4)); Zu = np.unique(np.round(Z, 4))
    dx = float(np.median(np.diff(Xu))) if len(Xu) > 1 else 0.0
    dz = float(np.median(np.diff(Zu))) if len(Zu) > 1 else 0.0
    out["geom"] = {
        "positions": int(len(set(zip(np.round(X, 4), np.round(Z, 4))))),
        "nx": int(len(Xu)), "nz": int(len(Zu)),
        "span_x_um": round(float(Xu.max()-Xu.min()), 3) if len(Xu) > 1 else 0.0,
        "span_z_lab_um": round(float(Zu.max()-Zu.min()), 3) if len(Zu) > 1 else 0.0,
        "step_x_um": round(dx, 4), "step_z_lab_um": round(dz, 4),
        # sample-frame Z is the stage Z de-projected by the mount (LAUE_MOUNT_DEG)
        "span_z_sample_um": round(float(Zu.max()-Zu.min())*ZSCALE, 3) if len(Zu) > 1 else 0.0,
        "step_z_sample_um": round(dz*ZSCALE, 4),
        "mount_deg": mount_deg(),
    }
    out["above_nullmax"] = int((nhit > nullmax).sum()) if nullmax else None
    out["above_nullmax_pct"] = round(100*float((nhit > nullmax).mean()), 1) if nullmax else None
    out["median_hits"] = int(np.median(nhit))
    for other in GATE_STATS:                         # both counts, where the npz has them
        if other in z.files:
            out[f"median_{other}"] = int(np.median(z[other]))

    # contiguity-aware grains
    Xi = {v: i for i, v in enumerate(Xu)}; Zi = {v: i for i, v in enumerate(Zu)}
    gi = np.array([Zi[round(v, 4)] for v in Z]); gj = np.array([Xi[round(v, 4)] for v in X])
    shape = (len(Zu), len(Xu)); st = structure()
    grains, gold = 0, 0
    ge5 = 0
    for c in range(lab.max()+1):
        idx = np.where(lab == c)[0]
        if not len(idx):
            continue
        m = np.zeros(shape, bool); m[gi[idx], gj[idx]] = True
        cc, n = ndi.label(m, structure=st)
        for k in range(1, max(n, 1)+1):
            keep = cc[gi[idx], gj[idx]] == k if n >= 1 else np.ones(len(idx), bool)
            if not keep.any():
                continue
            g = idx[keep]
            grains += 1
            npos = len(set(zip(gi[g], gj[g])))
            if npos >= 5:
                ge5 += 1
                if nullmax and (nhit[g] > nullmax).any():
                    gold += 1
    out["clusters"] = int(lab.max()+1)
    out["grains"] = grains
    out["grains_ge5"] = ge5
    out["gold"] = gold
    return out

print(f"statistic {STAT} (LAUE_GATE_STAT), connectivity {connectivity()}-neighbour "
      f"(LAUE_CONNECTIVITY)", file=sys.stderr)
res = {}
dirs = sorted(glob.glob(f"{ANA}/*/"))
if not dirs:
    sys.exit(f"no scan directories under {ANA}/")
for d in dirs:
    scan = os.path.basename(d.rstrip("/"))
    va = glob.glob(f"{d}/peel_map/*_alpha_validated.npz")
    if not va:
        continue
    pref = os.path.basename(va[0]).replace("_alpha_validated.npz", "")
    logf = sorted(glob.glob(f"{d}/*analysis*.log"))
    txt = open(logf[0]).read() if logf else ""

    nm = {ph: scan_null(d.rstrip("/"), pref, txt, ph) for ph in ("alpha", "beta")}

    entry = {
        "prefix": pref,
        "null": nm,
        "alpha_claim_pct": grab(txt, r"alpha claims ([0-9.]+)% of peaks"),
        "burgers_parents": grab(txt, r"=> (\d+) significant prior-beta grain", int),
        "burgers_explained_pct": grab(txt, r"significant-alpha instances \((\d+)%\)", int),
        "coherence_obs": grab(txt, r"OBSERVED same-variant fraction: ([0-9.]+)"),
        "coherence_null": grab(txt, r"SHUFFLED null: mean ([0-9.]+)"),
        "coherence_z": grab(txt, r"\nz = ([0-9.]+)", float),
    }
    for ph in ("alpha", "beta"):
        entry[ph] = phase_metrics(d, pref, ph, nm[ph]["max"])
    res[scan] = entry
    print(f"collected {scan}", file=sys.stderr)

if not res:
    sys.exit(f"{len(dirs)} directories under {ANA}/ but none has peel_map/*_alpha_validated.npz")
json.dump(res, sys.stdout, indent=1)
