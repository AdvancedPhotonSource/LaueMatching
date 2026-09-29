"""The SEARCH null (invariant 29): run the SAME indexer and the SAME validator
on spot-scrambled frames of THIS scan, and record the best they can do.

null_model.py measures a per-DRAW null: how many peaks ONE random orientation
explains. A gate at that bar is not safe against the indexer, which keeps the
best of its whole orientation database (~1e8) per frame. The honest bar is what
that search reaches on a frame with the same spots in random places. For each
sampled frame, and for each of ``scrambles`` scrambles of it:

1. read the indexer's own segmented input for the frame
   (``/entry/data/cleaned_data_threshold_filtered`` in the frame's
   ``image_*.output.h5``), and the raw frame for the validator's peaks;
2. scramble it with ``scramble_frames.scramble_frame`` (same component count,
   intensities and lit-pixel total; positions random inside the frame's 2theta
   band, never onto a masked pixel or gap -- invariant 28), carrying the
   validator's peaks with their components;
3. blur it exactly as the indexer's preprocessing does
   (``laue_index.preprocess.calculate_gaussian_sigma`` on the scrambled
   components, capped by ``GaussSigmaMax``) and run the SAME binary
   (``laue_index.indexer.run_indexer``) with the scan's own parameter file,
   orientation database and reflection list;
4. re-score every solution with the SAME validator statistic
   (``frame_peaks.count_matched_peaks`` against the scrambled peaks, TOL 8 px):
   ``nhit`` and ``nhit_distinct``, plus the indexer's ``NMatches``.

One SEARCH is one draw of this null: its value is the best score among that
search's solutions (0 if the indexer returned none). The unscrambled frame is run
through the same steps as a POSITIVE CONTROL (``real_control`` in the output): if
the pipeline cannot reproduce the scan's own solutions on the real frame, the
null it measures on scrambled frames means nothing.

Output: a ``search_null`` block added to ``$LAUE_WORK/peel_map/<prefix>_null.json``
beside null_model.py's per-draw entries (the per-draw block is left untouched):

    phases/<phase>/search_null = {
        "nhit": {"statistic", "n_draws" (= searches), "mean", "median", "p99",
                 "p999", "max", "kind": "search"},
        "nhit_distinct": {...}, "nmatches": {...},
        "n_searches", "searches_with_solution", "n_solutions", "n_validated",
        "real_control": {...}, "per_search": [...], provenance ...}

The gates (frame_peaks.load_null) use this block when present; LAUE_NULL_KIND=draw
forces the per-draw null.

``--keep-positions`` is the NEGATIVE CONTROL for the method: the "scramble"
keeps every spot where it is, so the search must find the real crystals and the
null must come out at the real frame's level. The block it writes is marked and
refused by every gate. It writes to ``--out`` only (required with it).

usage: search_null.py <phase> [nframes] [scrambles_per_frame] [ncpus]
           [--support band|detector] [--mask mask.npy] [--seed N]
           [--compute CPU|GPU] [--keep-positions --out other.json]
env:   LAUE_WORK, LAUE_SCAN_DATA, LAUE_SCAN_<PHASE>, LAUE_PARAMS_<PHASE> (or
       LAUE_PARAMS), LAUE_OUT_PREFIX
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_peaks import (GATE_STATS, SEARCH_KEY, count_matched_peaks, detect_peaks,
                         image_number, null_json_path, orientation_block, out_prefix,
                         poisson_lambda)
from scramble_frames import ScrambleError, scramble_frame, support_mask, twotheta_map

H5LOC = "/entry1/data/data"
SEG_KEY = "/entry/data/cleaned_data_threshold_filtered"
BLUR_KEY = "/entry/data/input_blurred"
TOL = 8.0            # the validator's tolerance (parentbeta_validate.py)
PGATE = 1e-4         # the validator's per-frame Poisson gate


def read_params(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            tok = line.split("#", 1)[0].split()
            if tok:
                out[tok[0]] = tok[1:]
    return out


class Indexer:
    """The scan's own indexer, run on prepared images.

    The parameter file is copied VERBATIM except for two things the null must not
    share with the scan: file paths are made absolute, and a DoFwd 1 run writes
    its forward cache into the work directory (first search only; later searches
    read it with DoFwd 0) so the scan's cache is never rewritten.
    """

    def __init__(self, params_path, workdir, ncpus=1, compute="CPU"):
        from laue_index import indexer
        self.indexer = indexer
        self.params_path = os.path.abspath(params_path)
        self.workdir = workdir
        self.ncpus = int(ncpus)
        self.compute = compute
        p = read_params(params_path)
        base = os.path.dirname(self.params_path)

        def absify(v):
            # relative to the cwd (as the C and laue_material read it), else to
            # the parameter file's own directory
            if os.path.isabs(v):
                return v
            return os.path.abspath(v) if os.path.exists(v) else os.path.join(base, v)

        for key in ("OrientationFile", "HKLFile"):
            if key not in p:
                raise SystemExit(f"{params_path} has no {key}: the search null runs the "
                                 f"indexer with the scan's own {key}")
        self.orient_db = absify(p["OrientationFile"][0])
        self.hkl_file = absify(p["HKLFile"][0])
        for f in (self.orient_db, self.hkl_file):
            if not os.path.isfile(f):
                raise SystemExit(f"{f} (from {params_path}) does not exist")
        self.n_orient = os.path.getsize(self.orient_db) // 72
        self.do_fwd = int(float(p.get("DoFwd", ["1"])[0]))
        if self.do_fwd:
            self.fwd = os.path.join(workdir, "search_null_forward.bin")
        else:
            self.fwd = absify(p.get("ForwardFile", [""])[0])
            if not os.path.isfile(self.fwd):
                raise SystemExit(f"DoFwd 0 in {params_path} but its ForwardFile {self.fwd} "
                                 f"does not exist")
        self.lines = open(params_path).read().splitlines()
        self.first = True
        self.binary = str(indexer.binary_path(compute, False))

    def _config(self):
        dofwd = 1 if (self.do_fwd and self.first) else 0
        out = []
        for ln in self.lines:
            k = ln.split()[0] if ln.split() else ""
            if k in ("DoFwd", "ForwardFile", "OrientationFile", "HKLFile"):
                continue
            out.append(ln)
        out += [f"DoFwd {dofwd}", f"ForwardFile {self.fwd}",
                f"OrientationFile {self.orient_db}", f"HKLFile {self.hkl_file}"]
        path = os.path.join(self.workdir, f"params_search_null_dofwd{dofwd}.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(out) + "\n")
        return path

    def run(self, blurred, tag):
        """Solutions (n, 34) of the indexer on ``blurred`` (float64, (NrPxY, NrPxX))."""
        img = os.path.join(self.workdir, f"{tag}.bin")
        np.ascontiguousarray(blurred, dtype=np.float64).tofile(img)
        res = self.indexer.run_indexer(
            repo_root=None, config_file=self._config(), orient_db_file=self.orient_db,
            hkl_file=self.hkl_file, image_bin=img, ncpus=self.ncpus,
            output_path=img, compute_type=self.compute, do_forward=False)
        if not res.success:
            tail = ""
            if res.stdout_log and os.path.isfile(res.stdout_log):
                tail = open(res.stdout_log).read()[-1500:]
            raise RuntimeError(f"indexer failed on {tag}: {res.error}\n{tail}")
        self.first = False
        sol = f"{img}.solutions.txt"
        if not os.path.isfile(sol):
            raise RuntimeError(f"indexer wrote no {sol}")
        arr = np.loadtxt(sol, comments="%", ndmin=2)
        return arr.reshape(-1, 34) if arr.size else np.zeros((0, 34))


def blur_like_the_indexer(seg, params):
    """``gaussian_filter(seg, sigma)`` with sigma from the SAME function RunImage and
    the streaming preprocessor use, on this image's own components."""
    from laue_index.preprocess import calculate_gaussian_sigma
    lab, n = ndi.label(seg > 0, structure=np.ones((3, 3), bool))
    if n:
        com = ndi.center_of_mass(seg.astype(float), lab, np.arange(1, n + 1))
        centers = [[k + 1, (float(c[1]), float(c[0])), 0] for k, c in enumerate(com)]
    else:
        centers = []
    px = float(params.get("PxX", ["0.0002"])[0])
    dist = float(params.get("P_Array", ["0", "0", "0.513"])[2])
    spacing = float(params.get("OrientationSpacing", ["0.4"])[0])
    sigma = calculate_gaussian_sigma(centers, px, dist, spacing)
    smax = float(params.get("GaussSigmaMax", ["0"])[0])
    if smax > 0:
        sigma = min(sigma, smax)
    return ndi.gaussian_filter(seg.astype(np.float64), sigma), float(sigma)


def score(sols, peaks, phase):
    """Validator statistics for every solution against ``peaks`` (n, 2) (x, y).

    Returns dict of arrays: nhit, nhit_distinct, nmatches, validated (the
    validator's own Poisson gate, p < 1e-4, on nhit).
    """
    n = len(sols)
    out = {"nhit": np.zeros(n, int), "nhit_distinct": np.zeros(n, int),
           "nmatches": sols[:, 5].astype(int) if n else np.zeros(0, int),
           "validated": np.zeros(n, bool)}
    if not n or len(peaks) < 5:
        return out
    tree = cKDTree(peaks)
    for i, om in enumerate(orientation_block(sols, "solutions").reshape(-1, 3, 3)):
        pr = phase.project(om)
        if not len(pr):
            continue
        hd, h = count_matched_peaks(tree, pr, TOL)
        out["nhit"][i], out["nhit_distinct"][i] = h, hd
        lam = poisson_lambda(len(pr), len(peaks), TOL, phase.npx_x, phase.npx_y)
        out["validated"][i] = poisson.sf(h - 1, lam) < PGATE
    return out


def _stats(arr, name):
    arr = np.asarray(arr, float)
    return {"statistic": name, "kind": "search", "n_draws": int(len(arr)),
            "mean": round(float(arr.mean()), 4), "median": float(np.median(arr)),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "p999": round(float(np.percentile(arr, 99.9)), 3), "max": int(arr.max())}


def measure(frames, phase, params_path, workdir, scrambles=1, ncpus=1, compute="CPU",
            support="band", mask=None, seed=0, keep_positions=False, log=print):
    """Run the search null over ``frames`` and return the ``search_null`` block.

    ``frames`` is a list of dicts with ``name``, ``seg`` (the indexer's segmented
    input), ``raw`` (the raw frame, for the validator's peaks) and optionally
    ``blurred`` (the stored indexer input, checked against our re-blur).
    """
    params = read_params(params_path)
    idx = Indexer(params_path, workdir, ncpus, compute)
    tth = twotheta_map(phase) if support == "band" else None
    rng = np.random.default_rng(seed)
    per_search, real = [], []
    n_fail = 0
    for fr in frames:
        seg = np.asarray(fr["seg"])
        if seg.shape != (phase.npx_y, phase.npx_x):
            raise SystemExit(f"{fr['name']}: segmented image {seg.shape} is not the params' "
                             f"(NrPxY, NrPxX) = {(phase.npx_y, phase.npx_x)}")
        raw = np.asarray(fr["raw"], float)
        xs, ys, _ = detect_peaks(raw)
        peaks = np.c_[xs, ys].astype(float)
        fmask = raw < 0 if mask is None else (np.asarray(mask, bool) | (raw < 0))
        # positive control: the unscrambled frame through the same steps
        blur, sigma = blur_like_the_indexer(seg, params)
        if fr.get("blurred") is not None:
            ref = np.asarray(fr["blurred"], float)
            dev = float(np.abs(ref - blur).max() / max(np.abs(ref).max(), 1e-12))
            if dev > 1e-6:
                log(f"  WARNING {fr['name']}: re-blurred input differs from the stored "
                    f"input_blurred by {dev:.2g} (relative); the null's preprocessing is "
                    f"not the scan's")
        sols = idx.run(blur, f"real_{len(real)}")
        s = score(sols, peaks, phase)
        best_om = (orientation_block(sols, "solutions")[int(np.argmax(s["nhit"]))].tolist()
                   if len(sols) else None)
        real.append({"frame": fr["name"], "n_solutions": int(len(sols)), "sigma": sigma,
                     "best_om": best_om,
                     "best_nhit": int(s["nhit"].max()) if len(sols) else 0,
                     "best_nhit_distinct": int(s["nhit_distinct"].max()) if len(sols) else 0,
                     "best_nmatches": int(s["nmatches"].max()) if len(sols) else 0,
                     "n_validated": int(s["validated"].sum())})
        allowed = support_mask(seg > 0, fmask, tth)
        for k in range(scrambles):
            try:
                sseg, spk, info = scramble_frame(seg, rng, allowed=allowed, peaks=peaks,
                                                 keep_positions=keep_positions)
            except ScrambleError as e:
                n_fail += 1
                log(f"  {fr['name']} scramble {k}: {e}")
                continue
            assert info["n_components_out"] == info["n_components"], info
            assert info["lit_px_out"] == info["lit_px"], info
            sblur, ssig = blur_like_the_indexer(sseg, params)
            ssols = idx.run(sblur, f"scr_{len(per_search)}")
            ss = score(ssols, spk, phase)
            per_search.append({
                "frame": fr["name"], "scramble": k, "n_components": info["n_components"],
                "lit_px": info["lit_px"], "sigma": ssig, "n_solutions": int(len(ssols)),
                "best_nhit": int(ss["nhit"].max()) if len(ssols) else 0,
                "best_nhit_distinct": int(ss["nhit_distinct"].max()) if len(ssols) else 0,
                "best_nmatches": int(ss["nmatches"].max()) if len(ssols) else 0,
                "n_validated": int(ss["validated"].sum())})
            log(f"  {fr['name']} scramble {k}: {len(ssols)} solutions, best nhit "
                f"{per_search[-1]['best_nhit']} (real frame {real[-1]['best_nhit']}, "
                f"{real[-1]['n_solutions']} solutions)")
    if not per_search:
        raise SystemExit(f"no scrambled search completed ({n_fail} scrambles failed to "
                         f"place); nothing to write")
    blk = {"schema": 1, "kind": "search", "keep_positions": bool(keep_positions),
           "tol_px": TOL, "validator_pgate": PGATE, "support": support,
           "n_frames": len(real), "scrambles_per_frame": int(scrambles),
           "n_searches": len(per_search), "n_scramble_failures": n_fail,
           "searches_with_solution": int(sum(p["n_solutions"] > 0 for p in per_search)),
           "n_solutions": int(sum(p["n_solutions"] for p in per_search)),
           "n_validated": int(sum(p["n_validated"] for p in per_search)),
           "params": os.path.abspath(params_path), "orientation_db": idx.orient_db,
           "n_orientations": int(idx.n_orient), "hkl_file": idx.hkl_file,
           "binary": idx.binary, "compute": compute, "seed": int(seed),
           "real_control": {"frames": len(real),
                            "median_best_nhit": float(np.median([r["best_nhit"] for r in real])),
                            "per_frame": real},
           "per_search": per_search}
    for name, key in (("nhit", "best_nhit"), ("nhit_distinct", "best_nhit_distinct"),
                      ("nmatches", "best_nmatches")):
        blk[name] = _stats([p[key] for p in per_search], name)
    return blk


def write_block(path, phase, blk, prefix):
    """Merge ``blk`` into the null json under phases/<phase>/search_null, keeping
    every other key (null_model.py's per-draw entries in particular)."""
    js = {"schema": 1, "prefix": prefix, "phases": {}}
    if os.path.isfile(path):
        with open(path) as fh:
            js = json.load(fh)
    js.setdefault("phases", {}).setdefault(phase, {})[SEARCH_KEY] = blk
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(js, fh, indent=1)
    os.replace(tmp, path)


def _load_frames(run, data, nframes, log):
    mapping = json.load(open(os.path.join(run, "frame_mapping.json")))
    img2file = {int(k): v["file"] for k, v in mapping.items()
                if isinstance(v, dict) and "file" in v}
    outs = {}
    for h5 in glob.glob(os.path.join(run, "results", "image_*.output.h5")):
        try:
            outs[image_number(h5)] = h5
        except ValueError:
            continue
    have = sorted(set(img2file) & set(outs))
    if not have:
        raise SystemExit(f"no frame has both a frame_mapping.json entry and an "
                         f"image_*.output.h5 under {run}/results")
    sel = have[::max(1, len(have) // nframes)][:nframes]
    import h5py
    frames = []
    for inum in sel:
        with h5py.File(outs[inum], "r") as h:
            if SEG_KEY not in h:
                log(f"  {outs[inum]}: no {SEG_KEY}; skipped")
                continue
            seg = h[SEG_KEY][()]
            blurred = h[BLUR_KEY][()] if BLUR_KEY in h else None
        with h5py.File(os.path.join(data, img2file[inum]), "r") as h:
            raw = h[H5LOC][()]
        if raw.ndim == 3:
            raw = raw[0]
        frames.append({"name": img2file[inum], "seg": seg, "raw": raw, "blurred": blurred})
    if not frames:
        raise SystemExit(f"none of the {len(sel)} selected frames carries {SEG_KEY}")
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase")
    ap.add_argument("nframes", nargs="?", type=int, default=20)
    ap.add_argument("scrambles", nargs="?", type=int, default=1)
    ap.add_argument("ncpus", nargs="?", type=int, default=8)
    ap.add_argument("--support", choices=("band", "detector"), default="band")
    ap.add_argument("--mask", default=None, help=".npy bool mask (True = masked)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compute", default="CPU")
    ap.add_argument("--keep-positions", action="store_true",
                    help="NEGATIVE CONTROL: do not move any spot (requires --out)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default=None,
                    help="scratch for indexer inputs/outputs (default: a temp dir under "
                         "$LAUE_WORK/peel_map, removed afterwards)")
    a = ap.parse_args(argv)
    if a.keep_positions and not a.out:
        sys.exit("--keep-positions is a negative control and must not overwrite the "
                 "scan's null: give --out")
    W = os.environ.get("LAUE_WORK") or sys.exit("LAUE_WORK is not set")
    DATA = os.environ.get("LAUE_SCAN_DATA") or sys.exit("LAUE_SCAN_DATA is not set")
    var = f"LAUE_SCAN_{a.phase.upper()}"
    RUN = os.environ.get(var) or sys.exit(f"{var} is not set (indexing-run directory)")
    prefix = out_prefix()
    from laue_material import Phase
    ph = Phase.load(a.phase)
    print(f"[search_null] {a.phase}: {ph}", flush=True)
    mask = np.load(a.mask).astype(bool) if a.mask else None
    frames = _load_frames(RUN, DATA, a.nframes, print)
    os.makedirs(os.path.join(W, "peel_map"), exist_ok=True)
    work = a.workdir or tempfile.mkdtemp(prefix="search_null_", dir=os.path.join(W, "peel_map"))
    os.makedirs(work, exist_ok=True)
    try:
        blk = measure(frames, ph, ph.params_path, work, a.scrambles, a.ncpus, a.compute,
                      a.support, mask, a.seed, a.keep_positions)
    finally:
        if not a.workdir:
            shutil.rmtree(work, ignore_errors=True)
    out = a.out or null_json_path(W, prefix)
    write_block(out, a.phase, blk, prefix)
    print(f"\n[{a.phase}] SEARCH NULL over {blk['n_searches']} scrambled searches "
          f"({blk['n_frames']} frames x {a.scrambles}), DB {blk['n_orientations']:,} "
          f"orientations{'  [NEGATIVE CONTROL: positions kept]' if a.keep_positions else ''}")
    for st in GATE_STATS + ("nmatches",):
        r = blk[st]
        print(f"   best {st:>13}: mean {r['mean']:.2f}  median {r['median']:g}  "
              f"p99 {r['p99']:g}  max {r['max']}")
    print(f"   searches with any solution: {blk['searches_with_solution']}/{blk['n_searches']}; "
          f"solutions passing the validator's Poisson gate: {blk['n_validated']}")
    print(f"   real-frame control: median best nhit {blk['real_control']['median_best_nhit']:g} "
          f"over {blk['n_frames']} frames")
    if blk["searches_with_solution"] == 0:
        print(f"   NOTE: the search returned nothing on any scrambled frame, so its max is 0 "
              f"and the gate reduces to the indexer's own MinNrSpots and the validator's "
              f"Poisson test; the resolution is 1/{blk['n_searches']} per search.")
    print(f"wrote {out} [{SEARCH_KEY}]")


if __name__ == "__main__":
    main()
