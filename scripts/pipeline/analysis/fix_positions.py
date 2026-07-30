"""Correct the stage-readback race in a raster scan's position labels.

WHY. On a unidirectional raster the fast-axis readback can be sampled *after* the move
has begun, so one frame takes the next position's label. The signature is acquisition-order
diffs containing 0 and 2*step as adjacent pairs. Seen on both 34-ID-E Zn scans (sampleH: 180 of
20,301 frames, 0.89%; the same signature is in sampleG), so treat it as beamline behaviour and
check every new scan.

WHAT IT IS NOT. It is tempting to read `X, X+2, X+2, X+3` as the motor skipping a position
and dwelling twice -- which would mean a real hole plus a real duplicate. That reading is
wrong, and the fix it implies is the opposite of the correct one. Established from the raw
images on sampleH: the two frames sharing a label have residual-pattern similarity +0.8742
against a matched 1-step control of +0.8738 (identical to four decimals), with a matched
2-step control at +0.7488 proving the measure resolves one step from two; and 0 of 20 pairs
were bit-identical, ruling out a duplicated readout. So the stage visited every position
exactly once: **no gaps, no duplicates, only wrong labels**.

THE FIX. On a unidirectional raster of fixed row length the true position is determined by
the frame index alone:

    col = (N - 1) % NR ,  X_true = X0 + col * step

This module derives that, then VALIDATES it: it must reproduce the recorded X on every
unaffected frame and differ by exactly one step on precisely the frames the diff census
flags. Anything else means the model does not fit and the correction must not be applied.

Run this BEFORE regrain.py. Uncorrected labels put one false hole and one doubled pixel per
affected frame straight into the contiguity definition the grain count depends on.

usage:
    fix_positions.py --scan <folder> --nr 201 [--prefix G21_scan1_]      # census + validate
    fix_positions.py --scan <folder> --nr 201 --npz <validated.npz> [--out <fixed.npz>]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import h5py
except ImportError:  # census/validate need h5py; applying a cached table does not
    h5py = None


def frame_number(name):
    """'G21_scan1_016081.h5' -> 16081. Tolerates zero padding from symlink shards."""
    m = re.search(r"_(\d+)\.h5$", str(name))
    if not m:
        raise ValueError(f"cannot parse a frame number from {name!r}")
    return int(m.group(1))


def read_fast_axis(scan, prefix=None, key="entry1/sample/sampleX", nworkers=4):
    """Return (frames, recorded_X) sorted by frame number."""
    if h5py is None:
        raise SystemExit("h5py is required to read the scan")
    pat = f"{scan}/{prefix}*.h5" if prefix else f"{scan}/*.h5"
    files = glob.glob(pat)
    if not files:
        raise SystemExit(f"no frames matched {pat}")
    nums = sorted(frame_number(f) for f in files)
    byn = {frame_number(f): f for f in files}
    from concurrent.futures import ThreadPoolExecutor

    def rd(n):
        try:
            with h5py.File(byn[n], "r") as h:
                return n, float(h[key][0])
        except Exception:
            return n, np.nan

    with ThreadPoolExecutor(nworkers) as ex:
        got = dict(ex.map(rd, nums))
    F = np.array([n for n in nums if np.isfinite(got[n])])
    return F, np.array([got[n] for n in F])


def census(F, X, nr):
    """Diff census in acquisition order. Returns (step, skip_frames, n_repeat, n_flyback)."""
    d = np.diff(X)
    pos = d[d > 0]
    step = float(np.median(pos[pos > 0])) if len(pos) else 1.0
    zero = np.where(np.abs(d) < 1e-9)[0]
    skip = np.where(np.abs(d - 2 * step) < 1e-6)[0]
    fly = np.where(d < -1e-9)[0]
    skip_frames = F[skip + 1]
    return step, skip_frames, len(zero), len(fly)


def index_derived(F, nr, x0, step):
    return x0 + ((F - 1) % nr) * step


def validate(F, X, nr, verbose=True):
    """Fit and check the index-derived model. Returns (x_true, flagged_mask, step, x0)."""
    step, skip_frames, n_zero, n_fly = census(F, X, nr)
    col = (F - 1) % nr
    x0 = float(np.median(X[col == 0])) if (col == 0).any() else float(X.min())
    xt = index_derived(F, nr, x0, step)
    diff = X - xt
    bad = np.abs(diff) > 1e-9
    ok_exact = int((~bad).sum())
    if verbose:
        print(f"  frames {len(F)}, row length {nr}, step {step:g}, X0 {x0:g}")
        print(f"  diff census: {len(skip_frames)} skips, {n_zero} repeats, {n_fly} flybacks")
        print(f"  index-derived model: {ok_exact} frames exact, {int(bad.sum())} disagree")
        if bad.any():
            u = np.unique(np.round(diff[bad], 6))
            print(f"    disagreements: {u}")
    consistent = (set(F[bad].tolist()) == set(skip_frames.tolist())
                  and bool(np.allclose(diff[bad], step)) if bad.any() else True)
    if verbose:
        print(f"  flagged set == census skip set and all off by exactly one step: {consistent}")
        if not consistent:
            print("  *** MODEL DOES NOT FIT -- do not apply the correction ***")
    return xt, bad, step, x0, consistent


def apply_to_npz(src, F, xt, dst=None, verbose=True):
    lut = dict(zip(F.tolist(), xt.tolist()))
    z = np.load(src, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    if "frames" not in d:
        raise SystemExit(f"{src} has no 'frames' field")
    new = np.array([lut.get(frame_number(f), np.nan) for f in d["frames"]], float)
    if np.isnan(new).any():
        raise SystemExit(f"{int(np.isnan(new).sum())} frames absent from the validated table")
    if "X" in d:
        old = np.asarray(d["X"], float)
        moved = np.abs(new - old) > 1e-9
        if verbose:
            print(f"  {os.path.basename(src)}: {len(old)} rows, {int(moved.sum())} corrected "
                  f"({100 * moved.mean():.2f}%)")
            if moved.any():
                print(f"    shifts: {np.unique(np.round(new[moved] - old[moved], 6))}")
        d["X_recorded"] = old
    d["X"] = new
    d["x_race_corrected"] = np.array(True)
    dst = dst or src.replace(".npz", "_xfix.npz")
    np.savez(dst, **d)
    if verbose:
        print(f"    -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="folder of raw .h5 frames")
    ap.add_argument("--nr", type=int, required=True, help="frames per raster row")
    ap.add_argument("--prefix", default=None, help="filename prefix, e.g. G21_scan1_")
    ap.add_argument("--key", default="entry1/sample/sampleX", help="HDF5 path to the fast axis")
    ap.add_argument("--npz", default=None, help="validated npz to correct (needs a 'frames' field)")
    ap.add_argument("--out", default=None, help="output npz (default: <in>_xfix.npz)")
    a = ap.parse_args()

    print(f"reading fast axis from {a.scan}")
    F, X = read_fast_axis(a.scan, a.prefix, a.key)
    if len(F) % a.nr:
        print(f"  NOTE: {len(F)} frames is not a whole number of {a.nr}-frame rows "
              f"({len(F) % a.nr} extra) -- scan may be incomplete")
    xt, bad, step, x0, ok = validate(F, X, a.nr)
    if not ok:
        sys.exit(2)
    if a.npz:
        apply_to_npz(a.npz, F, xt, a.out)
    else:
        print("  (no --npz given; census and validation only)")


if __name__ == "__main__":
    main()
