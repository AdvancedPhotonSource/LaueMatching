"""Shared peak detection for one raw frame: saturation, halo and streak handling.

`null_model.py`, `parentbeta_validate.py` and every other script that counts hits
against a frame's peaks import `detect_peaks` and `count_matched_peaks` from here.
They must detect and count identically -- the null is what gates the validator's
output, so any drift between them silently compares a count against a null
measured with a different detector or a different statistic.

NOTHING IS DELETED. Every genuine reflection stays in the list, carrying flags
that say what is trustworthy about it. Downstream code decides: indexing wants
POSITIONS (valid even for a clipped peak), intensity analysis must skip saturated
ones. Removing detections was the wrong instinct -- on deposit-on-substrate(111) the weak
reflections sitting beside the saturated Cu spot are exactly the ones carrying the
orientation relationship.

THREE DISTINCT ARTEFACTS, MEASURED NOT ASSUMED
----------------------------------------------
1. FLAT-TOP PLATEAUS. A clipped reflection has a flat top, so every pixel on it
   equals the local maximum and `sub == maximum_filter(sub)` flags all of them:
   one reflection becomes dozens of peaks. On a sampleD frame a single 117 px
   saturated Cu spot produced 58 detections at identical intensity; whole-frame
   counts fell 35-45% once collapsed, and 189 -> 62 on the bare-Cu reference.
   Handled by `saturated_peaks`: one peak per saturated region, positioned from
   the UNSATURATED SHOULDERS so the clipped top does not bias it.

2. ISOTROPIC HALO. The wing of a very intense reflection. Measured around the
   saturated Cu spot on sampleD frame 2500, in ADU above background:

       distance   along column   along row
        15 px        1480          2142
        25 px         508           363
        40 px          74            79
        60 px          34            20
       100 px          14             4      (frame noise sigma = 50)

   It is essentially ISOTROPIC and reaches noise by ~40-60 px, so it is a halo,
   NOT a directional streak. The standing background (a 25 px median on a 4x
   downsample, i.e. ~100 px scale) is far too coarse to follow it, so it leaks
   into the residual, raises the local bar and manufactures maxima. Handled by
   `subtract_halos`: an azimuthal radial profile per bright spot, subtracted.
   This is what lets a weak neighbour stand clear -- on that frame the nearer Zn
   spot (I=579) sat on ~500 ADU of halo.

3. VERTICAL BLOOMING. Charge overflow running a bright column hundreds of rows
   from its source. Real, but rarer than expected: present on the bare-Cu
   reference frame, absent from every sampleD frame sampled. Handled by
   `directional_streaks`, which is SHAPE-selective -- a morphological opening
   removes structures that are thin in one axis and long in the other, and cannot
   touch a compact spot regardless of where it sits. That is strictly safer than
   the position-based band filter it replaces, which could only decide by column
   and so risked deleting real reflections that happened to lie near a bloom.

Do not conflate 1 and 2: on sampleD the saturated Cu spot produces plateau duplicates
and a halo but no detectable bloom at all.

The indexer is unaffected by all of this -- it runs its own percentile + MinArea +
watershed detection. Two separate code paths; check which produced a peak count
before comparing numbers.
"""
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage as ndi

# --- saturation ---------------------------------------------------------------
# Raw ADU at or above which a pixel is clipped (16-bit panel, full scale 65535).
SAT_LEVEL = 65000
# Shoulder band used to position a clipped peak, as a fraction of the clip level.
# Below SHOULDER_LO the wings are noise-dominated; above SHOULDER_HI they are
# already rolling into the flat top and carry no position information.
SHOULDER_LO, SHOULDER_HI = 0.20, 0.90
# How far around a saturated region to look for its shoulders.
SHOULDER_PAD = 12
# A clipped reflection covers at least this many pixels. Below it, a saturated
# region is a hot pixel or a small cluster of them, not a reflection.
SAT_MIN_PX = 4
# ...and it must show at least this many shoulder pixels at intermediate level.
# A hot pixel jumps straight from full scale to background with no shoulder.
SHOULDER_MIN_PX = 8

# --- halo ---------------------------------------------------------------------
# Radius out to which the halo of a bright spot is modelled and subtracted.
# Measured decay reaches frame noise by 40-60 px; 80 gives margin.
HALO_RMAX = 80
# Radial bin width, px.
HALO_BIN = 4
# Only spots at least this bright (ADU above background) get a halo model; the
# wings of anything fainter are already under the noise.
HALO_MIN_PEAK = 8000.0

# --- streaks ------------------------------------------------------------------
# A streak is THIN in one axis and LONG in the other. THIN must exceed the widest
# real spot (p95 aspect on these scans is ~4.9 with cores under ~15 px) and LONG
# must exceed the tallest real spot by a wide margin.
STREAK_THIN = 15
STREAK_LONG = 81
# A streak must stand this far above frame noise to be subtracted at all.
STREAK_K = 3.0


def _bg_sub(raw):
    """Background-subtracted frame and its robust noise sigma."""
    med = np.median(raw)
    mad = 1.4826 * np.median(np.abs(raw - med))
    bg4 = ndi.median_filter(raw[::4, ::4], 25)
    bg = np.kron(bg4, np.ones((4, 4)))[:raw.shape[0], :raw.shape[1]]
    return raw - bg, float(mad)


def directional_streaks(sub, mad, thin=STREAK_THIN, long_=STREAK_LONG, k=STREAK_K):
    """Map of thin-and-long structures (blooming), by morphological opening.

    Shape-selective: opening with a horizontal element erases anything narrower
    than `thin` in x, so `sub - opening` isolates thin-in-x structure; opening
    that with a vertical element keeps only what also runs `long_` rows. A
    compact reflection survives both and contributes nothing, wherever it sits.
    """
    out = np.zeros_like(sub)
    # vertical streaks: thin in x, long in y
    thin_x = sub - ndi.grey_opening(sub, size=(1, thin))
    out += ndi.grey_opening(thin_x, size=(long_, 1))
    # horizontal streaks: thin in y, long in x
    thin_y = sub - ndi.grey_opening(sub, size=(thin, 1))
    out += ndi.grey_opening(thin_y, size=(1, long_))
    out[out < k * mad] = 0.0
    return out


def saturated_peaks(raw, sub, sat_level=SAT_LEVEL,
                    lo=SHOULDER_LO, hi=SHOULDER_HI, pad=SHOULDER_PAD):
    """One peak per clipped region, positioned from its unsaturated shoulders.

    Returns (list of (x, y), labelled saturated mask). The plateau centroid is
    biased whenever the spot is asymmetric, so the position comes from the
    surrounding shoulder band instead; that turns a clipped reflection from a
    liability into a well-determined position, which is what an orientation
    relationship needs. Falls back to the plateau centroid if the shoulders are
    unusable (e.g. two spots merged).
    """
    satmask = raw >= sat_level
    if not satmask.any():
        return [], np.zeros(raw.shape, dtype=np.int32)
    slab, n = ndi.label(satmask)
    objs = ndi.find_objects(slab)
    pts = []
    keep_lab = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(objs, start=1):
        # A clipped REFLECTION has unsaturated shoulders around it. A HOT PIXEL is
        # saturated with ordinary background immediately adjacent. These panels
        # carry ~34 permanently hot pixels, so without this test every frame gains
        # ~34 fabricated "reflections" (and a halo model around each). Size alone
        # is not enough -- hot pixels can cluster -- so require both.
        area = int((slab[sl] == i).sum())
        y0 = max(sl[0].start - pad, 0); y1 = min(sl[0].stop + pad, raw.shape[0])
        x0 = max(sl[1].start - pad, 0); x1 = min(sl[1].stop + pad, raw.shape[1])
        win_raw, win_sub = raw[y0:y1, x0:x1], sub[y0:y1, x0:x1]
        band = (win_raw >= lo * sat_level) & (win_raw < hi * sat_level)
        if area < SAT_MIN_PX or band.sum() < SHOULDER_MIN_PX:
            continue                                # hot pixel, not a reflection
        w = np.clip(win_sub, 0, None) * band
        tot = w.sum()
        if tot <= 0:
            continue
        yy, xx = np.nonzero(band)
        cy = float((w[yy, xx] * yy).sum() / tot)
        cx = float((w[yy, xx] * xx).sum() / tot)
        pts.append((int(round(cx)) + x0, int(round(cy)) + y0))
        keep_lab[i] = True
    # only genuinely clipped reflections take part in plateau collapsing
    slab = np.where(keep_lab[slab], slab, 0)
    return pts, slab


def subtract_halos(sub, centres, mad, rmax=HALO_RMAX, rbin=HALO_BIN):
    """Subtract an azimuthally-averaged radial profile around each bright spot.

    The halo is isotropic (measured), so a median-per-annulus profile removes it
    without touching neighbouring spots: a compact neighbour occupies a small
    fraction of its annulus and cannot move that annulus's median.
    """
    if not len(centres):
        return sub, 0
    out = sub.copy()
    H, W = sub.shape
    yy, xx = np.mgrid[-rmax:rmax + 1, -rmax:rmax + 1]
    rr = np.hypot(yy, xx)
    nbin = int(rmax // rbin) + 1
    which = np.clip((rr / rbin).astype(int), 0, nbin - 1)
    done = 0
    for (cx, cy) in centres:
        y0, y1 = cy - rmax, cy + rmax + 1
        x0, x1 = cx - rmax, cx + rmax + 1
        sy0, sx0 = max(y0, 0), max(x0, 0)
        sy1, sx1 = min(y1, H), min(x1, W)
        if sy1 - sy0 < 8 or sx1 - sx0 < 8:
            continue
        win = out[sy0:sy1, sx0:sx1]
        wsel = which[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        prof = ndi.median(win, wsel, index=np.arange(nbin))
        prof = np.nan_to_num(prof)
        prof[prof < 0] = 0.0
        # never subtract below the noise floor -- that would eat real signal
        prof[prof < mad] = 0.0
        win -= prof[wsel]
        done += 1
    return out, done


def detect_peaks(raw, npx=None, snr=8.0, maxfilt=9,
                 drop_streaks=True, collapse_plateaus=True, remove_halos=True,
                 sat_level=SAT_LEVEL, return_flags=False):
    """Detect reflections on one raw frame.

    ``npx`` is DEPRECATED and IGNORED: it was never used (every size comes from
    ``raw.shape``). It stays the second positional parameter, now optional, so
    existing ``detect_peaks(raw, NPX)`` and ``detect_peaks(raw, NPX, 8.0)`` calls
    keep working unchanged; new code should omit it.

    Returns
    -------
    xs, ys : int arrays        peak pixel coordinates
    info : dict                'mad', 'sub' (cleaned residual), 'n_raw',
                               'n_plateau_dropped', 'n_streak_px', 'n_halos',
                               'flags' -> {'saturated': bool array}
    If `return_flags`, returns (xs, ys, flags, info) where flags['saturated']
    marks peaks whose intensity is not trustworthy (position still is).
    """
    sub, mad = _bg_sub(raw)
    raw_sub = sub
    info = {"mad": mad, "n_plateau_dropped": 0, "n_streak_px": 0, "n_halos": 0}

    # 1. clipped reflections: one peak each, positioned from the shoulders
    sat_pts, slab = ([], None)
    if collapse_plateaus:
        sat_pts, slab = saturated_peaks(raw, sub, sat_level)

    # 2. thin-and-long structures (blooming), removed by shape not by position
    if drop_streaks:
        st = directional_streaks(sub, mad)
        info["n_streak_px"] = int((st > 0).sum())
        sub = sub - st

    # 3. halo of each very bright spot, so weak neighbours stand clear
    if remove_halos:
        bright = [p for p in sat_pts]
        if not bright:
            loc = (raw_sub == ndi.maximum_filter(raw_sub, 25)) & (raw_sub > HALO_MIN_PEAK)
            ys_b, xs_b = np.where(loc)
            bright = list(zip(xs_b.tolist(), ys_b.tolist()))[:40]
        sub, nh = subtract_halos(sub, bright, mad)
        info["n_halos"] = nh

    # 4. detect on the cleaned residual
    pk = (sub == ndi.maximum_filter(sub, maxfilt)) & (sub > snr * mad)
    ys, xs = np.where(pk)
    info["n_raw"] = int(len(xs))

    # 5. replace every detection inside a clipped region with the single
    #    shoulder-fitted peak for that region -- nothing is lost, the duplicates
    #    collapse to the one reflection that produced them
    sat_flag = np.zeros(len(xs), dtype=bool)
    if collapse_plateaus and slab is not None and slab.max() > 0:
        inplateau = slab[ys, xs] > 0
        info["n_plateau_dropped"] = int(inplateau.sum())
        xs, ys = xs[~inplateau], ys[~inplateau]
        sat_flag = np.zeros(len(xs), dtype=bool)
        if sat_pts:
            sx = np.array([p[0] for p in sat_pts], dtype=xs.dtype)
            sy = np.array([p[1] for p in sat_pts], dtype=ys.dtype)
            xs = np.concatenate([xs, sx]); ys = np.concatenate([ys, sy])
            sat_flag = np.concatenate([sat_flag, np.ones(len(sx), dtype=bool)])

    info["sub"] = sub
    info["flags"] = {"saturated": sat_flag}
    if return_flags:
        return xs, ys, info["flags"], info
    return xs, ys, info



def count_matched_peaks(tree, predicted, tol, exclude=None):
    """Count how many of ``predicted`` land on a detected peak, BOTH ways.

    Returns ``(n_distinct, n_predicted)``:

    * ``n_distinct``  -- distinct DETECTED PEAKS explained. This is the quantity
      handbook invariant 15b asks a gate to use. Carried as ``nhit_distinct``.
    * ``n_predicted`` -- predicted reflections that found support. This is what
      the pipeline has always counted (``nhit``), and what the measured null
      maxima in use (e.g. nhit > 11 on sampleH) were calibrated against.

    ``exclude``, if given, is a boolean mask over the peaks in ``tree``: a
    prediction whose nearest peak is excluded does not count either way. This is
    the alpha-exclusion census's "lands on a peak no alpha grain claimed", with
    the same nearest-peak rule the census always used.

    They differ for two independent reasons, both real and both measured on sampleH:

    1. HARMONICS. (001), (002), (003)... share a scattering direction exactly, so
       they land on ONE pixel. ``Phase.project`` returns one row per hkl, so all
       of them are counted. The C indexer already removes these -- ``calcOverlap``
       in ``LaueMatchingHeaders.h`` rejects any reflection whose q-hat matches one
       already recorded to 1e-6 -- so the C and Python sides have been counting
       differently. Measured on the Zn hkl list: 7,984 entries reduce to 6,638
       distinct directions, and on the detector a typical orientation gives 60.1
       predicted rows at 50.7 distinct positions, a 1.186x inflation.
    2. UNRESOLVED DISTINCT REFLECTIONS. Two genuinely different reflections can
       fall within ``tol`` of the same detected peak. You still only saw one peak.
       Measured residual ~1.18x, which matches the indexer's own post-dedup
       stacking of 1.15-1.22x.

    Together: 1.186 x 1.18 = 1.40, which is the median inflation measured in
    ``nhit`` on sampleH (94.3% of instances affected).

    BOTH are returned deliberately. Redefining ``nhit`` in place would silently
    invalidate every gate threshold measured against the old statistic; a gate on
    ``n_distinct`` needs its own null, measured with this same function.
    """
    if predicted is None or len(predicted) == 0:
        return 0, 0
    d, idx = tree.query(predicted)
    m = d < tol
    if exclude is not None:
        m &= ~np.asarray(exclude, bool)[np.minimum(idx, len(exclude) - 1)]
    if not m.any():
        return 0, 0
    return int(np.unique(idx[m]).size), int(m.sum())


def poisson_lambda(n_predicted, n_peaks, tol, npx_x, npx_y):
    """Analytic chance expectation of ``nhit``: predicted reflections landing
    within ``tol`` px of one of ``n_peaks`` peaks scattered uniformly over an
    ``npx_x`` x ``npx_y`` detector.

    The scripts wrote ``n_pred * n_peaks * pi * tol**2 / (NPX * NPX)`` with NPX
    the x size, which is wrong for any non-square detector (a 1024 x 2048 panel
    had its area halved, doubling lambda). Both dimensions, always.
    """
    return float(n_predicted) * float(n_peaks) * np.pi * tol * tol / (float(npx_x) * float(npx_y))


def presence_p(tree, npeaks, pr, tol, npx_x, npx_y):
    """Analytic Poisson p that pattern ``pr`` is present (nhit, both dimensions)."""
    from scipy.stats import poisson
    if pr is None or not len(pr) or npeaks < 1:
        return 1.0
    _, h = count_matched_peaks(tree, pr, tol)
    return float(poisson.sf(h - 1, poisson_lambda(len(pr), npeaks, tol, npx_x, npx_y)))


def control_orientations(oms, seed=0):
    """One RANDOMLY ROTATED control per grain: ``R_k @ OM_k`` with R_k uniform.

    The backfills test every master grain in every frame at an analytic Poisson
    gate and used to quote ``n_grains * n_frames * PGATE`` as the expected false
    count. That assumes uniformly scattered peaks; real peak fields cluster, so it
    is optimistic. A control grain has the same phase and reflection structure as
    a real one but an orientation no frame supports, so the number of control
    "presences" at the same gate is a MEASURED false count.
    """
    rng = np.random.default_rng(seed)
    oms = np.asarray(oms, float).reshape(-1, 3, 3)
    q = rng.normal(size=(len(oms), 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                  np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                  np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)
    return R @ oms


def mask_disks(img, xy, radius, value):
    """Set a disk of ``radius`` px around each (x, y) in ``xy`` to ``value``, in place.

    Clips x against ``img.shape[1]`` and y against ``img.shape[0]``. The peel
    driver clipped y with the x size, which on a non-square frame either walked
    off the array or left the bottom rows unmasked.
    """
    H, W = img.shape
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius
    for x, y in np.asarray(xy, float).reshape(-1, 2):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        img[y0:y1, x0:x1][disk[(y0 - yi + radius):(y1 - yi + radius),
                               (x0 - xi + radius):(x1 - xi + radius)]] = value
    return img


# --- which hit statistic a gate uses, and the null measured for it -----------
# A gate and its null must be the SAME statistic: an nhit_distinct count compared
# with an nhit null maximum is optimistic by the ~1.4x stacking factor above, and
# the reverse is pessimistic by the same. Everything that gates on a hit count
# goes through these three functions so the choice is made once and printed.
GATE_STATS = ("nhit", "nhit_distinct")


# --- indexer output layout ----------------------------------------------------
def solution_format(n_cols, where="solution table"):
    """The ``laue_index.records.SolutionFormat`` whose column count is ``n_cols``.

    The indexer writes two layouts (``laue_index.records.SOLUTION_FORMATS``, the
    authoritative map): ``runimage`` (34 columns, orientation matrix at 22..30) and
    ``stream`` (35 columns, ImageNr prepended, matrix at 23..31). Scripts used to
    hard-code the stream columns (``23:32``), which on a RunImage file reads the
    wrong numbers as a matrix without any error. Anything else exits.
    """
    try:
        from laue_index.records import SOLUTION_FORMATS
    except ImportError as exc:                      # pragma: no cover - env problem
        raise SystemExit(f"laue_index is not importable ({exc}); it holds the solution "
                         f"column map (laue_index.records.SOLUTION_FORMATS)")
    for fmt in SOLUTION_FORMATS.values():
        if fmt.n_cols == int(n_cols):
            return fmt
    known = {f.name: f.n_cols for f in SOLUTION_FORMATS.values()}
    raise SystemExit(f"{where}: {n_cols}-column solution table matches no known layout "
                     f"{known}; refusing to guess the column map")


def orientation_block(filt, where="filtered_orientations"):
    """``(n, 9)`` orientation matrices (row-major) from a solution table of either layout."""
    filt = np.atleast_2d(np.asarray(filt))
    fmt = solution_format(filt.shape[1], where)
    return filt[:, fmt.om_start:fmt.om_start + 9]


def spot_columns(fmt):
    """Column map of the ``filtered_spots`` table that accompanies layout ``fmt``.

    ``records`` pins grain / x / y. The h, k, l and intensity columns sit at fixed
    offsets from x in both layouts (the stream layout is the RunImage one with
    ImageNr prepended, every field +1): h,k,l = x-3..x-1, intensity = x+5 -- the
    stream positions 3,4,5 and 11 that spot_energy.py and separate_layers.py
    hard-coded.
    """
    x = fmt.spot_x
    return {"grain": fmt.spot_grain, "h": x - 3, "k": x - 2, "l": x - 1,
            "x": x, "y": fmt.spot_y, "intensity": x + 5}


def image_number(path):
    """``.../image_000123.output.h5`` -> 123, from the FULL digit run after ``image_``.

    The scripts used ``int(h5.split("image_")[1][:5])``, which truncates to five
    digits: image 100000 and above silently mapped to the wrong frame.
    """
    m = re.search(r"image_(\d+)", os.path.basename(str(path)))
    if not m:
        raise ValueError(f"no image_<number> in {path!r}")
    return int(m.group(1))


DEFAULT_OUT_PREFIX = "scan"


def out_prefix():
    """``LAUE_OUT_PREFIX``, default ``scan``: THE basename prefix of every
    ``peel_map/<prefix>_*`` file the chain reads and writes.

    One helper so a producer and its consumer cannot default differently. Before
    2026-09 the census defaulted to ``env``, its null (exclusion_null.py) to
    ``scan`` and exposure_signal_check.py to ``parentbeta``, so with the variable
    unset the census and its null read different files.

    The default only serves a script run by hand: it names output files, it does
    not select anyone's data. run_analysis_chain.sh requires LAUE_OUT_PREFIX and
    passes it through, so a chain run never reaches this default.
    """
    p = os.environ.get("LAUE_OUT_PREFIX", DEFAULT_OUT_PREFIX).strip()
    if not p or "/" in p:
        raise SystemExit(f"LAUE_OUT_PREFIX must be a non-empty file-name prefix, got {p!r}")
    return p


DEFAULT_CLUSTER_TOL = 1.0


def cluster_tol():
    """``LAUE_CLUSTER_TOL`` (degrees, default 1.0): THE orientation-clustering cut.

    Before 2026-09 four scripts clustered at their own literals -- 1.0
    (parentbeta_validate, parentbeta_backfill), 0.7 (beta_map_validate,
    map_validate_cluster) and 1.5 (scan_map) -- so their grain counts described
    different grain definitions. ``LAUE_CLUSTER_TOL=0.7`` / ``1.5`` reproduces
    those scripts' old counts. (batch_peel_driver's 0.7 deg is a per-frame DEDUP
    of re-found orientations between peel passes, not a grain definition, and stays.)
    """
    raw = os.environ.get("LAUE_CLUSTER_TOL", "").strip()
    if not raw:
        return DEFAULT_CLUSTER_TOL
    try:
        v = float(raw)
    except ValueError:
        raise SystemExit(f"LAUE_CLUSTER_TOL must be a positive number of degrees, got {raw!r}")
    if not np.isfinite(v) or v <= 0:
        raise SystemExit(f"LAUE_CLUSTER_TOL must be a positive number of degrees, got {raw!r}")
    return v


def require_labels(labels, where):
    """Exit if any instance is unclustered (label -1), naming the fix.

    ``LAUE_SKIP_CLUSTER=1`` stops parentbeta_validate.py with every label -1. The
    consumers used to crash after a full frame pass (census), crash (anchor_null,
    regrain), report nothing without saying why (empirical_gate) or silently
    re-cluster O(n^2) (parentbeta_reconstruct).
    """
    labels = np.asarray(labels)
    if len(labels) and (labels < 0).any():
        raise SystemExit(
            f"{where}: {int((labels < 0).sum())} of {len(labels)} instances are unclustered "
            f"(label -1, e.g. after LAUE_SKIP_CLUSTER=1). Cluster first and write the "
            f"labels back to the same file: cluster_orientations.py {where} {where} "
            f"<tol_deg> <phase>")
    return labels


def clustered_npz(work):
    """The clustered-instance npz the map scripts read.

    ``$LAUE_CLUSTERED_NPZ`` if set; else the one ``peel_map/<prefix>_*_clustered.npz``
    for ``LAUE_OUT_PREFIX``. The scripts hard-coded ``full_zn_clustered.npz``, i.e.
    one campaign's prefix and phase. Several matches or none exit.
    """
    import glob
    env = os.environ.get("LAUE_CLUSTERED_NPZ", "").strip()
    if env:
        if not os.path.isfile(env):
            raise SystemExit(f"LAUE_CLUSTERED_NPZ={env} does not exist")
        return env
    pat = os.path.join(work, "peel_map", f"{out_prefix()}_*_clustered.npz")
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} files match {pat}"
                         + (f" ({', '.join(os.path.basename(h) for h in hits)})" if hits else "")
                         + ": set LAUE_CLUSTERED_NPZ to the clustered npz to use")
    return hits[0]


def analytic_gate_note(script):
    """Say, at run time, that a per-frame ANALYTIC Poisson gate ignores LAUE_GATE_STAT.

    The analytic lambda (n_predicted * n_peaks * pi * TOL^2 / Npx^2) is the
    expected number of PREDICTED reflections landing on a peak by chance -- a
    model of ``nhit``. There is no matching closed form for ``nhit_distinct``
    (it depends on how predictions cluster on the detector), so these gates stay
    on ``nhit`` whatever ``LAUE_GATE_STAT`` says. Printed rather than silently
    ignored; the empirical-null gates downstream do follow the variable.
    """
    stat = os.environ.get("LAUE_GATE_STAT", "nhit").strip() or "nhit"
    msg = (f"[{script}] per-frame analytic Poisson gate is on nhit (the analytic lambda "
           f"models predicted reflections; no closed form exists for nhit_distinct)")
    if stat != "nhit":
        msg += (f" -- LAUE_GATE_STAT={stat} is NOT applied here; it applies to the "
                f"empirical-null gates downstream. Both counts are saved.")
    print(msg, flush=True)


def gate_statistic():
    """``LAUE_GATE_STAT`` = ``nhit`` (default, reproduces existing results) or
    ``nhit_distinct``. Anything else exits."""
    s = os.environ.get("LAUE_GATE_STAT", "nhit").strip()
    if s not in GATE_STATS:
        raise SystemExit(f"LAUE_GATE_STAT must be one of {GATE_STATS}, got {s!r}")
    return s


def null_json_path(work, prefix):
    """Where null_model.py (per-draw) and search_null.py (search) write a scan's null."""
    return os.path.join(work, "peel_map", f"{prefix}_null.json")


# --- which null: the SEARCH or one DRAW ----------------------------------------
# Invariant 29: a gate safe against one random orientation is not safe against the
# best of the indexer's whole search. null_model.py measures the per-DRAW null
# (random orientations, one at a time); search_null.py measures the SEARCH null
# (the same indexer and validator run on spot-scrambled frames) and writes it as a
# "search_null" block beside the per-draw entries of the same json. The gates use
# the search null when it is present (default since 2026-09) and fall back to the
# per-draw null with a loud warning; LAUE_NULL_KIND=search|draw forces one.
NULL_KINDS = ("search", "draw")
SEARCH_KEY = "search_null"


def null_kind():
    """``LAUE_NULL_KIND``: ``search``, ``draw``, or None (unset: search if measured,
    else draw with a warning). Anything else exits."""
    raw = os.environ.get("LAUE_NULL_KIND", "").strip()
    if not raw:
        return None
    if raw not in NULL_KINDS:
        raise SystemExit(f"LAUE_NULL_KIND must be one of {NULL_KINDS}, got {raw!r}")
    return raw


def select_null(ent, stat, path, phase):
    """``(rec, rec_other, kind)`` from one phase's entry of the null json.

    ``ent`` is ``json["phases"][phase]`` (may be empty). Follows :func:`null_kind`.
    A search block written as a NEGATIVE CONTROL (``keep_positions``: the scramble
    kept every spot where it was) is never a null and is refused.
    """
    other = [s for s in GATE_STATS if s != stat][0]
    want = null_kind()
    search = ent.get(SEARCH_KEY) or {}
    if search.get("keep_positions"):
        raise SystemExit(f"{path}: the {phase} {SEARCH_KEY} block was written with "
                         f"--keep-positions (a negative control, spots NOT scrambled); "
                         f"it can never gate. Re-run search_null.py without it.")
    if want in (None, "search") and search.get(stat) is not None:
        return search[stat], search.get(other), "search"
    if want == "search":
        raise SystemExit(
            f"LAUE_NULL_KIND=search but {path} has no {phase} {SEARCH_KEY}/{stat}: run "
            f"search_null.py on this scan, or set LAUE_NULL_KIND=draw to gate on the "
            f"per-draw null (invariant 29: it is not safe against the search).")
    rec = ent.get(stat)
    if rec is not None and want is None:
        msg = (f"WARNING: no SEARCH null for {phase}/{stat} in {path}; falling back to "
               f"the PER-DRAW null (null_model.py). Invariant 29: a gate safe against "
               f"one random orientation is not safe against the indexer's search. Run "
               f"search_null.py, or set LAUE_NULL_KIND=draw to make this choice explicit.")
        print(msg, file=sys.stderr, flush=True)
        print(msg, flush=True)
    return rec, ent.get(other), "draw"


def load_null(phase, work, prefix, stat=None):
    """The measured null for ``phase``, for statistic ``stat``.

    Sources, in order:

    * ``$LAUE_NULLMAX_<PHASE>`` -- overrides ``max``. It must be the maximum of the
      statistic in force; if the null json is present and the value equals the
      OTHER statistic's maximum instead, this exits rather than gate one
      statistic against the other's null.
    * ``<work>/peel_map/<prefix>_null.json``: its ``search_null`` block (written
      by search_null.py -- the SEARCH null, invariant 29) when present, else the
      per-draw entries written by null_model.py, with a warning. ``LAUE_NULL_KIND``
      (``search`` | ``draw``) forces one; see :func:`select_null`.

    Neither present -> exit. There is deliberately no built-in fallback: a null
    maximum is a property of one scan's peak field and reflection list.

    Returns a dict with at least ``statistic``, ``max``, ``kind`` and ``source``,
    and prints one line saying which null it is.
    """
    stat = stat or gate_statistic()
    other = [s for s in GATE_STATS if s != stat][0]
    path = null_json_path(work, prefix)
    rec = rec_other = None
    kind = None
    if os.path.isfile(path):
        with open(path) as fh:
            ent = json.load(fh).get("phases", {}).get(phase) or {}
        rec, rec_other, kind = select_null(ent, stat, path, phase)
        if rec is not None and rec.get("statistic", stat) != stat:
            raise SystemExit(f"{path}: entry {phase}/{stat} says statistic "
                             f"{rec.get('statistic')!r} -- the file is inconsistent")
    elif null_kind() == "search":
        raise SystemExit(f"LAUE_NULL_KIND=search but {path} does not exist: run "
                         f"search_null.py on this scan")
    var = f"LAUE_NULLMAX_{phase.upper()}"
    env = os.environ.get(var)
    if env is not None:
        try:
            mx = int(env)
        except ValueError:
            raise SystemExit(f"{var}={env!r} is not an integer")
        if rec is not None and mx != int(rec["max"]):
            if rec_other is not None and mx == int(rec_other["max"]):
                raise SystemExit(
                    f"{var}={mx} is the {other} null maximum in {path}, but the gate "
                    f"statistic is {stat} (null max {rec['max']}). A gate and its null "
                    f"must be the same statistic; set LAUE_GATE_STAT={other} or fix {var}.")
            print(f"WARNING: {var}={mx} overrides the measured {stat} null max "
                  f"{rec['max']} in {path}", flush=True)
        out = dict(rec) if rec is not None else {}
        out.update(statistic=stat, max=mx, source=f"${var}", kind=kind or "override")
        _say_null(phase, out)
        return out
    if rec is None:
        raise SystemExit(
            f"no measured {stat} null for phase {phase!r}: set {var} or run "
            f"search_null.py (the search null) and/or null_model.py (the per-draw null) "
            f"on THIS scan (they write {path}). Do not inherit a null from another scan.")
    out = dict(rec)
    out.update(statistic=stat, source=path, kind=kind)
    _say_null(phase, out)
    return out


def _say_null(phase, rec):
    """Every gate prints which null it gates on: kind, statistic, max, source."""
    what = {"search": "SEARCH null (indexer on spot-scrambled frames, search_null.py)",
            "draw": "PER-DRAW null (random orientations, null_model.py)"}.get(
                rec.get("kind"), f"{rec.get('kind')} null")
    print(f"[null] {phase}: {what}; {rec['statistic']} max {rec['max']} "
          f"[{rec['source']}]", flush=True)


def gate_counts(z, stat, where="npz"):
    """The per-instance hit counts for ``stat`` from a validated npz; exits if absent."""
    if stat not in z.files:
        raise SystemExit(
            f"{where} has no {stat!r} column (it was written before {stat} existed). "
            f"Re-run parentbeta_validate.py, or gate on LAUE_GATE_STAT=nhit.")
    return np.asarray(z[stat]).astype(int)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 512
    f = rng.normal(100, 3, size=(N, N))

    def spot(cy, cx, amp, w=2.0):
        yy, xx = np.mgrid[cy-6:cy+7, cx-6:cx+7]
        f[cy-6:cy+7, cx-6:cx+7] += amp*np.exp(-((yy-cy)**2 + (xx-cx)**2)/(2*w*w))

    for (cy, cx) in ((100, 120), (300, 400), (420, 90)):
        spot(cy, cx, 4000)
    # a clipped reflection with an isotropic halo, and a WEAK neighbour on it
    spot(250, 256, 400000, w=3.0)
    yy, xx = np.mgrid[0:N, 0:N]
    f += 9000.0 * np.exp(-np.hypot(yy-250, xx-256)/14.0)     # halo
    np.clip(f, None, 65535, out=f)
    spot(250, 291, 2600, w=2.0)                              # weak neighbour, 35 px away
    f[60:460, 255:258] += 260                                # a vertical bloom

    xs, ys, flags, info = detect_peaks(f, N, return_flags=True)
    print(f"peaks {len(xs)}  saturated-flagged {int(flags['saturated'].sum())}  "
          f"plateau dups removed {info['n_plateau_dropped']}  "
          f"streak px {info['n_streak_px']}  halos {info['n_halos']}")

    def near(px, py, tol=6):
        return [i for i in range(len(xs)) if abs(xs[i]-px) <= tol and abs(ys[i]-py) <= tol]

    assert len(near(256, 250)) == 1, "clipped spot must give exactly one peak"
    assert flags["saturated"][near(256, 250)[0]], "clipped spot must be flagged saturated"
    assert len(near(291, 250)) == 1, (
        "the WEAK neighbour beside the saturated spot was lost -- this is the "
        "Zn-next-to-Cu case and losing it destroys the orientation relationship")
    for cx, cy in ((120, 100), (400, 300), (90, 420)):
        assert len(near(cx, cy)) >= 1, f"lost the real spot at ({cx},{cy})"
    tail = [i for i in range(len(xs)) if 252 <= xs[i] <= 260 and (ys[i] < 230 or ys[i] > 275)]
    assert not tail, f"bloom tail survived at {[(xs[i], ys[i]) for i in tail]}"

    g = rng.normal(100, 3, size=(N, N))
    for cy in range(60, 460, 30):
        spot_y, spot_x = cy, 400
        yy2, xx2 = np.mgrid[spot_y-6:spot_y+7, spot_x-6:spot_x+7]
        g[spot_y-6:spot_y+7, spot_x-6:spot_x+7] += 4000*np.exp(
            -((yy2-spot_y)**2 + (xx2-spot_x)**2)/8.0)
    xg0, _, _ = detect_peaks(g, N, drop_streaks=False, collapse_plateaus=False,
                             remove_halos=False)
    xg1, _, ig = detect_peaks(g)
    # npx is deprecated and ignored: any value (or none) gives the same peaks
    xg2, _, _ = detect_peaks(g, 12345)
    assert np.array_equal(xg1, xg2), "detect_peaks must ignore its deprecated npx argument"
    assert abs(len(xg1) - len(xg0)) <= 1, (
        f"a tall column of REAL unsaturated spots was altered: {len(xg0)} -> {len(xg1)}")

    # count_matched_peaks: a harmonic pair (two predicted rows on one pixel) is
    # ONE distinct peak but two predicted hits; two separate peaks are two of each.
    from scipy.spatial import cKDTree
    tr = cKDTree(np.array([[100.0, 100.0], [300.0, 300.0]]))
    assert count_matched_peaks(tr, np.array([[100.0, 100.0], [100.4, 99.8]]), 8.0) == (1, 2)
    assert count_matched_peaks(tr, np.array([[100.0, 100.0], [300.0, 301.0]]), 8.0) == (2, 2)
    assert count_matched_peaks(tr, np.array([[500.0, 500.0]]), 8.0) == (0, 0)
    assert count_matched_peaks(tr, np.array([[100.0, 100.0], [300.0, 301.0]]), 8.0,
                               exclude=np.array([True, False])) == (1, 1)
    print("frame_peaks selftest OK (clipped spot -> one flagged peak, weak "
          "neighbour kept, bloom removed, real spots untouched; harmonic pair "
          "counts (1 distinct, 2 predicted))")

