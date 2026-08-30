"""Shared peak detection for one raw frame: saturation, halo and streak handling.

`null_model.py` and `parentbeta_validate.py` both import `detect_peaks` from here.
They must detect identically -- the null is what gates the validator's output, so
any drift between them silently compares a count against a null measured with a
different detector.

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


def detect_peaks(raw, npx, snr=8.0, maxfilt=9,
                 drop_streaks=True, collapse_plateaus=True, remove_halos=True,
                 sat_level=SAT_LEVEL, return_flags=False):
    """Detect reflections on one raw frame.

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
    xg1, _, ig = detect_peaks(g, N)
    assert abs(len(xg1) - len(xg0)) <= 1, (
        f"a tall column of REAL unsaturated spots was altered: {len(xg0)} -> {len(xg1)}")
    print("frame_peaks selftest OK (clipped spot -> one flagged peak, weak "
          "neighbour kept, bloom removed, real spots untouched)")
