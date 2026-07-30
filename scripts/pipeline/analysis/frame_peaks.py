"""Shared SNR peak detection for one raw frame, with a blooming-streak filter.

`null_model.py` and `parentbeta_validate.py` previously each carried their own copy
of the same four lines:

    med = np.median(raw); mad = 1.4826*np.median(np.abs(raw-med))
    bg4 = ndi.median_filter(raw[::4,::4], 25); bg = np.kron(bg4, np.ones((4,4)))
    sub = raw - bg
    pk = (sub == ndi.maximum_filter(sub, 9)) & (sub > 8*mad)

They must stay identical -- the null is what gates the validator's output, so any
drift between the two silently compares a count against a null measured with a
different detector. Hence one function, imported by both.

WHY THE STREAK FILTER
---------------------
The Perkin Elmer panel blooms VERTICALLY out of a saturated reflection: a bright
column running many hundreds of rows away from the spot that caused it. The bare
local-maximum test above has no notion of shape, so it happily returns a stack of
"peaks" spaced ~9 px apart all the way down the bloom. On the 34-ID-E bare-Cu
reference frame this turned two saturated reflections into dozens of detections
and inflated the reported peak count of that frame by roughly 2x.

Consequences if left in:
  * the measured random-orientation null is inflated (more peaks -> more chance
    hits), which biases the gate conservative rather than permissive, so it
    corrupts the number without making results look better; and
  * any per-frame peak count quoted from a frame containing a saturated
    reflection is wrong.

It does NOT reach the indexer: the indexer runs its own percentile + MinArea +
watershed detection, which finds the compact blob and does not walk the tail.
The two detections are separate code paths and only this one had the defect.

THE FILTER
----------
A bloom is the one feature on these frames that is tall and narrow: it occupies a
handful of columns and hundreds of rows, whereas a reflection (even a streaky one,
median aspect ratio ~1.8, p95 ~4.9 on these scans) spans a few px in both. So:

  1. count lit pixels per column at a LOW threshold, to trace the faint tail;
  2. flag columns whose lit-count is a large outlier AND exceeds MIN_ROWS;
  3. group contiguous flagged columns into bands;
  4. REQUIRE the band to contain a saturated blob, i.e. confirm the bloom against
     its cause;
  5. inside each confirmed band keep only the single strongest peak -- that is the
     saturated reflection the bloom came from, which is real and often the most
     intense spot on the frame -- and drop the rest of the stack.

Step 4 is not optional, and leaving it out is worse than the original bug. A tall
lit-column count alone also fires on an ordinary column that happens to cross
several real reflections down the panel. Measured on these scans, steps 1-3 by
themselves flagged 4 bands on one sampleA frame (one of them 48 columns wide) and 5 on
an sampleB frame, none of which contained a saturated pixel: keeping one peak per band
there would have destroyed ~40 real reflections per frame. Blooming is charge
overflow, so a genuine bloom always has a saturated source; the confirmed bands on
the bare-Cu frame carry 71 and 57 saturated pixels while every false positive
carried 0 or 1.

The threshold is a COUNT of saturated pixels, not merely one, because these panels
carry 34 permanently hot pixels that sit at full scale in every frame; a single
saturated pixel in a band proves nothing.

Step 5 is what preserves the physics. Dropping the whole band would throw away a
real and usually very strong reflection; on the bare-Cu frame both bloom sources
were genuine indexed reflections at I ~ 59,000-60,000 with sensible hkl.

On a frame with no saturated reflection nothing is confirmed and this is a no-op,
so the filter cannot change results on clean frames.
"""
import numpy as np
from scipy import ndimage as ndi

# Tracing threshold for the bloom tail, in units of the frame's own MAD sigma.
# Lower than the 8-sigma detection bar on purpose: the tail is faint, and it is
# the tail's EXTENT that identifies it, not its height.
STREAK_TRACE_SIGMA = 4.0
# A band must span at least this many lit rows to count as a bloom. The panel is
# 2048 rows; a genuine reflection occupies a few tens of rows at the very most.
STREAK_MIN_ROWS = 60
# Outlier bar on the per-column lit count, in robust sigmas above the median.
STREAK_SIGMA = 6.0
# Peaks within this many columns of a flagged band are treated as in the band.
STREAK_PAD = 3
# Raw ADU at or above which a pixel counts as saturated (16-bit panel, full scale
# 65535). Blooming is charge overflow, so a real bloom always has a saturated source.
SAT_LEVEL = 65000
# A band must contain at least this many saturated pixels to be confirmed a bloom.
# More than one, because these panels carry ~34 permanently hot pixels at full
# scale in every frame and a lone saturated pixel in a band proves nothing.
STREAK_MIN_SAT = 10
# Widest a bloom band may be. Blooming follows a few readout columns; anything
# broad is a crowded region, not a bloom.
STREAK_MAX_WIDTH = 32


def streak_columns(sub, mad,
                   trace_sigma=STREAK_TRACE_SIGMA,
                   min_rows=STREAK_MIN_ROWS,
                   nsigma=STREAK_SIGMA):
    """Columns occupied by a vertical blooming streak.

    Returns a sorted int array of column indices (empty if the frame is clean).
    """
    lit = (sub > trace_sigma * mad).sum(axis=0)
    if not lit.size:
        return np.empty(0, dtype=int)
    med = np.median(lit)
    sd = 1.4826 * np.median(np.abs(lit - med))
    if sd <= 0:                      # near-empty frame; fall back to plain std
        sd = float(np.std(lit)) or 1.0
    bar = max(med + nsigma * sd, min_rows)
    return np.flatnonzero(lit > bar)


def _bands(cols, pad=STREAK_PAD):
    """Group contiguous (within 2*pad) columns into [lo, hi] bands."""
    if not len(cols):
        return []
    out, lo, prev = [], int(cols[0]), int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c - prev > 2 * pad:
            out.append((lo - pad, prev + pad)); lo = c
        prev = c
    out.append((lo - pad, prev + pad))
    return out


def detect_peaks(raw, npx, snr=8.0, maxfilt=9, drop_streaks=True,
                 sat_level=SAT_LEVEL, min_sat=STREAK_MIN_SAT):
    """SNR peak detection on one raw frame.

    Parameters
    ----------
    raw : (n, n) float array   the raw frame
    npx : int                  detector width, to trim the upsampled background
    snr : float                detection bar in MAD sigmas
    maxfilt : int              local-maximum window
    drop_streaks : bool        apply the blooming-streak filter
    sat_level : float          raw ADU counting as saturated
    min_sat : int              saturated pixels a band needs to be confirmed a bloom

    Returns
    -------
    xs, ys : int arrays        peak pixel coordinates
    info : dict                'mad', 'sub', 'n_raw', 'n_streak_dropped',
                               'streak_bands' (confirmed), 'streak_candidates'
    """
    med = np.median(raw)
    mad = 1.4826 * np.median(np.abs(raw - med))
    bg4 = ndi.median_filter(raw[::4, ::4], 25)
    # trim to the frame's own shape; identical to the incumbent's [:npx,:npx] on a
    # square panel, but does not silently mis-broadcast on a non-square one
    bg = np.kron(bg4, np.ones((4, 4)))[:raw.shape[0], :raw.shape[1]]
    sub = raw - bg

    pk = (sub == ndi.maximum_filter(sub, maxfilt)) & (sub > snr * mad)
    ys, xs = np.where(pk)
    info = {"mad": mad, "sub": sub, "n_raw": len(xs),
            "n_streak_dropped": 0, "streak_bands": [], "streak_candidates": []}
    if not drop_streaks or not len(xs):
        return xs, ys, info

    cand = _bands(streak_columns(sub, mad))
    # Confirm each candidate against the thing that causes blooming. Without this
    # the filter also fires on ordinary columns crossing several real reflections.
    bands = []
    for lo, hi in cand:
        if hi - lo + 1 > STREAK_MAX_WIDTH:
            continue
        nsat = int((raw[:, max(lo, 0):hi + 1] >= sat_level).sum())
        if nsat >= min_sat:
            bands.append((lo, hi))
    info["streak_bands"] = bands
    info["streak_candidates"] = cand
    if not bands:
        return xs, ys, info

    keep = np.ones(len(xs), dtype=bool)
    inten = sub[ys, xs]
    for lo, hi in bands:
        inband = np.flatnonzero((xs >= lo) & (xs <= hi))
        if len(inband) <= 1:
            continue                      # nothing stacked; leave it alone
        # keep the bloom SOURCE (strongest peak in the band), drop the tail
        keep[inband] = False
        keep[inband[np.argmax(inten[inband])]] = True
    info["n_streak_dropped"] = int((~keep).sum())
    return xs[keep], ys[keep], info


if __name__ == "__main__":
    # Selftest: a synthetic frame with three compact spots and one bloom column.
    rng = np.random.default_rng(0)
    N = 512
    f = rng.normal(100, 3, size=(N, N))
    for (cy, cx) in ((100, 120), (300, 400), (420, 90)):
        yy, xx = np.mgrid[cy-4:cy+5, cx-4:cx+5]
        f[cy-4:cy+5, cx-4:cx+5] += 4000*np.exp(-((yy-cy)**2 + (xx-cx)**2)/4.0)
    # Bloom runs along the columns of the saturated pixels, so the source is no
    # wider than the streak it produces -- keep the fixture that way.
    f[60:460, 255:258] += 220           # bloom column
    f[250:262, 254:259] = 65535         # its SATURATED source, same columns

    x0, y0, i0 = detect_peaks(f, N, drop_streaks=False)
    x1, y1, i1 = detect_peaks(f, N, drop_streaks=True)
    print(f"unfiltered peaks : {len(x0)}")
    print(f"filtered peaks   : {len(x1)}  (dropped {i1['n_streak_dropped']}, "
          f"bands {i1['streak_bands']})")
    assert len(x1) < len(x0), "filter removed nothing"
    lo, hi = i1["streak_bands"][0]
    src = [(int(a), int(b)) for a, b in zip(x1, y1) if lo <= a <= hi]
    assert len(src) == 1, f"bloom source not preserved exactly once in band: {src}"
    assert 246 <= src[0][1] <= 266, f"kept a tail peak instead of the source: {src}"
    for cx in (120, 400, 90):
        assert any(abs(int(a) - cx) <= 2 for a in x1), f"lost the real spot at x={cx}"
    xc, yc, ic = detect_peaks(f[:240, :240], 240, drop_streaks=True)
    assert ic["n_streak_dropped"] == 0, "filter is not a no-op on a clean frame"

    # Negative control: a tall column of real, UNSATURATED spots must survive.
    g = rng.normal(100, 3, size=(N, N))
    for cy in range(60, 460, 30):
        yy, xx = np.mgrid[cy-4:cy+5, 396:405]
        g[cy-4:cy+5, 396:405] += 4000*np.exp(-((yy-cy)**2 + (xx-400)**2)/4.0)
    xg0, _, _ = detect_peaks(g, N, drop_streaks=False)
    xg1, _, ig = detect_peaks(g, N, drop_streaks=True)
    assert ig["n_streak_dropped"] == 0, (
        f"filter ate {ig['n_streak_dropped']} real unsaturated spots in a tall column; "
        f"candidates={ig['streak_candidates']} confirmed={ig['streak_bands']}")
    assert len(xg1) == len(xg0)
    print("frame_peaks selftest OK (bloom tail removed, source and real spots kept, "
          "no-op on a clean frame)")
