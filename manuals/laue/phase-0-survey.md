# Phase 0 — Survey the experiment folder

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 0 — Survey the experiment folder (do this before promising anything)

Goal: a written `SURVEY.md` in the work directory answering *what is actually here*, with numbers
read from the files, never from folder names.

```bash
E=<EXPERIMENT_FOLDER>

# 1. what scan folders exist, and how big is each (ls|wc -l lies past ~40k files: ARG_MAX)
for d in "$E"/*/; do
  n=$(find "$d" -maxdepth 1 -name '*.h5' | wc -l)
  [ "$n" -gt 0 ] && printf '%8d  %s\n' "$n" "$d"
done | sort -rn

# 2. the HDF5 layout of ONE frame -- do not assume /entry1/data/data
python - <<'PY'
import h5py, sys, glob
f = sorted(glob.glob(sys.argv[1] + "/*.h5"))[0] if len(sys.argv) > 1 else None
def show(n, o):
    if isinstance(o, h5py.Dataset): print(f"{n:60s} {str(o.shape):20s} {o.dtype}")
h5py.File(f).visititems(show)
PY
```

Record, per scan folder:

| field | how to get it | why it matters |
|---|---|---|
| frame count | `find … -name '*.h5' \| wc -l` | queue order, runtime estimate |
| image dataset path + shape | `visititems` above | goes in the params / `run_laue.sh` arg |
| **stage coordinates** | `entry1/sample/sampleX`, `sampleZ` (or the local equivalent) | **the real raster** |
| step size and span | `np.unique(np.diff(sorted(X)))` | see the 45° trap below |
| **position-label integrity** | `np.unique(np.diff(X))` in **acquisition order** — look for `0` and `2*step` | a readback race, see below. Present on **both** Zn scans, so assume it until checked |
| exposure | frame header / folder name, then confirm against counts | 0.25 s vs 1 s changes what is detectable |
| still growing? | frame count twice, 120 s apart | never index a scan still being collected |
| peaks on one frame | detect on a background-subtracted frame, SNR>8, **area ≥ 4 px** | density regime (below) |
| hot pixels | pixels saturated in ≥90% of ~60 frames spread over the scan | Zn scan: **36 permanent** hot pixels; only ~8% of saturated pixels were real reflections. Every frame's `max` looked like signal and was not. |
| background, decomposed | median of four detector **corners** (flat) vs a central box (halo) | the flat part is isotropic — but TEST whether it is fluorescence (tracks amount) or diffuse scattering (tracks grain-size/disorder); on Zn/Zn it was scattering, see §Same-phase and invariant on the background |
| spot shape | blob aspect ratio, median **and** p95 | Zn *looked* heavily streaked; measured median AR was 1.6 with only ~10% above 3. The eye reads the p95 tail. |

**The stage-readback race — mislabels, NOT gaps.** If the acquisition-order diffs of the fast axis
contain `0` and `2*step` as adjacent pairs, the readback is being sampled *after* the move begins and
one frame takes the next position's label. On sampleH this hit 180 of 20,301 frames (0.89%); the same
signature is in sampleG, so it is beamline/macro behaviour, not a one-off.

Decide the mechanism from the **images**, because the two readings have opposite fixes. Labels read
`X, X+2, X+2, X+3`. If the motor genuinely skipped, the two frames sharing a label are *co-located*;
if the readback merely led, they are one step apart. Compare their residual-pattern similarity
against a 1-step and a 2-step control taken **from the same four frames**, so row, time and
brightness are shared and both controls hold under either hypothesis. On sampleH: suspect pair
**+0.8742** against a 1 µm control of **+0.8738** — identical to four decimals — with the 2 µm
control at **+0.7488**, which proves the measure resolves one step from two. 0 of 20 pairs were
bit-identical, so not a duplicated readout either.

**Conclusion: the stage visited every position exactly once. No gaps, no duplicates — only wrong
labels.** True position is therefore index-derived, `X = X0 + ((N-1) mod NR)`. Validate that it
reproduces the recorded value on every unaffected frame and differs by exactly one step on precisely
the flagged ones (sampleH: 20,121 exact, 180 off by exactly +1, nothing unexplained). Apply it with
`analysis/fix_positions.py` **before `regrain.py`** — raw labels inject one false hole and one
doubled pixel per affected frame directly into the contiguity test the grain count rests on.

**The 45° trap.** A folder called `10x10um_0p25umStepSize` measured 20.000 µm in X at 0.2500 µm and
14.142 µm in Z at 0.1768 µm — exactly 1/√2, because the sample sits at 45° to the beam. It is a
20×20 µm region in the sample frame, 400 µm² not 100 µm². **Measure the raster from the stage
coordinates for every scan and quote that.** Any area, density, or grains-per-µm² taken from a
folder name is wrong by a factor you will not notice.

**Peak-count check, and why the area filter is not optional.** A bare SNR>8 local-maximum test on a
weak frame returns single-pixel noise spikes as "peaks": on one Si calibration frame it gave 159,
of which 47 were extended reflections. Requiring a connected area of ≥4 px above 4σ is what makes
the count mean something. `pattern_complexity_figure.py` in the report scripts does exactly this
for both frames it compares.

**The area filter is necessary but NOT sufficient — a very bright spot breaks detection three
different ways.** All three were found on the 34-ID-E Perkin Elmer panel and all three are handled
in `analysis/frame_peaks.py`, the shared detector that `null_model.py` and `parentbeta_validate.py`
both import (so a count and the null it is gated against cannot drift apart).

1. **FLAT-TOP PLATEAUS — the big one, and the least obvious.** A clipped reflection has a *flat*
   top, so every pixel on it equals the local maximum and `sub == maximum_filter(sub)` flags all of
   them: one reflection is reported as dozens of peaks. One 117 px saturated Cu spot produced **58
   detections at identical intensity**; collapsing plateaus cut whole-frame counts by **35–45%**
   (197→111 on sampleD, 189→62 on the bare-Cu reference, 228→145 on a sampleA scan). This inflates the peak
   list, inflates any null built from it, and buries weak neighbours. One connected saturated region
   is ONE reflection — and position it from the **unsaturated shoulders** (the 20–90%-of-clip band),
   which is also *less* biased than the plateau centroid whenever the spot is asymmetric.
2. **ISOTROPIC HALO.** The wing of an intense reflection, measured in ADU above background at
   15/25/40/60/100 px: 1480/508/74/34/14 along the column and 2142/363/79/20/4 along the row, with
   frame noise σ = 50. Vertical and horizontal decay *together*, so it is a halo, not a streak. The
   standing background (a 25 px median on a 4× downsample, ~100 px scale) cannot follow something
   decaying over 40 px, so it leaks into the residual, raises the local bar and manufactures maxima.
   Subtract an azimuthal radial profile per bright spot. This is what lets a weak neighbour survive:
   on that frame the nearer neighbour (I = 579) sat on ~500 ADU of halo.
3. **VERTICAL BLOOMING.** Charge overflow running a bright column hundreds of rows from its source.
   Real but **rarer than expected** — present on the bare-Cu reference frame, absent from every sampleD
   frame sampled. Remove it by **shape**: a morphological opening that erases structures thin in one
   axis and long in the other cannot touch a compact reflection wherever it sits.

**Never delete a detection; flag it.** Position stays valid for a clipped peak even when intensity
does not, so indexing can use it and intensity analysis can skip it. Two filters that *did* delete
were both wrong, and only measurement revealed it: one keyed on tall-and-narrow columns with no
saturation test and removed ~40 real reflections per frame from ordinary crowded columns; another
fabricated ~34 reflections per frame by treating the panel's permanently hot pixels as clipped
reflections. **A clipped reflection has unsaturated shoulders; a hot pixel jumps straight from full
scale to background** — that, plus a minimum area, is what separates them.

This matters beyond bookkeeping. On deposit-on-substrate(111) the substrate reflection saturates and the
reflections sitting a few tens of pixels from it are exactly the ones carrying the orientation
relationship, so a filter that clears the neighbourhood of a bright spot destroys the measurement it
was meant to protect. Verify explicitly that near-neighbours survive: on sampleD they sit at r = 22.0,
29.8 and 53.3 px and must come through unchanged.

The indexer is *not* affected by any of this — it runs its own percentile + `MinArea` + watershed
detection. Two separate code paths; check which detector produced a peak count before comparing.

**Density regime** — sets expectations and the peel depth:

| peaks/frame | regime | consequence |
|---|---|---|
| ≲ 50 | single crystal / few grains | classical indexing works; this pipeline is overkill but fine |
| 100–500 | many grains | the design case; nulls matter |
| ≳ 900 | dense/streaky | expect the iterative peel (`batch_peel_driver.py`) and several s/frame |

---
