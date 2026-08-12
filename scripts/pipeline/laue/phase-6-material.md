# Phase 6 — Material configuration

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 6 — Material configuration (the port is done; verify it)

`analysis/laue_material.py` is the single source of truth. It parses the **indexing parameter
file** — the same file the indexer consumed — for lattice, space group, symmetry, detector
geometry, energy window and the reflection-list path, so the analysis physically cannot be run
against a different material's reflections than the indexing was.

```python
from laue_material import Phase
ph = Phase.load("zn")            # -> $LAUE_PARAMS_ZN, else $LAUE_PARAMS
ph.B, ph.hkls, ph.sym_ops        # reciprocal matrix, reflections, proper rotations
ph.project(OM, with_energy=True) # (n,3): px, py, E_keV
ph.misorientation(A, Bs)         # DEGREES, symmetry-reduced
```

There is **no built-in default material**: failing to resolve `LAUE_PARAMS_<PHASE>` raises rather
than falling back, because silently analysing one material with another's reflection list is the
exact failure this module exists to prevent.

**Orientation maths comes from `midas_stress`**, the canonical MIDAS implementation (a
byte-for-byte port of the C `GetMisorientation.h`): `misorientation_om_batch` for misorientation
(it returns **radians** — `laue_material` converts to degrees) and `make_symmetries(sg)` for the
operator set. Install it with the rest of the deps: `pip install midas-stress midas-hkls`
(both are **torch-free** — `pip install midas-stress` needs no torch as of **0.8.1**; torch is an
opt-in `midas-stress[torch]` extra). `laue_material` keeps a local operator fallback only for
environments where midas_stress is not installed at all. On midas_stress **>= 0.8.1** the operators
are exact and agree with the fallback to machine precision; older (<= 0.8.0) installs stored 5-decimal
symmetry quaternions and agreed to **0.032 deg** worst case (still **0 pairs** across the 1.0 deg
clustering cut, measured on 30,000 real Zn pairs), which is why the selftest tolerance is 1e-4 not
machine epsilon.

Setting up a new material:

```bash
python ../GenerateHKLs.py -resultFileName $WORK/params/valid_hkls_<M>.csv \
   -sym <F|I|C|A|R|P|B> -sgnum <SG> -latticeParameter a b c al be ga \
   -RArray ... -PArray ... -NumPxX 2048 -NumPxY 2048 -dx 200e-6 -dy 200e-6 -Ehi <keV>
# then let the daemon build the forward cache once (~10 min), and
export LAUE_PHASES=<m>  LAUE_PARAMS_<M>=$WORK/params/params_<M>.txt
```

Still per-material and *not* automatic:

- **The orientation relationship**, if one applies: `burgers_Cv()` in
  `parentbeta_reconstruct.py:49`, returning a `(12,3,3)` variant set. Everything downstream is
  generic in that array. K-S gives `(24,3,3)`; the **accept threshold must be re-derived** —
  "11 of 12" is an empirical cut for Burgers, not a law.
- The 100M-orientation database is **material-independent** (an SO(3) grid) — symlink it, never
  copy 7.2 GB. Only the hkl list and forward cache are per-phase.

### Same-phase problems (substrate/deposit, weld/parent, deposit on like substrate)

If the two things you want to separate are the *same phase* — e.g. Zn electroplated on Zn — then
no phase fraction, no exclusion census and no parent reconstruction applies (chain steps 4–8 are
out). What is left:

- **Laue footprint (orientation coherence)** — the size of a contiguous single-orientation cluster;
  the cleanest same-phase discriminator (Zn/Zn: substrate = large coherent grains ~213 positions,
  deposit = fine ~36). Footprint measures crystallographic coherence, not necessarily physical grain
  size — a terraced/twinned crystal can over-segment — so if you have an SEM, check whether "fine
  footprint" means genuinely fine grains or a fragmented large crystal. (On Zn/Zn the deposit read as
  fine, and the off-region SEM could not settle which morphology it was; the assignment came from the
  co-registered maps + IPF, not the SEM.)
- **the flat detector background** — do NOT assume it is fluorescence tracking "how much material".
  Split it from the forward-peaked halo by corners-vs-centre, then TEST what it tracks: if it scales
  with grain size / Laue footprint it is **diffuse scattering** from disordered material (fluorescence
  from a thick sample is escape-depth *saturated* and therefore blind to grain size). On Zn/Zn the flat
  floor tracked footprint (corr −0.33, spatial p=0) and is scattering — HIGHER over the rough,
  fragmented deposit — not a path-length/fluorescence gauge. Decide which mechanism by the grain-size
  dependence, not by assumption.
- **per-spot energy** — `Phase.project(OM, with_energy=True)`, or from the stored per-spot `hkl`
  (spots-table cols 3,4,5) and its grain's orientation. In reflection geometry a reflection from under
  an overlayer round-trips through it and hardens; but this is a SEPARATE effect from the background
  and was too small to detect on Zn/Zn (thin deposit, wire parked).

Use them together: two independent observables that must move in the predicted directions is a far
stronger claim than either alone, and each needs its own null (§ below).

**Do not let a threshold define the groups you then test.** It is tempting to *label* each
orientation substrate/deposit by a footprint-or-energy cutoff and then compare the two labels'
energies or footprints — but that contrast is **circular**: the label was built from the very
quantity being contrasted (on Zn/Zn the circular split reads a spectacular −0.64 keV / −0.73, an
artefact of the cutoff, not a measurement). The honest test asks whether two *independently
measured* signatures **agree**: `corr(log footprint, median energy)`, footprint from the clustering
geometry and energy from the spot wavelengths, nothing shared. On Zn/Zn that is −0.10 (wrong sign,
r²~1%): the layers are real on the *map* but not separable *per position*. Keep any threshold-defined
split only as map colouring, and store the independent-test statistic (and a warning) next to it —
see `separate_layers.py`.

**State the aggregation of every map correlation, and keep it fixed.** "footprint vs pedestal" is
two different numbers: *per grain* (one point per cluster, −0.12 on Zn/Zn) and *per scan position*
(one point per occupied pixel, −0.33). Both are legitimate and point the same way, but they are not
interchangeable — the per-position measure is the one comparable to the optical/ground-truth map. Pick
one aggregation for the narrative claim and use it everywhere, labelled. (The *sign* of the
optical-comparison also depends on the registration flip — see invariant #11; anchor direction on
ground truth, not on which flip maximises |corr|.)

### Deposit on a SINGLE-CRYSTAL substrate — peel before you ask about epitaxy

The most natural question about a deposit on a single-crystal substrate ("is there an orientation
relationship?") is the one this pipeline answers *wrongly by construction*, and it answers it with
a large, clean-looking effect. **Validation scores PREDICTED reflections**, not distinct observed
peaks. So when a candidate orientation of the deposit can be rotated to overlay the substrate, it
collects the substrate's peaks as evidence for itself.

Two mechanisms, both measured on bt_34ide_jul26 sampleD (Zn electrodeposited on the fcc substrate):

- **Harmonic stacking.** A Laue spot's position depends only on the *direction* of **g**. Putting
  Zn's c\* on Cu[111] — i.e. exactly the epitaxial relationship under test — sends all seven Zn
  (000ℓ) harmonics inside the 5–30 keV window onto the **single** the fcc substrate pixel (separation
  0.00 px). One observed substrate peak then scores ~7 for the Zn candidate.
- **Generic vector coincidence.** 52.8% of Zn *hki*0 vectors sit within 0.1° of some Cu vector, so
  the overlay is rewarded far beyond the harmonics.

The result: the deposit orientation that best *overlays* the substrate wins on score with no
deposit diffraction present at all. On sampleD this produced "26.4% of 53 grains within 5° of Cu⟨111⟩
at 18.3× the null", which met a pre-registered bar and was **retracted**: removing Cu-explained
peaks dropped the epitaxial grain's pass rate 72%→20%, and the largest apparent Zn grain (56% of
all validated Zn) went 94%→0%. Honest ratio 2.3×.

**Also**: a (111)-polished substrate is parallel to (111) *by construction*, so out-of-plane
alignment of the deposit with the surface normal carries **no lattice information**. Only the
in-plane relationship (e.g. Zn⟨11-20⟩‖Cu⟨110⟩) tests epitaxy.

**Procedure — peel the substrate, then re-index:**

1. Measure the substrate orientation from its own gated instances (dominant cluster).
2. **Measure how far its predicted spots wander across the raster** — do not assume a mask radius.
   Project the cluster's orientations, nearest-neighbour match to a reference projection, and take
   the p99 of the displacement. It differs wildly between scans that look alike: sampleD 6.2 px median
   (p99 12.4), sampleA scan 1 **19.0 px median (p99 89.1)**. A fixed 15 px disc is right for the first
   and leaks the substrate straight back in for the second — recreating the artefact while looking
   like it removed it. (Ignore the *max*: when a reflection leaves the energy window the
   nearest-neighbour match jumps to a different spot.)
3. Build the mask as the **union** of the discs predicted by the cluster's own orientations, after
   dropping the most deviant ~5% so a few bad fits cannot inflate it. Spots that wander get an
   elongated mask, stable spots stay tight, and the geometry sets the shape. sampleA scan 2: 1.14% of
   the detector.
4. **Fill masked pixels with the local background, never zero** — a field of zeros skews the
   indexer's percentile threshold and manufactures false edges.
5. **Rebuild the background from the peeled frames.** Reusing the unpeeled background leaves the
   substrate in the background model and partly undoes the peel.
6. Re-index the deposit, and report the mask fraction next to the result.

The peel also removes any genuine deposit reflection that coincides with a substrate one. That is
unavoidable and it is the point: the test becomes **stricter**, which is the direction it must err.
A relationship that survives is real; one that vanishes is consistent with either artefact or
over-masking, and the mask fraction is what separates those. The real fix is to score distinct
**observed** peaks.
