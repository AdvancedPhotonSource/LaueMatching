# IndexFile Text Output

LaueMatching writes a Tischler-style `$filetype IndexFile` text next to
every per-image HDF5 result. It is **on by default**; pass
`--no-indexfile` to `RunImage.py`, `laue_postprocess.py`, or
`laue_orchestrator.py` to disable.

## Output location

| Entry point                   | Default output path                                    |
|-------------------------------|--------------------------------------------------------|
| `RunImage.py process`         | `<image>.output.h5` → `<image>.output.indexing.txt`    |
| `laue_postprocess.py`         | `<output-dir>/image_XXXXX.output.h5` → `<output-dir>/image_XXXXX.indexing.txt` |
| `laue_orchestrator.py`        | As postprocess, inside the timestamped run dir         |

Override the directory with `--indexfile-out DIR`.

## Schema

Opens with:

```
$filetype	IndexFile
// Found N patterns, indexed M out of K spots in HH:MM:SS = (X.XX sec)
// ------------------------------------------------------------
$peakFile         '<input H5 basename>'
$keVmaxCalc       <Ehi from config>
$angleTolerance   <MaxAngle>
$keVmaxTest       <Ehi>
$hklPrefer        '{0,0,2}'
$cone             72
$NpatternsFound   N
$Nindexed         M
$NiData           K
$executionTime    X.XX
// ------------------------------------------------------------
$structureDesc    Ni                       # (optional; from config StructureDesc)
$SpaceGroup       225
$latticeParameters { a, b, c, α, β, γ }
$lengthUnit       nm
$AtomDesctiption1 {...}                    # (optional; preserves original typo)
$xtalFileName     ...xml                   # (optional; from config XtalFile)
$inputImage       ...
$xdim 2048  $ydim 2048  $xDimDet 2048  $yDimDet 2048
$startx 0  $endx 2047  $groupx 1
$starty 0  $endy 2047  $groupy 1
<beamline metadata — only if supplied>
<peak-search params — only if supplied>
$programName      'LaueMatching'
$geoFileName      ...                      # (optional)
// ------------------------------------------------------------
```

Per indexed grain:

```
$patternN
$EulerAnglesN { φ1, Φ, φ2 }                // Bunge ZXZ, degrees
$goodnessN    <quality>
$rms_errorN   <RMS of per-spot err, deg>
$rotation_matrixN  {{row0}{row1}{row2}}
//   rotation matrix   <col 0 components>
//   column vectors    <col 1 components>
//                     <col 2 components>
$recip_latticeN    {{row0}{row1}{row2}}
//   reciprocal matrix <col 0 components>
//   column vectors    <col 1 components>
//                     <col 2 components>
//
$arrayN  <N>  <N>     G^             (hkl)   intens     E(keV)    err(deg)   PkIndex
    [  0]  (gx gy gz)  (h k l)   int,  E,   err   pkindex
    ...
```

## Field derivations

| Field                   | Source                                          |
|-------------------------|--------------------------------------------------|
| `$EulerAngles`          | `scripts/laue_indexfile.orient_matrix_to_euler_deg` — a faithful port of the C `OrientMat2Euler` in `src/LaueMatchingHeaders.h`, with the 1st/3rd angles wrapped into (-180°, 180°]. |
| `$goodness`             | `NMatches * sqrt(Intensity)` — col 5 of `solutions.txt` / filtered_orientations. |
| `$rms_error`            | RMS of per-spot angular errors (see below). Falls back to col-33 `misOrientationPostRefinement` when no spots are present. |
| `$rotation_matrix`      | Cols 23–31 of `solutions.txt` (OrientMatrix). |
| `$recip_lattice`        | Cols 8–16 of `solutions.txt` (Recip = OrientFit @ RecipFit). |
| per-spot `G^`           | Cols 8–10 of `spots.txt` (Qhat, fit-predicted). |
| per-spot `(hkl)`        | Cols 3–5 of `spots.txt`. |
| per-spot `intens`       | Col 11 of `spots.txt` (observed pixel value). |
| per-spot `E(keV)`       | **Computed**: `hc * |Q| / (4π · -Q̂_z)` with `Q = recip_lattice @ hkl`. Mirrors the C formula at `LaueMatchingHeaders.h:609`. |
| per-spot `err(deg)`     | **Computed**: angle between fit-predicted G-hat (spots.txt) and observed G-hat obtained by inverting the observed pixel `(X, Y)` through the detector geometry. |
| per-spot `PkIndex`      | Nearest-neighbour lookup against `/entry/data/component_centers` in the HDF5 output, with a small distance threshold. `-1` if no center is within 5 px. |

## Optional config keys for richer headers

Add to `params.txt`:

```
StructureDesc      Ni
XtalFile           /path/to/Ni.xml
AtomDesctiption    Ni001  0 0 0 1        # preserves original spelling
```

Beamline metadata (exposure, sample XYZ, scan number, detector ID, …)
can be merged in via the Python API (`build_from_h5(..., beamline_meta={...})`)
— no CLI flag yet.
