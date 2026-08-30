# Phase 5 — Report

> Part of the **Laue doc set**. The spine — invariants, done-means and the phase
> order — is [`README.md`](README.md).

---

## Phase 5 — Report

Two deliverables, from the same numbers:

- **PDF** — LaTeX, one per scan or one per campaign. Every number reported with its null. State the
  measured raster (not the folder name), the null maxima, the grain definition used, and what was
  *not* measurable (e.g. no depth resolution → no per-grain depth).
- **HTML artifact** — the shareable version. Keep slides to one screen each; embed figures as
  data-URI JPEGs (a strict CSP blocks every external request). To update an existing artifact, pass
  its URL back — do not mint a new one for the same deliverable.

### Artifact structure: ONE overview, linking OUT to one page per sample

A single artifact that grows with the campaign becomes a *chronology of the analysis* rather than a
description of the samples, and the experimenter cannot find their own sample in it. On
bt_34ide_jul26 the single page reached 2.3 MB and had to be split under protest from the reader.
**Split it at the second dataset, not when it hurts.**

```
OVERVIEW  (the URL you share first; keep this one stable and re-publish in place)
  ├── what the campaign is, one table of samples with links
  ├── findings that SPAN samples (method problems, cross-sample comparisons)
  └── what is still open
        ├──> per-sample page: sampleA scan 1        ├──> per-sample page: sampleB (deposit + bare)
        ├──> per-sample page: sampleA scan 2        └──> per-sample page: sampleD
```

Rules that make it work:

- **One page per SCAN, not per specimen**, when scans differ in raster or condition. Two scans of
  one specimen get two pages and are compared *in the overview*, never silently averaged.
- **Combine only what the reader treats as one question** — deposit and its bare substrate belong
  on one page because they are read against each other.
- **Each page is self-contained**: it repeats the method section and the caveats. Readers arrive
  from a link, not from the overview, and a page that assumes the overview was read will be
  misread.
- **Keep the per-sample pages descriptive**: what orientations are there and where. Cross-cutting
  interpretation (relationships between phases, comparisons between samples) lives in the overview.
  When the experimenter says "just the orientations, we don't need the relationship" — that is
  exactly this split, and it is the right instinct.
- **Every per-sample page carries the same three diagnostics**, because they travel: the tolerance
  sweep, the effective (Kish) n beside the nominal grain count, and the count of objects spanning
  more than half the map. A reader comparing two samples needs to know that one has effective n=51
  and the other n=7.5.
- **Link back** from each page to the overview; put the sample links in a grid near the top of the
  overview, not buried at the bottom.
- **Export the numbers next to the pictures** — a `<key>_grains.csv` per sample with grain id,
  position, size and the full orientation matrix. "Orientations extracted" usually means the reader
  wants the table, not only the map.
- Generate all pages from ONE builder with a shared stylesheet (`build_reports.py` pattern:
  `dataset_page(key, ...)` reading a per-dataset `_stats.json`), so a fix to the method text or the
  palette lands everywhere at once.
- Publish the per-sample pages FIRST, collect their URLs, then build the overview with the links in
  it. The overview is re-published in place afterwards whenever a sample page changes.
- **Every spatial map is drawn to TRUE SCALE — `aspect="equal"`, never `aspect="auto"`.** A
  stretched map misrepresents grain shape and elongation, which is exactly what the reader is
  looking at. On bt_34ide_jul26 a 100 × 150 µm scan was rendered nearly square and a beamline scientist
  caught it before we did. Size the figure from the map's own aspect ratio so equal-scale panels
  do not leave a band of whitespace; where two panels share coordinates, give them the same
  extent and the same aspect.

Reusable figure generators live in the report scripts: `validated_figures.py` (report plates),
`catalog_figures.py`, `scan_map.py` (quick-look map), and the survey figure that puts a
single-crystal frame beside a many-grain frame **under identical detection and scaling** — the
honest way to show why a dataset is hard.

Report only validated, recurring quantities. Cluster catalogs (e.g. "28,063 orientation clusters")
are pipeline intermediates; on a 400 µm² region half of them were single-position. Quote the
doubly-supported subset: beyond the measured null **and** recurring across positions.

Deliverable layout that worked (13 scans, ~293k result files):

```
DELIVERABLE_<campaign>/
  MANIFEST.md          contents, key numbers, method caveats
  indexing_output/     per-frame output.h5 (hardlinked), indexing.txt, frame_mapping.json, provenance.json
  per_scan_analysis/   per scan: peel_map/*.npz, figures/, logs
  cross_series/        metrics.json + summary
  reports/             PDFs, HTML artifacts, LaTeX sources
  scripts/             the exact analysis scripts used
```

Hardlink the per-frame outputs rather than copying (same inode, zero extra disk, becomes an
independent copy when tarred) — but only within one filesystem; check `stat -c %d` on both paths.

---
