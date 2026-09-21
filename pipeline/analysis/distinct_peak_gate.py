"""Per-instance spot counts from the indexer's own outputs, for gate comparisons.

CORRECTED 2026-09-21. This script was written reading
`unique_spots_per_orientation` as "distinct OBSERVED peaks", which is what handbook
invariant 15b said at the time. That is wrong, and so was the "stacking ratio" this
script printed. What the two columns actually are:

    filtered_orientations[:, n_matches]  NMatches (col 6 stream, 5 RunImage): distinct observed pixels matched by
                                         THIS orientation. The C indexer dedups by q-hat,
                                         so it does not stack -- on sampleH, 0 of 25,172
                                         solutions put two matches on one pixel.
    unique_spots_per_orientation[:, 1]   WINNER-TAKE-ALL across the frame's orientations
                                         (laue_index.filtering.calculate_unique_spots):
                                         strongest first, a claimed pixel is unavailable to
                                         weaker ones. Measures INDEPENDENT evidence.

So `matched / unique` is a SHARING ratio -- how much of an orientation's evidence a
stronger orientation already claimed -- not stacking. The output keys keep their old
names (`distinct`, `ratio`) so existing npz files stay readable; read them as
`wta_unique` and `sharing_ratio`. For a genuine per-orientation distinct-peak count
against the analysis peak list, use frame_peaks.count_matched_peaks, which is what
`nhit_distinct` in parentbeta_validate.py now carries.

JOIN ON THE ORIENTATION ID WITHIN THE FRAME:

    filtered_orientations[:, grain]  <->  unique_spots_per_orientation[:, 0]

``grain`` is column 1 in the 35-column stream layout, where column 0 is the IMAGE
number (joining on it returns zeros), and column 0 in the 34-column RunImage layout.
The layout is chosen by column count (frame_peaks.solution_format); an unknown one exits.

Sanity anchor: winner-take-all can only remove evidence, so matched / unique >= 1
for every row. A value below 1 means the join is wrong, and the script refuses to write.

usage: distinct_peak_gate.py <results_dir> <out.npz> [--nw N] [--limit N]
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

# Column map by the table's own column count: laue_index.records.SOLUTION_FORMATS
# (stream 35 columns, RunImage 34) via frame_peaks.solution_format. This used to
# hard-code the stream map; on a RunImage file column 1 is not the grain id, the
# join missed, and a wrong orientation block was still written.
from frame_peaks import image_number, solution_format


def scan_one(path: str):
    """Return (n, 6) rows [image, grain, matched, wta_unique, quality, intensity]
    plus the (n, 9) orientation block, for one frame. (wta_unique is saved under
    the legacy key ``distinct``; it is winner-take-all, not a distinct count.)"""
    with h5py.File(path, "r") as h:
        res = h["entry/results"]
        if "filtered_orientations" not in res:
            return None
        fo = res["filtered_orientations"][:]
        if fo.size == 0:
            return None
        us = (
            res["unique_spots_per_orientation"][:]
            if "unique_spots_per_orientation" in res
            else np.empty((0, 2), int)
        )
    fo = np.atleast_2d(fo)
    fmt = solution_format(fo.shape[1], path)
    lut = {int(a): int(b) for a, b in us}
    ids = fo[:, fmt.grain].astype(int)
    # RunImage tables carry no ImageNr column: take it from the file name
    image = (fo[:, fmt.image_nr] if fmt.image_nr >= 0
             else np.full(len(fo), float(image_number(path))))
    distinct = np.array([lut.get(int(i), -1) for i in ids], float)
    rows = np.column_stack(
        [
            image,
            fo[:, fmt.grain],
            fo[:, fmt.n_matches],
            distinct,
            fo[:, fmt.quality],
            fo[:, fmt.intensity],
        ]
    )
    return rows, fo[:, fmt.om_start:fmt.om_start + 9], os.path.basename(path)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if "=" in a and a.startswith("--")}
    if len(args) < 2:
        sys.exit(__doc__)
    resdir, out = Path(args[0]), args[1]
    nw = int(opts.get("--nw", os.cpu_count() or 8))
    limit = int(opts.get("--limit", 0))

    files = sorted(str(p) for p in resdir.glob("image_*.output.h5"))
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"no image_*.output.h5 under {resdir}")
    print(f"{len(files)} frames, {nw} workers", flush=True)

    rows, oms, srcs = [], [], []
    with ProcessPoolExecutor(nw) as ex:
        for k, r in enumerate(ex.map(scan_one, files, chunksize=64)):
            if r is not None:
                rows.append(r[0])
                oms.append(r[1])
                srcs.extend([r[2]] * len(r[0]))
            if k % 2000 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    R = np.concatenate(rows)
    OM = np.concatenate(oms).reshape(-1, 3, 3)
    srcs = np.array(srcs)

    matched, distinct = R[:, 2], R[:, 3]
    found = distinct > 0
    n_missing = int((~found).sum())

    # Sanity anchor. Winner-take-all can only REMOVE an orientation's evidence,
    # never add to it, so matched >= wta_unique on every row.
    ratio = np.full(len(R), np.nan)
    ratio[found] = matched[found] / distinct[found]
    bad = found & (ratio < 1.0 - 1e-9)
    if bad.any():
        sys.exit(
            f"JOIN IS WRONG: {bad.sum()} of {found.sum()} rows have "
            f"matched/wta_unique < 1 (min {np.nanmin(ratio):.3f}). Winner-take-all "
            f"can only remove an orientation's evidence, never add to it. Refusing "
            f"to write. Check the join is on the layout's grain column, not the image column."
        )

    # Map the shard-local image number to the map-wide source frame, if available.
    fm = resdir.parent / "frame_mapping.json"
    frames = np.array([""] * len(R), dtype=object)
    if fm.exists():
        m = json.loads(fm.read_text())
        # keys may be str or int; values a dict with "file" or a bare path
        def _file(v):
            return os.path.basename(v["file"] if isinstance(v, dict) else v)
        lut = {int(k): _file(v) for k, v in m.items()}
        frames = np.array([lut.get(int(i), "") for i in R[:, 0]], dtype=object)
        print(f"frame_mapping.json: {len(lut)} entries, "
              f"{int((frames != '').sum())}/{len(R)} rows mapped", flush=True)
    else:
        print("WARNING: no frame_mapping.json; rows carry shard-local image numbers "
              "ONLY and cannot be merged across shards", flush=True)

    np.savez(
        out,
        image=R[:, 0], grain=R[:, 1], matched=matched, distinct=distinct,
        quality=R[:, 4], intensity=R[:, 5], ratio=ratio,
        oms=OM, frames=frames.astype(str), src=srcs,
    )

    q = np.nanpercentile(ratio[found], [50, 90, 99])
    print(f"\nwrote {out}")
    print(f"  instances               {len(R)}")
    print(f"  no wta_unique count     {n_missing}")
    print(f"  matched  median {np.median(matched):.0f}  p10 {np.percentile(matched,10):.0f}")
    print(f"  wta_unique (key 'distinct') median {np.median(distinct[found]):.0f}  "
          f"p10 {np.percentile(distinct[found],10):.0f}")
    print(f"  SHARING RATIO (matched/wta_unique -- NOT stacking) median {q[0]:.3f}  p90 {q[1]:.3f}  p99 {q[2]:.3f}")
    for lo, hi in ((1.0, 1.25), (1.25, 2.0), (2.0, 3.0), (3.0, np.inf)):
        n = int(((ratio >= lo) & (ratio < hi)).sum())
        print(f"    ratio [{lo:>4}, {hi:>4}): {n:7d}  ({100*n/len(R):5.1f}%)")


if __name__ == "__main__":
    main()
