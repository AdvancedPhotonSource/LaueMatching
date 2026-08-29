"""laue_index command-line entry.

``run`` drives the full image→index pipeline; ``fetch-db`` gets the orientation
database it needs; ``parse``/``filter``/``calibrate`` are the post-indexing
operations the package owns — inspect a solutions table, re-run post-processing
(unique-spots → filter → spot-filter) on existing C output without re-indexing,
or solve a detector pose from labelled spots.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __all__ as _api  # noqa: F401  (kept for `info`)
from . import __version__ as _VERSION
from .records import SOLUTION_FORMATS, parse_solutions
from .postprocess import PostProcessor

__all__ = ["main"]

#: Where `fetch-db` gets the orientation database, and in how many parts.
ORIENT_DB_RELEASE = (
    "https://github.com/AdvancedPhotonSource/LaueMatching/releases/download/v1.0-data")
ORIENT_DB_PARTS = ("100MilOrients.part.aa", "100MilOrients.part.ab",
                   "100MilOrients.part.ac", "100MilOrients.part.ad")
#: 100,000,000 orientations x 9 doubles. The count is derivable from the size,
#: which is why the check below is arithmetic rather than a magic number.
ORIENT_DB_BYTES = 100_000_000 * 9 * 8
#: Consulted by the pipeline when a config names no OrientationFile.
ORIENT_DB_ENV = "LAUEMATCHING_ORIENT_DB"


def _cmd_parse(args: argparse.Namespace) -> int:
    sols = parse_solutions(args.solutions, fmt=args.fmt)
    print(f"{len(sols)} solutions ({args.fmt} format)")
    print(f"{'grain':>6} {'matches':>8} {'quality':>14} {'row':>12}")
    n = len(sols) if args.all else min(len(sols), args.top)
    for s in sorted(sols, key=lambda x: x.quality, reverse=True)[:n]:
        print(f"{s.grain_nr:>6} {s.n_matches:>8} {s.quality:>14.3f} "
              f"{s.orientation_row_nr:>12}")
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:
    fmt = SOLUTION_FORMATS[args.fmt]
    sols = np.atleast_2d(np.loadtxt(args.solutions, skiprows=1))
    spots = np.atleast_2d(np.loadtxt(args.spots, skiprows=1))
    if args.labels:
        labels = np.load(args.labels)
    else:
        # per-spot-position fallback labels
        labels = np.zeros((args.nr_px_y, args.nr_px_x), dtype=np.int32)
        c = 1
        xs = spots[:, fmt.spot_x].astype(int)
        ys = spots[:, fmt.spot_y].astype(int)
        ok = (xs >= 0) & (xs < args.nr_px_x) & (ys >= 0) & (ys < args.nr_px_y)
        for x, y in zip(xs[ok], ys[ok]):
            if labels[y, x] == 0:
                labels[y, x] = c
                c += 1
    res = PostProcessor(robust=args.robust, min_unique=args.min_unique,
                        min_total_spots=args.min_total_spots,
                        max_angle_deg=args.max_angle, space_group=args.space_group,
                        fmt=fmt)(sols, spots, labels)
    print(f"kept {res.filtered_orientations.shape[0]} of {sols.shape[0]} "
          f"orientations ({'robust' if args.robust else 'legacy'} filter); "
          f"grains {sorted(res.kept_grain_nrs)}")
    if args.out:
        with open(args.solutions) as f:
            header = f.readline().rstrip("\n")
        np.savetxt(args.out, res.filtered_orientations, header=header, comments="")
        print(f"wrote {args.out}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import Anchor, DetectorSpec, calibrate

    # anchors: one per line, "h k l px py [energy_keV]"
    anchors, held_hkl = [], []
    with open(args.anchors) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            v = line.split()
            if len(v) < 5:
                raise SystemExit(f"anchor line needs 'h k l px py [E]': {line!r}")
            anchors.append(Anchor(
                hkl=(int(float(v[0])), int(float(v[1])), int(float(v[2]))),
                pixel=(float(v[3]), float(v[4])),
                energy_kev=float(v[5]) if len(v) > 5 else None))

    recip = np.loadtxt(args.recip).reshape(3, 3)
    lattice = [float(x) for x in args.lattice.split(",")]
    spec = DetectorSpec(n_pix=(args.nr_px_x, args.nr_px_y),
                        px_size=(args.px_x, args.px_y))

    spots = np.loadtxt(args.spots)[:, :2] if args.spots else None
    if args.held_out_hkl:
        held_hkl = np.loadtxt(args.held_out_hkl).reshape(-1, 3)

    guess = None
    if args.initial_guess:
        guess = [float(x) for x in args.initial_guess.split(",")]
        if len(guess) != 6:
            raise SystemExit("--initial-guess needs 6 comma-separated values")

    res = calibrate(
        anchors, recip=recip, lattice=lattice, spec=spec,
        frame_provenance=args.frame_provenance,
        initial_guess=guess,
        convention=args.convention or None,
        observed_spots=spots,
        held_out_hkl=held_hkl if len(held_hkl) else None,
        tolerance_px=args.tolerance_px, null_trials=args.null_trials)

    print(f"P_Array  {' '.join(f'{x:.9f}' for x in res.p_array)}")
    print(f"R_Array  {' '.join(f'{x:.9f}' for x in res.r_array)}")
    print(f"distance {res.distance_mm:.3f} mm   rms {res.rms_px:.4f} px")
    print(f"recip1 read as {res.convention.upper()}  "
          f"(scores: " + ", ".join(
              f"{k} {v['rms_px']:.4g} px" for k, v in res.convention_scores.items())
          + ")")
    print(res.conditioning.describe())
    if res.energy_check.get("n"):
        e = res.energy_check
        print(f"energy check (orientation, NOT pose): {e['n']} reflections, "
              f"{e['mean_ppm']:+.1f} +- {e['sd_ppm']:.1f} ppm, "
              f"max |dE| {e['max_abs_ev']:.2f} eV")
    if res.validation is not None:
        v = res.validation
        print(f"held out {v.n_held_out_matched}/{v.n_held_out} within "
              f"{v.tolerance_px:g} px, median {v.held_out_median_px:.3f} px; "
              f"null max {v.null_max_matched} over {v.null_trials} trials "
              f"-> {'CLEARS' if v.clears_null else 'DOES NOT CLEAR'} the null")
        print(f"residuals dx {v.residual_dx[0]:+.3f} +- {v.residual_dx[1]:.3f}, "
              f"dy {v.residual_dy[0]:+.3f} +- {v.residual_dy[1]:.3f} px")
    if args.out:
        with open(args.out, "w") as f:
            f.write(res.params_text(lattice, args.space_group, args.symmetry,
                                    spec=spec))
        print(f"wrote {args.out}")
    return 0


def _cmd_run(rest: list[str]) -> int:
    """Hand off to the orchestrator, which ships inside the package."""
    from .pipeline import run_module
    try:
        run_module("RunImage", rest)
    except SystemExit as exc:                     # argparse / normal exit
        return int(exc.code or 0)
    except ImportError as exc:
        print(f"error: the pipeline needs dependencies that are not installed ({exc}).\n"
              "       pip install 'laue-index[run]'", file=sys.stderr)
        return 1
    return 0


def _download(url: str, dest, expected: int | None = None) -> int:
    """Fetch one URL to a file, skipping it if already complete. Returns bytes."""
    import urllib.request

    have = dest.stat().st_size if dest.exists() else 0
    with urllib.request.urlopen(url) as resp:     # noqa: S310 (fixed release URL)
        total = int(resp.headers.get("Content-Length") or 0)
        if have and total and have == total:
            print(f"  {dest.name}: already complete ({have:,} B)")
            return have
        print(f"  {dest.name}: {total:,} B" if total else f"  {dest.name}")
        written = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 22)        # 4 MiB
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
    if total and written != total:
        raise IOError(f"{dest.name}: got {written:,} B, expected {total:,} B")
    if expected is not None and written != expected:
        raise IOError(f"{dest.name}: got {written:,} B, expected {expected:,} B")
    return written


def _cmd_fetch_db(args: argparse.Namespace) -> int:
    """Download and reassemble the orientation database.

    `pip install laue-index` ships the binaries but not the 6.7 GB database
    they index against, and a pip user has no build.sh to fetch it. This is
    that step, and nothing more: it does not decide where the database should
    live, it reports where it put it.
    """
    from pathlib import Path

    dest = Path(args.dest).expanduser().resolve()
    if dest.is_dir():
        dest = dest / "100MilOrients.bin"
    if dest.exists() and not args.force:
        size = dest.stat().st_size
        print(f"{dest} already exists ({size:,} B). Use --force to replace it.")
        return 0 if size == ORIENT_DB_BYTES else 1

    parts_dir = Path(args.parts_dir).expanduser().resolve() if args.parts_dir else dest.parent
    parts_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(ORIENT_DB_PARTS)} parts to {parts_dir}")
    paths = []
    for name in ORIENT_DB_PARTS:
        p = parts_dir / name
        try:
            _download(f"{ORIENT_DB_RELEASE}/{name}", p)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("       parts already downloaded are kept; re-run to resume.",
                  file=sys.stderr)
            return 1
        paths.append(p)

    print(f"reassembling -> {dest}")
    with open(dest, "wb") as out:
        for p in paths:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)

    size = dest.stat().st_size
    if size % 72:
        print(f"error: {dest} is {size:,} B, not a whole number of orientations "
              f"(9 doubles = 72 B each). The download is corrupt.", file=sys.stderr)
        return 1
    print(f"{dest}: {size:,} B = {size // 72:,} orientations")
    if size != ORIENT_DB_BYTES:
        print(f"warning: expected {ORIENT_DB_BYTES:,} B for the 100M database.",
              file=sys.stderr)

    if not args.keep_parts:
        for p in paths:
            p.unlink()

    print(f"\nPoint runs at it with {ORIENT_DB_ENV}={dest}, or set OrientationFile "
          f"in the config.\nCopying it to /dev/shm first makes the indexer mmap it "
          f"instead of reading it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Pass `run` straight through: argparse would eat --help and RunImage's own
    # subcommand names before they ever reached it.
    if argv and argv[0] == "run":
        return _cmd_run(argv[1:])

    p = argparse.ArgumentParser(
        prog="laue-index", description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=f"laue-index {_VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # `run` is intercepted before parsing (see main) so that every argument --
    # including --help and RunImage's own subcommands -- reaches RunImage
    # verbatim. Registered here only so it appears in `laue-index --help`.
    sub.add_parser(
        "run", add_help=False,
        help="run the full pipeline (RunImage): laue-index run process -c ... -i ...")

    pd = sub.add_parser("fetch-db", help="download the 6.7 GB orientation database")
    pd.add_argument("--dest", default=".",
                    help="file, or a directory to write 100MilOrients.bin into")
    pd.add_argument("--parts-dir", dest="parts_dir", default="",
                    help="where to stage the 4 downloaded parts (default: next to --dest)")
    pd.add_argument("--keep-parts", dest="keep_parts", action="store_true",
                    help="keep the parts after reassembly")
    pd.add_argument("--force", action="store_true", help="overwrite an existing database")
    pd.set_defaults(func=_cmd_fetch_db)

    pp = sub.add_parser("parse", help="parse a solutions table and summarise")
    pp.add_argument("solutions")
    pp.add_argument("--fmt", choices=sorted(SOLUTION_FORMATS), default="runimage")
    pp.add_argument("--top", type=int, default=10)
    pp.add_argument("--all", action="store_true")
    pp.set_defaults(func=_cmd_parse)

    pf = sub.add_parser("filter", help="re-run post-processing on existing output")
    pf.add_argument("--solutions", required=True)
    pf.add_argument("--spots", required=True)
    pf.add_argument("--labels", default="", help="optional labels .npy")
    pf.add_argument("--fmt", choices=sorted(SOLUTION_FORMATS), default="runimage")
    pf.add_argument("--robust", action="store_true", default=True)
    pf.add_argument("--legacy", dest="robust", action="store_false")
    pf.add_argument("--min-unique", dest="min_unique", type=int, default=2)
    pf.add_argument("--min-total-spots", dest="min_total_spots", type=int, default=5)
    pf.add_argument("--max-angle", dest="max_angle", type=float, default=5.0)
    pf.add_argument("--space-group", dest="space_group", type=int, default=225)
    pf.add_argument("--nr-px-x", dest="nr_px_x", type=int, default=2048)
    pf.add_argument("--nr-px-y", dest="nr_px_y", type=int, default=2048)
    pf.add_argument("--out", default="", help="write filtered solutions here")
    pf.set_defaults(func=_cmd_filter)

    pc = sub.add_parser(
        "calibrate",
        help="detector pose from labelled spots and a SUPPLIED orientation")
    pc.add_argument("--anchors", required=True,
                    help="text file, one spot per line: 'h k l px py [E_keV]'. "
                         "At least 4 are required: 3 is exactly determined and "
                         "admits a second exact solution the residual cannot "
                         "distinguish from the true one.")
    pc.add_argument("--recip", required=True,
                    help="3x3 reciprocal matrix (text). Both row and column "
                         "readings are tried; the projection decides.")
    pc.add_argument("--frame-provenance", dest="frame_provenance", required=True,
                    help="REQUIRED. Where the supplied orientation's frame came "
                         "from: the rotation about the beam is inherited from "
                         "it, not measured from the pattern.")
    pc.add_argument("--lattice", default="0.543102,0.543102,0.543102,90,90,90",
                    help="a,b,c,alpha,beta,gamma (nm, degrees)")
    pc.add_argument("--nr-px-x", dest="nr_px_x", type=int, default=1028)
    pc.add_argument("--nr-px-y", dest="nr_px_y", type=int, default=1062)
    pc.add_argument("--px-x", dest="px_x", type=float, default=75e-6)
    pc.add_argument("--px-y", dest="px_y", type=float, default=75e-6)
    pc.add_argument("--initial-guess", dest="initial_guess", default="",
                    help="P0,P1,P2,r0,r1,r2 (metres, radians)")
    pc.add_argument("--convention", choices=["columns", "rows"], default="",
                    help="force a reading instead of letting the fit choose")
    pc.add_argument("--spots", default="",
                    help="all observed spots (px py per line) for validation")
    pc.add_argument("--held-out-hkl", dest="held_out_hkl", default="",
                    help="reflections to predict but NOT fit (h k l per line)")
    pc.add_argument("--tolerance-px", dest="tolerance_px", type=float, default=3.0)
    pc.add_argument("--null-trials", dest="null_trials", type=int, default=2000)
    pc.add_argument("--space-group", dest="space_group", type=int, default=227)
    pc.add_argument("--symmetry", default="F")
    pc.add_argument("--out", default="", help="write a params file here")
    pc.set_defaults(func=_cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
