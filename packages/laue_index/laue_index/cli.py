"""laue_index command-line entry (REFACTOR_PLAN §3 — mirrors laue_torch/cli.py).

Thin wrapper over the package; the full image→index pipeline lives in
``scripts/RunImage.py``.  This exposes the post-indexing operations the package
owns — inspect a solutions table, or re-run post-processing (unique-spots →
filter → spot-filter) on existing C output without re-indexing.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __all__ as _api  # noqa: F401  (kept for `info`)
from .records import SOLUTION_FORMATS, parse_solutions
from .postprocess import PostProcessor

__all__ = ["main"]

_VERSION = "0.1.0"


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="laue-index", description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=f"laue-index {_VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

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
