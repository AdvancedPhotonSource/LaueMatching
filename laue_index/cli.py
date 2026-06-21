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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
