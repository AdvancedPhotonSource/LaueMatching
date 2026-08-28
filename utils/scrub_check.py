#!/usr/bin/env python3
"""Fail if a user's name, campaign, sample, or data path reaches a tracked file.

The LaueMatching repo is PUBLIC. Campaign directories are named after the user,
sample labels plus the material identify a specific study, and the analysis
scripts carried the real data root as an env-var *default*. Anyone reading them
learns whose data a number came from and can reproduce unpublished science.
This gate stops new ones landing.

    python utils/scrub_check.py            # scan tracked files, exit 1 on a hit
    python utils/scrub_check.py --staged   # scan only staged files (pre-commit)
    python utils/scrub_check.py --install-hook

Resolve pseudonyms with the private BEAMTIME_KEY.md (git-excluded, never committed).

Ported from ~/opt/MIDAS/utils/scrub_check.py. **Adapt NAME_PATTERNS per repo** --
MIDAS's list is its own users, and running MIDAS's copy against this tree reports
"clean" while every hit below is present. A detector that does not know the
relevant names is not evidence of absence.

Four traps this deliberately handles, all hit during the 2026-08 scrub:

1. **Underscore is a word character.** ``work`` is NOT matched by
   ``\\bhemant\\b``. A first name embedded in a path needs its own pattern.

2. **Initials, and quoted speech.** The original notebook attributed direct
   quotations of a user's research questions to ``SF``. A surname deny-list
   misses initials entirely, and the quote leaks more than the name would.
   Initials are too short to pattern-match safely -- they are caught by reading,
   which is why the review step is not optional.

3. **Filenames leak independently of content.** ``LAB_NOTEBOOK.md`` and
   ``parentbeta_reconstruct.py`` disclose without a single bad line inside. The
   scanner checks ``path.as_posix()`` before it opens the file.

4. **base64 in notebooks.** Embedded PNGs in ``.ipynb`` outputs contain arbitrary
   letter runs. Notebooks are parsed as JSON and only cell *source* is scanned,
   never ``outputs`` or ``attachments``. Scanning them as flat text produces
   false positives and, worse, tempts a blanket sed that corrupts the image.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Personal names that have appeared as DATA SOURCES (campaign dirs, sample
# filenames, notebook attributions, third-party analysis paths). Case-insensitive.
#
# NOT listed, deliberately -- method attributions and citations, where scrubbing
# would be a misattribution rather than a privacy win:
#   Laue, Bragg, Rodrigues, Gauss, von Mises  -- physics.
#   Sharma  -- the repository author, in LICENSE/CONTRIBUTING/pyproject metadata.
# A campaign named after any of them is still caught by BEAMTIME_RE below.
# NOTE: this list is deliberately EMPTY in the committed file. The real names
# are loaded at runtime from PRIVATE_PATTERNS (git-excluded) -- see
# load_private_patterns() below and the rationale in the module docstring.
NAME_PATTERNS: list[str] = []

# Materials, systems and sample labels are ALSO private -- same reasoning as
# NAME_PATTERNS. Loaded from PRIVATE_PATTERNS at runtime.
MATERIAL_PATTERNS: list[str] = []
SAMPLE_PATTERNS: list[str] = []
# NOT listed, deliberately: `parentbeta`, `beta_alpha`, and the bare phase names
# `alpha`/`beta`. Parent-beta reconstruction is a standard, published method in
# titanium and steel metallurgy -- the filenames name a technique, not a user's
# science, and renaming them would break the chain script, the output-file
# prefixes, and any existing user scripts for no privacy gain. The material that
# DID identify a campaign (a two-phase hcp/bcc alloy in the params templates) is scrubbed above.

# Real data roots. These leaked as argparse/env defaults in ~30 analysis
# scripts; the scripts already read $LAUE_WORK, only the fallback disclosed.
#
# These stay PUBLIC: a beamline store mount point names no person, it is the
# same for every user of the sector, and having it in the committed gate is what
# stops the next default path landing.
PATH_PATTERNS = [
    r"$LAUE_ROOT",
    r"/data34[a-z]?/(?!\$)[a-z]",   # /data34c/<something>, but not a variable
    r"the-analysis-host",
]

#: Git-excluded file holding the identifying patterns. One regex per line;
#: ``#`` comments and blank lines ignored; section headers select the group.
PRIVATE_PATTERNS = Path(".scrub_patterns")


def load_private_patterns() -> bool:
    """Populate NAME/MATERIAL/SAMPLE patterns from the git-excluded file.

    Returns True if the file was found and loaded.

    Why the names are not in this file: a committed deny-list naming the people
    it hides is self-defeating for a repo being de-identified -- it hands a
    reader the exact search terms to run against an old clone or a cached
    commit. MIDAS's copy of this script does inline its list; that is an
    accepted tradeoff there, and the wrong one here.

    Fails OPEN for the structural checks (paths, beamtime shape) so CI, which
    has no private file, still gates the class of mistake that matters most --
    and says loudly that it is running reduced.
    """
    section = None
    groups = {
        "names": NAME_PATTERNS,
        "materials": MATERIAL_PATTERNS,
        "samples": SAMPLE_PATTERNS,
    }
    try:
        text = PRIVATE_PATTERNS.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section not in groups:
                raise SystemExit(
                    f"{PRIVATE_PATTERNS}: unknown section [{section}]; "
                    f"expected one of {sorted(groups)}"
                )
            continue
        if section is None:
            raise SystemExit(
                f"{PRIVATE_PATTERNS}: pattern before any [section] header"
            )
        groups[section].append(line)
    return True

# Any name_monYY / nameMonYY beamtime that is not on the institutional allow-list.
BEAMTIME_RE = re.compile(
    r"\b([a-z0-9]+)_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[0-9]{2}\b",
    re.IGNORECASE,
)
ALLOWED_BEAMTIME_PREFIXES = {
    # pseudonyms
    "bt", "dataset", "sample",
    # non-personal: facility, sector, programme, or descriptive
    "mpe", "afrl", "hpldrd", "nfdev",
}

# (path substring, regex) pairs that are known-good and must not fail the gate.
ALLOWLIST = [
    # This file names the patterns it searches for.
    ("utils/scrub_check.py", r".*"),
    # The public notebook explains WHY these were scrubbed, in prose that has to
    # be able to say what it is talking about ("an hcp deposit on an fcc
    # substrate"). It carries no real label -- verified by the same gate.
    ("scripts/pipeline/laue/LAB_NOTEBOOK.md", r"parentbeta|beta_alpha"),
]

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".ge1", ".ge2", ".ge3",
    ".ge5", ".edf", ".tif", ".tiff", ".h5", ".hdf5", ".bin", ".pptx", ".docx",
    ".so", ".dylib", ".o", ".a", ".pyc",
}


def tracked_files(staged: bool) -> list[Path]:
    cmd = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
        if staged
        else ["git", "ls-files"]
    )
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [Path(p) for p in out.splitlines() if p]


def notebook_source_lines(path: Path):
    """Yield (lineno, text) for notebook cell SOURCE only -- never outputs."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return
    for ci, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        blob = src if isinstance(src, str) else "".join(src)
        for li, line in enumerate(blob.splitlines(), start=1):
            yield f"cell{ci}:{li}", line


def text_lines(path: Path):
    try:
        for i, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            yield str(i), line
    except OSError:
        return


def allowlisted(path: Path, line: str) -> bool:
    p = str(path)
    return any(
        frag in p and re.search(rx, line, re.IGNORECASE) for frag, rx in ALLOWLIST
    )


def scan(paths: list[Path]) -> list[tuple[str, str, str, str]]:
    # NEVER re.compile("|".join([])) -- an empty pattern matches at every
    # position, so an unloaded group would flag every line of every file. A
    # never-matching sentinel is the safe empty value.
    def _alt(patterns: list[str], flags: int = 0):
        if not patterns:
            return re.compile(r"(?!x)x")     # matches nothing, ever
        return re.compile("|".join(patterns), flags)

    name_re = _alt(NAME_PATTERNS, re.IGNORECASE)
    mat_re = _alt(MATERIAL_PATTERNS, re.IGNORECASE)
    path_re = _alt(PATH_PATTERNS, re.IGNORECASE)
    # Sample labels are deliberately NOT case-insensitive: `\bG21\b` folded to
    # lowercase also matches the English word "g21"-free text far less often
    # than it matches hex digests and version strings. The patterns already
    # spell out the case variants that occur (sampleA and sampleK both appear).
    samp_re = _alt(SAMPLE_PATTERNS)
    hits: list[tuple[str, str, str, str]] = []

    for path in paths:
        # `git ls-files` lists files whose DELETION is not yet staged. Those no
        # longer exist on disk and cannot leak anything -- flagging them makes
        # the gate un-passable until you commit, which is backwards. Staged mode
        # already excludes deletions via --diff-filter=ACM.
        if not path.exists():
            continue

        # Filenames leak too: a LAB_NOTEBOOK_<PI>.md or a parentbeta_*.py
        # discloses just as much as its contents, and a content-only scan never
        # sees it. This is how LAB_NOTEBOOK.md and the five parentbeta/
        # beta_alpha scripts were found.
        posix = path.as_posix()
        for kind, rx in (
            ("name-in-path", name_re),
            ("material-in-path", mat_re),
            ("datapath-in-path", path_re),
            ("sample-in-path", samp_re),
        ):
            m = rx.search(posix)
            if m and not allowlisted(path, posix):
                hits.append((posix, "<filename>", kind, m.group(0)))

        if path.suffix.lower() in BINARY_SUFFIXES or not path.exists():
            continue
        reader = notebook_source_lines if path.suffix == ".ipynb" else text_lines
        for loc, line in reader(path):
            if allowlisted(path, line):
                continue
            for kind, rx in (
                ("name", name_re),
                ("material", mat_re),
                ("datapath", path_re),
                ("sample", samp_re),
            ):
                m = rx.search(line)
                if m:
                    hits.append((str(path), loc, kind, m.group(0)))
            for m in BEAMTIME_RE.finditer(line):
                prefix = m.group(1).lower()
                if not any(prefix.startswith(a) for a in ALLOWED_BEAMTIME_PREFIXES):
                    hits.append((str(path), loc, "beamtime", m.group(0)))
    return hits


def install_hook() -> int:
    hook = Path(".git/hooks/pre-commit")
    body = (
        "#!/bin/sh\n"
        "# Added by utils/scrub_check.py --install-hook\n"
        'exec python3 "$(git rev-parse --show-toplevel)/utils/scrub_check.py" --staged\n'
    )
    if hook.exists() and "scrub_check.py" not in hook.read_text():
        print(f"refusing to overwrite existing hook: {hook}", file=sys.stderr)
        print("add this line to it yourself:\n  " + body.splitlines()[-1])
        return 1
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(body)
    hook.chmod(0o755)
    print(f"installed {hook}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged", action="store_true", help="scan staged files only")
    ap.add_argument("--install-hook", action="store_true")
    args = ap.parse_args()

    if args.install_hook:
        return install_hook()

    if not load_private_patterns():
        print(
            f"scrub-check: WARNING -- {PRIVATE_PATTERNS} not found. Running "
            "STRUCTURAL checks only (data paths, beamtime shape); name, "
            "material and sample checks are DISABLED. This is expected in CI "
            "and is NOT a clean bill of health on a developer machine.",
            file=sys.stderr,
        )

    hits = scan(tracked_files(args.staged))
    if not hits:
        print("scrub-check: clean")
        return 0

    print("scrub-check: FAILED -- identifying strings in tracked files\n", file=sys.stderr)
    for path, loc, kind, tok in hits:
        print(f"  {path}:{loc}  [{kind}]  {tok}", file=sys.stderr)
    print(
        f"\n{len(hits)} hit(s). Replace with a pseudonym and record the mapping in "
        "BEAMTIME_KEY.md (git-excluded). If this is a literature citation rather "
        "than a data source, add it to ALLOWLIST in utils/scrub_check.py.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
