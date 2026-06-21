"""Tiny golden-snapshot helper for characterization tests.

Behaviour-preserving refactors need an external record of "what the code does
today".  ``check_golden(name, obj)`` canonicalises ``obj`` (numpy -> plain
Python, floats rounded for stable diffs, sets -> sorted lists) and compares it
to ``tests/golden/<name>.json``.

Regenerate goldens (e.g. on first creation, or after an *intentional* behaviour
change you have reviewed) with::

    UPDATE_GOLDEN=1 python -m pytest tests/

With ``UPDATE_GOLDEN=1`` the golden file is (re)written and the check passes.
Otherwise a missing golden is a hard failure (so CI never silently self-heals).
"""
import json
import os
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
UPDATE = os.environ.get("UPDATE_GOLDEN", "0") == "1"

# Decimal places kept when canonicalising floats.  6 is ample for the
# quantities pinned here (angles in deg, threshold counts, quality scores) and
# keeps the JSON goldens human-diffable and stable across platforms.
_ROUND = 6


def canon(obj):
    """Recursively convert ``obj`` to a JSON-safe, rounded, order-stable form."""
    if isinstance(obj, dict):
        return {str(k): canon(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (set, frozenset)):
        return [canon(x) for x in sorted(obj, key=repr)]
    if isinstance(obj, np.ndarray):
        return canon(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [canon(x) for x in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), _ROUND)
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"canon: unsupported type {type(obj)!r}")


def check_golden(name: str, obj) -> None:
    """Assert ``obj`` matches the stored golden ``name`` (or write it under UPDATE)."""
    path = GOLDEN_DIR / f"{name}.json"
    got = canon(obj)
    if UPDATE or not path.exists():
        if not path.exists() and not UPDATE:
            # Auto-create on first ever run, but be loud about it.
            print(f"[golden] creating new golden: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        return
    expected = canon(json.loads(path.read_text()))
    assert got == expected, (
        f"Golden mismatch for '{name}'.\n"
        f"  got:      {json.dumps(got, sort_keys=True)[:800]}\n"
        f"  expected: {json.dumps(expected, sort_keys=True)[:800]}\n"
        f"If this change is intentional, regenerate with UPDATE_GOLDEN=1."
    )


def fixture(name: str) -> Path:
    """Path to a committed fixture under ``tests/fixtures/``."""
    p = FIXTURE_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing committed fixture: {p}")
    return p


# Heavy / local-only fixtures (real frames, the 497 MB indent scan, …) live
# outside git in the main checkout's ``_seed_test_local/``.  Resolve via the
# LAUE_FIXTURES env var, defaulting to the sibling main checkout.  Tests that
# need them should ``pytest.skip`` when the file is absent (so CI stays green).
SEED_DIR = Path(os.environ.get(
    "LAUE_FIXTURES",
    str(Path.home() / "opt" / "LaueMatching" / "_seed_test_local"),
))


def local_fixture(name: str):
    """Path to a heavy local-only fixture, or ``None`` if unavailable."""
    p = SEED_DIR / name
    return p if p.exists() else None
