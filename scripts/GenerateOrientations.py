#!/usr/bin/env python3
"""Entry point for GenerateOrientations.

The implementation moved to ``laue_index/pipeline/GenerateOrientations.py`` so that
``pip install laue-index`` ships it -- the package used to install the library
and the C binaries but nothing that could run an image through them. This shim
keeps ``python scripts/GenerateOrientations.py ...`` and the shell pipeline working from a
checkout, unchanged.
"""
import sys
from pathlib import Path

try:
    from laue_index.pipeline import run_module
except ImportError:  # not installed -- use the package in this checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "laue_index"))
    from laue_index.pipeline import run_module

if __name__ == "__main__":
    run_module("GenerateOrientations", sys.argv[1:])
