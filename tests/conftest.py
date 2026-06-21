"""Shared pytest setup for the LaueMatching test suite.

Puts the legacy ``scripts/`` directory on ``sys.path`` so the characterization
tests can import the *current* implementation (``laue_stream_utils``,
``laue_config``, …) as the behaviour baseline.  During the refactor the new
``laue_index`` package will be imported the same way and these golden anchors
must keep passing unchanged.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(_HERE, os.pardir, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
