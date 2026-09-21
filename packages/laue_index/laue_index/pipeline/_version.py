"""Central version string for LaueMatching.

Read by ``laue_provenance.collect`` and emitted into every generated artifact
so that consumers can trace the exact code version that produced them.

WARNING: this is NOT the laue-index package version and it is NOT bumped on
release. It read "2.2.0" across at least 84 commits and every 0.x release, so it
identifies nothing. Provenance now records the real identity under
``build``: ``laue_index.__version__`` and the ``c_src_sha256`` from the CMake
build manifest. Kept only so existing records stay readable.
"""

__version__ = "2.2.0"
