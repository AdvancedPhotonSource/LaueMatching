"""The `if (xMapped) munmap / else free` pairings must stay adjacent.

This guards a bug class that already cost two months of silent breakage.
`LaueMatchingCPU` exited -11 (SIGSEGV) on every run that WROTE the forward cache
-- the first run on any machine -- while producing complete, correct output, so
RunImage reported the image as failed and the user got nothing. The cause was not
in the cleanup logic itself: commit 654957a inserted an unrelated `outArr`
munmap BETWEEN the two halves of

    if (orientsMapped)
      munmap(orients, szFile);
    else
      free(orients);

which silently re-parented the `else` onto the new `if`. Writing the cache leaves
`outArr` NULL, so the `else` fired and `free()` ran on a pointer `munmap`'d one
line above. Fixed in 87896b7.

Nothing caught it because every test that ran the binary supplied a prebuilt
cache, i.e. the whole suite was green on the one path that could not fail. A
runtime test for it needs both `/dev/shm` and a cold cache
(`test_cold_forward_cache.py`, Linux only). This one is static, runs everywhere,
and fails the moment a statement is inserted into a pairing again.

The rule it encodes: releasing a mapped-or-malloc'd buffer EARLY is fine, but do
it where the buffer is finished with, not by reaching into this block.
"""
import os
import re

import pytest

C_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "c_src"))
FILES = ("LaueMatchingCPU.c", "LaueMatchingGPU.cu", "LaueMatchingGPUStream.cu")

# flag name -> the pointer its else-branch must free
PAIRINGS = {
    "orientsMapped": "orients",
    "outArrMapped": "outArr",
}


def _read(name):
    with open(os.path.join(C_SRC, name)) as f:
        return f.read()


def _strip_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _match_brace(src, i):
    """Index just past the `}` closing the `{` at src[i]."""
    assert src[i] == "{"
    d = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0:
                return j + 1
    raise AssertionError("unbalanced braces")


def _then_and_else(src, after):
    """Split an if-statement's then-branch and else-branch starting at `after`."""
    i = after
    while src[i].isspace():
        i += 1
    if src[i] == "{":
        end = _match_brace(src, i)
        then = src[i:end]
    else:
        end = src.index(";", i) + 1
        then = src[i:end]
    j = end
    while j < len(src) and src[j].isspace():
        j += 1
    if not src.startswith("else", j):
        return then, None, src[end:end + 120]
    j += 4
    while src[j].isspace():
        j += 1
    if src[j] == "{":
        els = src[j:_match_brace(src, j)]
    else:
        els = src[j:src.index(";", j) + 1]
    return then, els, ""


@pytest.mark.parametrize("fname", FILES)
def test_mapped_free_pairings_are_intact(fname):
    """Each `if (xMapped)` must munmap in its then-branch and free in its else.

    A BRACED if/else cannot be re-parented, so extra statements inside the
    braces are fine. The dangerous form is the braceless one 654957a broke: an
    inserted statement there silently moves the `else` to a different `if`. So
    the check is structural -- match the branches and look in them -- rather
    than "nothing may appear between the halves", which would forbid the safe
    braced form too.
    """
    src = _strip_comments(_read(fname))
    seen = 0
    for flag, ptr in PAIRINGS.items():
        for m in re.finditer(r"if\s*\(\s*%s\s*\)" % flag, src):
            seen += 1
            then, els, tail = _then_and_else(src, m.end())
            assert els is not None, (
                f"{fname}: `if ({flag})` has no `else`. If its `else` was "
                f"re-parented onto another `if`, that is the 654957a bug "
                f"exactly.\nAfter the then-branch: {tail!r}")
            assert "munmap" in then, (
                f"{fname}: `if ({flag})` no longer munmaps in its then-branch; "
                f"saw {then[:160]!r}")
            assert re.search(r"free\s*\(\s*%s\s*\)" % re.escape(ptr), els), (
                f"{fname}: the `else` of `if ({flag})` no longer frees {ptr}; "
                f"saw {els[:160]!r}")
            # A braceless then-branch must be the munmap alone -- that is the
            # form an insertion can break.
            if not then.lstrip().startswith("{"):
                assert then.count(";") == 1, (
                    f"{fname}: braceless `if ({flag})` then-branch has more "
                    f"than one statement, which is how the else gets "
                    f"re-parented. Brace it.\nSaw: {then!r}")
    assert seen >= 1, f"{fname}: no mapped/free pairing found at all"


def test_stream_releases_the_host_cache_inside_the_upload_branch():
    """The 12.2 GB host copy is freed where it becomes dead, not in cleanup.

    In unchunked mode the host `outArr` is read exactly once, by the cudaMemcpy
    that uploads it; the per-image re-upload is guarded by `!outArrOnGPU`.
    Releasing it there drops steady-state RSS 18.35 -> 6.99 GB (measured, 40
    frames, 100M cache, solutions and spots identical). Doing it by editing the
    end-of-main cleanup instead is the move that caused the SIGSEGV above.
    """
    src = _strip_comments(_read("LaueMatchingGPUStream.cu"))
    i = src.index("if (outArrBytes <= usableForOutArr)")
    b = src.index("{", i)
    branch = src[b:_match_brace(src, b)]
    assert "cudaMemcpy(d_outArr" in branch, "wrong branch located"
    assert "munmap(outArr" in branch and re.search(r"free\s*\(\s*outArr\s*\)", branch), (
        "the host forward cache is no longer released in the upload branch; "
        "it would sit resident for the daemon's whole life (12.2 GB at 100M)")
    assert re.search(r"outArr\s*=\s*NULL\s*;", branch), (
        "outArr must be NULLed after release so the end-of-main cleanup is a "
        "no-op via free(NULL) -- otherwise it is a double free")
    assert re.search(r"outArrMapped\s*=\s*0\s*;", branch), (
        "outArrMapped must be cleared, or cleanup will munmap(NULL)")


def test_the_release_is_guarded_by_the_unchunked_flag():
    """Freeing the host copy is only safe when the device holds all of it.

    In chunked mode the per-image path re-reads `outArr + offset * stride`, so a
    release there would be a use-after-free on every image.
    """
    src = _strip_comments(_read("LaueMatchingGPUStream.cu"))
    assert "if (!outArrOnGPU)" in src, (
        "the per-image chunk re-upload has lost its !outArrOnGPU guard; the "
        "host-copy release in the upload branch is only safe because of it")
