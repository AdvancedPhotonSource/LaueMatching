#!/usr/bin/env bash
# Release a new version of laue-index.
#
# Usage:
#   ./release.sh <new_version>            # prepare locally only (default)
#   ./release.sh <new_version> --publish  # prepare + push + GitHub release + PyPI
#   ./release.sh <new_version> --dry-run  # prepare, but DON'T commit or tag
#
# Example:
#   ./release.sh 0.1.1 --publish
#
# Adapted from MIDAS packages/midas_dct_tt/release.sh. Two guards there were
# learned the hard way and are kept verbatim in spirit:
#   * the commit is pathspec-limited to the version files, so whatever happened
#     to be staged does not get swept into the bump commit (and, under
#     --publish, pushed and tagged with it);
#   * the tag -> package map in the publish workflow is checked BEFORE pushing,
#     because a tag it does not match builds nothing and silently publishes
#     nothing.

set -e

PKG_NAME="laue-index"
PKG_DIR_NAME="laue_index"
MAIN_BRANCH="main"

# --- Arg parsing ---
if [ -z "$1" ]; then
    echo "Usage: $0 <new_version> [--publish | --dry-run]"
    echo "  <new_version>    e.g. 0.1.1"
    echo "  --publish        push to GitHub + create release + upload to PyPI"
    echo "  --dry-run        prepare artifacts but don't commit/tag"
    exit 1
fi

NEW_VERSION="$1"
MODE="${2:-prepare}"

if [ "$MODE" != "prepare" ] && [ "$MODE" != "--publish" ] && [ "$MODE" != "--dry-run" ]; then
    echo "ERROR: unknown flag '$MODE'. Use --publish or --dry-run."
    exit 1
fi

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG_DIR"
TAG="${PKG_NAME}-v${NEW_VERSION}"
WORKFLOW="../../.github/workflows/python-packages.yml"

echo "=== Releasing ${PKG_NAME} v${NEW_VERSION} (mode: ${MODE}) ==="
echo

# --- 1. Safety checks ---
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$MAIN_BRANCH" ]; then
    echo "ERROR: not on ${MAIN_BRANCH} (on $CURRENT_BRANCH). Switch branches first."
    exit 1
fi

if ! git diff --quiet HEAD -- .; then
    echo "ERROR: uncommitted changes in packages/${PKG_DIR_NAME}/. Commit or stash first."
    git status -s -- .
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag $TAG already exists locally. Pick a different version or delete it:"
    echo "  git tag -d $TAG"
    exit 1
fi

if [ "$MODE" = "--publish" ] && git ls-remote --tags origin "$TAG" | grep -q "$TAG"; then
    echo "ERROR: tag $TAG already exists on origin. Pick a different version."
    exit 1
fi

# The publish job's tag -> package map is a hand-maintained elif chain; a tag it
# does not match produces a silent no-op build and nothing reaches PyPI.
if [ "$MODE" = "--publish" ] && ! grep -q "${PKG_NAME}-v\*" "$WORKFLOW"; then
    echo "ERROR: python-packages.yml has no branch for '${PKG_NAME}-v*'."
    echo "       The release would build nothing and publish nothing."
    exit 1
fi

# --- 2. Bump version ---
BACKUP_DIR="$(mktemp -d)"
cp pyproject.toml "$BACKUP_DIR/pyproject.toml"
cp "${PKG_DIR_NAME}/__init__.py" "$BACKUP_DIR/__init__.py"

restore_version() {
    cp "$BACKUP_DIR/pyproject.toml" pyproject.toml
    cp "$BACKUP_DIR/__init__.py" "${PKG_DIR_NAME}/__init__.py"
    echo "  Version restored to $(grep '^version = ' pyproject.toml | cut -d'"' -f2)."
}

echo "[1/6] Bumping version to ${NEW_VERSION}..."
sed -i.bak "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" "${PKG_DIR_NAME}/__init__.py"
rm -f pyproject.toml.bak "${PKG_DIR_NAME}/__init__.py.bak"

PYPROJ_VER=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
INIT_VER=$(grep '^__version__ = ' "${PKG_DIR_NAME}/__init__.py" | cut -d'"' -f2)
if [ "$PYPROJ_VER" != "$NEW_VERSION" ] || [ "$INIT_VER" != "$NEW_VERSION" ]; then
    echo "ERROR: version bump failed."
    restore_version
    exit 1
fi

# --- 3. Run tests ---
echo "[2/6] Running tests..."
# macOS conda envs ship duplicate libomp.dylib (numba + torch); SIGABRTs at
# import without this.
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q --tb=short || {
    echo "ERROR: tests failed. Aborting."
    restore_version
    exit 1
}

# --- 4. Build ---
echo "[3/6] Building package..."
rm -rf dist/ build/ ./*.egg-info/

if ! python -c "import build" 2>/dev/null; then
    echo "  Installing 'build' and 'twine'..."
    pip install --quiet build twine
fi

set -o pipefail
# SDIST ONLY. scikit-build-core compiles the C, so `python -m build` emits a
# PLATFORM wheel (cp312-cp312-linux_x86_64 on CI) and PyPI rejects bare
# linux_x86_64 -- only manylinux/musllinux are accepted. Publishing a platform
# wheel would also silently shadow the sdist for matching users and start a
# wheel matrix we have deliberately chosen not to maintain (see the MIDAS
# cibuildwheel post-mortem). midas-index, the model for this package, is
# sdist-only on PyPI for the same reason.
python -m build --sdist 2>&1 | tail -5
set +o pipefail

if [ ! -d dist ] || [ -z "$(ls -A dist 2>/dev/null)" ]; then
    echo "ERROR: build did not produce artifacts."
    restore_version
    exit 1
fi

# --- 4b. Leak check -------------------------------------------------------
# The unpublished research tree (report/, experiments/, ~6 GB) used to live
# INSIDE this package. It has been moved to research/ and git-excluded, and
# pyproject uses an explicit packages= list rather than find:, but verify the
# artifact itself rather than trusting either. Cheap; catches a whole class of
# disaster.
echo "[4/6] Checking the sdist for research leaks..."
SDIST=$(ls dist/*.tar.gz 2>/dev/null | head -1)
if [ -z "$SDIST" ]; then
    echo "ERROR: no sdist produced; cannot verify contents."
    restore_version
    exit 1
fi
LEAKS=$(tar -tzf "$SDIST" | grep -icE "(^|/)(report|experiments|research)/" || true)
if [ "$LEAKS" -ne 0 ]; then
    echo "ERROR: sdist contains ${LEAKS} path(s) under report/ experiments/ research/."
    echo "       Refusing to publish unpublished research. Offending entries:"
    tar -tzf "$SDIST" | grep -iE "(^|/)(report|experiments|research)/" | head -20
    restore_version
    exit 1
fi
SDIST_MB=$(( $(wc -c < "$SDIST") / 1048576 ))
if [ "$SDIST_MB" -gt 10 ]; then
    echo "ERROR: sdist is ${SDIST_MB} MB -- far larger than the ~1 MB expected."
    echo "       Something bulky got swept in. Inspect: tar -tzf $SDIST"
    restore_version
    exit 1
fi
echo "  sdist clean ($(tar -tzf "$SDIST" | wc -l | tr -d ' ') entries, ${SDIST_MB} MB)."

# --- 5. If dry-run, stop here ---
if [ "$MODE" = "--dry-run" ]; then
    echo
    echo "=== Dry run complete ==="
    ls -1 dist/
    echo
    echo "Reverting the version bump (a dry run must leave no trace):"
    restore_version
    exit 0
fi

# --- 6. Commit + tag ---
echo "[5/6] Committing version bump..."
# Pathspec-limited on purpose: without it, anything staged before running this
# gets swept into the bump commit -- and under --publish, pushed and tagged.
VERSION_FILES=(pyproject.toml "${PKG_DIR_NAME}/__init__.py")

git add -- "${VERSION_FILES[@]}"
if git diff --cached --quiet -- "${VERSION_FILES[@]}"; then
    echo "  Version was already at ${NEW_VERSION} on disk; skipping commit."
else
    git commit -m "${PKG_NAME}: bump version to ${NEW_VERSION}" -- "${VERSION_FILES[@]}"
fi

echo "[6/6] Tagging as ${TAG}..."
git tag -a "$TAG" -m "${PKG_NAME} v${NEW_VERSION}"

if [ "$MODE" = "--publish" ]; then
    echo "Pushing to GitHub..."
    git push origin "${MAIN_BRANCH}" --follow-tags

    # The GitHub release is created by the workflow, which attaches the SAME
    # sdist it publishes to PyPI. This script used to upload its own local
    # build here, so the two artifacts were built on different machines and
    # could differ -- 0.2.0's GitHub asset carried a stray 32 MB median.bin
    # from the working tree that PyPI's did not. The local build above stays,
    # as a pre-flight: it is what the leak and size checks inspect before a
    # tag exists to regret.
    echo "The workflow will publish to PyPI and attach that same sdist to the release."

    echo
    echo "=== Release prepared ==="
    echo "Watch: https://github.com/AdvancedPhotonSource/LaueMatching/actions"
    echo
    echo "When the workflow completes, verify FROM PyPI (not from CI status):"
    echo "  pip install -U ${PKG_NAME} && \\"
    echo "    python -c 'import ${PKG_DIR_NAME}; print(${PKG_DIR_NAME}.__version__)'"
    exit 0
fi

echo
echo "=== Release prepared locally ==="
ls -1 dist/
echo
echo "To publish:"
echo "  git push origin ${MAIN_BRANCH} --follow-tags"
echo
echo "That is the whole of it: the workflow builds the sdist from the tagged"
echo "checkout, publishes it to PyPI, and attaches THAT file to the release."
echo "Do not upload dist/ by hand -- it is a pre-flight for the checks above,"
echo "and hand-uploading is how a GitHub asset and a PyPI artifact drift apart."
echo
echo "Or re-run with --publish to do all of this automatically."
