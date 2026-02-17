#!/usr/bin/env bash
#
# build.sh — Build LaueMatching using CMake
#
# Usage:
#   ./build.sh          # CPU only (default)
#   ./build.sh gpu      # CPU + GPU (requires CUDA)
#   ./build.sh clean    # Remove build directory
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Detect number of cores
if command -v nproc &>/dev/null; then
  NPROC=$(nproc)
elif command -v sysctl &>/dev/null; then
  NPROC=$(sysctl -n hw.ncpu)
else
  NPROC=4
fi

case "${1:-cpu}" in
  clean)
    echo "Removing ${BUILD_DIR} ..."
    rm -rf "${BUILD_DIR}"
    echo "Done."
    exit 0
    ;;
  gpu)
    USE_CUDA=ON
    ;;
  cpu|*)
    USE_CUDA=OFF
    ;;
esac

echo "=== LaueMatching Build ==="
echo "  CUDA:  ${USE_CUDA}"
echo "  Jobs:  ${NPROC}"
echo "  Build: ${BUILD_DIR}"
echo ""

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
  -DUSE_CUDA="${USE_CUDA}" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j "${NPROC}"

echo ""
echo "=== Build complete ==="
echo "Binaries are in: ${SCRIPT_DIR}/bin/"
ls -lh "${SCRIPT_DIR}/bin/"
