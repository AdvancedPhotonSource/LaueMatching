#!/bin/bash

#
# Copyright (c) 2024, UChicago Argonne, LLC
# See LICENSE file.
# Hemant Sharma, hsharma@anl.gov
#

set -e  # Exit on error

# Parse arguments
USE_CUDA=ON
DOWNLOAD_ORIENTATION=ON
BUILD_TYPE="Release"
CORES=$(nproc)

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --no-cuda               Disable CUDA support"
    echo "  --no-orientation-file   Don't download the orientation file"
    echo "  --debug                 Build in Debug mode"
    echo "  --cores N               Use N cores for building (default: all available)"
    echo "  --help                  Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cuda)
            USE_CUDA=OFF
            shift
            ;;
        --no-orientation-file)
            DOWNLOAD_ORIENTATION=OFF
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo "=== LaueMatching Build Script ==="
echo "CUDA support: $USE_CUDA"
echo "Download orientation file: $DOWNLOAD_ORIENTATION"
echo "Build type: $BUILD_TYPE"
echo "============================"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake -DUSE_CUDA=$USE_CUDA \
      -DDOWNLOAD_ORIENTATION_FILE=$DOWNLOAD_ORIENTATION \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      ..

# Build
make

# Install Python requirements
echo "Installing Python requirements..."
pip install -r ../requirements.txt

echo "=== Build completed successfully ==="
echo "Executables are in: $(pwd)/bin"
echo ""
echo "To run an example, navigate to the simulation directory:"
echo "cd ../simulation"
echo "python RunExample.py"