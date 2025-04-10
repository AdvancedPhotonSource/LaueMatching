# LaueMatching

LaueMatching is a software package to find orientations in Laue Diffraction Images.

## Overview

This project provides tools for indexing and matching orientations in Laue diffraction patterns. It includes both CPU and GPU implementations for maximum performance, as well as a reusable library component.

## Features

- Fast indexing of Laue diffraction patterns
- CPU and GPU implementations for optimal performance
- Shared library for integration with other applications
- Automatic orientation matching algorithms
- Support for various crystal symmetries
- Integration with Python for data processing and visualization

## Project Structure

- `src/` - Main executable source files
- `laue_matching_lib/` - Reusable library code
  - `src/` - Library implementation files
  - `include/` - Public header files
- `build/` - Build directory (created during build)
  - `bin/` - Compiled executables
  - `lib/` - Compiled libraries
- `simulation/` - Example data and configuration files

## Installation

LaueMatching can be installed using multiple methods:

### Using CMake (Recommended)

See the detailed [installation guide](INSTALL.md) for CMake-based installation.

```bash
# Quick start with CMake
git clone https://github.com/AdvancedPhotonSource/LaueMatching
cd LaueMatching
mkdir build && cd build
cmake ..
make -j8
```

### Using the Build Script

We provide a convenient build script that handles the compilation process:

```bash
./build.sh
```

Run `./build.sh --help` to see all available options.

### Using Docker

We also provide Docker support for easy deployment:

```bash
# Build and run the Docker container
docker-compose up -d
docker exec -it lauematching bash
```

## Build Options

LaueMatching provides several build options:

- `USE_CUDA=ON/OFF` - Enable/disable CUDA support (default: ON)
- `DOWNLOAD_ORIENTATION_FILE=ON/OFF` - Automatically download the orientation file if needed (default: ON)
- `USE_SYSTEM_NLOPT=ON/OFF` - Use system-installed NLopt instead of building from source (default: OFF)
- `BUILD_LIBRARY=ON/OFF` - Build the shared library component (default: ON)

Example:
```bash
cmake -DUSE_CUDA=OFF -DBUILD_LIBRARY=ON ..
```

## Python Requirements

The following Python packages are required:

- numpy
- h5py
- scipy
- pillow
- scikit-image
- diplib
- matplotlib
- opencv-python-headless

You can install all requirements with:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

The basic workflow involves:

1. Preparing your data
2. Running the LaueMatching algorithms
3. Analyzing the results

```bash
# Example for CPU version
./bin/LaueMatchingCPU config_file.txt orientations.bin hkls.txt image.bin 8

# Example for GPU version
./bin/LaueMatchingGPU config_file.txt orientations.bin hkls.txt image.bin 8
```

### Python Interface

```python
# Example Python usage
import os
import subprocess

# Setup parameters
config_file = "config.txt"
image_file = "laue_image.h5"
n_cpus = 8
compute_type = "GPU"  # or "CPU"

# Run the analysis
subprocess.call(f"python RunImage.py -configFile {config_file} -imageFile {image_file} -nCPUs {n_cpus} -computeType {compute_type}", shell=True)
```

### Library Usage

If you've built the library component, you can link against it in your own C applications:

```c
#include <laue_matching/laue_matching.h>

// Initialize the library
laue_init();

// Use library functions
laue_process_image(...);

// Cleanup
laue_cleanup();
```

Compile and link with:
```bash
gcc -o myapp myapp.c -I/path/to/laue_matching/include -L/path/to/laue_matching/lib -llaue_matching
```

## Examples

Example configurations and test data can be found in the `simulation` directory.

```bash
cd simulation
python RunExample.py
```

## Best Practices

- LaueMatching runs only on Linux computers. Windows support is not planned.
- The CPU code can be compiled to run on macOS, but requires `brew install gcc` to get OpenMP support.
- If you want to reduce initialization times, your `OrientationFile` and `ForwardFile` should be in `/dev/shm`. This allows for memory mapping and is millions of times faster.

## Troubleshooting

Common issues:

- **Memory errors**: Make sure you have enough RAM to handle the orientation matrices (≥8GB recommended).
- **CUDA errors**: Verify your CUDA toolkit is compatible with your GPU.
- **Missing Dependencies**: Check that all required libraries are installed.
- **Library Loading Issues**: Ensure the library path is correctly set (e.g., `export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH`).

For more help, contact the developer: hsharma@anl.gov

## Citation

If you use LaueMatching in your work, please cite:

```
Citation coming soon. For now, please cite as:
    LaueMatching, 2024. https://github.com/AdvancedPhotonSource/LaueMatching
```

## License

See the LICENSE file for details.

## Contact

Hemant Sharma (hsharma@anl.gov)