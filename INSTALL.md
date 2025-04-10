# Installing LaueMatching

LaueMatching is a software package to find orientations in Laue Diffraction Images. This guide provides instructions for installing LaueMatching using CMake.

## Prerequisites

Before installing, make sure you have the following installed on your system:

- CMake (≥ 3.15)
- C compiler (gcc recommended)
- CUDA toolkit (optional, for GPU acceleration)
- Python 3 with pip

## Building with CMake

### Step 1: Clone the repository

```bash
git clone https://github.com/AdvancedPhotonSource/LaueMatching
cd LaueMatching
```

### Step 2: Create a build directory

```bash
mkdir build
cd build
```

### Step 3: Configure and build

You can configure the build with various options:

For a standard build with CUDA support:

```bash
cmake ..
make
```

#### Build options:

If you don't want CUDA support:

```bash
cmake -DUSE_CUDA=OFF ..
make
```

If you want a custom CUDA compiler:

```bash
cmake -DCMAKE_CUDA_COMPILER=path_to_nvcc -DCMAKE_CUDA_COMPILER_FORCED=ON -DCMAKE_CUDA_ARCHITECTURES=90   ..
make
```

If you don't want to download the 7GB orientation file (for example, if you already have it):

```bash
cmake -DDOWNLOAD_ORIENTATION_FILE=OFF ..
make
```

If you want to use a system-installed NLopt library instead of building it:

```bash
cmake -DUSE_SYSTEM_NLOPT=ON ..
make
```

If you want to build without the shared library:

```bash
cmake -DBUILD_LIBRARY=OFF ..
make
```

You can also use the provided build script which offers a more user-friendly interface:

```bash
./build.sh                     # Default build with all features
./build.sh --no-cuda           # Build without CUDA support
./build.sh --no-orientation-file  # Skip downloading orientation file
./build.sh --help              # Show all options
```

### Step 4: Install Python dependencies

```bash
pip install -r ../requirements.txt
```

## Project Structure

The LaueMatching project is organized as follows:

- `src/` - Main executable source files (LaueMatchingCPU.c, LaueMatchingGPU.cu)
- `laue_matching_lib/` - Library source code
  - `src/` - Library implementation files
  - `include/` - Public header files
- `build/` - Build directory (created during build)
  - `bin/` - Compiled executables
  - `lib/` - Compiled libraries
- `cmake/` - CMake modules and scripts

## Running the software

After building, the executables will be available in the `build/bin` directory:

- `LaueMatchingCPU` - CPU version
- `LaueMatchingGPU` - GPU version (if CUDA was enabled)

### Command-line usage

Run the example:
```bash
cd ..
cd simulation
python RunExample.py
```

## Best Practices

- LaueMatching runs only on Linux computers. Windows support is not planned.
- The CPU code can be compiled to run on macOS, but requires `brew install gcc` to get OpenMP support.
- If you want to reduce initialization times, your `OrientationFile` and `ForwardFile` should be in `/dev/shm`. This allows for memory mapping and is millions of times faster.
- Look inside the **simulation** folder for instructions to run examples.

## Troubleshooting

- If you encounter issues with CUDA, make sure your CUDA toolkit version is compatible with your GPU.
- For memory errors, check if you have enough RAM to handle the orientation matrices (especially the 7GB orientation file).
- For any other issues, please contact the developer: hsharma@anl.gov

## Citation

If you use LaueMatching in your work, please cite:

```
Citation coming soon. For now, please cite as:
    LaueMatching, 2024. https://github.com/AdvancedPhotonSource/LaueMatching
```