# LaueMatching v1.0

LaueMatching is a software package for indexing orientations in Laue diffraction images.

## Version History

### v1.0 (2026-02-17)
- **Code Refactor**: Consolidated ~700 lines of duplicated code into shared `LaueMatchingHeaders.h`.
- **Bug Fixes**:
  - Fixed c/a ratio fitting (was integer division `1/3`).
  - Fixed negative pixel handling (uint16_t underflow).
  - Fixed trigonal symmetry definition (consistent between CPU/GPU).
  - Fixed memory leaks and file descriptor handling.
  - Fixed GPU unique-solution indexing bug.
- **Build System**: Improved CMake configuration with working strict warning flags.

## Features

- Fast indexing of Laue diffraction patterns from polychromatic X-ray data
- CPU (OpenMP) and GPU (CUDA) implementations
- Orientation matching with configurable tolerance and crystal symmetry support
- Optional lattice-parameter refinement (including c/a ratio fitting)
- Python utilities for HKL generation, image cleanup, simulation, and analysis

## Project Structure

```
├── src/                     # C / CUDA source code
│   ├── LaueMatchingCPU.c    # CPU implementation (OpenMP)
│   ├── LaueMatchingGPU.cu   # GPU implementation (CUDA)
│   └── LaueMatchingHeaders.h
├── bin/                     # Compiled binaries (created by build)
├── LIBS/NLOPT/              # NLopt dependency (auto-downloaded)
├── simulation/              # Example data and parameter files
├── GenerateHKLs.py          # Generate valid HKL list for a crystal
├── GenerateSimulation.py    # Create synthetic Laue patterns
├── ImageCleanup.py          # Pre-process raw detector images
├── RunImage.py              # End-to-end indexing pipeline
├── CMakeLists.txt           # CMake build system
├── build.sh                 # Convenience build script
├── Makefile                 # Legacy Makefile (deprecated)
└── 100MilOrients.bin        # Pre-computed candidate orientations (~6.7 GB)
```

## Prerequisites

- **C compiler** with C99 support (GCC recommended)
- **CMake** ≥ 3.18
- **OpenMP** (usually bundled with GCC; on macOS use `brew install gcc`)
- **CUDA toolkit** (optional, only needed for the GPU build)
- **Python 3** with packages listed in `requirements.txt`

## Building

### Quick Start (CPU Only)

```bash
./build.sh
```

### Using CMake Directly

```bash
mkdir -p build && cd build
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

NLOPT is automatically downloaded and built into `LIBS/NLOPT/` if not already present.

### GPU Build (Requires CUDA)

```bash
./build.sh gpu
```

Or manually:
```bash
mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Default CUDA architectures: sm_70, sm_80, sm_86, sm_90. Override with `-DCMAKE_CUDA_ARCHITECTURES="80;90"`.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CUDA` | `OFF` | Build the GPU executable |
| `BUILD_OMP` | `ON` | Enable OpenMP parallelism |

### Clean Build

```bash
./build.sh clean
```

## Python Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

LaueMatching requires five positional arguments:

```bash
./bin/LaueMatchingCPU \
    parameterFile.txt \
    orientations.bin \
    valid_hkls.csv \
    image.bin \
    nCPUs
```

| Argument | Description |
|----------|-------------|
| `parameterFile` | Text file with experiment geometry and matching parameters |
| `orientations.bin` | Binary file of candidate orientation matrices (doubles) |
| `valid_hkls.csv` | Space-separated HKL list (preferably sorted by structure factor) |
| `image.bin` | Binary detector image (doubles) |
| `nCPUs` | Number of OpenMP threads |

### Key Parameter File Settings

| Parameter | Description |
|-----------|-------------|
| `LatticeParameter` | a, b, c (nm), α, β, γ (°) |
| `SpaceGroup` | Space group number (1–230) |
| `Elo`, `Ehi` | Energy range (keV) for spot simulation |
| `MaxNrLaueSpots` | Max spots per orientation |
| `MinNrSpots` | Minimum matching spots to qualify a grain |
| `MinIntensity` | Minimum total intensity threshold |
| `MaxAngle` | Misorientation tolerance (°) for merging candidates |

See `simulation/params_sim.txt` for a complete example.

## Examples

Example data and parameter files are in `simulation/`:

```bash
cd simulation
cat README.md    # Usage instructions for the example dataset
```

## Best Practices

- **Linux** is the primary platform. macOS CPU builds work with `brew install gcc`.
- Place `OrientationFile` and `ForwardFile` in `/dev/shm` (tmpfs) for dramatically faster memory-mapped I/O.
- Ensure ≥ 8 GB RAM for the full 100-million orientation file.

## Citation

```bibtex
@article{LaueMatching,
  author  = {Sharma, Hemant and Sheyfer, Dina and Harder, Ross and Tischler, Jonathan Z.},
  title   = {LaueMatching: A Tool for rapid and robust indexing of Laue diffraction patterns},
  year    = {2026; in print},
  journal = {Journal of Applied Crystallography},
  url     = {https://github.com/AdvancedPhotonSource/LaueMatching}
}
```

## License

See the [LICENSE](LICENSE) file for details.


## Workflow

```mermaid
graph TD
    A[Parameter File] --> LM(LaueMatching Binary)
    B[Orientation List] --> LM
    C[Valid HKLs] --> LM
    D[Detector Image] --> LM
    
    LM --> E{Found Forward Simulation?}
    E -- No --> F[Run Forward Simulation]
    E -- Yes --> G[Load Forward Simulation]
    F --> G
    
    G --> H[Parallel Matching]
    H --> I[Unique Solutions]
    I --> J[Refinement (Nelder-Mead)]
    J --> K[Output: .solutions.txt]
```

## Contact

Hemant Sharma — hsharma@anl.gov