# Contributing to LaueMatching

Thank you for your interest in contributing to LaueMatching! This document provides guidelines for contributing to this project.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/LaueMatching.git
   cd LaueMatching
   ```
3. **Build** the project:
   ```bash
   ./build.sh
   ```
4. Create a **feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- **C code**: Follow existing style — K&R braces, 4-space tabs, descriptive variable names.
- **Python code**: Follow PEP 8. Use type hints where practical.
- **Headers**: Shared declarations go in `LaueMatchingHeaders.h` to avoid duplication between CPU and GPU code.

### Commit Messages

Use clear, descriptive commit messages:

```
Short summary (50 chars or less)

Optional longer description explaining the motivation
for the change and any important details.
```

### Testing

- Test changes with the simulation dataset in `simulation/`.
- Verify both CPU (`./build.sh`) and GPU (`./build.sh gpu`) builds if modifying shared code.
- For algorithm changes, compare indexing results against a known-good baseline.

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](https://github.com/AdvancedPhotonSource/LaueMatching/issues/new?template=bug_report.md) issue template.
- Include your OS, compiler version, and build configuration.
- Attach sample data or parameter files if possible.

### Suggesting Features

- Use the [Feature Request](https://github.com/AdvancedPhotonSource/LaueMatching/issues/new?template=feature_request.md) issue template.
- Describe the use case and expected behavior.

### Submitting Pull Requests

1. Ensure your code builds cleanly with no new warnings (`-Wall -Wextra`).
2. Update documentation if adding new parameters or changing behavior.
3. Keep PRs focused — one feature or fix per PR.
4. Reference any related issues in your PR description.

## Project Architecture

| File | Purpose |
|------|---------|
| `src/LaueMatchingCPU.c` | CPU implementation (OpenMP parallelized) |
| `src/LaueMatchingGPU.cu` | GPU implementation (CUDA) |
| `src/LaueMatchingHeaders.h` | Shared structs, constants, and utility functions |
| `RunImage.py` | End-to-end Python pipeline |
| `GenerateHKLs.py` | HKL generation for a given crystal structure |
| `GenerateSimulation.py` | Synthetic Laue pattern generator |
| `ImageCleanup.py` | Image preprocessing utilities |

## Contact

If you have questions about contributing, please contact [Hemant Sharma](mailto:hsharma@anl.gov).
