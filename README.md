# LaueMatching

LaueMatching code to find orientations in Laue Diffraction Images.

To install, you would need to compile C-codes in src.

Installation:
    
    git clone https://github.com/AdvancedPhotonSource/LaueMatching
    cd LaueMatching
    pip install -r requirements.txt

Python packages needed:

    - numpy
    - h5py
    - scipy
    - pillow
    - scikit-image
    - diplib
    - matplotlib

C-libraries downloaded:

    - nlopt

**NOTE:** It will download a ~7GB file once when installing. This file consists of a list of 100 million orientation matrices. These were used in the paper.

For GPU Version, provide CUDA executable as follows:

    make NCC=PATH_TO_NVCC lauegpu

For CPU Version:

    make lauecpu

To make both:

    make all

**simulation** folder consists of files needed to generate a simulation and run indexing.

## Best Practices:

> LaueMatching runs only on Linux computers for now.

> If you want to reduce initialization times (from a few seconds per image to a few microseconds), your `OrientationFile` and `ForwardFile` should be in `/dev/shm`. You can either just provide the path to these files and `LaueMatching` will generate these, or copy already generated files to `/dev/shm`. Any files on `/dev/shm` can be directly memory mapped instead of being read, and this process is millions of times faster. This is particularly useful for multiple app calls using the same mapping files, but different images. ***Note:*** This will not work on MAC.

> The CPU code can be compiled to run on MACOS, but need to `brew install gcc` to get `OpenMP` and possibly other dependencies.

> Windows support is not planned.

> Look inside **simulation** folder for instructions to run the example.

# CITATION

If you use `LaueMatching` in your work, please cite the paper:

    Citation coming soon. For now, please cite as:

        LaueMatching, 2024. https://github.com/AdvancedPhotonSource/LaueMatching