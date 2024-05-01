# LaueMatching

LaueMatching code to find orientations in Laue Diffraction Images.

To install, you would need to compile C-codes in src. Makefile is coming soon.

Python packages needed:
    - numpy
    - h5py
    - scipy
    - pillow
    - scikit-image
    - diplib
    - matplotlib

C-libraries needed:
    - nlopt
    - CUDA (for GPU version only)

TODO:
    - Example files to run indexing, one way is to give a list of grains and a config file. 
    - Streamline installation of NLOPT, then compilation.
