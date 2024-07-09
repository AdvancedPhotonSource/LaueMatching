# LaueMatching

LaueMatching code to find orientations in Laue Diffraction Images.

To install, you would need to compile C-codes in src.

Installation:
    
    git clone https://https://github.com/AdvancedPhotonSource/LaueMatching
    cd LaueMatching
    python install -r requirements.txt

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

For GPU Version, provide CUDA executable as follows:
    make NCC=PATH_TO_NCC lauegpu

For CPU Version:
    make lauecpu

TODO:

    - Example files to run indexing, one way is to 
      give a list of images and a config file. 
