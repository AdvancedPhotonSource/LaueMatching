#!/bin/bash

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

installDir=$(pwd)/LIBS
mkdir -p ${installDir}

if [ ! -d ${installDir}/NLOPT/include ]; then # Install NLOPT
    mkdir ${installDir}/NLOPT
    cd ${installDir}/NLOPT
    git clone https://github.com/stevengj/nlopt.git
    cd nlopt
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=${installDir}/NLOPT ..
    cmake --build . -j 8
	cmake --build . -j 8 --target install
fi

if [ ! -f $(pwd)/100MilOrients.bin ]; then # Download 100 Million Orientation File
    echo "Downloading orientation file. ~7GB, might take long."
    wget -O 100MilOrients.bin https://anl.box.com/shared/static/qhao454ub2nh5t89zymj1bhlxw1q4obu.bin
fi