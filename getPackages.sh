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