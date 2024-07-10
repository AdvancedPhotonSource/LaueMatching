#
# Copyright (c) 2024, UChicago Argonne, LLC
# See LICENSE file.
# Hemant Sharma, hsharma@anl.gov
#

CC=gcc
NCC=$${HOME}/opt/localcuda/bin/nvcc
CFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w -Wall -fopenmp
current_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CFLAGSNLOPT=-I$(current_dir)/LIBS/NLOPT/include -L$(current_dir)/LIBS/NLOPT/lib -L$(current_dir)/LIBS/NLOPT/lib64 -lnlopt
NCFLAGS=-lgomp -O3 -G -g -w -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_86,code=sm_86 \
	-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_80,code=sm_80 -lm -rdc=true -Xcompiler=-fopenmp
SRC=src/
BIN=bin/

all: predep help bindircheck lauecpu lauegpu

predep:
	./getPackages.sh

help:
	@echo NLOPT compiler flags in $(CFLAGSNLOPT)

bindircheck:
	mkdir -p $(BIN)

lauecpu: predep help bindircheck $(SRC)LaueMatchingCPU.c
	$(CC) $(SRC)LaueMatchingCPU.c -o $(BIN)LaueMatchingCPU $(CFLAGS) $(CFLAGSNLOPT)

lauegpu: predep help bindircheck $(SRC)LaueMatchingGPU.cu
	$(NCC) $(SRC)LaueMatchingGPU.cu -o $(BIN)LaueMatchingGPU $(NCFLAGS) $(CFLAGSNLOPT)
