#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

CC=gcc
NCC=$${HOME}/opt/localcuda/bin/nvcc
CFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w -Wall
current_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CFLAGSNLOPT=-I$(current_dir)/LIBS/NLOPT/include -L$(current_dir)/LIBS/NLOPT/lib -L$(current_dir)/LIBS/NLOPT/lib64 -lnlopt
NCFLAGS=-O3 -G -g -w -arch sm_90 -gencode=arch=compute_90,code=sm_90 -lm -rdc=true
SRC=src/
BIN=bin/

all: predep help bindircheck lauecpu lauegpu

predep:
	./getPackages.sh

help:
	@echo NLOPT compiler flags in $(CFLAGSNLOPT)
	@echo 

bindircheck:
	mkdir -p $(BIN)

lauecpu: predep help bindircheck $(SRC)LaueMatchingCPU.c
	$(CC) $(SRC)LaueMatchingCPU.c -o $(BIN)LaueMatchingCPU $(CFLAGS) $(CFLAGSNLOPT) -fopenmp

lauegpu: predep help bindircheck $(SRC)LaueMatchingGPU.cu
	$(NCC) $(SRC)LaueMatchingGPU.cu -o $(BIN)LaueMatchingGPU $(NCFLAGS)
