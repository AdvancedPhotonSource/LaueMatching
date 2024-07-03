#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

CC=gcc
NCC=nvcc
CFLAGS=-fPIC -g -ldl -lm -fgnu89-inline -O3 -w -Wall
current_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CFLAGSNLOPT=-I$(current_dir)/LIBS/NLOPT/include -L$(current_dir)/LIBS/NLOPT/lib -lnlopt
SRC=src/
BIN=bin/

all: predep help bindircheck lauecpu

predep:
	./getPackages.sh

help:
	@echo NLOPT compiler flags in $(CFLAGSNLOPT)
	@echo 

bindircheck:
	mkdir -p $(BIN)

lauecpu: $(SRC)LaueMatchingCPU.c
	$(CC) $(SRC)LaueMatchingCPU.c -o $(BIN)LaueMatchingCPU $(CFLAGS) $(CFLAGSNLOPT) -fopenmp

lauegpu: $(SRC)LaueMatchingGPU.cu
	$(NCC) $(SRC)LaueMatchingGPU.cu -o $(BIN)LaueMatchingCPU $(CFLAGS) $(CFLAGSNLOPT) -fopenmp
