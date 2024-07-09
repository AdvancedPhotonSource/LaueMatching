//
// Copyright (c) 2023, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <nlopt.h>

#ifdef __linux__ 
#include <malloc.h>
#endif

#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define NrValsResults 2
#define MaxNHKLS 200000
#define CalcLength(x,y,z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
#define hc_keVnm 1.2398419739
#define EPS 1E-12

struct dataFit{
	double *image;
	int *hkls;
	int nhkls;
	int nrPxX;
	int nrPxY;
	double LatCOrig[6];
	double recip[3][3];
	double *outArrThis;
	int maxNrSpots;
	double rotTranspose[3][3];
	double pArr[3];
	double pxX;
	double pxY;
	double Elo;
	double Ehi;
};

#if defined(__CUDA_ARCH__)
#include<cuda.h>
size_t MaxNrSpots;
size_t nrPixels;
__global__
void compare(size_t nrPx, size_t nOr, size_t nrMaxSpots, double minInt, size_t minSps, uint16_t *oA, double *im, double *mA);
#endif