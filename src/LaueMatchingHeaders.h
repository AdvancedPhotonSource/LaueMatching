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

inline void calcV(double LatC[6]);
inline void calcRecipArray(double Lat[6], int SpaceGroup, double recip[3][3]);
inline double calcOverlap(double *image, double euler[3], int *hkls, int nhkls, int nrPxX, int nrPxY,
	double recip[3][3], double *outArrThis, int maxNrSpots, double rotTranspose[3][3], double pArr[3], double pxX,
	double pxY, double Elo, double Ehi);
inline double problem_function(unsigned n, const double *x, double *grad, void* f_data_supplied);
inline double FitOrientation(double *image, double euler[3], int *hkls, int nhkls, int nrPxX, int nrPxY,
	double recip[3][3], double *outArrThis, int maxNrSpots, double rotTranspose[3][3], double pArr[3], double pxX, 
	double pxY, double Elo, double Ehi, double tol, double latc[6], double eulerFit[3], double latCUpd[6], double *minVal, int doCrystalFit);
inline int writeCalcOverlap(double *image, double euler[3], int *hkls, int nhkls, int nrPxX, int nrPxY,
	double recip[3][3], double *outArrThis, int maxNrSpots, double rotTranspose[3][3], double pArr[3], double pxX,
	double pxY, double Elo, double Ehi, FILE *ExtraInfo, int saveExtraInfo, int *simulNrSps);
inline void MatrixMultF(double m[3][3],double v[3],double r[3]);
inline double zeroOut(double val);
inline void MatrixMultF33(double m[3][3],double n[3][3],double res[3][3]);
inline void Euler2OrientMat(double Euler[3], double m_out[3][3]);
inline void OrientMat2Euler(double m[3][3],double Euler[3]);
inline void OrientMat2Quat33(double OM[3][3], double Quat[4]);
inline void OrientMat2Quat(double OrientMat[9], double Quat[4]);
inline double GetMisOrientation(double quat1[4], double quat2[4], int SGNr);
inline void BringDownToFundamentalRegion(double QuatIn[4], double QuatOut[4]);
inline int MakeSymmetries(int SGNr, double Sym[24][4]);
inline void QuaternionProduct(double q[4], double r[4], double Q[4]);
inline void normalizeQuat(double quat[4]);

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

