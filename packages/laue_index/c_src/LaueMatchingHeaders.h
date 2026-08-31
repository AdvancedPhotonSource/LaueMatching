//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//
// Shared header for LaueMatchingCPU and LaueMatchingGPU.
// All symmetry tables, math utilities, and diffraction functions live
// here so they are compiled once per translation unit and stay in sync.
//

#ifndef LAUE_MATCHING_HEADERS_H
#define LAUE_MATCHING_HEADERS_H

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include "nelder_mead.h"
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <malloc.h>
#endif

// ── Constants ───────────────────────────────────────────────────────────
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define NrValsResults 2
#define MaxNHKLS 200000

// ── Forward-simulation cache ────────────────────────────────────────────
// Can an existing cache be used? ONE implementation for all three binaries.
//
// This check used to live only in LaueMatchingCPU.c. The CUDA binaries mapped
// whatever sat at the path, so a 0-byte leftover -- which the C itself creates
// with O_CREAT when a run is interrupted mid-write -- made LaueMatchingGPU
// print "file was found. Will not do forward simulation." and then take SIGBUS
// on first touch, 12.2 GB of mapping backed by an empty file. Three copies of
// a check is how one of them goes missing; hence one function, three callers.
//
// The size IS the contract: entry orientNr occupies (1 + 2*maxNrSpots) uint16
// at a fixed offset, so a file of the right length is structurally valid and a
// file of any other length cannot be.
//
// It does NOT prove the cache came from THIS geometry. A right-sized cache
// built with a different detector, energy range or HKL list is accepted and is
// silently wrong -- the same limit the CPU check always had. Regenerate on any
// doubt; the cost is minutes, and a wrong cache is not visibly wrong.
static inline bool forwardCacheUsable(const char *outfn, size_t nrOrients,
                                      int maxNrSpots) {
  if (outfn == NULL || outfn[0] == '\0') {
    // Don't blindly open a blank/default path and pick up a stale cache.
    printf("No ForwardFile specified. Running in simulation mode!\n");
    return false;
  }
  printf("Trying to see if the forward simulation exists. Looking for %s "
         "file.\n",
         outfn);
  int fd = open(outfn, O_RDONLY);
  if (fd < 0) { // open returns -1 on error, never 0
    printf("Could not read the forward simulation file (%s). Running in "
           "simulation mode!\n",
           strerror(errno));
    return false;
  }
  const size_t expected =
      nrOrients * (size_t)(1 + 2 * maxNrSpots) * sizeof(uint16_t);
  struct stat cst;
  if (fstat(fd, &cst) != 0) {
    printf("Could not stat the forward cache %s (%s). Running in simulation "
           "mode!\n",
           outfn, strerror(errno));
    close(fd);
    return false;
  }
  if ((size_t)cst.st_size != expected) {
    printf("Forward cache %s is %lld B, expected %zu B for %zu orientations x "
           "(1+2*%d) uint16. Ignoring it and running in simulation mode.\n",
           outfn, (long long)cst.st_size, expected, nrOrients, maxNrSpots);
    close(fd);
    return false;
  }
  close(fd);
  printf("%s was found and is the expected size. Will not do forward "
         "simulation.\n",
         outfn);
  return true;
}

static inline double CalcLength(double x, double y, double z) {
  return sqrt(x * x + y * y + z * z);
}

#define hc_keVnm 1.2398419739
#define EPS 1E-12

// ── Global state (defined once per translation unit) ────────────────────
// Each .c / .cu file that includes this header must provide one definition
// of these variables.  We declare them here so all shared functions can
// reference them.
extern double tol_LatC[6];
extern double tol_c_over_a;
extern double c_over_a_orig;
extern int sg_num;
extern double cellVol;
extern double phiVol;
extern int nSym;
extern double Symm[24][4];
// Retained only so an existing `Optimizer BOBYQA` line in a parameter file
// still parses. BOBYQA is gone -- Nelder-Mead is used unconditionally, and
// measured BETTER on this objective (see FitOrientation). Requesting BOBYQA
// now prints a notice and proceeds with Nelder-Mead.
extern int useBobyqa; // vestigial: reported, never selects an algorithm

// ── Optimization data bundle ────────────────────────────────────────────
struct dataFit {
  float *image;
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
  int *validHKLIdx; // prefiltered HKL indices (NULL = iterate all)
  int nValidHKL;    // number of valid HKLs
  // A predicted reflection counts as MATCHED when the pixel under it exceeds
  // this. 0.0 reproduces the historical `> 0` exactly, so the default is a
  // no-op; see MinSpotIntensity in the parameter file.
  double minSpotIntensity;
};

// ── Symmetry tables ─────────────────────────────────────────────────────
// `static` gives each TU its own copy, avoiding multiple-definition
// linker errors while keeping the data in the header.

static double TricSym[2][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {1.00000, 0.00000, 0.00000, 0.00000}};

static double MonoSym[2][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.00000, 1.00000, 0.00000, 0.00000}};

static double OrtSym[4][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                              {1.00000, 1.00000, 0.00000, 0.00000},
                              {0.00000, 0.00000, 1.00000, 0.00000},
                              {0.00000, 0.00000, 0.00000, 1.00000}};

static double TetSym[8][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                              {0.70711, 0.00000, 0.00000, 0.70711},
                              {0.00000, 0.00000, 0.00000, 1.00000},
                              {0.70711, -0.00000, -0.00000, -0.70711},
                              {0.00000, 1.00000, 0.00000, 0.00000},
                              {0.00000, 0.00000, 1.00000, 0.00000},
                              {0.00000, 0.70711, 0.70711, 0.00000},
                              {0.00000, -0.70711, 0.70711, 0.00000}};

// Correct TrigSym (from CPU code — the GPU file previously had a stale copy)
static double TrigSym[6][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.00000, 0.86603, -0.50000, 0.00000},
                               {0.50000, 0.00000, 0.00000, 0.86603},
                               {0.00000, 0.00000, 1.00000, 0.00000},
                               {0.50000, -0.00000, -0.00000, -0.86603},
                               {0.00000, 0.86603, 0.50000, 0.00000}};

static double HexSym[12][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.86603, 0.00000, 0.00000, 0.50000},
                               {0.50000, 0.00000, 0.00000, 0.86603},
                               {0.00000, 0.00000, 0.00000, 1.00000},
                               {0.50000, -0.00000, -0.00000, -0.86603},
                               {0.86603, -0.00000, -0.00000, -0.50000},
                               {0.00000, 1.00000, 0.00000, 0.00000},
                               {0.00000, 0.86603, 0.50000, 0.00000},
                               {0.00000, 0.50000, 0.86603, 0.00000},
                               {0.00000, 0.00000, 1.00000, 0.00000},
                               {0.00000, -0.50000, 0.86603, 0.00000},
                               {0.00000, -0.86603, 0.50000, 0.00000}};

static double CubSym[24][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.70711, 0.70711, 0.00000, 0.00000},
                               {0.00000, 1.00000, 0.00000, 0.00000},
                               {0.70711, -0.70711, 0.00000, 0.00000},
                               {0.70711, 0.00000, 0.70711, 0.00000},
                               {0.00000, 0.00000, 1.00000, 0.00000},
                               {0.70711, 0.00000, -0.70711, 0.00000},
                               {0.70711, 0.00000, 0.00000, 0.70711},
                               {0.00000, 0.00000, 0.00000, 1.00000},
                               {0.70711, 0.00000, 0.00000, -0.70711},
                               {0.50000, 0.50000, 0.50000, 0.50000},
                               {0.50000, -0.50000, -0.50000, -0.50000},
                               {0.50000, -0.50000, 0.50000, 0.50000},
                               {0.50000, 0.50000, -0.50000, -0.50000},
                               {0.50000, 0.50000, -0.50000, 0.50000},
                               {0.50000, -0.50000, 0.50000, -0.50000},
                               {0.50000, -0.50000, -0.50000, 0.50000},
                               {0.50000, 0.50000, 0.50000, -0.50000},
                               {0.00000, 0.70711, 0.70711, 0.00000},
                               {0.00000, -0.70711, 0.70711, 0.00000},
                               {0.00000, 0.70711, 0.00000, 0.70711},
                               {0.00000, 0.70711, 0.00000, -0.70711},
                               {0.00000, 0.00000, 0.70711, 0.70711},
                               {0.00000, 0.00000, 0.70711, -0.70711}};

// ── Utility functions ───────────────────────────────────────────────────
// `static inline` is the correct C99 way to define functions in a header
// that is included in multiple translation units.

static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

static inline void normalizeQuat(double quat[4]) {
  double norm = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] +
                     quat[3] * quat[3]);
  quat[0] /= norm;
  quat[1] /= norm;
  quat[2] /= norm;
  quat[3] /= norm;
}

static inline void QuaternionProduct(double q[4], double r[4], double Q[4]) {
  Q[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3];
  Q[1] = r[1] * q[0] + r[0] * q[1] + r[3] * q[2] - r[2] * q[3];
  Q[2] = r[2] * q[0] + r[0] * q[2] + r[1] * q[3] - r[3] * q[1];
  Q[3] = r[3] * q[0] + r[0] * q[3] + r[2] * q[1] - r[1] * q[2];
  if (Q[0] < 0) {
    Q[0] = -Q[0];
    Q[1] = -Q[1];
    Q[2] = -Q[2];
    Q[3] = -Q[3];
  }
  normalizeQuat(Q);
}

static inline int MakeSymmetries(int SGNr, double Sym[24][4]) {
  int i, j, NrSymmetries;
  if (SGNr <= 2) {
    NrSymmetries = 1;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = TricSym[i][j];
  } else if (SGNr <= 15) {
    NrSymmetries = 2;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = MonoSym[i][j];
  } else if (SGNr <= 74) {
    NrSymmetries = 4;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = OrtSym[i][j];
  } else if (SGNr <= 142) {
    NrSymmetries = 8;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = TetSym[i][j];
  } else if (SGNr <= 167) {
    NrSymmetries = 6;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = TrigSym[i][j];
  } else if (SGNr <= 194) {
    NrSymmetries = 12;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = HexSym[i][j];
  } else {
    NrSymmetries = 24;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = CubSym[i][j];
  }
  return NrSymmetries;
}

static inline void BringDownToFundamentalRegion(double QuatIn[4],
                                                double QuatOut[4]) {
  int i, maxCosRowNr = 0;
  double qps[24][4], q2[4], qt[4], maxCos = -10000;
  for (i = 0; i < nSym; i++) {
    q2[0] = Symm[i][0];
    q2[1] = Symm[i][1];
    q2[2] = Symm[i][2];
    q2[3] = Symm[i][3];
    QuaternionProduct(QuatIn, q2, qt);
    qps[i][0] = qt[0];
    qps[i][1] = qt[1];
    qps[i][2] = qt[2];
    qps[i][3] = qt[3];
    if (maxCos < qt[0]) {
      maxCos = qt[0];
      maxCosRowNr = i;
    }
  }
  QuatOut[0] = qps[maxCosRowNr][0];
  QuatOut[1] = qps[maxCosRowNr][1];
  QuatOut[2] = qps[maxCosRowNr][2];
  QuatOut[3] = qps[maxCosRowNr][3];
  normalizeQuat(QuatOut);
}

static inline double GetMisOrientation(double quat1[4], double quat2[4]) {
  double q1FR[4], q2FR[4], q1Inv[4], QP[4], MisV[4];
  BringDownToFundamentalRegion(quat1, q1FR);
  BringDownToFundamentalRegion(quat2, q2FR);
  q1Inv[0] = -q1FR[0];
  q1Inv[1] = q1FR[1];
  q1Inv[2] = q1FR[2];
  q1Inv[3] = q1FR[3];
  QuaternionProduct(q1Inv, q2FR, QP);
  BringDownToFundamentalRegion(QP, MisV);
  if (MisV[0] > 1)
    MisV[0] = 1;
  return 2 * (acos(MisV[0])) * rad2deg;
}

static inline void OrientMat2Quat(double OrientMat[9], double Quat[4]) {
  double trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
  if (trace > 0) {
    double s = 0.5 / sqrt(trace + 1.0);
    Quat[0] = 0.25 / s;
    Quat[1] = (OrientMat[7] - OrientMat[5]) * s;
    Quat[2] = (OrientMat[2] - OrientMat[6]) * s;
    Quat[3] = (OrientMat[3] - OrientMat[1]) * s;
  } else {
    if (OrientMat[0] > OrientMat[4] && OrientMat[0] > OrientMat[8]) {
      double s = 2.0 * sqrt(1.0 + OrientMat[0] - OrientMat[4] - OrientMat[8]);
      Quat[0] = (OrientMat[7] - OrientMat[5]) / s;
      Quat[1] = 0.25 * s;
      Quat[2] = (OrientMat[1] + OrientMat[3]) / s;
      Quat[3] = (OrientMat[2] + OrientMat[6]) / s;
    } else if (OrientMat[4] > OrientMat[8]) {
      double s = 2.0 * sqrt(1.0 + OrientMat[4] - OrientMat[0] - OrientMat[8]);
      Quat[0] = (OrientMat[2] - OrientMat[6]) / s;
      Quat[1] = (OrientMat[1] + OrientMat[3]) / s;
      Quat[2] = 0.25 * s;
      Quat[3] = (OrientMat[5] + OrientMat[7]) / s;
    } else {
      double s = 2.0 * sqrt(1.0 + OrientMat[8] - OrientMat[0] - OrientMat[4]);
      Quat[0] = (OrientMat[3] - OrientMat[1]) / s;
      Quat[1] = (OrientMat[2] + OrientMat[6]) / s;
      Quat[2] = (OrientMat[5] + OrientMat[7]) / s;
      Quat[3] = 0.25 * s;
    }
  }
  if (Quat[0] < 0) {
    Quat[0] = -Quat[0];
    Quat[1] = -Quat[1];
    Quat[2] = -Quat[2];
    Quat[3] = -Quat[3];
  }
  normalizeQuat(Quat);
}

static inline void OrientMat2Quat33(double OM[3][3], double Quat[4]) {
  double OrientMat[9];
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      OrientMat[i * 3 + j] = OM[i][j];
  OrientMat2Quat(OrientMat, Quat);
}

static inline void OrientMat2Euler(double m[3][3], double Euler[3]) {
  double psi, phi, theta, sph;
  if (fabs(m[2][2] - 1.0) < EPS) {
    phi = 0;
  } else {
    phi = acos(m[2][2]);
  }
  sph = sin(phi);
  if (fabs(sph) < EPS) {
    psi = 0.0;
    theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0])
                                        : sin_cos_to_angle(-m[1][0], m[0][0]);
  } else {
    psi = (fabs(-m[1][2] / sph) <= 1.0)
              ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph)
              : sin_cos_to_angle(m[0][2] / sph, 1);
    theta = (fabs(m[2][1] / sph) <= 1.0)
                ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph)
                : sin_cos_to_angle(m[2][0] / sph, 1);
  }
  Euler[0] = psi;
  Euler[1] = phi;
  Euler[2] = theta;
}

static inline void Euler2OrientMat(double Euler[3], double m_out[3][3]) {
  double psi = Euler[0], phi = Euler[1], theta = Euler[2];
  double cps = cos(psi), cph = cos(phi), cth = cos(theta);
  double sps = sin(psi), sph = sin(phi), sth = sin(theta);
  m_out[0][0] = cth * cps - sth * cph * sps;
  m_out[0][1] = -cth * cph * sps - sth * cps;
  m_out[0][2] = sph * sps;
  m_out[1][0] = cth * sps + sth * cph * cps;
  m_out[1][1] = cth * cph * cps - sth * sps;
  m_out[1][2] = -sph * cps;
  m_out[2][0] = sth * sph;
  m_out[2][1] = cth * sph;
  m_out[2][2] = cph;
}

static inline void MatrixMultF33(double m[3][3], double n[3][3],
                                 double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

static inline double zeroOut(double val) { return (fabs(val) < EPS) ? 0 : val; }

static inline void MatrixMultF(double m[3][3], double v[3], double r[3]) {
  int i, j;
  r[0] = 0;
  r[1] = 0;
  r[2] = 0;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      r[i] += m[i][j] * v[j];
}

static inline void calcV(double LatC[6]) {
  double ca = cos(LatC[3] * deg2rad);
  double cb = cos(LatC[4] * deg2rad);
  double cg = cos(LatC[5] * deg2rad);
  phiVol = sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);
  cellVol = LatC[0] * LatC[1] * LatC[2] * phiVol;
}

static inline void calcRecipArray(double Lat[6], int SpaceGroup,
                                  double recip[3][3]) {
  double a = Lat[0], b = Lat[1], c = Lat[2];
  double alpha = Lat[3], beta = Lat[4], gamma = Lat[5];
  int rhomb = 0;
  if (SpaceGroup == 146 || SpaceGroup == 148 || SpaceGroup == 155 ||
      SpaceGroup == 160 || SpaceGroup == 161 || SpaceGroup == 166 ||
      SpaceGroup == 167)
    rhomb = 1;
  double ca = cos(alpha * deg2rad);
  double cb = cos(beta * deg2rad);
  double cg = cos(gamma * deg2rad);
  double sg = sin(gamma * deg2rad);
  double phi = sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);
  double Vc = a * b * c * phi;
  double pv = (2 * M_PI) / Vc;
  double a0, a1, a2, b0, b1, b2, c0, c1, c2;
  if (rhomb == 0) {
    a0 = a;
    a1 = 0.0;
    a2 = 0.0;
    b0 = b * cg;
    b1 = b * sg;
    b2 = 0;
    c0 = c * cb;
    c1 = c * (ca - cb * cg) / sg;
    c2 = c * phi / sg;
    a0 = zeroOut(a0);
    a1 = zeroOut(a1);
    a2 = zeroOut(a2);
    b1 = zeroOut(b1);
    b2 = zeroOut(b2);
    c2 = zeroOut(c2);
  } else {
    double p = sqrt(1.0 + 2 * ca);
    double q = sqrt(1.0 - ca);
    double pmq = (a / 3.0) * (p - q);
    double p2q = (a / 3.0) * (p + 2 * q);
    a0 = p2q;
    a1 = pmq;
    a2 = pmq;
    b0 = pmq;
    b1 = p2q;
    b2 = pmq;
    c0 = pmq;
    c1 = pmq;
    c2 = p2q;
  }
  recip[0][0] = zeroOut((b1 * c2 - b2 * c1) * pv);
  recip[1][0] = zeroOut((b2 * c0 - b0 * c2) * pv);
  recip[2][0] = zeroOut((b0 * c1 - b1 * c0) * pv);
  recip[0][1] = zeroOut((c1 * a2 - c2 * a1) * pv);
  recip[1][1] = zeroOut((c2 * a0 - c0 * a2) * pv);
  recip[2][1] = zeroOut((c0 * a1 - c1 * a0) * pv);
  recip[0][2] = zeroOut((a1 * b2 - a2 * b1) * pv);
  recip[1][2] = zeroOut((a2 * b0 - a0 * b2) * pv);
  recip[2][2] = zeroOut((a0 * b1 - a1 * b0) * pv);
}

// ── Diffraction overlap calculation ─────────────────────────────────────

static inline double calcOverlap(float *image, double euler[3], int *hkls,
                                 int nhkls, int nrPxX, int nrPxY,
                                 double recip[3][3], double *outArrThis,
                                 int maxNrSpots, double rotTranspose[3][3],
                                 double pArr[3], double pxX, double pxY,
                                 double Elo, double Ehi,
                                 double minSpotIntensity) {
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  double ki[3] = {0, 0, 1.0};
  MatrixMultF33(OMt, recip, OM);
  int hklnr, badSpot;
  double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp, px, py,
      sinTheta, E, result = 0;
  int spotNr = 0, iterNr, nrPos = 0;
  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    qhat[0] = qvec[0] / qlen;
    qhat[1] = qvec[1] / qlen;
    qhat[2] = qvec[2] / qlen;
    dot = qhat[2];
    kf[0] = ki[0] - 2 * dot * qhat[0];
    kf[1] = ki[1] - 2 * dot * qhat[1];
    kf[2] = ki[2] - 2 * dot * qhat[2];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    xyz[0] = xyz[0] * pArr[2] / xyz[2];
    xyz[1] = xyz[1] * pArr[2] / xyz[2];
    xyz[2] = pArr[2];
    xp = xyz[0] - pArr[0];
    yp = xyz[1] - pArr[1];
    px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    sinTheta = -qhat[2];
    E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    badSpot = 0;
    for (iterNr = 0; iterNr < spotNr; iterNr++) {
      if ((fabs(qhat[0] - outArrThis[3 * iterNr + 0]) * 100000 < 0.1) &&
          (fabs(qhat[1] - outArrThis[3 * iterNr + 1]) * 100000 < 0.1) &&
          (fabs(qhat[2] - outArrThis[3 * iterNr + 2]) * 100000 < 0.1)) {
        badSpot = 1;
        break;
      }
    }
    if (badSpot == 0) {
      outArrThis[3 * spotNr + 0] = qhat[0];
      outArrThis[3 * spotNr + 1] = qhat[1];
      outArrThis[3 * spotNr + 2] = qhat[2];
      if (image[(size_t)((size_t)py * nrPxX + (size_t)px)] >
          minSpotIntensity) {
        result += image[(size_t)((size_t)py * nrPxX + (size_t)px)];
        nrPos++;
      }
      spotNr++;
      if (spotNr == maxNrSpots)
        break;
    }
  }
  result = nrPos * sqrt(result);
  return result;
}

// ── HKL prefiltering ────────────────────────────────────────────────────
// Identify which HKLs produce valid diffraction spots at a given
// orientation. During NLopt optimisation the Euler angles vary by only
// ±3°, so the valid-HKL set stays essentially constant.  Running the
// inner loop over ~60 HKLs instead of 3058 is the main speedup.

static inline int prefilterHKLs(int *hkls, int nhkls, double euler[3],
                                double recip[3][3], int nrPxX, int nrPxY,
                                double rotTranspose[3][3], double pArr[3],
                                double pxX, double pxY, double Elo, double Ehi,
                                int *validIdx, int maxValid) {
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  MatrixMultF33(OMt, recip, OM);
  int nValid = 0;
  for (int hklnr = 0; hklnr < nhkls && nValid < maxValid; hklnr++) {
    double hkl[3], qvec[3];
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    double qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    double qhat[3] = {qvec[0] / qlen, qvec[1] / qlen, qvec[2] / qlen};
    double dot = qhat[2];
    double kf[3] = {-2 * dot * qhat[0], -2 * dot * qhat[1],
                    1.0 - 2 * dot * qhat[2]};
    double xyz[3];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    double xp = xyz[0] * pArr[2] / xyz[2] - pArr[0];
    double yp = xyz[1] * pArr[2] / xyz[2] - pArr[1];
    double px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    double py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    double sinTheta = -qhat[2];
    double E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    validIdx[nValid++] = hklnr;
  }
  return nValid;
}

// ── Overlap using prefiltered HKLs ──────────────────────────────────────

static inline double
calcOverlapFiltered(float *image, double euler[3], int *hkls, int *validIdx,
                    int nValid, int nrPxX, int nrPxY, double recip[3][3],
                    double *outArrThis, int maxNrSpots,
                    double rotTranspose[3][3], double pArr[3], double pxX,
                    double pxY, double Elo, double Ehi,
                    double minSpotIntensity) {
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  double ki[3] = {0, 0, 1.0};
  MatrixMultF33(OMt, recip, OM);
  int badSpot;
  double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp, px, py,
      sinTheta, E, result = 0;
  int spotNr = 0, iterNr, nrPos = 0;
  for (int vi = 0; vi < nValid; vi++) {
    int hklnr = validIdx[vi];
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    qhat[0] = qvec[0] / qlen;
    qhat[1] = qvec[1] / qlen;
    qhat[2] = qvec[2] / qlen;
    dot = qhat[2];
    kf[0] = ki[0] - 2 * dot * qhat[0];
    kf[1] = ki[1] - 2 * dot * qhat[1];
    kf[2] = ki[2] - 2 * dot * qhat[2];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    xyz[0] = xyz[0] * pArr[2] / xyz[2];
    xyz[1] = xyz[1] * pArr[2] / xyz[2];
    xyz[2] = pArr[2];
    xp = xyz[0] - pArr[0];
    yp = xyz[1] - pArr[1];
    px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    sinTheta = -qhat[2];
    E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    badSpot = 0;
    for (iterNr = 0; iterNr < spotNr; iterNr++) {
      if ((fabs(qhat[0] - outArrThis[3 * iterNr + 0]) * 100000 < 0.1) &&
          (fabs(qhat[1] - outArrThis[3 * iterNr + 1]) * 100000 < 0.1) &&
          (fabs(qhat[2] - outArrThis[3 * iterNr + 2]) * 100000 < 0.1)) {
        badSpot = 1;
        break;
      }
    }
    if (badSpot == 0) {
      outArrThis[3 * spotNr + 0] = qhat[0];
      outArrThis[3 * spotNr + 1] = qhat[1];
      outArrThis[3 * spotNr + 2] = qhat[2];
      if (image[(size_t)((size_t)py * nrPxX + (size_t)px)] >
          minSpotIntensity) {
        result += image[(size_t)((size_t)py * nrPxX + (size_t)px)];
        nrPos++;
      }
      spotNr++;
      if (spotNr == maxNrSpots)
        break;
    }
  }
  result = nrPos * sqrt(result);
  return result;
}

// ── NLopt problem function ──────────────────────────────────────────────

static inline double problem_function(unsigned n, const double *x, double *grad,
                                      void *f_data_supplied) {
  int i, j;
  struct dataFit *f_data = (struct dataFit *)f_data_supplied;
  double rotTranspose[3][3], pArr[3], recip[3][3];
  for (i = 0; i < 3; i++) {
    pArr[i] = f_data->pArr[i];
    for (j = 0; j < 3; j++) {
      rotTranspose[i][j] = f_data->rotTranspose[i][j];
      if (n == 3)
        recip[i][j] = f_data->recip[i][j];
    }
  }
  if (n > 3) {
    double latCNew[6];
    int cntr = 0;
    for (i = 0; i < 6; i++) {
      if (tol_LatC[i] != 0) {
        latCNew[i] = x[3 + cntr];
        cntr++;
      } else {
        latCNew[i] = f_data->LatCOrig[i];
      }
    }
    if (tol_c_over_a != 0) {
      // FIX: was pow(x, 1/3) — integer division gave 0.  Use cbrt().
      double aNew = cbrt(cellVol / (x[3] * phiVol));
      latCNew[0] = aNew;
      latCNew[1] = aNew;
      latCNew[2] = x[3] * aNew;
    }
    calcRecipArray(latCNew, sg_num, recip);
  }
  float *image = f_data->image;
  int *hkls = f_data->hkls;
  double *outArrThis = f_data->outArrThis;
  double Euler[3];
  for (i = 0; i < 3; i++)
    Euler[i] = x[i];
  double overlap;
  if (f_data->validHKLIdx != NULL) {
    overlap = calcOverlapFiltered(
        image, Euler, hkls, f_data->validHKLIdx, f_data->nValidHKL,
        f_data->nrPxX, f_data->nrPxY, recip, outArrThis, f_data->maxNrSpots,
        rotTranspose, pArr, f_data->pxX, f_data->pxY, f_data->Elo, f_data->Ehi,
        f_data->minSpotIntensity);
  } else {
    overlap = calcOverlap(image, Euler, hkls, f_data->nhkls, f_data->nrPxX,
                          f_data->nrPxY, recip, outArrThis, f_data->maxNrSpots,
                          rotTranspose, pArr, f_data->pxX, f_data->pxY,
                          f_data->Elo, f_data->Ehi, f_data->minSpotIntensity);
  }
  return -overlap;
}

// ── Orientation fitting ─────────────────────────────────────────────────

static inline double
FitOrientation(float *image, double euler[3], int *hkls, int nhkls, int nrPxX,
               int nrPxY, double recip[3][3], double *outArrThis,
               int maxNrSpots, double rotTranspose[3][3], double pArr[3],
               double pxX, double pxY, double Elo, double Ehi, double tol,
               double latc[6], double eulerFit[3], double latCUpd[6],
               double *minVal, int doCrystalFit, int *validHKLIdx,
               int nValidHKL, int forceNelderMead, double initStepRad,
               double minSpotIntensity) {
  int i, j;
  unsigned n;
  if (doCrystalFit == 0) {
    n = 3;
  } else {
    int non_zero = 0;
    for (i = 0; i < 6; i++)
      if (tol_LatC[i] != 0)
        non_zero++;
    n = 3 + non_zero;
    if (tol_c_over_a != 0)
      n = 4;
  }
  double minf;
  double x[n], xl[n], xu[n];
  x[0] = euler[0];
  xl[0] = euler[0] - tol;
  xu[0] = euler[0] + tol;
  x[1] = euler[1];
  xl[1] = euler[1] - tol;
  xu[1] = euler[1] + tol;
  x[2] = euler[2];
  xl[2] = euler[2] - tol;
  xu[2] = euler[2] + tol;
  if (doCrystalFit != 0) {
    int cntr = 3;
    for (i = 0; i < 6; i++) {
      if (tol_LatC[i] != 0) {
        x[cntr] = latc[i];
        xl[cntr] = latc[i] * (1 - tol_LatC[i]);
        xu[cntr] = latc[i] * (1 + tol_LatC[i]);
        cntr++;
      }
    }
    if (tol_c_over_a != 0) {
      x[3] = c_over_a_orig;
      xl[3] = c_over_a_orig * (1 - tol_c_over_a);
      xu[3] = c_over_a_orig * (1 + tol_c_over_a);
    }
  }

  struct dataFit f_data = {0};
  f_data.image = image;
  f_data.hkls = hkls;
  f_data.nhkls = nhkls;
  f_data.nrPxX = nrPxX;
  f_data.nrPxY = nrPxY;
  f_data.outArrThis = outArrThis;
  f_data.maxNrSpots = maxNrSpots;
  f_data.validHKLIdx = validHKLIdx;
  f_data.nValidHKL = nValidHKL;
  f_data.minSpotIntensity = minSpotIntensity;
  for (i = 0; i < 3; i++) {
    f_data.pArr[i] = pArr[i];
    for (j = 0; j < 3; j++) {
      f_data.rotTranspose[i][j] = rotTranspose[i][j];
      f_data.recip[i][j] = recip[i][j];
    }
  }
  for (i = 0; i < 6; i++)
    f_data.LatCOrig[i] = latc[i];
  f_data.pxX = pxX;
  f_data.pxY = pxY;
  f_data.Elo = Elo;
  f_data.Ehi = Ehi;
  void *trp = (void *)&f_data;
  // Nelder-Mead for BOTH stages, via the vendored simplex -- no NLopt.
  //
  // The objective (calcOverlap) samples observed intensity at the INTEGER pixel
  // of each predicted reflection, so it is piecewise-constant: zero gradient
  // except where a spot crosses a pixel boundary. BOBYQA fits a smooth
  // quadratic model to that staircase, which is why the coarse stage always
  // forced Nelder-Mead. Measured on 198 paired synthetic seeds perturbed by up
  // to one 0.4 deg grid spacing, scored as misorientation to a known truth:
  //
  //   Nelder-Mead   median 0.004103 deg   p95 0.009076   max 0.013942   0.609 s
  //   BOBYQA        median 0.005432 deg   p95 0.029568   max 0.345720   0.610 s
  //
  // Nelder-Mead is better on every statistic, wins 132/198 paired, and costs
  // the same wall-clock. BOBYQA's worst case is its own seed error -- on that
  // seed it did not refine at all, which is the quadratic model failing exactly
  // as the comment above always predicted. Dropping it removes the last
  // external dependency, so the C builds with no NLopt to fetch.
  // See PREREGISTER.md / RESULTS.md in the optimiser-comparison analysis dir.
  //
  // `forceNelderMead` is retained in the signature: it is now always true in
  // effect, and callers still pass it to document which stage they are in.
  (void)forceNelderMead;
  NLoptConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.objective_function = problem_function;
  cfg.obj_data = trp;
  cfg.dimension = n;
  cfg.lower_bounds = xl;
  cfg.upper_bounds = xu;
  cfg.initial_guess = x;          /* IN/OUT: holds the best point on return */
  cfg.ftol_rel = 1e-6;
  cfg.xtol_rel = 1e-6;
  cfg.max_evaluations = 200;
  double nmStep[n];               /* same VLA form as x/xl/xu above */
  if (initStepRad > 0.0) {
    /* nlopt_set_initial_step1 set ONE scalar for every dimension; the vendored
     * API takes a per-dimension array, so broadcast it. NULL leaves the NLopt
     * default-step heuristic, which is what the fine stage always used. */
    for (i = 0; i < (int)n; i++)
      nmStep[i] = initStepRad;
    cfg.step_sizes = nmStep;
  }
  run_nlopt_optimization(MIDAS_LN_NELDERMEAD, &cfg);
  minf = cfg.min_function_val;
  for (i = 0; i < 3; i++)
    eulerFit[i] = x[i];
  if (doCrystalFit != 0) {
    int cntr2 = 0;
    for (i = 0; i < 6; i++) {
      if (tol_LatC[i] != 0) {
        latCUpd[i] = x[3 + cntr2];
        cntr2++;
      } else {
        latCUpd[i] = latc[i];
      }
    }
    if (tol_c_over_a != 0) {
      // FIX: was pow(x, 1/3) — use cbrt()
      double aNew = cbrt(cellVol / (x[3] * phiVol));
      latCUpd[0] = aNew;
      latCUpd[1] = aNew;
      latCUpd[2] = x[3] * aNew;
    }
  }
  *minVal = -minf;
  return 0;
}

// ── Write overlap with optional spot info ───────────────────────────────
// Always buffers spot data. Caller decides whether to flush based on
// nrSps return value, or passes saveExtraInfo != 0 to auto-flush.

static inline int writeCalcOverlap(float *image, double euler[3], int *hkls,
                                   int nhkls, int nrPxX, int nrPxY,
                                   double recip[3][3], double *outArrThis,
                                   int maxNrSpots, double rotTranspose[3][3],
                                   double pArr[3], double pxX, double pxY,
                                   double Elo, double Ehi,
                                   double minSpotIntensity, FILE *ExtraInfo,
                                   int saveExtraInfo, int *simulNrSps,
                                   int imageNr) {
  int nrSps = 0;
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  double ki[3] = {0, 0, 1.0};
  MatrixMultF33(OMt, recip, OM);
  int hklnr, badSpot;
  double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp, px, py,
      sinTheta, E, result = 0;
  int spotNr = 0, iterNr, nrPos = 0;

  char *outputBuf = NULL;
  size_t currentOffset = 0;
  size_t outputBufCap = (size_t)maxNrSpots * 256;  // per-spot line budget
  if (saveExtraInfo != 0) {
    outputBuf = (char *)malloc(outputBufCap);
    if (outputBuf)
      outputBuf[0] = '\0';
  }

  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    qhat[0] = qvec[0] / qlen;
    qhat[1] = qvec[1] / qlen;
    qhat[2] = qvec[2] / qlen;
    dot = qhat[2];
    kf[0] = ki[0] - 2 * dot * qhat[0];
    kf[1] = ki[1] - 2 * dot * qhat[1];
    kf[2] = ki[2] - 2 * dot * qhat[2];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    xyz[0] = xyz[0] * pArr[2] / xyz[2];
    xyz[1] = xyz[1] * pArr[2] / xyz[2];
    xyz[2] = pArr[2];
    xp = xyz[0] - pArr[0];
    yp = xyz[1] - pArr[1];
    px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    sinTheta = -qhat[2];
    E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    badSpot = 0;
    for (iterNr = 0; iterNr < spotNr; iterNr++) {
      if ((fabs(qhat[0] - outArrThis[3 * iterNr + 0]) * 100000 < 0.1) &&
          (fabs(qhat[1] - outArrThis[3 * iterNr + 1]) * 100000 < 0.1) &&
          (fabs(qhat[2] - outArrThis[3 * iterNr + 2]) * 100000 < 0.1)) {
        badSpot = 1;
        break;
      }
    }
    if (badSpot == 0) {
      outArrThis[3 * spotNr + 0] = qhat[0];
      outArrThis[3 * spotNr + 1] = qhat[1];
      outArrThis[3 * spotNr + 2] = qhat[2];
      if (image[(size_t)((size_t)py * nrPxX + (size_t)px)] >
          minSpotIntensity) {
        if (saveExtraInfo != 0) {
          if (outputBuf != NULL) {
            // snprintf with remaining space: never overrun outputBuf even if a
            // line is unexpectedly wide (large hkl / imageNr / intensity).
            size_t remBuf = (currentOffset < outputBufCap)
                                ? outputBufCap - currentOffset : 0;
            int wrote = 0;
            if (imageNr > 0)
              wrote = snprintf(
                  outputBuf + currentOffset, remBuf,
                  "%d\t%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                  imageNr, saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                  (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                  (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
            else
              wrote = snprintf(
                  outputBuf + currentOffset, remBuf,
                  "%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                  saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1], (int)hkl[2],
                  (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                  (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
            if (wrote > 0)
              currentOffset += ((size_t)wrote < remBuf) ? (size_t)wrote
                                                        : (remBuf ? remBuf - 1 : 0);
          } else {
#pragma omp critical
            {
              if (imageNr > 0)
                fprintf(
                    ExtraInfo,
                    "%d\t%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                    imageNr, saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                    (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                    (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
              else
                fprintf(ExtraInfo,
                        "%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                        saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                        (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1],
                        qhat[2],
                        (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
            }
          }
        }
        result += image[(size_t)((size_t)py * nrPxX + (size_t)px)];
        nrPos++;
      }
      spotNr++;
      if (spotNr == maxNrSpots)
        break;
    }
  }

  if (saveExtraInfo != 0 && outputBuf != NULL) {
    if (currentOffset > 0) {
#pragma omp critical
      {
        fputs(outputBuf, ExtraInfo);
      }
    }
    free(outputBuf);
  }

  *simulNrSps = spotNr;
  nrSps = nrPos;
  return nrSps;
}

// Cap on the number of coarse match results fed into the O(N^2) merge below.
// Far above any realistic per-frame count (hundreds to a few thousand), so it
// is a no-op for normal data; it only fires on a pathological frame to keep the
// pairwise misorientation matrix (nrResults^2) from exhausting memory.
#define MERGE_MAX_RESULTS 50000

typedef struct {
  double score;
  size_t row;
} ScoreRow_t;

static int cmpScoreRowDesc(const void *a, const void *b) {
  double sa = ((const ScoreRow_t *)a)->score;
  double sb = ((const ScoreRow_t *)b)->score;
  return (sa < sb) - (sa > sb); // descending by score
}

// ── Merge duplicate orientations (parallel) ──────────────────────────────
// Clusters match results by misorientation angle using OMP-parallel
// pairwise precomputation.  Returns number of unique solutions.
static inline int mergeDuplicateOrientations(double *orients, size_t *rowNrs,
                                             double *matchScores, int nrResults,
                                             double maxAngle, int numProcs,
                                             double *FinOrientArr, int *dArr,
                                             int *bsArr, double *bsScoreArr) {
  // Cap pathological result counts: keep the top MERGE_MAX_RESULTS by coarse
  // score before the O(N^2) clustering.  No-op for realistic frames.
  if (nrResults > MERGE_MAX_RESULTS) {
    fprintf(stderr,
            "WARNING: %d coarse matches exceed merge cap %d; keeping the "
            "top-scored before clustering.\n",
            nrResults, MERGE_MAX_RESULTS);
    ScoreRow_t *sr = (ScoreRow_t *)malloc((size_t)nrResults * sizeof(ScoreRow_t));
    if (sr == NULL) {
      fprintf(stderr, "FATAL: could not allocate score-sort buffer.\n");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nrResults; i++) {
      sr[i].score = matchScores[i];
      sr[i].row = rowNrs[i];
    }
    qsort(sr, (size_t)nrResults, sizeof(ScoreRow_t), cmpScoreRowDesc);
    for (int i = 0; i < MERGE_MAX_RESULTS; i++) {
      matchScores[i] = sr[i].score;
      rowNrs[i] = sr[i].row;
    }
    free(sr);
    nrResults = MERGE_MAX_RESULTS;
  }
  int *doneArr = (int *)calloc(nrResults, sizeof(int));
  // Step 1: Precompute quaternions for all matches (parallel)
  double *quats = (double *)malloc((size_t)nrResults * 4 * sizeof(double));
  if ((doneArr == NULL || quats == NULL) && nrResults > 0) {
    fprintf(stderr, "FATAL: mergeDuplicateOrientations could not allocate "
            "quaternion arrays (nrResults=%d).\n", nrResults);
    exit(EXIT_FAILURE);
  }
#pragma omp parallel for num_threads(numProcs)
  for (int qi = 0; qi < nrResults; qi++) {
    double or9[9];
    for (int k = 0; k < 9; k++)
      or9[k] = orients[rowNrs[qi] * 9 + k];
    OrientMat2Quat(or9, &quats[qi * 4]);
  }
  // Step 2: Precompute pairwise misorientations (parallel)
  size_t nPairs = (size_t)nrResults * (nrResults - 1) / 2;
  float *misoDist = (float *)malloc(nPairs * sizeof(float));
  // NOTE(perf): this pairwise matrix is O(nrResults^2) memory — for a
  // pathological frame with very many candidate matches it can be enormous.
  // Fail loudly here rather than segfault; bounding nrResults before the
  // merge is a tracked follow-up (see C_CUDA_HARDENING.md).
  if (misoDist == NULL && nPairs > 0) {
    fprintf(stderr, "FATAL: mergeDuplicateOrientations could not allocate the "
            "%zu-pair misorientation matrix (nrResults=%d). Too many matches.\n",
            nPairs, nrResults);
    exit(EXIT_FAILURE);
  }
#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int i = 1; i < nrResults; i++) {
    for (int j = 0; j < i; j++) {
      size_t idx = (size_t)i * (i - 1) / 2 + j;
      misoDist[idx] = (float)GetMisOrientation(&quats[i * 4], &quats[j * 4]);
    }
  }
  // Step 3: Greedy cluster using precomputed distances (serial, fast)
  int iterNr = 0;
  for (int gi = 0; gi < nrResults; gi++) {
    if (doneArr[gi] != 0)
      continue;
    doneArr[gi] = 1;
    int bestSol = rowNrs[gi];
    double bestIntensity = matchScores[gi];
    for (int l = gi + 1; l < nrResults; l++) {
      if (doneArr[l] > 0)
        continue;
      size_t idx = (size_t)l * (l - 1) / 2 + gi;
      if (misoDist[idx] <= maxAngle) {
        doneArr[l] = 1;
        doneArr[gi]++;
        if (matchScores[l] > bestIntensity) {
          bestIntensity = matchScores[l];
          bestSol = rowNrs[l];
        }
      }
    }
    for (int k = 0; k < 9; k++)
      FinOrientArr[iterNr * 9 + k] = orients[bestSol * 9 + k];
    dArr[iterNr] = doneArr[gi];
    bsArr[iterNr] = bestSol;
    bsScoreArr[iterNr] = bestIntensity;
    iterNr++;
  }
  free(quats);
  free(misoDist);
  free(doneArr);
  return iterNr;
}

// ── Fit and write orientations (parallel) ────────────────────────────────
// OMP-parallel fitting loop: prefilter HKLs, single-pass FitOrientation,
// writeCalcOverlap, fprintf results.
// imageNum > 0: prepend image number column (streaming mode)
// imageNum <= 0: no image column (batch mode)
// Geometry-scaled coarse-fit blur width (px).  Chosen so ~3 sigma covers the
// spot displacement of a worst-case orientation-grid seed error (~1.3x the
// grid spacing): displacement_px = 1.3 * gridDeg * (Lsd/pxSize).  Clamped to
// the empirically safe window [4,12] px -- below ~4 the blur cannot bridge the
// gap; above ~12 it over-smooths and merges neighbouring spots.  Callers may
// override with an explicit CoarseFitSigma.
static inline double autoCoarseSigma(double Lsd, double pxSize,
                                     double gridDeg) {
  if (gridDeg <= 0.0)
    gridDeg = 0.4; // default 100M-orientation DB spacing
  double sigma = 1.3 * gridDeg * deg2rad * (Lsd / pxSize) / 3.0;
  if (sigma < 4.0)
    sigma = 4.0;
  if (sigma > 12.0)
    sigma = 12.0;
  return sigma;
}

// Separable Gaussian blur of a float image (row-major, width nx).  Used to
// widen the match "capture radius" for the coarse stage of orientation
// refinement so a coarse-grid seed (up to ~one grid spacing off, i.e. tens of
// pixels of spot displacement) still has a smooth gradient path to the true
// peak.  Without this, a seed whose spots land outside the sharp-image spot
// blobs sits on a flat objective and the local optimizer cannot converge.
static inline void gaussianBlurImage(const float *in, float *out, int nx,
                                     int ny, double sigma) {
  int rad = (int)(3.0 * sigma + 0.5);
  if (rad < 1)
    rad = 1;
  int klen = 2 * rad + 1;
  double *kern = (double *)malloc((size_t)klen * sizeof(double));
  float *tmp = (float *)malloc((size_t)nx * ny * sizeof(float));
  if (kern == NULL || tmp == NULL) {
    fprintf(stderr, "FATAL: gaussianBlurImage allocation failed.\n");
    exit(EXIT_FAILURE);
  }
  double ksum = 0.0;
  for (int i = -rad; i <= rad; i++) {
    double w = exp(-(double)(i * i) / (2.0 * sigma * sigma));
    kern[i + rad] = w;
    ksum += w;
  }
  for (int i = 0; i < klen; i++)
    kern[i] /= ksum;
  // Horizontal pass (clamped edges).
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      double acc = 0.0;
      for (int k = -rad; k <= rad; k++) {
        int xx = x + k;
        if (xx < 0)
          xx = 0;
        else if (xx >= nx)
          xx = nx - 1;
        acc += kern[k + rad] * in[(size_t)y * nx + xx];
      }
      tmp[(size_t)y * nx + x] = (float)acc;
    }
  }
  // Vertical pass (clamped edges).
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      double acc = 0.0;
      for (int k = -rad; k <= rad; k++) {
        int yy = y + k;
        if (yy < 0)
          yy = 0;
        else if (yy >= ny)
          yy = ny - 1;
        acc += kern[k + rad] * tmp[(size_t)yy * nx + x];
      }
      out[(size_t)y * nx + x] = (float)acc;
    }
  }
  free(tmp);
  free(kern);
}

static inline void fitAndWriteOrientations(
    float *image, double *FinOrientArr, int *dArr, int *bsArr,
    double *bsScoreArr, int totalSols, int *hkls, int nhkls, int nrPxX,
    int nrPxY, double recip[3][3], double rotTranspose[3][3], double pArr[3],
    double pxX, double pxY, double Elo, double Ehi, double tol,
    double *LatticeParameter, int maxNrSpots, int minNrSpots, int numProcs,
    FILE *outF, FILE *ExtraInfo, int imageNum, double coarseFitSigma,
    double minSpotIntensity) {
  int iterNr;
  // Blur the match image once (shared, read-only across threads) to give the
  // coarse fit stage a wide capture radius.
  float *imageCoarse = (float *)malloc((size_t)nrPxX * nrPxY * sizeof(float));
  if (imageCoarse == NULL) {
    fprintf(stderr, "FATAL: could not allocate coarse-fit image buffer.\n");
    exit(EXIT_FAILURE);
  }
  double coarseSigma = (coarseFitSigma > 0.0)
                           ? coarseFitSigma
                           : autoCoarseSigma(pArr[2], pxX, 0.4);
  gaussianBlurImage(image, imageCoarse, nrPxX, nrPxY, coarseSigma);
#pragma omp parallel for num_threads(numProcs)
  for (iterNr = 0; iterNr < totalSols; iterNr++) {
    double orientBest[3][3], eulerBest[3], eulerFit[3], orientFit[3][3];
    double q1[4], q2[4];
    int iJ, iK;
    double *outArrThisFit = (double *)calloc(3 * maxNrSpots, sizeof(double));
    if (outArrThisFit == NULL) {
      fprintf(stderr, "FATAL: could not allocate fit spot buffer.\n");
      exit(EXIT_FAILURE);
    }
    for (iJ = 0; iJ < 3; iJ++)
      for (iK = 0; iK < 3; iK++)
        orientBest[iJ][iK] = FinOrientArr[iterNr * 9 + 3 * iJ + iK];
    OrientMat2Euler(orientBest, eulerBest);
    for (iJ = 0; iJ < 3; iJ++)
      eulerFit[iJ] = eulerBest[iJ];
    int doCrystalFit = 1;
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(double));
    double latCFit[6], recipFit[3][3], mv = 0;
    // Prefilter HKLs at coarse orientation
    int *validIdx = (int *)malloc(nhkls * sizeof(int));
    if (validIdx == NULL) {
      fprintf(stderr, "FATAL: could not allocate validIdx (nhkls=%d).\n", nhkls);
      exit(EXIT_FAILURE);
    }
    int nValid =
        prefilterHKLs(hkls, nhkls, eulerBest, recip, nrPxX, nrPxY, rotTranspose,
                      pArr, pxX, pxY, Elo, Ehi, validIdx, nhkls);
    // Coarse-to-fine fit.  Stage 1: orientation-only against the blurred
    // image (wide capture radius) to bridge a coarse-grid seed error that
    // would otherwise leave the true peak outside the sharp-image spot basin.
    double eulerCoarse[3], latCdummy[6], mvCoarse = 0;
    FitOrientation(imageCoarse, eulerBest, hkls, nhkls, nrPxX, nrPxY, recip,
                   outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo,
                   Ehi, tol, LatticeParameter, eulerCoarse, latCdummy, &mvCoarse,
                   0 /*doCrystalFit*/, NULL /*all HKLs*/, 0,
                   1 /*forceNelderMead*/, 0.2 * deg2rad /*initStep*/,
                   minSpotIntensity);
    // Re-prefilter HKLs at the coarse solution (spots can move on/off the
    // detector after a ~grid-spacing correction), then fine-fit on the sharp
    // image for full precision (orientation + crystal).
    free(validIdx);
    validIdx = (int *)malloc(nhkls * sizeof(int));
    nValid = prefilterHKLs(hkls, nhkls, eulerCoarse, recip, nrPxX, nrPxY,
                           rotTranspose, pArr, pxX, pxY, Elo, Ehi, validIdx,
                           nhkls);
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(double));
    // Stage 2: full-precision fit on the sharp image, seeded from the coarse
    // solution.
    FitOrientation(image, eulerCoarse, hkls, nhkls, nrPxX, nrPxY, recip,
                   outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo,
                   Ehi, tol, LatticeParameter, eulerFit, latCFit, &mv,
                   doCrystalFit, validIdx, nValid, 0 /*BOBYQA*/, 0.0,
                   minSpotIntensity);
    free(validIdx);
    Euler2OrientMat(eulerFit, orientFit);
    OrientMat2Quat33(orientBest, q1);
    OrientMat2Quat33(orientFit, q2);
    int simulNrSps = 0;
    calcRecipArray(latCFit, sg_num, recipFit);
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(double));
    int saveExtraInfo = iterNr + 1;
    int nrSps = writeCalcOverlap(
        image, eulerFit, hkls, nhkls, nrPxX, nrPxY, recipFit, outArrThisFit,
        maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo, Ehi, minSpotIntensity,
        ExtraInfo, saveExtraInfo, &simulNrSps, imageNum);
    if (nrSps >= minNrSpots) {
      int bs = bsArr[iterNr];
      double miso = GetMisOrientation(q1, q2);
      double OF[3][3];
      MatrixMultF33(orientFit, recipFit, OF);
#pragma omp critical
      {
        if (imageNum > 0)
          fprintf(outF, "%d\t", imageNum);
        fprintf(outF, "%d\t%d\t", iterNr + 1, dArr[iterNr]);
        fprintf(outF, "%-13.4lf\t", (mv / nrSps) * (mv / nrSps));
        fprintf(outF, "%-13.4lf\t", nrSps * (mv / nrSps) * (mv / nrSps));
        fprintf(outF, "%-13.4lf\t", mv);
        fprintf(outF, "%d\t", nrSps);
        fprintf(outF, "%d\t", simulNrSps);
        for (int k = 0; k < 3; k++)
          for (int l = 0; l < 3; l++)
            fprintf(outF, "%-13.7lf\t\t", OF[k][l]);
        for (int k = 0; k < 6; k++)
          fprintf(outF, "%-13.7lf\t\t", latCFit[k]);
        for (int k = 0; k < 3; k++)
          for (int l = 0; l < 3; l++)
            fprintf(outF, "%-13.7lf\t\t", orientFit[k][l]);
        fprintf(outF, "%-13.4lf\t%-13.7lf\t%d\n", bsScoreArr[iterNr], miso, bs);
      }
    }
    free(outArrThisFit);
  }
  free(imageCoarse);
}

#endif /* LAUE_MATCHING_HEADERS_H */
