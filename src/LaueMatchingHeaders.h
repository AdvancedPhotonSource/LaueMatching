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

#include <fcntl.h>
#include <math.h>
#include <nlopt.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
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
#define CalcLength(x, y, z) sqrt((x) * (x) + (y) * (y) + (z) * (z))
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
extern int useBobyqa; // 1 = BOBYQA (default), 0 = Nelder-Mead

// ── Optimization data bundle ────────────────────────────────────────────
struct dataFit {
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

static inline double calcOverlap(double *image, double euler[3], int *hkls,
                                 int nhkls, int nrPxX, int nrPxY,
                                 double recip[3][3], double *outArrThis,
                                 int maxNrSpots, double rotTranspose[3][3],
                                 double pArr[3], double pxX, double pxY,
                                 double Elo, double Ehi) {
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
      if (image[(int)((int)py * nrPxX + (int)px)] > 0) {
        result += image[(int)((int)py * nrPxX + (int)px)];
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
  double *image = f_data->image;
  int *hkls = f_data->hkls;
  double *outArrThis = f_data->outArrThis;
  double Euler[3];
  for (i = 0; i < 3; i++)
    Euler[i] = x[i];
  double overlap = calcOverlap(
      image, Euler, hkls, f_data->nhkls, f_data->nrPxX, f_data->nrPxY, recip,
      outArrThis, f_data->maxNrSpots, rotTranspose, pArr, f_data->pxX,
      f_data->pxY, f_data->Elo, f_data->Ehi);
  return -overlap;
}

// ── Orientation fitting ─────────────────────────────────────────────────

static inline double
FitOrientation(double *image, double euler[3], int *hkls, int nhkls, int nrPxX,
               int nrPxY, double recip[3][3], double *outArrThis,
               int maxNrSpots, double rotTranspose[3][3], double pArr[3],
               double pxX, double pxY, double Elo, double Ehi, double tol,
               double latc[6], double eulerFit[3], double latCUpd[6],
               double *minVal, int doCrystalFit) {
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

  struct dataFit f_data;
  f_data.image = image;
  f_data.hkls = hkls;
  f_data.nhkls = nhkls;
  f_data.nrPxX = nrPxX;
  f_data.nrPxY = nrPxY;
  f_data.outArrThis = outArrThis;
  f_data.maxNrSpots = maxNrSpots;
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
  nlopt_opt opt;
  opt = nlopt_create(useBobyqa ? NLOPT_LN_BOBYQA : NLOPT_LN_NELDERMEAD, n);
  nlopt_set_lower_bounds(opt, xl);
  nlopt_set_upper_bounds(opt, xu);
  nlopt_set_min_objective(opt, problem_function, trp);
  nlopt_set_ftol_rel(opt, 1e-6);
  nlopt_set_xtol_rel(opt, 1e-6);
  nlopt_set_maxeval(opt, 200);
  nlopt_optimize(opt, x, &minf);
  nlopt_destroy(opt);
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
  return 0; // FIX: was missing return statement in CPU code
}

// ── Write overlap with optional spot info ───────────────────────────────

static inline int writeCalcOverlap(double *image, double euler[3], int *hkls,
                                   int nhkls, int nrPxX, int nrPxY,
                                   double recip[3][3], double *outArrThis,
                                   int maxNrSpots, double rotTranspose[3][3],
                                   double pArr[3], double pxX, double pxY,
                                   double Elo, double Ehi, FILE *ExtraInfo,
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
      if (image[(int)((int)py * nrPxX + (int)px)] > 0) {
        if (saveExtraInfo != 0) {
#pragma omp critical
          {
            if (imageNr > 0)
              fprintf(ExtraInfo,
                      "%d\t%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                      imageNr, saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                      (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                      image[(int)((int)py * nrPxX + (int)px)]);
            else
              fprintf(ExtraInfo,
                      "%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                      saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                      (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                      image[(int)((int)py * nrPxX + (int)px)]);
          }
        }
        result += image[(int)((int)py * nrPxX + (int)px)];
        nrPos++;
      }
      spotNr++;
      if (spotNr == maxNrSpots)
        break;
    }
  }
  *simulNrSps = spotNr;
  nrSps = nrPos;
  return nrSps;
}

#endif /* LAUE_MATCHING_HEADERS_H */
