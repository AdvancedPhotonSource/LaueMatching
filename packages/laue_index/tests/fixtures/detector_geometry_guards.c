/* Parameter-file geometry helpers in LaueMatchingHeaders.h.
 *
 * WHAT IS EXERCISED: the real detectorRotationTranspose(), paramLineComplete()
 * and validateDetectorDistance().
 *   - R_Array (0,0,0) must give the EXACT identity. The inline code it replaced
 *     computed the axis as r/|r| = 0/0 = NaN, and 0 * NaN kept it NaN through
 *     Rodrigues, so every predicted spot was NaN.
 *   - A non-zero R_Array must give a proper rotation (orthonormal, det +1)
 *     AND be bit-identical to the inline formula the helper replaced (kept
 *     verbatim below as old_inline_rotation). Orthonormality alone would pass
 *     a transposed (inverse) rotation or a flipped sin sign; the golden
 *     comparison fails both. Inputs go through `volatile` so the compiler
 *     cannot constant-fold one side differently from the other.
 *   - paramLineComplete: a complete line passes; `P_Array 0 0 __SET_ME__`
 *     (sscanf returns 3 of 4) is refused.
 *   - validateDetectorDistance: 0 and NaN refused, 50 accepted.
 *   - validateEnergyBand: 5-30 accepted; reversed, empty, zero and NaN refused.
 * WHAT IS NOT: the three main()s (the CPU one is run by the Python test; the
 * .cu ones are covered by source checks only).
 */
#include "LaueMatchingHeaders.h"

double tol_LatC[6];
double tol_c_over_a;
double c_over_a_orig;
int sg_num;
double cellVol;
double phiVol;
int nSym;
double Symm[24][4];

/* VERBATIM the rotation code every main() carried before
 * detectorRotationTranspose() (it divided by |r| unguarded). */
static void old_inline_rotation(double rArr[3], double rotTranspose[3][3]) {
  double rotang = CalcLength(rArr[0], rArr[1], rArr[2]);
  double rotvect[3] = {rArr[0] / rotang, rArr[1] / rotang, rArr[2] / rotang};
  double rot[3][3] = {
      {cos(rotang) + (1 - cos(rotang)) * (rotvect[0] * rotvect[0]),
       (1 - cos(rotang)) * rotvect[0] * rotvect[1] - sin(rotang) * rotvect[2],
       (1 - cos(rotang)) * rotvect[0] * rotvect[2] + sin(rotang) * rotvect[1]},
      {(1 - cos(rotang)) * rotvect[1] * rotvect[0] + sin(rotang) * rotvect[2],
       cos(rotang) + (1 - cos(rotang)) * (rotvect[1] * rotvect[1]),
       (1 - cos(rotang)) * rotvect[1] * rotvect[2] - sin(rotang) * rotvect[0]},
      {(1 - cos(rotang)) * rotvect[2] * rotvect[0] - sin(rotang) * rotvect[1],
       (1 - cos(rotang)) * rotvect[2] * rotvect[1] + sin(rotang) * rotvect[0],
       cos(rotang) + (1 - cos(rotang)) * (rotvect[2] * rotvect[2])}};
  double t[3][3] = {{rot[0][0], rot[1][0], rot[2][0]},
                    {rot[0][1], rot[1][1], rot[2][1]},
                    {rot[0][2], rot[1][2], rot[2][2]}};
  memcpy(rotTranspose, t, sizeof(t));
}

int main(void) {
  int bad = 0;
  /* Golden: helper vs the verbatim old code, bit for bit, on real-looking
   * R_Array values (34-ID-E-like, small tilts, a large angle) and a sweep. */
  {
    volatile double golden[6][3] = {{-1.2, -1.2, -1.2},
                                    {0.3, -1.1, 0.7},
                                    {1e-4, 0.0, 0.0},
                                    {0.0, 0.0, 0.0001},
                                    {2.5, -0.4, 1.9},
                                    {-3.0, 0.2, 0.05}};
    int ndiff = 0;
    for (int k = 0; k < 6 + 1000; k++) {
      double r[3];
      if (k < 6) {
        for (int i = 0; i < 3; i++)
          r[i] = golden[k][i];
      } else {
        volatile double v = (double)(k - 6);
        r[0] = sin(v * 0.37) * 2.0;
        r[1] = cos(v * 0.11) * 1.3;
        r[2] = sin(v * 0.05 + 1.0) * 0.7;
      }
      double a[3][3], b[3][3];
      old_inline_rotation(r, a);
      detectorRotationTranspose(r, b);
      if (memcmp(a, b, sizeof(a)) != 0)
        ndiff++;
    }
    if (ndiff) {
      printf("rotation differs from the old inline formula on %d of 1006\n",
             ndiff);
      bad = 1;
    }
  }
  double z[3] = {0.0, 0.0, 0.0}, T[3][3];
  detectorRotationTranspose(z, T);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (T[i][j] != (i == j ? 1.0 : 0.0)) {
        printf("zero R_Array: T[%d][%d] = %g, want %g\n", i, j, T[i][j],
               i == j ? 1.0 : 0.0);
        bad = 1;
      }

  double r[3] = {0.3, -1.1, 0.7};
  detectorRotationTranspose(r, T);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      double d = T[i][0] * T[j][0] + T[i][1] * T[j][1] + T[i][2] * T[j][2];
      if (fabs(d - (i == j ? 1.0 : 0.0)) > 1e-12) {
        printf("rotation not orthonormal at (%d,%d): %g\n", i, j, d);
        bad = 1;
      }
    }
  double det = T[0][0] * (T[1][1] * T[2][2] - T[1][2] * T[2][1]) -
               T[0][1] * (T[1][0] * T[2][2] - T[1][2] * T[2][0]) +
               T[0][2] * (T[1][0] * T[2][1] - T[1][1] * T[2][0]);
  if (fabs(det - 1.0) > 1e-12) {
    printf("det = %g, want 1\n", det);
    bad = 1;
  }

  double p[3];
  char dummy[64];
  const char *full = "P_Array 0 0 50\n", *unfilled = "P_Array 0 0 __SET_ME__\n";
  if (!paramLineComplete(sscanf(full, "%s %lf %lf %lf", dummy, &p[0], &p[1],
                                &p[2]),
                         4, "P_Array", full)) {
    printf("complete line refused\n");
    bad = 1;
  }
  if (paramLineComplete(sscanf(unfilled, "%s %lf %lf %lf", dummy, &p[0],
                               &p[1], &p[2]),
                        4, "P_Array", unfilled)) {
    printf("unfilled line accepted\n");
    bad = 1;
  }

  double d0[3] = {0, 0, 0}, dn[3] = {0, 0, NAN}, d50[3] = {0, 0, 50};
  if (!validateDetectorDistance(d0) || !validateDetectorDistance(dn) ||
      validateDetectorDistance(d50)) {
    printf("validateDetectorDistance wrong\n");
    bad = 1;
  }
  if (validateEnergyBand(5.0, 30.0) || !validateEnergyBand(30.0, 5.0) ||
      !validateEnergyBand(0.0, 30.0) || !validateEnergyBand(5.0, 5.0) ||
      !validateEnergyBand(NAN, 30.0) || !validateEnergyBand(5.0, NAN)) {
    printf("validateEnergyBand wrong\n");
    bad = 1;
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad ? 2 : 0;
}
