/* Print what the shipped header builds for one space group and lattice:
 * the reciprocal matrix from calcRecipArray and the symmetry quaternions from
 * MakeSymmetries. test_c_symmetry_tables.py checks both against the lattice
 * (every operator must map the lattice onto itself, and preserve R-centring
 * for the R groups) and against midas_stress.
 *
 * WHAT IS EXERCISED: the REAL calcRecipArray, MakeSymmetries and
 * validateTrigonalSetting from LaueMatchingHeaders.h. Nothing is copied here.
 *
 * Usage: lattice_symmetry SG a b c alpha beta gamma
 * Output lines:
 *   VALID <0|1>             validateTrigonalSetting (1 = refused)
 *   RECIP r00 r01 ... r22   row-major, columns are a*, b*, c*
 *   NSYM n
 *   Q w x y z               n lines
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

int main(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr, "usage: %s SG a b c alpha beta gamma\n", argv[0]);
    return 2;
  }
  int sg = atoi(argv[1]);
  double lat[6];
  for (int i = 0; i < 6; i++)
    lat[i] = atof(argv[2 + i]);
  int bad = validateTrigonalSetting(sg, lat);
  printf("VALID %d\n", bad);
  if (bad)
    return 0;
  double recip[3][3];
  calcRecipArray(lat, sg, recip);
  printf("RECIP");
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      printf(" %.12g", recip[i][j]);
  printf("\n");
  double sym[24][4];
  int n = MakeSymmetries(sg, lat, sym);
  printf("NSYM %d\n", n);
  for (int i = 0; i < n; i++)
    printf("Q %.12g %.12g %.12g %.12g\n", sym[i][0], sym[i][1], sym[i][2],
           sym[i][3]);
  return 0;
}
