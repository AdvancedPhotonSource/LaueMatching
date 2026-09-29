/* Does the duplicate merge depend on the ORDER candidates arrive in?
 *
 * The GPU kernels append candidates in atomicAdd arrival order, which varies
 * run to run; the CPU appends in orientation-database row order. The merge is
 * greedy (each cluster is seeded by the first unmerged candidate), so a chain
 * A-B-C with A-B and B-C inside MaxAngle but A-C outside it clusters as
 * {A,B}+{C} when A comes first and {A,B,C} when B comes first. Before 0.8.0
 * that made GPU output non-deterministic and CPU output differ from GPU.
 *
 * WHAT IS EXERCISED: the REAL mergeDuplicateOrientations from the header.
 * Candidates: rotations about z by 0, 0.9 and 1.8 deg (rows 0, 1, 2); B (0.9)
 * scores highest; triclinic symmetry, MaxAngle 1.0 (degrees, as in the params).
 *
 * Output: one line per permutation: "PERM p0p1p2 n | row:size:score ..."
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

static void rotz(double deg, double *m) {
  double t = deg * deg2rad, c = cos(t), s = sin(t);
  double r[9] = {c, -s, 0, s, c, 0, 0, 0, 1};
  memcpy(m, r, sizeof(r));
}

int main(void) {
  double lat[6] = {0.5, 0.6, 0.7, 80, 95, 105};
  sg_num = 1;
  nSym = MakeSymmetries(sg_num, lat, Symm);
  double orients[27];
  rotz(0.0, orients);
  rotz(0.9, orients + 9);
  rotz(1.8, orients + 18);
  const double score[3] = {100.0, 300.0, 200.0};
  const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                           {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (int p = 0; p < 6; p++) {
    size_t rows[3];
    double sc[3];
    for (int i = 0; i < 3; i++) {
      rows[i] = (size_t)perms[p][i];
      sc[i] = score[perms[p][i]];
    }
    double fin[27], bsScore[3];
    int dArr[3], bsArr[3];
    int n = mergeDuplicateOrientations(orients, rows, sc, 3, 1.0, 1,
                                       fin, dArr, bsArr, bsScore);
    printf("PERM %d%d%d %d |", perms[p][0], perms[p][1], perms[p][2], n);
    for (int i = 0; i < n; i++)
      printf(" %d:%d:%.1f", bsArr[i], dArr[i], bsScore[i]);
    printf("\n");
  }
  return 0;
}
