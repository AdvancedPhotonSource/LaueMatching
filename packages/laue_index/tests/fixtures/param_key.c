/* paramKeyCmp: a parameter line matches a key only on the WHOLE first token.
 * Before 0.8.0 the binaries used strncmp(line, key, strlen(key)), a prefix
 * match, so a future key such as "PxXY" would have been read as "PxX".
 * Prints "KEY <line> <key> <0 match | nonzero>" for each case. */
#include "LaueMatchingHeaders.h"

double tol_LatC[6];
double tol_c_over_a;
double c_over_a_orig;
int sg_num;
double cellVol;
double phiVol;
int nSym;
double Symm[24][4];

int main(void) {
  const char *cases[][2] = {
      {"PxX 0.0002\n", "PxX"},        {"PxX\t0.0002\n", "PxX"},
      {"PxX\n", "PxX"},               {"PxXY 3\n", "PxX"},
      {"MinIntensityFrac 1\n", "MinIntensity"},
      {"MinIntensity 50\n", "MinIntensity"}, {"Px 1\n", "PxX"}};
  for (int i = 0; i < 7; i++)
    printf("KEY %d %d\n", i, paramKeyCmp(cases[i][0], cases[i][1]) == 0);
  return 0;
}
