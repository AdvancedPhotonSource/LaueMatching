/* Spot lines reach spots.txt only for grains that pass the MinNrSpots gate,
 * and the streaming layout (leading ImageNr column) is chosen by an explicit
 * sentinel, not by imageNr > 0.
 *
 * WHAT IS EXERCISED: the REAL writeCalcOverlapGated from the header, with a
 * constant image (every predicted, in-band spot is lit) and the (111) + (1-11)
 * pair of the harmonic fixture's geometry: 2 matches.
 *
 * Usage: spots_gate MIN_TO_WRITE IMAGE_NR  -> prints "LINES n COLS c"
 * (number of spot lines written, whitespace-separated fields in the first one).
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

#define NPX 2048

int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  int minW = atoi(argv[1]), imageNr = atoi(argv[2]);
  double lat[6] = {0.2, 0.2, 0.2, 90.0, 90.0, 90.0}, recip[3][3];
  double rotT[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  double pArr[3] = {100.0, 0.0, 50.0}, euler[3] = {0.0, M_PI, 0.0};
  sg_num = 225;
  calcRecipArray(lat, sg_num, recip);
  float *img = (float *)malloc((size_t)NPX * NPX * sizeof(float));
  if (img == NULL)
    return 1;
  for (size_t i = 0; i < (size_t)NPX * NPX; i++)
    img[i] = 1.0f;
  const int hk[] = {1, 1, 1, 1, -1, 1};
  double out[30];
  memset(out, 0, sizeof(out));
  FILE *f = tmpfile();
  int nSim = 0;
  int n = writeCalcOverlapGated(img, euler, (int *)hk, 2, NPX, NPX, recip, out,
                                10, rotT, pArr, 0.2, 0.2, 5.0, 20.0, 0.0, f, 7,
                                &nSim, imageNr, minW);
  rewind(f);
  char line[1024];
  int lines = 0, cols = 0;
  while (fgets(line, sizeof(line), f)) {
    if (lines == 0)
      for (char *t = strtok(line, " \t\n"); t; t = strtok(NULL, " \t\n"))
        cols++;
    lines++;
  }
  printf("NMATCH %d LINES %d COLS %d\n", n, lines, cols);
  fclose(f);
  free(img);
  return 0;
}
