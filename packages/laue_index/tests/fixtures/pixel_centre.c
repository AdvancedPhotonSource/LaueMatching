/* Which pixel does the fit stage read for a predicted spot?
 *
 * Pixel k is CENTRED at detector coordinate k: calcRecipArray's projection
 * puts the panel centre at (N-1)/2, and GenerateSimulation, laue_torch and
 * laue_index.calibrate all place a spot at continuous (px, py) on that grid.
 * So a spot predicted at px = k + 0.7 lies in pixel k + 1. Before 0.8.0 the
 * header truncated (read pixel k), a 0.5 px mean bias toward -x, -y.
 *
 * WHAT IS EXERCISED: the REAL calcOverlap, calcOverlapFiltered and
 * writeCalcOverlap, compiled from the shipped header. The image is zero except
 * ONE lit pixel, so a count of 1 says exactly which pixel was read.
 *
 * Geometry (solved by hand in the test): cubic a = 0.2 nm, Euler (0, pi, 0),
 * identity detector rotation, P = (P0, P1, 50), 0.2-unit pixels on a 2048
 * panel. (111) then diffracts along (2/3, -2/3, 1/3), so
 *     fx = (100 - P0) / 0.2 + 1023.5,   fy = (-100 - P1) / 0.2 + 1023.5.
 *
 * Usage: pixel_centre P0 P1 litX litY    -> prints "COUNTS c f w X Y"
 * where X Y are the pixel writeCalcOverlap reports for the spot (-1 if none).
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
#define MAXSP 10

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s P0 P1 litX litY\n", argv[0]);
    return 2;
  }
  double pArr[3] = {atof(argv[1]), atof(argv[2]), 50.0};
  int litX = atoi(argv[3]), litY = atoi(argv[4]);
  double rotT[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  double euler[3] = {0.0, M_PI, 0.0};
  double lat[6] = {0.2, 0.2, 0.2, 90.0, 90.0, 90.0};
  double recip[3][3];
  sg_num = 225;
  calcRecipArray(lat, sg_num, recip);
  float *img = (float *)calloc((size_t)NPX * NPX, sizeof(float));
  if (img == NULL) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  img[(size_t)litY * NPX + litX] = 1.0f;
  const int h111[] = {1, 1, 1};
  int idx[1] = {0};
  double out[3 * MAXSP];
  memset(out, 0, sizeof(out));
  double r = calcOverlap(img, euler, (int *)h111, 1, NPX, NPX, recip, out,
                         MAXSP, rotT, pArr, 0.2, 0.2, 5.0, 20.0, 0.0);
  int c = r > 0.5;
  memset(out, 0, sizeof(out));
  r = calcOverlapFiltered(img, euler, (int *)h111, idx, 1, NPX, NPX, recip,
                          out, MAXSP, rotT, pArr, 0.2, 0.2, 5.0, 20.0, 0.0);
  int f = r > 0.5;
  memset(out, 0, sizeof(out));
  char path[] = "pixel_centre_spots_XXXXXX";
  int fd = mkstemp(path);
  FILE *spots = fd >= 0 ? fdopen(fd, "w+") : NULL;
  int nSim = 0;
  int w = writeCalcOverlap(img, euler, (int *)h111, 1, NPX, NPX, recip, out,
                           MAXSP, rotT, pArr, 0.2, 0.2, 5.0, 20.0, 0.0, spots,
                           1, &nSim, NOT_STREAMING);
  int X = -1, Y = -1;
  if (spots != NULL) {
    fflush(spots);
    rewind(spots);
    char line[1024];
    while (fgets(line, sizeof(line), spots)) {
      int g, s, h, k, l, x, y;
      if (sscanf(line, "%d %d %d %d %d %d %d", &g, &s, &h, &k, &l, &x, &y) == 7) {
        X = x;
        Y = y;
      }
    }
    fclose(spots);
    unlink(path);
  }
  printf("COUNTS %d %d %d %d %d\n", c, f, w, X, Y);
  free(img);
  return 0;
}
