/* Can FitOrientation reach a correction when the seed has Phi ~ 0?
 *
 * Before 0.8.0 the fit bounded the three ZXZ Euler angles by +-tol about the
 * seed. At Phi ~ 0, psi and theta both turn about z, so a correction about the
 * in-plane axis perpendicular to the line of nodes is out of reach unless
 * Phi_seed exceeds ~ delta / sin(tol). The fit now refines a small rotation
 * vector d about the seed, R = exp([d]x) R_seed, |d_i| <= tol: the same
 * neighbourhood for every orientation.
 *
 * WHAT IS EXERCISED: the REAL FitOrientation (vendored Nelder-Mead),
 * writeCalcOverlap, autoCoarseSigma and gaussianBlurImage from the header. The
 * image lights exactly the pixels the header predicts for the TRUTH
 * orientation (writeCalcOverlap on a constant image), then is blurred as the
 * pipeline's coarse stage blurs it, so the objective's optimum is the truth
 * and a grid-sized seed error is inside its capture radius.
 *
 * Usage: fit_near_phi0 PHI_SEED_DEG AXIS_X AXIS_Y AXIS_Z DELTA_DEG
 *   -> prints "MISO <deg between fit and truth> SEEDMISO <deg seed-truth> N <lit>"
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
#define MAXSP 400

static void axisAngle(const double ax[3], double deg, double R[3][3]) {
  double n = sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
  double u[3] = {ax[0] / n, ax[1] / n, ax[2] / n}, t = deg * deg2rad;
  double c = cos(t), s = sin(t), C = 1 - c;
  R[0][0] = c + u[0] * u[0] * C;
  R[0][1] = u[0] * u[1] * C - u[2] * s;
  R[0][2] = u[0] * u[2] * C + u[1] * s;
  R[1][0] = u[1] * u[0] * C + u[2] * s;
  R[1][1] = c + u[1] * u[1] * C;
  R[1][2] = u[1] * u[2] * C - u[0] * s;
  R[2][0] = u[2] * u[0] * C - u[1] * s;
  R[2][1] = u[2] * u[1] * C + u[0] * s;
  R[2][2] = c + u[2] * u[2] * C;
}

static double misoDeg(double A[3][3], double B[3][3]) {
  double tr = 0; /* angle of A^T B, no symmetry: truth and fit are close */
  for (int i = 0; i < 3; i++)
    for (int k = 0; k < 3; k++)
      tr += A[k][i] * B[k][i];
  double c = (tr - 1) / 2;
  c = c > 1 ? 1 : (c < -1 ? -1 : c);
  return acos(c) * rad2deg;
}

int main(int argc, char **argv) {
  if (argc != 6)
    return 2;
  double phiSeed = atof(argv[1]) * deg2rad;
  double ax[3] = {atof(argv[2]), atof(argv[3]), atof(argv[4])};
  double delta = atof(argv[5]);
  double lat[6] = {0.36, 0.36, 0.36, 90, 90, 90}, recip[3][3];
  sg_num = 225;
  calcRecipArray(lat, sg_num, recip);
  double rotT[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  double pArr[3] = {0.0, 0.0, 100.0};
  int nh = 0, hkls[3 * 729];
  for (int h = -4; h <= 4; h++)
    for (int k = -4; k <= 4; k++)
      for (int l = -4; l <= 4; l++)
        if (h || k || l) {
          hkls[3 * nh] = h;
          hkls[3 * nh + 1] = k;
          hkls[3 * nh + 2] = l;
          nh++;
        }
  double eulerSeed[3] = {0.3, phiSeed, 0.2}, Rs[3][3], Rd[3][3], Rt[3][3];
  Euler2OrientMat(eulerSeed, Rs);
  axisAngle(ax, delta, Rd);
  MatrixMultF33(Rd, Rs, Rt);
  double eulerTruth[3];
  OrientMat2Euler(Rt, eulerTruth);
  /* truth spot pixels: writeCalcOverlap on a constant image lists them all */
  float *ones = (float *)malloc((size_t)NPX * NPX * sizeof(float));
  float *img = (float *)calloc((size_t)NPX * NPX, sizeof(float));
  if (ones == NULL || img == NULL)
    return 1;
  for (size_t i = 0; i < (size_t)NPX * NPX; i++)
    ones[i] = 1.0f;
  double *out = (double *)calloc(3 * MAXSP, sizeof(double));
  FILE *f = tmpfile();
  int nSim = 0;
  writeCalcOverlap(ones, eulerTruth, hkls, nh, NPX, NPX, recip, out, MAXSP,
                   rotT, pArr, 0.2, 0.2, 5.0, 30.0, 0.0, f, 1, &nSim,
                   NOT_STREAMING);
  rewind(f);
  char line[512];
  int lit = 0;
  while (fgets(line, sizeof(line), f)) {
    int g, s, h, k, l, x, y;
    if (sscanf(line, "%d %d %d %d %d %d %d", &g, &s, &h, &k, &l, &x, &y) == 7) {
      img[(size_t)y * NPX + x] = 100.0f;
      lit++;
    }
  }
  fclose(f);
  float *blur = (float *)malloc((size_t)NPX * NPX * sizeof(float));
  if (blur == NULL)
    return 1;
  gaussianBlurImage(img, blur, NPX, NPX, autoCoarseSigma(pArr[2], 0.2, 0.4), 1);
  double eulerFit[3], latUpd[6], mv = 0;
  memset(out, 0, 3 * MAXSP * sizeof(double));
  FitOrientation(blur, eulerSeed, hkls, nh, NPX, NPX, recip, out, MAXSP, rotT,
                 pArr, 0.2, 0.2, 5.0, 30.0, 3 * deg2rad, lat, eulerFit, latUpd,
                 &mv, 0, NULL, 0, 1, 0.2 * deg2rad, 0.0);
  double Rf[3][3];
  Euler2OrientMat(eulerFit, Rf);
  printf("MISO %.6f SEEDMISO %.6f N %d\n", misoDeg(Rt, Rf), misoDeg(Rt, Rs),
         lit);
  free(ones);
  free(img);
  free(blur);
  free(out);
  return 0;
}
