/* Does the fit-stage objective count a harmonic pair ONCE?
 *
 * (111) and (222) of one orientation have the same unit q-hat, so they land on
 * the same detector pixel at two energies. If the fit stage counted both, a
 * harmonic-rich orientation would win on NMatches for reasons that have nothing
 * to do with how many distinct peaks it explains. calcOverlap,
 * calcOverlapFiltered and writeCalcOverlap in LaueMatchingHeaders.h each keep
 * only the first reflection with a given q-hat; this pins that.
 *
 * WHAT IS EXERCISED: the REAL functions, compiled from the shipped header --
 * calcOverlap (refinement objective, all HKLs), calcOverlapFiltered (the same
 * objective on a prefiltered index list) and writeCalcOverlap (whose return
 * value is the NMatches column of solutions.txt). Nothing is copied here.
 *
 * WHAT IS NOT: the coarse forward-cache stage (pixelClaimed, integer-pixel
 * dedup), the CUDA kernels, FitOrientation's optimiser, and any main(). The
 * geometry is a synthetic transmission set-up (rotTranspose = identity), not a
 * beamline pose -- dedup is a property of the q-hat test, not of the pose.
 *
 * The image is a CONSTANT 1.0 everywhere, so the result does not depend on
 * which pixel a spot rounds to: every predicted spot that is on the detector
 * and in band is lit. That is what makes the controls meaningful:
 *   - (111) alone and (222) alone each count 1: both are on the detector and
 *     inside [Elo, Ehi] (9.30 and 18.60 keV at a = 0.2 nm), so a pair count
 *     of 1 is dedup, not one of them silently falling out of band.
 *   - (111) with the NON-harmonic (1,-1,1) counts 2: the harness can see two
 *     distinct reflections, so "1" is not a harness that only ever says 1.
 *
 * Build and run standalone (Linux; on macOS add the libomp flags):
 *     cc -O2 -fopenmp -I../../c_src -o hns harmonic_no_stack.c \
 *        ../../c_src/nelder_mead.c -lm && ./hns
 */
#include "LaueMatchingHeaders.h"

/* The globals every translation unit that includes the header must define. */
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

static float *g_img;
static double g_recip[3][3];
static double g_rotT[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
/* Detector 50 units downstream, offset 100 in x: (111) lands near pixel
 * (1023, 523) and (1,-1,1) near (1023, 1523) with 0.2-unit pixels. */
static double g_pArr[3] = {100.0, 0.0, 50.0};
static const double PX = 0.2, ELO = 5.0, EHI = 20.0;
/* 180 deg about x: sends (111) to q-hat (1,-1,-1)/sqrt(3), i.e. a Bragg angle
 * of 35.26 deg with the diffracted beam going downstream. */
static double g_euler[3] = {0.0, M_PI, 0.0};

/* nrPos from each of the three fit-stage functions for one hkl list. */
static void count(const int *hkls, int n, int *nCalc, int *nFilt, int *nWrite,
                  int *nSim) {
  double out[3 * MAXSP];
  int idx[16];
  double r;
  memset(out, 0, sizeof(out));
  r = calcOverlap(g_img, g_euler, (int *)hkls, n, NPX, NPX, g_recip, out,
                  MAXSP, g_rotT, g_pArr, PX, PX, ELO, EHI, 0.0);
  /* Constant image of 1.0: result = nrPos * sqrt(nrPos), so recover nrPos. */
  *nCalc = (int)floor(cbrt(r * r) + 0.5);
  for (int i = 0; i < n; i++)
    idx[i] = i;
  memset(out, 0, sizeof(out));
  r = calcOverlapFiltered(g_img, g_euler, (int *)hkls, idx, n, NPX, NPX,
                          g_recip, out, MAXSP, g_rotT, g_pArr, PX, PX, ELO,
                          EHI, 0.0);
  *nFilt = (int)floor(cbrt(r * r) + 0.5);
  memset(out, 0, sizeof(out));
  *nWrite = writeCalcOverlap(g_img, g_euler, (int *)hkls, n, NPX, NPX, g_recip,
                             out, MAXSP, g_rotT, g_pArr, PX, PX, ELO, EHI, 0.0,
                             NULL, 0, nSim, 0);
}

static int check(const char *label, const int *hkls, int n, int want) {
  int a, b, c, sim = -1;
  count(hkls, n, &a, &b, &c, &sim);
  int ok = (a == want && b == want && c == want && sim == want);
  printf("%-28s calcOverlap=%d calcOverlapFiltered=%d writeCalcOverlap"
         "(NMatches)=%d NSpotsCalc=%d want=%d %s\n",
         label, a, b, c, sim, want, ok ? "OK" : "*** FAIL ***");
  return ok ? 0 : 1;
}

int main(void) {
  double lat[6] = {0.2, 0.2, 0.2, 90.0, 90.0, 90.0};
  sg_num = 225;
  calcRecipArray(lat, sg_num, g_recip);
  size_t np = (size_t)NPX * NPX;
  g_img = (float *)malloc(np * sizeof(float));
  if (g_img == NULL) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  for (size_t i = 0; i < np; i++)
    g_img[i] = 1.0f;

  const int h111[] = {1, 1, 1};
  const int h222[] = {2, 2, 2};
  const int pair[] = {1, 1, 1, 2, 2, 2};
  const int pairRev[] = {2, 2, 2, 1, 1, 1};
  const int ctrl[] = {1, 1, 1, 1, -1, 1};

  int bad = 0;
  bad |= check("(111) alone", h111, 1, 1);
  bad |= check("(222) alone", h222, 1, 1);
  bad |= check("(111)+(222) harmonic pair", pair, 2, 1);
  bad |= check("(222)+(111) harmonic pair", pairRev, 2, 1);
  bad |= check("(111)+(1-11) control", ctrl, 2, 2);
  free(g_img);
  if (bad) {
    printf("FAIL\n");
    return 2;
  }
  printf("PASS\n");
  return 0;
}
