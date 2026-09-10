/* Is the parallel, clamp-hoisted coarse-fit blur BIT-IDENTICAL to the serial
 * one it replaced?
 *
 * WHY THIS EXISTS AS A C TEST. The coarse-fit blur in fitAndWriteOrientations()
 * was the whole streaming pipeline's bottleneck: measured 2026-09-09 on sentosa
 * (H200, 64 logical cores) against the 100M-orientation sampleH cache, the daemon
 * spent a median 464 ms/image in the fitting stage of which 449 ms was this one
 * blur, single-threaded, while 41 ms went to the GPU and the other 15 fit
 * threads idled (n=150, one run). Parallelising it took the completion-bounded
 * 150-frame time from ~72.8 s to ~10.1 s, about 7x; independent replications
 * under load spanned 5.2-6.9x.
 *
 * That change is only safe if it is EXACT, and the end-to-end solutions
 * comparison CANNOT establish that: the streaming daemon assigns GrainNr
 * nondeterministically, so the same binary on the same input disagrees with
 * itself on 54 of 1661 solution lines. A gate that fails on unchanged code
 * cannot exonerate changed code. This compares the function directly.
 *
 * Both implementations are carried here in full: `blur_serial` is verbatim the
 * pre-change body, `blur_omp` verbatim the current one. Exactness holds because
 * every output pixel still accumulates the same klen products in the same k
 * order -- parallelising over y only splits disjoint output rows, and hoisting
 * the clamp only skips branches that could never have fired in the interior.
 *
 * Build and run standalone:
 *     cc -O3 -fopenmp -o blurx coarse_blur_exactness.c -lm && ./blurx
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#else
/* No OpenMP: the "parallel" path still runs, single-threaded, so exactness is
 * still checked -- but the speedup is not, and the Python test says so. Use
 * clock() rather than returning 0.0, which reads like broken instrumentation. */
#include <time.h>
static double omp_get_wtime(void) {
  return (double)clock() / (double)CLOCKS_PER_SEC;
}
#endif

/* ---- SERIAL: the implementation that shipped before 2026-09-09 ---------- */
static void blur_serial(const float *in, float *out, int nx, int ny,
                        double sigma) {
  int rad = (int)(3.0 * sigma + 0.5);
  if (rad < 1)
    rad = 1;
  int klen = 2 * rad + 1;
  double *kern = (double *)malloc((size_t)klen * sizeof(double));
  float *tmp = (float *)malloc((size_t)nx * ny * sizeof(float));
  if (!kern || !tmp) {
    fprintf(stderr, "alloc failed\n");
    exit(1);
  }
  double ksum = 0.0;
  for (int i = -rad; i <= rad; i++) {
    double w = exp(-(double)(i * i) / (2.0 * sigma * sigma));
    kern[i + rad] = w;
    ksum += w;
  }
  for (int i = 0; i < klen; i++)
    kern[i] /= ksum;
  for (int y = 0; y < ny; y++)
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
  for (int y = 0; y < ny; y++)
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
  free(tmp);
  free(kern);
}

/* ---- PARALLEL + CLAMP HOISTED: must stay in step with
 * gaussianBlurImage() in LaueMatchingHeaders.h ---------------------------- */
static void blur_omp(const float *in, float *out, int nx, int ny, double sigma,
                     int nThreads) {
  int rad = (int)(3.0 * sigma + 0.5);
  if (rad < 1)
    rad = 1;
  int klen = 2 * rad + 1;
  double *kern = (double *)malloc((size_t)klen * sizeof(double));
  float *tmp = (float *)malloc((size_t)nx * ny * sizeof(float));
  if (!kern || !tmp) {
    fprintf(stderr, "alloc failed\n");
    exit(1);
  }
  double ksum = 0.0;
  for (int i = -rad; i <= rad; i++) {
    double w = exp(-(double)(i * i) / (2.0 * sigma * sigma));
    kern[i + rad] = w;
    ksum += w;
  }
  for (int i = 0; i < klen; i++)
    kern[i] /= ksum;
  if (nThreads < 1)
    nThreads = 1;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for (int y = 0; y < ny; y++) {
    const float *inRow = in + (size_t)y * nx;
    float *tmpRow = tmp + (size_t)y * nx;
    int xlo = rad < nx ? rad : nx;
    int xhi = nx - rad > xlo ? nx - rad : xlo;
    for (int x = 0; x < xlo; x++) {
      double acc = 0.0;
      for (int k = -rad; k <= rad; k++) {
        int xx = x + k;
        if (xx < 0)
          xx = 0;
        else if (xx >= nx)
          xx = nx - 1;
        acc += kern[k + rad] * inRow[xx];
      }
      tmpRow[x] = (float)acc;
    }
    for (int x = xlo; x < xhi; x++) {
      double acc = 0.0;
      const float *w = inRow + x - rad;
      for (int k = 0; k < klen; k++)
        acc += kern[k] * w[k];
      tmpRow[x] = (float)acc;
    }
    for (int x = xhi; x < nx; x++) {
      double acc = 0.0;
      for (int k = -rad; k <= rad; k++) {
        int xx = x + k;
        if (xx < 0)
          xx = 0;
        else if (xx >= nx)
          xx = nx - 1;
        acc += kern[k + rad] * inRow[xx];
      }
      tmpRow[x] = (float)acc;
    }
  }
  {
    int ylo = rad < ny ? rad : ny;
    int yhi = ny - rad > ylo ? ny - rad : ylo;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for (int y = 0; y < ny; y++) {
      float *outRow = out + (size_t)y * nx;
      if (y >= ylo && y < yhi) {
        const float *base = tmp + (size_t)(y - rad) * nx;
        for (int x = 0; x < nx; x++) {
          double acc = 0.0;
          const float *col = base + x;
          for (int k = 0; k < klen; k++)
            acc += kern[k] * col[(size_t)k * nx];
          outRow[x] = (float)acc;
        }
      } else {
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
          outRow[x] = (float)acc;
        }
      }
    }
  }
  free(tmp);
  free(kern);
}

/* A Laue-like frame: sparse bright spots on zero. Exactness does not depend on
 * the data being real, but the sparsity and spot size match what the matcher
 * actually sees (measured: ~85k lit pixels of 4.2M, ~100 components). */
static void synth(float *img, int nx, int ny, int nspots, unsigned seed) {
  memset(img, 0, (size_t)nx * ny * sizeof(float));
  unsigned s = seed ? seed : 1u;
  for (int i = 0; i < nspots; i++) {
    s = s * 1103515245u + 12345u;
    int cy = 20 + (int)((s >> 8) % (unsigned)(ny - 40));
    s = s * 1103515245u + 12345u;
    int cx = 20 + (int)((s >> 8) % (unsigned)(nx - 40));
    s = s * 1103515245u + 12345u;
    double amp = 500.0 + (double)((s >> 8) % 4500u);
    for (int dy = -6; dy <= 6; dy++)
      for (int dx = -6; dx <= 6; dx++) {
        int y = cy + dy, x = cx + dx;
        if (y < 0 || y >= ny || x < 0 || x >= nx)
          continue;
        img[(size_t)y * nx + x] +=
            (float)(amp * exp(-(dx * dx + dy * dy) / (2.0 * 2.0 * 2.0)));
      }
  }
}

int main(int argc, char **argv) {
  int nx = (argc > 1) ? atoi(argv[1]) : 512;
  int ny = nx;
  int nth = (argc > 2) ? atoi(argv[2]) : 4;
  /* 4.0 and 12.0 are autoCoarseSigma()'s clamp floor and ceiling; 7.77 is the
   * value the 0.4-degree 100M database produces at 1-ID/34-ID-E geometry. */
  double sigmas[3] = {4.0, 7.77, 12.0};
  size_t n = (size_t)nx * ny;
  float *in = (float *)malloc(n * sizeof(float));
  float *a = (float *)malloc(n * sizeof(float));
  float *b = (float *)malloc(n * sizeof(float));
  if (!in || !a || !b) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  synth(in, nx, ny, nx * ny / 4096, 12345u);
  size_t lit = 0;
  for (size_t i = 0; i < n; i++)
    if (in[i] != 0.0f)
      lit++;
  printf("frame %dx%d, %zu lit pixels, %d threads\n", nx, ny, lit, nth);
  int bad = 0;
  for (int si = 0; si < 3; si++) {
    double sg = sigmas[si];
    double t0 = omp_get_wtime();
    blur_serial(in, a, nx, ny, sg);
    double ts = (omp_get_wtime() - t0) * 1000.0;
    double t1 = omp_get_wtime();
    blur_omp(in, b, nx, ny, sg, nth);
    double tp = (omp_get_wtime() - t1) * 1000.0;
    size_t ndiff = 0;
    double maxabs = 0.0;
    for (size_t i = 0; i < n; i++)
      if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
        ndiff++;
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > maxabs)
          maxabs = d;
      }
    printf("  sigma %5.2f radius %2d : serial %7.1f ms  parallel %7.1f ms  ",
           sg, (int)(3.0 * sg + 0.5), ts, tp);
    if (ndiff == 0) {
      printf("BIT-IDENTICAL (%zu px)\n", n);
    } else {
      printf("*** DIFFERS: %zu/%zu px, max|d|=%.3e ***\n", ndiff, n, maxabs);
      bad = 1;
    }
  }
  if (bad) {
    printf("FAIL\n");
    return 2;
  }
  printf("PASS\n");
  return 0;
}
