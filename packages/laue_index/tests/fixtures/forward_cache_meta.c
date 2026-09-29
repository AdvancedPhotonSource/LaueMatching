/* The forward cache's provenance record (<ForwardFile>.meta.json): format +
 * a key over everything the cached spot positions depend on, the full
 * generating configuration and the params file's text.
 *
 * WHAT IS EXERCISED: the REAL forwardCacheKey, writeForwardCacheMeta,
 * forwardCacheMetaStatus and forwardCacheMetaMatches from LaueMatchingHeaders.h.
 *
 * Usage: forward_cache_meta DIR PARAMFILE -> prints one "CHECK <name> <0|1>"
 * per case (1 = as expected) and exits 0 only if every case passed. The record
 * it writes is left at DIR/forward.bin.meta.json for the test to parse.
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

static int bad = 0;
static void check(const char *name, int ok) {
  printf("CHECK %s %d\n", name, ok);
  if (!ok)
    bad = 1;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s DIR PARAMFILE\n", argv[0]);
    return 2;
  }
  char cache[4096];
  snprintf(cache, sizeof(cache), "%s/forward.bin", argv[1]);
  double lat[6] = {0.2921, 0.2921, 0.4665, 90, 90, 120};
  double P[3] = {0.0288, 0.0027, 0.513}, R[3] = {-1.2, -1.21, -1.22};
  int hkls[6] = {1, 0, 0, 0, 0, 2};
  uint64_t k0 = forwardCacheKey(194, lat, P, R, 2e-4, 2e-4, 2048, 2048, 5, 30,
                                500, 1000, hkls, 2);
  /* Every input the cached pixels depend on must change the key. */
  double lat1[6] = {0.2921, 0.2921, 0.4666, 90, 90, 120};
  double P1[3] = {0.0288, 0.0027, 0.5131}, R1[3] = {-1.2, -1.21, -1.2201};
  int hkls1[6] = {1, 0, 0, 0, 0, 4};
  check("lattice", forwardCacheKey(194, lat1, P, R, 2e-4, 2e-4, 2048, 2048, 5,
                                   30, 500, 1000, hkls, 2) != k0);
  check("space_group", forwardCacheKey(191, lat, P, R, 2e-4, 2e-4, 2048, 2048,
                                       5, 30, 500, 1000, hkls, 2) != k0);
  check("P_Array", forwardCacheKey(194, lat, P1, R, 2e-4, 2e-4, 2048, 2048, 5,
                                   30, 500, 1000, hkls, 2) != k0);
  check("R_Array", forwardCacheKey(194, lat, P, R1, 2e-4, 2e-4, 2048, 2048, 5,
                                   30, 500, 1000, hkls, 2) != k0);
  check("pixel_size", forwardCacheKey(194, lat, P, R, 1e-4, 2e-4, 2048, 2048,
                                      5, 30, 500, 1000, hkls, 2) != k0);
  check("panel", forwardCacheKey(194, lat, P, R, 2e-4, 2e-4, 2048, 1024, 5, 30,
                                 500, 1000, hkls, 2) != k0);
  check("energy", forwardCacheKey(194, lat, P, R, 2e-4, 2e-4, 2048, 2048, 5,
                                  25, 500, 1000, hkls, 2) != k0);
  check("max_spots", forwardCacheKey(194, lat, P, R, 2e-4, 2e-4, 2048, 2048, 5,
                                     30, 400, 1000, hkls, 2) != k0);
  check("n_orients", forwardCacheKey(194, lat, P, R, 2e-4, 2e-4, 2048, 2048, 5,
                                     30, 500, 999, hkls, 2) != k0);
  check("hkls", forwardCacheKey(194, lat, P, R, 2e-4, 2e-4, 2048, 2048, 5, 30,
                                500, 1000, hkls1, 2) != k0);
  check("deterministic", forwardCacheKey(194, lat, P, R, 2e-4, 2e-4, 2048,
                                         2048, 5, 30, 500, 1000, hkls,
                                         2) == k0);
  /* No record (a pre-0.8.0 cache): status 1, missing -> the binaries rebuild. */
  FILE *f = fopen(cache, "wb");
  if (f == NULL)
    return 1;
  fwrite("0123456789", 1, 10, f);
  fclose(f);
  check("missing_is_status_1", forwardCacheMetaStatus(cache, k0) == 1);
  check("missing_does_not_match", !forwardCacheMetaMatches(cache, k0));
  FwdCacheInfo info = makeFwdCacheInfo(194, lat, P, R, 2e-4, 2e-4, 2048, 2048, 5,
                                       30, 500, 1000, 2, argv[2], "db.bin",
                                       "hkls.csv", "LaueMatchingCPU", 12.5, 8);
  check("write_record", writeForwardCacheMeta(cache, k0, &info) == 0);
  check("matching_is_status_0", forwardCacheMetaStatus(cache, k0) == 0);
  check("matching_matches", forwardCacheMetaMatches(cache, k0));
  /* A record for ANOTHER configuration: status 2 -> the binaries refuse. */
  check("other_key_is_status_2", forwardCacheMetaStatus(cache, k0 ^ 1) == 2);
  check("other_key_does_not_match", !forwardCacheMetaMatches(cache, k0 ^ 1));
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad ? 2 : 0;
}
