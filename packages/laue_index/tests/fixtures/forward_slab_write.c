/* The forward-cache writer all three binaries use: beginForwardCacheWrite(),
 * writeForwardSlabOrDie(), finishForwardCacheOrDie(), abandonForwardCache(),
 * releaseForwardCacheLock().
 *
 * WHAT IS EXERCISED: the real functions from LaueMatchingHeaders.h, and the
 * atomic-publication contract: the cache is written as
 * <out>.partial.<host>.<pid> and renamed onto <out> only after fsync/close
 * succeed, one writer at a time under an fcntl lock on <out>.lock.
 *
 * Every mode first puts a PRIOR final file at <out> containing "OLD" (except
 * `empty`), so "the final name is untouched" is checked against real content.
 *   ok <out>          open partial; check it exists and <out> still says OLD;
 *                     write a 1 MB slab at a non-zero offset; check <out> STILL
 *                     says OLD; finish; check the partial is gone and <out>
 *                     now holds the slab byte for byte. Exit 0 / 3.
 *   fail <out>        write through an invalid fd (EBADF): FATAL, the partial
 *                     removed, exit non-zero. Caller checks <out> is OLD.
 *   finish <out>      finish a 1-byte partial: <out> replaced, partial gone.
 *   finish-badfd <out> finish(-1): fsync EBADF is FATAL; partial removed, <out>
 *                     left OLD.
 *   finish-pipe <out> finish on a pipe: fsync EINVAL (Linux) / ENOTSUP (macOS)
 *                     only WARNS; the partial is still published.
 *   killed <out>      write, then SIGKILL itself before finish -- the SIGKILL
 *                     case the pattern exists for. Caller checks <out> is OLD
 *                     and only a .partial.<pid> leftover exists.
 *   empty <out>       ForwardFile "" must be refused before anything is created.
 *   sim <out> <ms>    NO prior file is created. begin(); if a sibling
 *                     published while this waited print REUSED; else print
 *                     SIMULATING, hold the lock <ms> ms, write a complete
 *                     4-orientation, MaxNrLaueSpots=2 cache (40 bytes), finish,
 *                     release, print PUBLISHED. Two of these run together are
 *                     the sibling-shard case.
 * WHAT IS NOT: a genuine short write (not reproducible without fault
 * injection; the retry loop is covered by a source check), or the
 * multi-threaded callers in the three mains.
 *
 * Build: cc -std=gnu99 -O2 -fopenmp -I../../c_src -o fsw forward_slab_write.c -lm
 */
#include "LaueMatchingHeaders.h"
#include <signal.h>

double tol_LatC[6];
double tol_c_over_a;
double c_over_a_orig;
int sg_num;
double cellVol;
double phiVol;
int nSym;
double Symm[24][4];

static void put_old(const char *out) {
  FILE *f = fopen(out, "wb");
  if (f == NULL) {
    perror("fopen prior");
    exit(2);
  }
  fputs("OLD", f);
  fclose(f);
}

static int says_old(const char *out) {
  char b[8] = {0};
  FILE *f = fopen(out, "rb");
  if (f == NULL)
    return 0;
  size_t n = fread(b, 1, sizeof(b) - 1, f);
  fclose(f);
  return n == 3 && strcmp(b, "OLD") == 0;
}

static int exists(const char *p) { return access(p, F_OK) == 0; }

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s ok|fail|finish|finish-badfd|finish-pipe|killed|"
                    "empty|sim <out> [ms]\n",
            argv[0]);
    return 2;
  }
  const char *mode = argv[1], *out = argv[2];
  static FwdCache fc;
  const char *partial = fc.partial;

  if (strcmp(mode, "empty") == 0) {
    int fd = beginForwardCacheWrite("", 4, 2, &fc);
    printf(fd < 0 ? "REFUSED\n" : "OPENED\n");
    return fd < 0 ? 0 : 3;
  }
  if (strcmp(mode, "sim") == 0) {
    long ms = (argc > 3) ? atol(argv[3]) : 0;
    int fd = beginForwardCacheWrite(out, 4, 2, &fc);
    if (fd == FWD_CACHE_REUSE) {
      printf("REUSED\n");
      return 0;
    }
    if (fd < 0)
      return 2;
    printf("SIMULATING %s locked=%d\n", fc.partial, fc.lockFd >= 0);
    fflush(stdout);
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
    uint16_t cache[20];
    for (int i = 0; i < 20; i++)
      cache[i] = (uint16_t)(getpid() & 0xffff);
    writeForwardSlabOrDie(fd, cache, sizeof(cache), 0, 0, fc.partial);
    finishForwardCacheOrDie(fd, fc.partial, fc.target);
    releaseForwardCacheLock(&fc);
    printf("PUBLISHED\n");
    return 0;
  }

  put_old(out);
  int fd = beginForwardCacheWrite(out, 4, 2, &fc);
  if (fd < 0)
    return 2;
  printf("PARTIAL %s PID %ld\n", partial, (long)getpid());
  fflush(stdout);
  if (!exists(partial) || !says_old(out)) {
    printf("partial missing, or <out> changed at open\n");
    return 3;
  }

  if (strcmp(mode, "ok") == 0) {
    const size_t n = 1u << 19; /* 512k uint16 = 1 MB */
    const size_t off = 4096;
    uint16_t *buf = (uint16_t *)malloc(n * sizeof(uint16_t));
    uint16_t *back = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (buf == NULL || back == NULL)
      return 2;
    for (size_t i = 0; i < n; i++)
      buf[i] = (uint16_t)(i * 2654435761u >> 7);
    writeForwardSlabOrDie(fd, buf, n * sizeof(uint16_t), off, 0, partial);
    if (!says_old(out)) {
      printf("<out> changed before finish\n");
      return 3;
    }
    finishForwardCacheOrDie(fd, partial, fc.target);
    if (exists(partial)) {
      printf("partial still present after finish\n");
      return 3;
    }
    int rfd = open(out, O_RDONLY);
    if (rfd < 0)
      return 3;
    ssize_t got = pread(rfd, back, n * sizeof(uint16_t), (off_t)off);
    close(rfd);
    if (got != (ssize_t)(n * sizeof(uint16_t)) ||
        memcmp(buf, back, n * sizeof(uint16_t)) != 0) {
      printf("MISMATCH\n");
      return 3;
    }
    free(buf);
    free(back);
    printf("PASS\n");
    return 0;
  }
  if (strcmp(mode, "fail") == 0) {
    uint16_t buf[64] = {0};
    close(fd);
    writeForwardSlabOrDie(-1, buf, sizeof(buf), 0, 7, partial);
    printf("returned without exiting\n"); /* must never be reached */
    return 0;
  }
  if (strcmp(mode, "killed") == 0) {
    uint16_t buf[64] = {0};
    writeForwardSlabOrDie(fd, buf, sizeof(buf), 0, 0, partial);
    kill(getpid(), SIGKILL);
    return 0; /* not reached */
  }
  if (write(fd, "x", 1) != 1) {
    perror("write");
    return 2;
  }
  if (strcmp(mode, "finish") == 0) {
    finishForwardCacheOrDie(fd, partial, fc.target);
  } else if (strcmp(mode, "finish-badfd") == 0) {
    close(fd);
    finishForwardCacheOrDie(-1, partial, fc.target);
    printf("returned without exiting\n"); /* must never be reached */
  } else if (strcmp(mode, "finish-pipe") == 0) {
    close(fd);
    int p[2];
    if (pipe(p) != 0) {
      perror("pipe");
      return 2;
    }
    close(p[0]);
    finishForwardCacheOrDie(p[1], partial, fc.target);
  } else {
    return 2;
  }
  printf("FINISHED\n");
  return 0;
}
