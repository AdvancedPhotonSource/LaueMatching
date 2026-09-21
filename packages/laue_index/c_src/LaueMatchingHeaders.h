//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//
// Shared header for LaueMatchingCPU and LaueMatchingGPU.
// All symmetry tables, math utilities, and diffraction functions live
// here so they are compiled once per translation unit and stay in sync.
//

#ifndef LAUE_MATCHING_HEADERS_H
#define LAUE_MATCHING_HEADERS_H

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include "nelder_mead.h"
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <malloc.h>
#endif
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

// ── Constants ───────────────────────────────────────────────────────────
#define deg2rad 0.0174532925199433
#define rad2deg 57.2957795130823
#define NrValsResults 2
#define MaxNHKLS 200000

// ── Forward-simulation cache ────────────────────────────────────────────
// Can an existing cache be used? ONE implementation for all three binaries.
//
// This check used to live only in LaueMatchingCPU.c. The CUDA binaries mapped
// whatever sat at the path, so a 0-byte leftover -- which the C itself creates
// with O_CREAT when a run is interrupted mid-write -- made LaueMatchingGPU
// print "file was found. Will not do forward simulation." and then take SIGBUS
// on first touch, 12.2 GB of mapping backed by an empty file. Three copies of
// a check is how one of them goes missing; hence one function, three callers.
//
// The size IS the contract: entry orientNr occupies (1 + 2*maxNrSpots) uint16
// at a fixed offset, so a file of the right length is structurally valid and a
// file of any other length cannot be.
//
// It does NOT prove the cache came from THIS geometry. A right-sized cache
// built with a different detector, energy range or HKL list is accepted and is
// silently wrong -- the same limit the CPU check always had. Regenerate on any
// doubt; the cost is minutes, and a wrong cache is not visibly wrong.
static inline bool forwardCacheUsable(const char *outfn, size_t nrOrients,
                                      int maxNrSpots) {
  if (outfn == NULL || outfn[0] == '\0') {
    // Don't blindly open a blank/default path and pick up a stale cache.
    printf("No ForwardFile specified. Running in simulation mode!\n");
    return false;
  }
  printf("Trying to see if the forward simulation exists. Looking for %s "
         "file.\n",
         outfn);
  int fd = open(outfn, O_RDONLY);
  if (fd < 0) { // open returns -1 on error, never 0
    printf("Could not read the forward simulation file (%s). Running in "
           "simulation mode!\n",
           strerror(errno));
    return false;
  }
  const size_t expected =
      nrOrients * (size_t)(1 + 2 * maxNrSpots) * sizeof(uint16_t);
  struct stat cst;
  if (fstat(fd, &cst) != 0) {
    printf("Could not stat the forward cache %s (%s). Running in simulation "
           "mode!\n",
           outfn, strerror(errno));
    close(fd);
    return false;
  }
  if ((size_t)cst.st_size != expected) {
    printf("Forward cache %s is %lld B, expected %zu B for %zu orientations x "
           "(1+2*%d) uint16. Ignoring it and running in simulation mode.\n",
           outfn, (long long)cst.st_size, expected, nrOrients, maxNrSpots);
    close(fd);
    return false;
  }
  close(fd);
  printf("%s was found and is the expected size. Will not do forward "
         "simulation.\n",
         outfn);
  return true;
}

// ── Forward-simulation cache: begin, write, finish, abandon ─────────────
// ATOMIC PUBLICATION. The cache is never written under its final name. It is
// written to `<ForwardFile>.partial.<host>.<pid>` in the SAME directory (so
// rename is atomic on the same filesystem) and renamed onto ForwardFile only
// after every slab is written and fsync/close succeeded. So the final name
// only ever holds a complete cache (or whatever was there before), and:
//   - a run that FAILS removes its partial file (abandonForwardCache);
//   - a run that is KILLED (SIGKILL, node loss, OOM) leaves only a
//     `*.partial.*` file, which forwardCacheUsable() never looks at. Such
//     leftovers are dead weight and can be deleted at any time; the next
//     writer that holds the lock deletes them itself.
// Before this, each thread pwrote into ForwardFile itself, so a failed or
// killed run could leave a file with a zero-filled hole that other threads'
// slabs had extended to exactly the size forwardCacheUsable() accepts; a later
// DoFwd 0 run then read the hole as "no spots".
//
// ONE WRITER AT A TIME. Sibling shards on one host (and shards on other hosts
// sharing the filesystem) share one ForwardFile and cold-start together.
// Unserialised, each would write its own ~12 GB partial -- 4 shards = 48 GB,
// and under /dev/shm that is RAM. So simulate-and-publish runs under an
// advisory fcntl() write lock on `<ForwardFile>.lock` (F_SETLKW). fcntl locks
// are what NFS supports (through lockd/NLM, or natively in NFSv4); flock() is
// not reliably cross-host on NFS. A waiter RE-CHECKS after acquiring the lock:
// if a sibling published a new ForwardFile meanwhile it reads that instead of
// simulating. If locking is unsupported (ENOLCK, ENOSYS, EOPNOTSUPP, EINVAL,
// no lockd) or the lock file cannot be opened, a WARNING is printed and the
// run proceeds UNLOCKED -- the pre-lock behaviour -- rather than risk hanging.
// The lock is released by the kernel if the holder dies, so it cannot be left
// held. The .lock file itself is left in place (deleting lock files races).
//
// Ordering in every main(): forwardCacheUsable() (DoFwd 0) runs BEFORE any of
// this, so an existing, possibly stale ForwardFile is judged before a write
// starts and is never read while one is in progress; the rename then replaces
// it whole. Only LaueMatchingGPUStream reads the cache back by name after
// simulating, and it does so after finishForwardCacheOrDie() has renamed it.
//
// Not covered: the directory entry of the rename is not fsynced, so after a
// power loss the final name may still hold the PREVIOUS file (never a mix).
#define FWD_CACHE_REUSE (-2)

typedef struct {
  char target[PATH_MAX];      // ForwardFile, symlinks resolved
  char partial[PATH_MAX + 320];
  char lock[PATH_MAX + 16];
  int lockFd;                 // -1: not locked (none taken, or unsupported)
} FwdCache;

// Resolve ForwardFile to where the bytes should live. A symlinked cache is
// replaced at its TARGET (on the target's filesystem), not by a regular file
// in the link's directory, which is what open() used to write through to.
static inline int resolveForwardCachePath(const char *outfn, char *target) {
  if (realpath(outfn, target) != NULL)
    return 0;
  if (errno != ENOENT) {
    fprintf(stderr, "FATAL: cannot resolve ForwardFile %s: %s\n", outfn,
            strerror(errno));
    return -1;
  }
  // Does not exist yet: resolve its directory and append the name.
  char dir[PATH_MAX], rdir[PATH_MAX];
  const char *slash = strrchr(outfn, '/');
  const char *base = slash ? slash + 1 : outfn;
  if (slash == NULL)
    strcpy(dir, ".");
  else if (slash == outfn)
    strcpy(dir, "/");
  else {
    size_t n = (size_t)(slash - outfn);
    if (n >= sizeof(dir)) {
      fprintf(stderr, "FATAL: ForwardFile path too long: %s\n", outfn);
      return -1;
    }
    memcpy(dir, outfn, n);
    dir[n] = '\0';
  }
  if (base[0] == '\0' || realpath(dir, rdir) == NULL) {
    fprintf(stderr, "FATAL: the directory of ForwardFile %s does not exist or "
                    "cannot be resolved.\n",
            outfn);
    return -1;
  }
  int n = snprintf(target, PATH_MAX, "%s%s%s", rdir,
                   strcmp(rdir, "/") == 0 ? "" : "/", base);
  if (n < 0 || n >= PATH_MAX) {
    fprintf(stderr, "FATAL: ForwardFile path too long: %s\n", outfn);
    return -1;
  }
  return 0;
}

// Take the writer lock. Blocks (F_SETLKW) while a sibling holds it; says so
// first. Returns the lock fd, or -1 after a WARNING when locking is not
// available -- the caller then proceeds unlocked.
static inline int lockForwardCache(const char *lockfn) {
  // 0666 subject to umask: every beamline account that shares the cache must
  // be able to open the lock file O_RDWR, which a write lock requires.
  int fd = open(lockfn, O_RDWR | O_CREAT, 0666);
  if (fd < 0) {
    fprintf(stderr, "WARNING: cannot open the forward-cache lock %s (%s); "
                    "simulating WITHOUT the lock.\n",
            lockfn, strerror(errno));
    return -1;
  }
  struct flock fl;
  memset(&fl, 0, sizeof(fl));
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET; // l_start = l_len = 0: the whole file
  if (fcntl(fd, F_SETLK, &fl) == 0)
    return fd;
  if (errno == EACCES || errno == EAGAIN) {
    printf("Another process is simulating this forward cache; waiting for "
           "%s ...\n",
           lockfn);
    fflush(stdout);
    int rc;
    do {
      rc = fcntl(fd, F_SETLKW, &fl);
    } while (rc != 0 && errno == EINTR);
    if (rc == 0) {
      printf("Forward-cache lock acquired.\n");
      return fd;
    }
  }
  fprintf(stderr, "WARNING: advisory locking is not available for %s (%s); "
                  "simulating WITHOUT the lock. Concurrent writers each need "
                  "space for a full partial cache.\n",
          lockfn, strerror(errno));
  close(fd);
  return -1;
}

static inline void releaseForwardCacheLock(FwdCache *fc) {
  if (fc->lockFd >= 0)
    close(fc->lockFd); // closing releases this process's fcntl lock
  fc->lockFd = -1;
}

// With the lock held no other (locking) writer is active, so any partial file
// for this ForwardFile belongs to a dead one. Delete them.
static inline void sweepStalePartials(const char *target) {
  char dir[PATH_MAX];
  const char *slash = strrchr(target, '/'); // target is absolute
  size_t n = (size_t)(slash - target);
  if (n == 0)
    n = 1; // "/"
  memcpy(dir, target, n);
  dir[n] = '\0';
  char prefix[PATH_MAX];
  snprintf(prefix, sizeof(prefix), "%s.partial.", slash + 1);
  size_t plen = strlen(prefix);
  DIR *d = opendir(dir);
  if (d == NULL)
    return;
  struct dirent *e;
  while ((e = readdir(d)) != NULL) {
    if (strncmp(e->d_name, prefix, plen) != 0)
      continue;
    char p[2 * PATH_MAX + 2];
    int pn = snprintf(p, sizeof(p), "%s/%s", strcmp(dir, "/") == 0 ? "" : dir,
                      e->d_name);
    if (pn < 0 || (size_t)pn >= sizeof(p))
      continue;
    if (unlink(p) == 0)
      printf("Removed a partial forward cache left by a dead writer: %s\n", p);
    else
      fprintf(stderr, "WARNING: could not remove stale partial %s: %s\n", p,
              strerror(errno));
  }
  closedir(d);
}

// Everything before the first slab write. Returns:
//   >= 0            the partial's fd: simulate, write, finishForwardCacheOrDie,
//                   then releaseForwardCacheLock;
//   FWD_CACHE_REUSE a sibling published a usable ForwardFile while this run
//                   waited for the lock: take the DoFwd 0 read path instead;
//   -1              fatal, reason printed; exit non-zero.
// The re-check accepts only a file that CHANGED (new inode) while waiting, so
// an explicit DoFwd 1 over an existing right-sized cache still re-simulates.
static inline int beginForwardCacheWrite(const char *outfn, size_t nrOrients,
                                         int maxNrSpots, FwdCache *fc) {
  fc->lockFd = -1;
  fc->partial[0] = '\0';
  if (outfn == NULL || outfn[0] == '\0') {
    fprintf(stderr, "FATAL: no ForwardFile specified; cannot write the "
                    "forward cache.\n");
    return -1;
  }
  if (resolveForwardCachePath(outfn, fc->target) != 0)
    return -1;
  snprintf(fc->lock, sizeof(fc->lock), "%s.lock", fc->target);
  struct stat before, after;
  int existedBefore = (stat(fc->target, &before) == 0);
  fc->lockFd = lockForwardCache(fc->lock);
  if (stat(fc->target, &after) == 0 &&
      (!existedBefore || after.st_ino != before.st_ino ||
       after.st_dev != before.st_dev) &&
      forwardCacheUsable(fc->target, nrOrients, maxNrSpots)) {
    printf("A sibling published %s while this run waited; reading it instead "
           "of simulating.\n",
           fc->target);
    releaseForwardCacheLock(fc);
    return FWD_CACHE_REUSE;
  }
  if (fc->lockFd >= 0)
    sweepStalePartials(fc->target);
  // uname(), not gethostname(): the .cu files define _XOPEN_SOURCE 500, under
  // which some libcs (macOS) do not declare gethostname.
  char host[256];
  struct utsname un;
  if (uname(&un) == 0 && un.nodename[0] != '\0')
    snprintf(host, sizeof(host), "%s", un.nodename);
  else
    strcpy(host, "unknownhost");
  for (char *c = host; *c; c++)
    if (*c == '/')
      *c = '_';
  int n = snprintf(fc->partial, sizeof(fc->partial), "%s.partial.%s.%ld",
                   fc->target, host, (long)getpid());
  if (n < 0 || (size_t)n >= sizeof(fc->partial)) {
    fprintf(stderr, "FATAL: ForwardFile path too long: %s\n", fc->target);
    return -1;
  }
  // 0644 subject to umask (the old open() used 0600, which locked out other
  // beamline accounts that read the same cache). O_EXCL: host+pid is unique,
  // so an existing file means something is badly wrong -- say so, never
  // truncate someone's in-progress slabs.
  int fd = open(fc->partial, O_CREAT | O_EXCL | O_WRONLY, 0644);
  if (fd < 0) {
    fprintf(stderr, "FATAL: could not create %s: %s\n", fc->partial,
            strerror(errno));
    return -1;
  }
  {
    struct stat cur;
    if (stat(fc->target, &cur) == 0) {
      // Keep the replaced cache's permissions (as writing in place did).
      if (fchmod(fd, cur.st_mode & 0777) != 0)
        fprintf(stderr, "WARNING: could not copy the mode of %s: %s\n",
                fc->target, strerror(errno));
      // Fail NOW, not after the whole simulation: in a sticky directory
      // (/dev/shm, /tmp) only the owner may replace the existing file.
      struct stat dst;
      char dir[PATH_MAX];
      const char *slash = strrchr(fc->target, '/');
      size_t dn = (size_t)(slash - fc->target);
      if (dn == 0)
        dn = 1;
      memcpy(dir, fc->target, dn);
      dir[dn] = '\0';
      if (cur.st_uid != geteuid() && stat(dir, &dst) == 0 &&
          (dst.st_mode & S_ISVTX)) {
        fprintf(stderr, "FATAL: %s is owned by another user in a sticky "
                        "directory; this run could not replace it.\n",
                fc->target);
        close(fd);
        unlink(fc->partial);
        return -1;
      }
    }
  }
  return fd;
}

// Remove the PARTIAL cache and exit. Every fatal path taken after the partial
// file was opened ends here: a failed slab write, a failed per-thread
// allocation, a failed fsync/close/rename. The final ForwardFile, if one
// existed before this run, is not touched.
//
// SINGLE ENTRY. On ENOSPC/EDQUOT every writer thread fails at about the same
// moment, and POSIX leaves concurrent exit() undefined (glibc >= 2.37
// serialises it; older beamline glibc may not). The named critical section
// admits one thread: it unlinks and calls exit(); any other thread arriving
// blocks at the critical and never returns -- the process ends under it. The
// fcntl lock, if held, is released by the kernel at exit.
static inline void abandonForwardCache(const char *partialfn) {
#pragma omp critical(laue_abandon_forward_cache)
  {
    if (unlink(partialfn) == 0)
      fprintf(stderr, "  Removed the partial forward cache %s; any existing "
                      "ForwardFile was left untouched.\n",
              partialfn);
    else if (errno != ENOENT)
      fprintf(stderr, "  Could NOT remove the partial forward cache %s (%s). "
                      "It is never read and can be deleted.\n",
              partialfn, strerror(errno));
    exit(EXIT_FAILURE);
  }
}

// Write one thread's slab: `nbytes` at `offset` of the partial file, retrying
// short writes; on failure abandon the partial file and exit. ONE
// implementation for all three binaries, for the same reason as
// forwardCacheUsable() above. A second failing thread gets ENOENT from the
// unlink, which is harmless; threads still writing to the unlinked inode are
// harmless too.
//
// The three copies this replaces each failed differently: the CPU loop spun
// forever on rc == 0, LaueMatchingGPU printed and carried on, and the stream
// daemon ignored the return value entirely.
static inline void writeForwardSlabOrDie(int fd, const void *buf,
                                         size_t nbytes, size_t offset,
                                         int procNr, const char *partialfn) {
  size_t done = 0;
  while (done < nbytes) {
    ssize_t rc = pwrite(fd, (const char *)buf + done, nbytes - done,
                        (off_t)(offset + done));
    if (rc < 0 && errno == EINTR)
      continue; // the stream daemon installs handlers without SA_RESTART
    if (rc <= 0) {
      const char *why = (rc < 0) ? strerror(errno) : "wrote 0 bytes";
      fprintf(stderr,
              "FATAL: thread %d forward-cache pwrite failed at offset %zu "
              "(wrote %zu of %zu bytes): %s\n",
              procNr, offset + done, done, nbytes, why);
      abandonForwardCache(partialfn);
    }
    done += (size_t)rc;
  }
}

// Make the partial cache durable, close it, and publish it by rename -- or
// abandon it.
//
// A cache the kernel has not actually committed is the same hazard as a failed
// write: fsync reporting EIO/ENOSPC/EDQUOT means data a later DoFwd 0 run would
// trust may never reach the disk, and on NFS a deferred write error can surface
// only here or at close(). So both take the write-failure path, and nothing is
// published.
//
// EXCEPTION, kept as a WARNING: EINVAL / EROFS / ENOTSUP / EOPNOTSUPP mean
// "this fd or filesystem does not implement fsync" (Linux fsync(2): special
// files; some FUSE and network mounts; tmpfs-like targets). There durability is
// not the filesystem's contract at all, the data written by pwrite is still
// what a reader will see, and treating it as fatal would make the indexer
// unusable on such a mount without making anything safer. EINTR is retried.
// A failing close() is fatal except EINTR, whose fd state is unspecified and
// which does not by itself mean data was lost.
static inline void finishForwardCacheOrDie(int fd, const char *partialfn,
                                           const char *outfn) {
  int rc;
  do {
    rc = fsync(fd);
  } while (rc != 0 && errno == EINTR);
  if (rc != 0) {
    int e = errno;
    if (e == EINVAL || e == EROFS || e == ENOTSUP || e == EOPNOTSUPP) {
      fprintf(stderr,
              "WARNING: fsync is not supported for the forward cache %s (%s); "
              "its durability is not guaranteed by this filesystem.\n",
              partialfn, strerror(e));
    } else {
      fprintf(stderr, "FATAL: fsync of the forward cache %s failed: %s\n",
              partialfn, strerror(e));
      close(fd);
      abandonForwardCache(partialfn);
    }
  }
  if (close(fd) != 0 && errno != EINTR) {
    fprintf(stderr, "FATAL: close of the forward cache %s failed: %s\n",
            partialfn, strerror(errno));
    abandonForwardCache(partialfn);
  }
  if (rename(partialfn, outfn) != 0) {
    fprintf(stderr, "FATAL: could not rename %s onto %s: %s\n", partialfn,
            outfn, strerror(errno));
    abandonForwardCache(partialfn);
  }
}

// ── Comparison: one spot row against the image ──────────────────────────
// The ONE place a predicted reflection is tested against the detector image.
//
// Both the fresh path (doFwd=1, spots just simulated) and the cached path
// (doFwd=0, spots read back from the forward cache) call this. They used to
// carry separate inlined copies and they DRIFTED: the quantized image_u8
// clamped faint pixels up to 1, inflating totInt and flipping the minIntensity
// test, so a cached run reported MORE solutions than a fresh one. The remedy at
// the time was a comment asserting the two were "identical to the fresh path".
// One function is the structural version of that comment.
//
// `row` points at the orientation's row in the cache layout:
//     row[0]           = number of spots
//     row[1 + 2*i + 0] = ipx of spot i
//     row[1 + 2*i + 1] = ipy of spot i
// Spots are visited in stored order, so totInt accumulates in the same order
// the inlined versions used and the result is bit-identical.
static inline void compareRowToImage(const uint16_t *row, const double *image,
                                     int nrPxX, double minSpotIntensity,
                                     int *nSpotsOut, double *totIntOut) {
  int nSpots = 0;
  double totInt = 0.0;
  int nsp = (int)row[0];
  for (int i = 0; i < nsp; i++) {
    size_t ipx = (size_t)row[1 + 2 * i + 0];
    size_t ipy = (size_t)row[1 + 2 * i + 1];
    double v = image[ipy * (size_t)nrPxX + ipx];
    if (v > minSpotIntensity) {
      totInt += v;
      nSpots++;
    }
  }
  *nSpotsOut = nSpots;
  *totIntOut = totInt;
}

// ── Per-orientation duplicate-pixel test ────────────────────────────────
// Has an earlier reflection of THIS orientation already claimed this pixel?
//
// The claimed set never exceeds maxNrSpots (30 in the shipped Laue configs), so
// a linear scan over the spots already written is EXACT and costs <= 30 integer
// compares, against an hkl loop that runs thousands of times per orientation.
//
// This replaces a full nrPxX*nrPxY bool mask PER THREAD -- 4.2 MB each at
// 2048^2, 67 MB of working set at 16 threads -- which was an O(1) membership
// structure for a <= 30 element set. It also removes the only obstacle to
// running the forward simulation on a GPU, where a per-thread mask of that size
// is impossible (100k threads x 4.2 MB = 420 GB).
//
// `spots` points at the first (ipx, ipy) pair of this orientation's row, i.e.
// one past the spot-count slot. Order is preserved: the FIRST reflection to
// claim a pixel keeps it, exactly as the mask behaved.
//
// TWO DEDUP RULES, ON PURPOSE. This coarse (forward-cache) stage dedups by
// INTEGER PIXEL and scores `totInt * sqrt(nSpots)` (Intensity*sqrt(N)). The
// fit stage -- calcOverlap / calcOverlapFiltered / writeCalcOverlap below --
// dedups by UNIT q-hat (|d| < 1e-6 per component) and scores
// `nrPos * sqrt(sum)` (N*sqrt(Intensity)). Both collapse harmonics ((111),
// (222), ... share q-hat, hence the pixel); they differ only when two
// NON-parallel reflections land on one pixel (one here, two there). The cache
// layout is (ipx, ipy) pairs with no q-hat to compare, and the fit stage needs
// sub-pixel identity, so do not "unify" them without re-measuring both.
static inline int pixelClaimed(const uint16_t *spots, int spotNr, int ipx,
                               int ipy) {
  for (int i = 0; i < spotNr; i++)
    if ((int)spots[2 * i] == ipx && (int)spots[2 * i + 1] == ipy)
      return 1;
  return 0;
}

static inline double CalcLength(double x, double y, double z) {
  return sqrt(x * x + y * y + z * z);
}

#define hc_keVnm 1.2398419739
// hc/(4 pi): E = hc|q|/(4 pi sin(theta)) = -hcOver4Pi*|q|^2/q_z
#define hcOver4Pi (hc_keVnm / (4.0 * M_PI))
#define EPS 1E-12

// ── Global state (defined once per translation unit) ────────────────────
// Each .c / .cu file that includes this header must provide one definition
// of these variables.  We declare them here so all shared functions can
// reference them.
extern double tol_LatC[6];
extern double tol_c_over_a;
extern double c_over_a_orig;
extern int sg_num;
extern double cellVol;
extern double phiVol;
extern int nSym;
extern double Symm[24][4];
// There is no optimiser switch. Refinement is Nelder-Mead unconditionally (see
// FitOrientation); an `Optimizer ...` line in a parameter file is still parsed
// so old files keep working, and `Optimizer BOBYQA` prints a notice and
// proceeds with Nelder-Mead. The former `useBobyqa` global was written by that
// parse and read by nothing -- and was initialised to 1 "default: BOBYQA",
// which described an algorithm that no longer exists -- so it is gone.

// ── Optimization data bundle ────────────────────────────────────────────
struct dataFit {
  float *image;
  int *hkls;
  int nhkls;
  int nrPxX;
  int nrPxY;
  double LatCOrig[6];
  double recip[3][3];
  double *outArrThis;
  int maxNrSpots;
  double rotTranspose[3][3];
  double pArr[3];
  double pxX;
  double pxY;
  double Elo;
  double Ehi;
  int *validHKLIdx; // prefiltered HKL indices (NULL = iterate all)
  int nValidHKL;    // number of valid HKLs
  // A predicted reflection counts as MATCHED when the pixel under it exceeds
  // this. 0.0 reproduces the historical `> 0` exactly, so the default is a
  // no-op; see MinSpotIntensity in the parameter file.
  double minSpotIntensity;
};

// ── Symmetry tables ─────────────────────────────────────────────────────
// `static` gives each TU its own copy, avoiding multiple-definition
// linker errors while keeping the data in the header.

static double TricSym[2][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {1.00000, 0.00000, 0.00000, 0.00000}};

static double MonoSym[2][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.00000, 1.00000, 0.00000, 0.00000}};

static double OrtSym[4][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                              {1.00000, 1.00000, 0.00000, 0.00000},
                              {0.00000, 0.00000, 1.00000, 0.00000},
                              {0.00000, 0.00000, 0.00000, 1.00000}};

static double TetSym[8][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                              {0.70711, 0.00000, 0.00000, 0.70711},
                              {0.00000, 0.00000, 0.00000, 1.00000},
                              {0.70711, -0.00000, -0.00000, -0.70711},
                              {0.00000, 1.00000, 0.00000, 0.00000},
                              {0.00000, 0.00000, 1.00000, 0.00000},
                              {0.00000, 0.70711, 0.70711, 0.00000},
                              {0.00000, -0.70711, 0.70711, 0.00000}};

// Correct TrigSym (from CPU code — the GPU file previously had a stale copy)
static double TrigSym[6][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.00000, 0.86603, -0.50000, 0.00000},
                               {0.50000, 0.00000, 0.00000, 0.86603},
                               {0.00000, 0.00000, 1.00000, 0.00000},
                               {0.50000, -0.00000, -0.00000, -0.86603},
                               {0.00000, 0.86603, 0.50000, 0.00000}};

static double HexSym[12][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.86603, 0.00000, 0.00000, 0.50000},
                               {0.50000, 0.00000, 0.00000, 0.86603},
                               {0.00000, 0.00000, 0.00000, 1.00000},
                               {0.50000, -0.00000, -0.00000, -0.86603},
                               {0.86603, -0.00000, -0.00000, -0.50000},
                               {0.00000, 1.00000, 0.00000, 0.00000},
                               {0.00000, 0.86603, 0.50000, 0.00000},
                               {0.00000, 0.50000, 0.86603, 0.00000},
                               {0.00000, 0.00000, 1.00000, 0.00000},
                               {0.00000, -0.50000, 0.86603, 0.00000},
                               {0.00000, -0.86603, 0.50000, 0.00000}};

static double CubSym[24][4] = {{1.00000, 0.00000, 0.00000, 0.00000},
                               {0.70711, 0.70711, 0.00000, 0.00000},
                               {0.00000, 1.00000, 0.00000, 0.00000},
                               {0.70711, -0.70711, 0.00000, 0.00000},
                               {0.70711, 0.00000, 0.70711, 0.00000},
                               {0.00000, 0.00000, 1.00000, 0.00000},
                               {0.70711, 0.00000, -0.70711, 0.00000},
                               {0.70711, 0.00000, 0.00000, 0.70711},
                               {0.00000, 0.00000, 0.00000, 1.00000},
                               {0.70711, 0.00000, 0.00000, -0.70711},
                               {0.50000, 0.50000, 0.50000, 0.50000},
                               {0.50000, -0.50000, -0.50000, -0.50000},
                               {0.50000, -0.50000, 0.50000, 0.50000},
                               {0.50000, 0.50000, -0.50000, -0.50000},
                               {0.50000, 0.50000, -0.50000, 0.50000},
                               {0.50000, -0.50000, 0.50000, -0.50000},
                               {0.50000, -0.50000, -0.50000, 0.50000},
                               {0.50000, 0.50000, 0.50000, -0.50000},
                               {0.00000, 0.70711, 0.70711, 0.00000},
                               {0.00000, -0.70711, 0.70711, 0.00000},
                               {0.00000, 0.70711, 0.00000, 0.70711},
                               {0.00000, 0.70711, 0.00000, -0.70711},
                               {0.00000, 0.00000, 0.70711, 0.70711},
                               {0.00000, 0.00000, 0.70711, -0.70711}};

// ── Utility functions ───────────────────────────────────────────────────
// `static inline` is the correct C99 way to define functions in a header
// that is included in multiple translation units.

static inline double sin_cos_to_angle(double s, double c) {
  return (s >= 0.0) ? acos(c) : 2.0 * M_PI - acos(c);
}

static inline void normalizeQuat(double quat[4]) {
  double norm = sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] +
                     quat[3] * quat[3]);
  quat[0] /= norm;
  quat[1] /= norm;
  quat[2] /= norm;
  quat[3] /= norm;
}

static inline void QuaternionProduct(double q[4], double r[4], double Q[4]) {
  Q[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3];
  Q[1] = r[1] * q[0] + r[0] * q[1] + r[3] * q[2] - r[2] * q[3];
  Q[2] = r[2] * q[0] + r[0] * q[2] + r[1] * q[3] - r[3] * q[1];
  Q[3] = r[3] * q[0] + r[0] * q[3] + r[2] * q[1] - r[1] * q[2];
  if (Q[0] < 0) {
    Q[0] = -Q[0];
    Q[1] = -Q[1];
    Q[2] = -Q[2];
    Q[3] = -Q[3];
  }
  normalizeQuat(Q);
}

static inline int MakeSymmetries(int SGNr, double Sym[24][4]) {
  int i, j, NrSymmetries;
  if (SGNr <= 2) {
    NrSymmetries = 1;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = TricSym[i][j];
  } else if (SGNr <= 15) {
    NrSymmetries = 2;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = MonoSym[i][j];
  } else if (SGNr <= 74) {
    NrSymmetries = 4;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = OrtSym[i][j];
  } else if (SGNr <= 142) {
    NrSymmetries = 8;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = TetSym[i][j];
  } else if (SGNr <= 167) {
    NrSymmetries = 6;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = TrigSym[i][j];
  } else if (SGNr <= 194) {
    NrSymmetries = 12;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = HexSym[i][j];
  } else {
    NrSymmetries = 24;
    for (i = 0; i < NrSymmetries; i++)
      for (j = 0; j < 4; j++)
        Sym[i][j] = CubSym[i][j];
  }
  return NrSymmetries;
}

static inline void BringDownToFundamentalRegion(double QuatIn[4],
                                                double QuatOut[4]) {
  int i, maxCosRowNr = 0;
  double qps[24][4], q2[4], qt[4], maxCos = -10000;
  for (i = 0; i < nSym; i++) {
    q2[0] = Symm[i][0];
    q2[1] = Symm[i][1];
    q2[2] = Symm[i][2];
    q2[3] = Symm[i][3];
    QuaternionProduct(QuatIn, q2, qt);
    qps[i][0] = qt[0];
    qps[i][1] = qt[1];
    qps[i][2] = qt[2];
    qps[i][3] = qt[3];
    if (maxCos < qt[0]) {
      maxCos = qt[0];
      maxCosRowNr = i;
    }
  }
  QuatOut[0] = qps[maxCosRowNr][0];
  QuatOut[1] = qps[maxCosRowNr][1];
  QuatOut[2] = qps[maxCosRowNr][2];
  QuatOut[3] = qps[maxCosRowNr][3];
  normalizeQuat(QuatOut);
}

static inline double GetMisOrientation(double quat1[4], double quat2[4]) {
  double q1FR[4], q2FR[4], q1Inv[4], QP[4], MisV[4];
  BringDownToFundamentalRegion(quat1, q1FR);
  BringDownToFundamentalRegion(quat2, q2FR);
  q1Inv[0] = -q1FR[0];
  q1Inv[1] = q1FR[1];
  q1Inv[2] = q1FR[2];
  q1Inv[3] = q1FR[3];
  QuaternionProduct(q1Inv, q2FR, QP);
  BringDownToFundamentalRegion(QP, MisV);
  if (MisV[0] > 1)
    MisV[0] = 1;
  return 2 * (acos(MisV[0])) * rad2deg;
}

static inline void OrientMat2Quat(double OrientMat[9], double Quat[4]) {
  double trace = OrientMat[0] + OrientMat[4] + OrientMat[8];
  if (trace > 0) {
    double s = 0.5 / sqrt(trace + 1.0);
    Quat[0] = 0.25 / s;
    Quat[1] = (OrientMat[7] - OrientMat[5]) * s;
    Quat[2] = (OrientMat[2] - OrientMat[6]) * s;
    Quat[3] = (OrientMat[3] - OrientMat[1]) * s;
  } else {
    if (OrientMat[0] > OrientMat[4] && OrientMat[0] > OrientMat[8]) {
      double s = 2.0 * sqrt(1.0 + OrientMat[0] - OrientMat[4] - OrientMat[8]);
      Quat[0] = (OrientMat[7] - OrientMat[5]) / s;
      Quat[1] = 0.25 * s;
      Quat[2] = (OrientMat[1] + OrientMat[3]) / s;
      Quat[3] = (OrientMat[2] + OrientMat[6]) / s;
    } else if (OrientMat[4] > OrientMat[8]) {
      double s = 2.0 * sqrt(1.0 + OrientMat[4] - OrientMat[0] - OrientMat[8]);
      Quat[0] = (OrientMat[2] - OrientMat[6]) / s;
      Quat[1] = (OrientMat[1] + OrientMat[3]) / s;
      Quat[2] = 0.25 * s;
      Quat[3] = (OrientMat[5] + OrientMat[7]) / s;
    } else {
      double s = 2.0 * sqrt(1.0 + OrientMat[8] - OrientMat[0] - OrientMat[4]);
      Quat[0] = (OrientMat[3] - OrientMat[1]) / s;
      Quat[1] = (OrientMat[2] + OrientMat[6]) / s;
      Quat[2] = (OrientMat[5] + OrientMat[7]) / s;
      Quat[3] = 0.25 * s;
    }
  }
  if (Quat[0] < 0) {
    Quat[0] = -Quat[0];
    Quat[1] = -Quat[1];
    Quat[2] = -Quat[2];
    Quat[3] = -Quat[3];
  }
  normalizeQuat(Quat);
}

static inline void OrientMat2Quat33(double OM[3][3], double Quat[4]) {
  double OrientMat[9];
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      OrientMat[i * 3 + j] = OM[i][j];
  OrientMat2Quat(OrientMat, Quat);
}

static inline void OrientMat2Euler(double m[3][3], double Euler[3]) {
  double psi, phi, theta, sph;
  if (fabs(m[2][2] - 1.0) < EPS) {
    phi = 0;
  } else {
    phi = acos(m[2][2]);
  }
  sph = sin(phi);
  if (fabs(sph) < EPS) {
    psi = 0.0;
    theta = (fabs(m[2][2] - 1.0) < EPS) ? sin_cos_to_angle(m[1][0], m[0][0])
                                        : sin_cos_to_angle(-m[1][0], m[0][0]);
  } else {
    psi = (fabs(-m[1][2] / sph) <= 1.0)
              ? sin_cos_to_angle(m[0][2] / sph, -m[1][2] / sph)
              : sin_cos_to_angle(m[0][2] / sph, 1);
    theta = (fabs(m[2][1] / sph) <= 1.0)
                ? sin_cos_to_angle(m[2][0] / sph, m[2][1] / sph)
                : sin_cos_to_angle(m[2][0] / sph, 1);
  }
  Euler[0] = psi;
  Euler[1] = phi;
  Euler[2] = theta;
}

static inline void Euler2OrientMat(double Euler[3], double m_out[3][3]) {
  double psi = Euler[0], phi = Euler[1], theta = Euler[2];
  double cps = cos(psi), cph = cos(phi), cth = cos(theta);
  double sps = sin(psi), sph = sin(phi), sth = sin(theta);
  m_out[0][0] = cth * cps - sth * cph * sps;
  m_out[0][1] = -cth * cph * sps - sth * cps;
  m_out[0][2] = sph * sps;
  m_out[1][0] = cth * sps + sth * cph * cps;
  m_out[1][1] = cth * cph * cps - sth * sps;
  m_out[1][2] = -sph * cps;
  m_out[2][0] = sth * sph;
  m_out[2][1] = cth * sph;
  m_out[2][2] = cph;
}

static inline void MatrixMultF33(double m[3][3], double n[3][3],
                                 double res[3][3]) {
  int r;
  for (r = 0; r < 3; r++) {
    res[r][0] = m[r][0] * n[0][0] + m[r][1] * n[1][0] + m[r][2] * n[2][0];
    res[r][1] = m[r][0] * n[0][1] + m[r][1] * n[1][1] + m[r][2] * n[2][1];
    res[r][2] = m[r][0] * n[0][2] + m[r][1] * n[1][2] + m[r][2] * n[2][2];
  }
}

static inline double zeroOut(double val) { return (fabs(val) < EPS) ? 0 : val; }

static inline void MatrixMultF(double m[3][3], double v[3], double r[3]) {
  int i, j;
  r[0] = 0;
  r[1] = 0;
  r[2] = 0;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      r[i] += m[i][j] * v[j];
}


static inline void calcV(double LatC[6]) {
  double ca = cos(LatC[3] * deg2rad);
  double cb = cos(LatC[4] * deg2rad);
  double cg = cos(LatC[5] * deg2rad);
  phiVol = sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);
  cellVol = LatC[0] * LatC[1] * LatC[2] * phiVol;
}

// ── Parameter-file geometry: refuse what nobody wrote ───────────────────
// Did a numeric parameter line parse completely? `got` is sscanf's return,
// `want` counts the key token too. sscanf stops at the first token it cannot
// convert, so an unfilled template value (`P_Array 0 0 __SET_ME__`) used to
// leave the remaining numbers at their 0 defaults and the run went ahead on a
// geometry nobody specified. Returns 1 if complete; else prints a FATAL naming
// the key and returns 0 (caller exits non-zero).
static inline int paramLineComplete(int got, int want, const char *key,
                                    const char *line) {
  if (got == want)
    return 1;
  size_t n = strlen(line);
  fprintf(stderr,
          "FATAL: %s needs %d number(s) but only %d parsed from this line:\n"
          "  %s%s",
          key, want - 1, got > 1 ? got - 1 : 0, line,
          (n > 0 && line[n - 1] == '\n') ? "" : "\n");
  return 0;
}

// P_Array[2] is the detector distance along its normal: every predicted spot
// is projected with `xyz * pArr[2] / xyz[2]`, so 0 collapses them all onto one
// point and the run "works" on nothing. Zero is what a missing P_Array line
// or an unparsed value leaves, never a real geometry. Written `!(fabs > 0)`
// so NaN is refused too. Returns 0 on success, 1 on a rejected value.
static inline int validateDetectorDistance(const double pArr[3]) {
  if (!(fabs(pArr[2]) > 0.0)) {
    fprintf(stderr,
            "FATAL: P_Array[2] (detector distance) is %g. It must be non-zero; "
            "a missing or unfilled P_Array line leaves it 0. Check the "
            "parameter file.\n",
            pArr[2]);
    return 1;
  }
  return 0;
}

// The energy band [Elo, Ehi] (keV) every predicted reflection must fall in.
// Both default to 5/30 when their line is missing or unparsed, so a template's
// `Elo __SET_ME__` used to run silently at 5 keV; paramLineComplete now refuses
// the unparsed line, and this refuses a band that cannot be real. Written
// `!(0 < Elo && Elo < Ehi)` so NaN is refused too. Returns 0 on success, 1 on
// a rejected band.
static inline int validateEnergyBand(double Elo, double Ehi) {
  if (!(Elo > 0.0 && Ehi > Elo)) {
    fprintf(stderr,
            "FATAL: energy band Elo = %g keV, Ehi = %g keV is invalid; it "
            "must satisfy 0 < Elo < Ehi. Check the parameter file.\n",
            Elo, Ehi);
    return 1;
  }
  return 0;
}

// Detector rotation from the R_Array rotation vector (axis * angle, radians).
//
// |r| == 0 is LEGITIMATE: a detector exactly perpendicular to its P_Array
// axis, with no rotation. The axis r/|r| is then 0/0 = NaN, and the NaN rode
// through Rodrigues (0 * NaN = NaN) into every predicted spot. With the angle
// zero the axis is irrelevant -- cos 0 = 1, sin 0 = 0, 1 - cos 0 = 0 -- so any
// unit axis gives the exact identity; use z. Guarded, not rejected. (An
// unfilled R_Array is caught separately by paramLineComplete.)
static inline void detectorRotationTranspose(const double rArr[3],
                                             double rotTranspose[3][3]) {
  double rotang = CalcLength(rArr[0], rArr[1], rArr[2]);
  double rotvect[3] = {0.0, 0.0, 1.0};
  if (rotang > 0.0) {
    rotvect[0] = rArr[0] / rotang;
    rotvect[1] = rArr[1] / rotang;
    rotvect[2] = rArr[2] / rotang;
  }
  double c = cos(rotang), s = sin(rotang), t = 1 - cos(rotang);
  double rot[3][3] = {
      {c + t * (rotvect[0] * rotvect[0]),
       t * rotvect[0] * rotvect[1] - s * rotvect[2],
       t * rotvect[0] * rotvect[2] + s * rotvect[1]},
      {t * rotvect[1] * rotvect[0] + s * rotvect[2],
       c + t * (rotvect[1] * rotvect[1]),
       t * rotvect[1] * rotvect[2] - s * rotvect[0]},
      {t * rotvect[2] * rotvect[0] - s * rotvect[1],
       t * rotvect[2] * rotvect[1] + s * rotvect[0],
       c + t * (rotvect[2] * rotvect[2])}};
  // Transpose = inverse for a proper rotation.
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      rotTranspose[i][j] = rot[j][i];
}

// Reject a crystal-fit tolerance that was written as a PERCENT.
//
// `tol_LatC` and `tol_c_over_a` are FRACTIONS: the bounds are formed as
// `value * (1 -/+ tol)`. A tolerance of 1.0 therefore puts the lower bound at
// ZERO and the upper at twice the seed -- a and c are then free to collapse
// through zero, and on a staircase objective the simplex wanders rather than
// failing. The usage strings said "in %" until 2026-09-20, so "1.0" meaning
// "1%" is the natural mistake and it produces a plausible-looking run, not an
// error. Anything at or above 1.0 cannot be a meaningful elastic tolerance --
// a 100% strain is not a refinement bound -- so refuse it and say why.
//
// Below 1.0 there is still a percent-minded mistake that passes: 0.5 meaning
// "0.5%" is read as +-50%. Legal as a fraction, but no elastic refinement needs
// more than ~10%, so anything above WARNTOL is accepted with a WARNING that
// states the percent it will actually be used as. Not fatal: a deliberately
// wide bound is the user's call.
//
// EFFECTIVE values only. When tol_c_over_a != 0 every main zeroes tol_LatC
// straight after this call (c/a at constant volume overrides the per-parameter
// tolerances), so tol_LatC never forms a bound and is not validated here --
// a NOTE says it is being ignored instead. Checking it anyway would abort a run
// over a value the fit never uses. Deciding this HERE, rather than by where
// each main places the call relative to that zeroing, keeps the three
// binaries in agreement by construction.
//
// The tests are written `!(tol >= 0 && tol < MAXTOL)` so a NaN (sscanf accepts
// "nan") is rejected rather than slipping through both comparisons.
//
// Returns 0 on success, 1 on a rejected value (caller should exit non-zero).
static inline int validateCrystalFitTolerances(void) {
  const double MAXTOL = 1.0;
  const double WARNTOL = 0.1;
  int bad = 0;
  if (!(tol_c_over_a >= 0.0 && tol_c_over_a < MAXTOL)) {
    fprintf(stderr,
            "FATAL: tol_c_over_a = %g is not a valid FRACTION.\n"
            "  It must satisfy 0 <= tol_c_over_a < 1 (0 disables the c/a fit).\n"
            "  The bounds are c/a * (1 -/+ tol), so %g would put the lower "
            "bound at %g.\n"
            "  If you meant one percent, write 0.01 -- not 1.0.\n",
            tol_c_over_a, tol_c_over_a, 1.0 - tol_c_over_a);
    bad = 1;
  } else if (tol_c_over_a > WARNTOL) {
    fprintf(stderr,
            "WARNING: tol_c_over_a = %g is a FRACTION: c/a may move by "
            "+-%g%%.\n"
            "  If you meant %g%%, write %g.\n",
            tol_c_over_a, 100.0 * tol_c_over_a, tol_c_over_a,
            tol_c_over_a / 100.0);
  }
  if (tol_c_over_a != 0.0) {
    int anyLatC = 0;
    for (int i = 0; i < 6; i++)
      if (tol_LatC[i] != 0.0)
        anyLatC = 1;
    if (anyLatC)
      fprintf(stderr,
              "NOTE: tol_c_over_a is set, so it overrides tol_LatC; "
              "tol_LatC (%g %g %g %g %g %g) is ignored and not validated.\n",
              tol_LatC[0], tol_LatC[1], tol_LatC[2], tol_LatC[3], tol_LatC[4],
              tol_LatC[5]);
    return bad;
  }
  for (int i = 0; i < 6; i++) {
    if (!(tol_LatC[i] >= 0.0 && tol_LatC[i] < MAXTOL)) {
      fprintf(stderr,
              "FATAL: tol_LatC[%d] = %g is not a valid FRACTION.\n"
              "  It must satisfy 0 <= tol < 1 (0 holds that parameter fixed).\n"
              "  If you meant one percent, write 0.01 -- not 1.0.\n",
              i, tol_LatC[i]);
      bad = 1;
    } else if (tol_LatC[i] > WARNTOL) {
      fprintf(stderr,
              "WARNING: tol_LatC[%d] = %g is a FRACTION: that parameter may "
              "move by +-%g%%.\n"
              "  If you meant %g%%, write %g.\n",
              i, tol_LatC[i], 100.0 * tol_LatC[i], tol_LatC[i],
              tol_LatC[i] / 100.0);
    }
  }
  return bad;
}

static inline void calcRecipArray(double Lat[6], int SpaceGroup,
                                  double recip[3][3]) {
  double a = Lat[0], b = Lat[1], c = Lat[2];
  double alpha = Lat[3], beta = Lat[4], gamma = Lat[5];
  int rhomb = 0;
  if (SpaceGroup == 146 || SpaceGroup == 148 || SpaceGroup == 155 ||
      SpaceGroup == 160 || SpaceGroup == 161 || SpaceGroup == 166 ||
      SpaceGroup == 167)
    rhomb = 1;
  double ca = cos(alpha * deg2rad);
  double cb = cos(beta * deg2rad);
  double cg = cos(gamma * deg2rad);
  double sg = sin(gamma * deg2rad);
  double phi = sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);
  double Vc = a * b * c * phi;
  double pv = (2 * M_PI) / Vc;
  double a0, a1, a2, b0, b1, b2, c0, c1, c2;
  if (rhomb == 0) {
    a0 = a;
    a1 = 0.0;
    a2 = 0.0;
    b0 = b * cg;
    b1 = b * sg;
    b2 = 0;
    c0 = c * cb;
    c1 = c * (ca - cb * cg) / sg;
    c2 = c * phi / sg;
    a0 = zeroOut(a0);
    a1 = zeroOut(a1);
    a2 = zeroOut(a2);
    b1 = zeroOut(b1);
    b2 = zeroOut(b2);
    c2 = zeroOut(c2);
  } else {
    double p = sqrt(1.0 + 2 * ca);
    double q = sqrt(1.0 - ca);
    double pmq = (a / 3.0) * (p - q);
    double p2q = (a / 3.0) * (p + 2 * q);
    a0 = p2q;
    a1 = pmq;
    a2 = pmq;
    b0 = pmq;
    b1 = p2q;
    b2 = pmq;
    c0 = pmq;
    c1 = pmq;
    c2 = p2q;
  }
  recip[0][0] = zeroOut((b1 * c2 - b2 * c1) * pv);
  recip[1][0] = zeroOut((b2 * c0 - b0 * c2) * pv);
  recip[2][0] = zeroOut((b0 * c1 - b1 * c0) * pv);
  recip[0][1] = zeroOut((c1 * a2 - c2 * a1) * pv);
  recip[1][1] = zeroOut((c2 * a0 - c0 * a2) * pv);
  recip[2][1] = zeroOut((c0 * a1 - c1 * a0) * pv);
  recip[0][2] = zeroOut((a1 * b2 - a2 * b1) * pv);
  recip[1][2] = zeroOut((a2 * b0 - a0 * b2) * pv);
  recip[2][2] = zeroOut((a0 * b1 - a1 * b0) * pv);
}

// ── Diffraction overlap calculation ─────────────────────────────────────
//
// HARMONICS ARE COUNTED ONCE. (111), (222), (333)... of one orientation share a
// unit q-hat and therefore one detector pixel; the loop below keeps only the
// FIRST reflection with a given q-hat (|d| < 1e-6 per component) and skips the
// rest before they reach the image. So nrPos -- and NMatches, which is
// writeCalcOverlap's nrPos under the same rule -- cannot stack harmonics. A
// handbook entry once claimed the opposite; it was wrong.
// tests/test_c_harmonic_no_stack.py pins this.
//
// This q-hat dedup and score (N*sqrt(Intensity)) intentionally differ from the
// coarse stage's integer-pixel dedup and Intensity*sqrt(N): see pixelClaimed().

static inline double calcOverlap(float *image, double euler[3], int *hkls,
                                 int nhkls, int nrPxX, int nrPxY,
                                 double recip[3][3], double *outArrThis,
                                 int maxNrSpots, double rotTranspose[3][3],
                                 double pArr[3], double pxX, double pxY,
                                 double Elo, double Ehi,
                                 double minSpotIntensity) {
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  double ki[3] = {0, 0, 1.0};
  MatrixMultF33(OMt, recip, OM);
  int hklnr, badSpot;
  double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp, px, py,
      sinTheta, E, result = 0;
  int spotNr = 0, iterNr, nrPos = 0;
  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    qhat[0] = qvec[0] / qlen;
    qhat[1] = qvec[1] / qlen;
    qhat[2] = qvec[2] / qlen;
    dot = qhat[2];
    kf[0] = ki[0] - 2 * dot * qhat[0];
    kf[1] = ki[1] - 2 * dot * qhat[1];
    kf[2] = ki[2] - 2 * dot * qhat[2];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    xyz[0] = xyz[0] * pArr[2] / xyz[2];
    xyz[1] = xyz[1] * pArr[2] / xyz[2];
    xyz[2] = pArr[2];
    xp = xyz[0] - pArr[0];
    yp = xyz[1] - pArr[1];
    px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    sinTheta = -qhat[2];
    E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    badSpot = 0;
    for (iterNr = 0; iterNr < spotNr; iterNr++) {
      if ((fabs(qhat[0] - outArrThis[3 * iterNr + 0]) * 100000 < 0.1) &&
          (fabs(qhat[1] - outArrThis[3 * iterNr + 1]) * 100000 < 0.1) &&
          (fabs(qhat[2] - outArrThis[3 * iterNr + 2]) * 100000 < 0.1)) {
        badSpot = 1;
        break;
      }
    }
    if (badSpot == 0) {
      outArrThis[3 * spotNr + 0] = qhat[0];
      outArrThis[3 * spotNr + 1] = qhat[1];
      outArrThis[3 * spotNr + 2] = qhat[2];
      if (image[(size_t)((size_t)py * nrPxX + (size_t)px)] >
          minSpotIntensity) {
        result += image[(size_t)((size_t)py * nrPxX + (size_t)px)];
        nrPos++;
      }
      spotNr++;
      if (spotNr == maxNrSpots)
        break;
    }
  }
  result = nrPos * sqrt(result);
  return result;
}

// ── HKL prefiltering ────────────────────────────────────────────────────
// Identify which HKLs produce valid diffraction spots at a given
// orientation. During NLopt optimisation the Euler angles vary by only
// ±3°, so the valid-HKL set stays essentially constant.  Running the
// inner loop over ~60 HKLs instead of 3058 is the main speedup.

static inline int prefilterHKLs(int *hkls, int nhkls, double euler[3],
                                double recip[3][3], int nrPxX, int nrPxY,
                                double rotTranspose[3][3], double pArr[3],
                                double pxX, double pxY, double Elo, double Ehi,
                                int *validIdx, int maxValid) {
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  MatrixMultF33(OMt, recip, OM);
  int nValid = 0;
  for (int hklnr = 0; hklnr < nhkls && nValid < maxValid; hklnr++) {
    double hkl[3], qvec[3];
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    double qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    double qhat[3] = {qvec[0] / qlen, qvec[1] / qlen, qvec[2] / qlen};
    double dot = qhat[2];
    double kf[3] = {-2 * dot * qhat[0], -2 * dot * qhat[1],
                    1.0 - 2 * dot * qhat[2]};
    double xyz[3];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    double xp = xyz[0] * pArr[2] / xyz[2] - pArr[0];
    double yp = xyz[1] * pArr[2] / xyz[2] - pArr[1];
    double px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    double py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    double sinTheta = -qhat[2];
    double E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    validIdx[nValid++] = hklnr;
  }
  return nValid;
}

// ── Overlap using prefiltered HKLs ──────────────────────────────────────

static inline double
calcOverlapFiltered(float *image, double euler[3], int *hkls, int *validIdx,
                    int nValid, int nrPxX, int nrPxY, double recip[3][3],
                    double *outArrThis, int maxNrSpots,
                    double rotTranspose[3][3], double pArr[3], double pxX,
                    double pxY, double Elo, double Ehi,
                    double minSpotIntensity) {
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  double ki[3] = {0, 0, 1.0};
  MatrixMultF33(OMt, recip, OM);
  int badSpot;
  double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp, px, py,
      sinTheta, E, result = 0;
  int spotNr = 0, iterNr, nrPos = 0;
  for (int vi = 0; vi < nValid; vi++) {
    int hklnr = validIdx[vi];
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    qhat[0] = qvec[0] / qlen;
    qhat[1] = qvec[1] / qlen;
    qhat[2] = qvec[2] / qlen;
    dot = qhat[2];
    kf[0] = ki[0] - 2 * dot * qhat[0];
    kf[1] = ki[1] - 2 * dot * qhat[1];
    kf[2] = ki[2] - 2 * dot * qhat[2];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    xyz[0] = xyz[0] * pArr[2] / xyz[2];
    xyz[1] = xyz[1] * pArr[2] / xyz[2];
    xyz[2] = pArr[2];
    xp = xyz[0] - pArr[0];
    yp = xyz[1] - pArr[1];
    px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    sinTheta = -qhat[2];
    E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    badSpot = 0;
    for (iterNr = 0; iterNr < spotNr; iterNr++) {
      if ((fabs(qhat[0] - outArrThis[3 * iterNr + 0]) * 100000 < 0.1) &&
          (fabs(qhat[1] - outArrThis[3 * iterNr + 1]) * 100000 < 0.1) &&
          (fabs(qhat[2] - outArrThis[3 * iterNr + 2]) * 100000 < 0.1)) {
        badSpot = 1;
        break;
      }
    }
    if (badSpot == 0) {
      outArrThis[3 * spotNr + 0] = qhat[0];
      outArrThis[3 * spotNr + 1] = qhat[1];
      outArrThis[3 * spotNr + 2] = qhat[2];
      if (image[(size_t)((size_t)py * nrPxX + (size_t)px)] >
          minSpotIntensity) {
        result += image[(size_t)((size_t)py * nrPxX + (size_t)px)];
        nrPos++;
      }
      spotNr++;
      if (spotNr == maxNrSpots)
        break;
    }
  }
  result = nrPos * sqrt(result);
  return result;
}

// ── NLopt problem function ──────────────────────────────────────────────

static inline double problem_function(unsigned n, const double *x, double *grad,
                                      void *f_data_supplied) {
  int i, j;
  struct dataFit *f_data = (struct dataFit *)f_data_supplied;
  double rotTranspose[3][3], pArr[3], recip[3][3];
  for (i = 0; i < 3; i++) {
    pArr[i] = f_data->pArr[i];
    for (j = 0; j < 3; j++) {
      rotTranspose[i][j] = f_data->rotTranspose[i][j];
      if (n == 3)
        recip[i][j] = f_data->recip[i][j];
    }
  }
  if (n > 3) {
    double latCNew[6];
    int cntr = 0;
    for (i = 0; i < 6; i++) {
      if (tol_LatC[i] != 0) {
        latCNew[i] = x[3 + cntr];
        cntr++;
      } else {
        latCNew[i] = f_data->LatCOrig[i];
      }
    }
    if (tol_c_over_a != 0) {
      // FIX: was pow(x, 1/3) — integer division gave 0.  Use cbrt().
      double aNew = cbrt(cellVol / (x[3] * phiVol));
      latCNew[0] = aNew;
      latCNew[1] = aNew;
      latCNew[2] = x[3] * aNew;
    }
    calcRecipArray(latCNew, sg_num, recip);
  }
  float *image = f_data->image;
  int *hkls = f_data->hkls;
  double *outArrThis = f_data->outArrThis;
  double Euler[3];
  for (i = 0; i < 3; i++)
    Euler[i] = x[i];
  double overlap;
  if (f_data->validHKLIdx != NULL) {
    overlap = calcOverlapFiltered(
        image, Euler, hkls, f_data->validHKLIdx, f_data->nValidHKL,
        f_data->nrPxX, f_data->nrPxY, recip, outArrThis, f_data->maxNrSpots,
        rotTranspose, pArr, f_data->pxX, f_data->pxY, f_data->Elo, f_data->Ehi,
        f_data->minSpotIntensity);
  } else {
    overlap = calcOverlap(image, Euler, hkls, f_data->nhkls, f_data->nrPxX,
                          f_data->nrPxY, recip, outArrThis, f_data->maxNrSpots,
                          rotTranspose, pArr, f_data->pxX, f_data->pxY,
                          f_data->Elo, f_data->Ehi, f_data->minSpotIntensity);
  }
  return -overlap;
}

// ── Orientation fitting ─────────────────────────────────────────────────

static inline double
FitOrientation(float *image, double euler[3], int *hkls, int nhkls, int nrPxX,
               int nrPxY, double recip[3][3], double *outArrThis,
               int maxNrSpots, double rotTranspose[3][3], double pArr[3],
               double pxX, double pxY, double Elo, double Ehi, double tol,
               double latc[6], double eulerFit[3], double latCUpd[6],
               double *minVal, int doCrystalFit, int *validHKLIdx,
               int nValidHKL, int forceNelderMead, double initStepRad,
               double minSpotIntensity) {
  int i, j;
  unsigned n;
  if (doCrystalFit == 0) {
    n = 3;
  } else {
    int non_zero = 0;
    for (i = 0; i < 6; i++)
      if (tol_LatC[i] != 0)
        non_zero++;
    n = 3 + non_zero;
    if (tol_c_over_a != 0)
      n = 4;
  }
  double minf;
  double x[n], xl[n], xu[n];
  x[0] = euler[0];
  xl[0] = euler[0] - tol;
  xu[0] = euler[0] + tol;
  x[1] = euler[1];
  xl[1] = euler[1] - tol;
  xu[1] = euler[1] + tol;
  x[2] = euler[2];
  xl[2] = euler[2] - tol;
  xu[2] = euler[2] + tol;
  if (doCrystalFit != 0) {
    int cntr = 3;
    for (i = 0; i < 6; i++) {
      if (tol_LatC[i] != 0) {
        x[cntr] = latc[i];
        xl[cntr] = latc[i] * (1 - tol_LatC[i]);
        xu[cntr] = latc[i] * (1 + tol_LatC[i]);
        cntr++;
      }
    }
    if (tol_c_over_a != 0) {
      x[3] = c_over_a_orig;
      xl[3] = c_over_a_orig * (1 - tol_c_over_a);
      xu[3] = c_over_a_orig * (1 + tol_c_over_a);
    }
  }

  struct dataFit f_data = {0};
  f_data.image = image;
  f_data.hkls = hkls;
  f_data.nhkls = nhkls;
  f_data.nrPxX = nrPxX;
  f_data.nrPxY = nrPxY;
  f_data.outArrThis = outArrThis;
  f_data.maxNrSpots = maxNrSpots;
  f_data.validHKLIdx = validHKLIdx;
  f_data.nValidHKL = nValidHKL;
  f_data.minSpotIntensity = minSpotIntensity;
  for (i = 0; i < 3; i++) {
    f_data.pArr[i] = pArr[i];
    for (j = 0; j < 3; j++) {
      f_data.rotTranspose[i][j] = rotTranspose[i][j];
      f_data.recip[i][j] = recip[i][j];
    }
  }
  for (i = 0; i < 6; i++)
    f_data.LatCOrig[i] = latc[i];
  f_data.pxX = pxX;
  f_data.pxY = pxY;
  f_data.Elo = Elo;
  f_data.Ehi = Ehi;
  void *trp = (void *)&f_data;
  // Nelder-Mead for BOTH stages, via the vendored simplex -- no NLopt.
  //
  // The objective (calcOverlap) samples observed intensity at the INTEGER pixel
  // of each predicted reflection, so it is piecewise-constant: zero gradient
  // except where a spot crosses a pixel boundary. BOBYQA fits a smooth
  // quadratic model to that staircase, which is why the coarse stage always
  // forced Nelder-Mead. Measured on 198 paired synthetic seeds perturbed by up
  // to one 0.4 deg grid spacing, scored as misorientation to a known truth:
  //
  //   Nelder-Mead   median 0.004103 deg   p95 0.009076   max 0.013942   0.609 s
  //   BOBYQA        median 0.005432 deg   p95 0.029568   max 0.345720   0.610 s
  //
  // Nelder-Mead is better on every statistic, wins 132/198 paired, and costs
  // the same wall-clock. BOBYQA's worst case is its own seed error -- on that
  // seed it did not refine at all, which is the quadratic model failing exactly
  // as the comment above always predicted. Dropping it removes the last
  // external dependency, so the C builds with no NLopt to fetch.
  // See PREREGISTER.md / RESULTS.md in the optimiser-comparison analysis dir.
  //
  // `forceNelderMead` is retained in the signature: it is now always true in
  // effect, and callers still pass it to document which stage they are in.
  (void)forceNelderMead;
  NLoptConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.objective_function = problem_function;
  cfg.obj_data = trp;
  cfg.dimension = n;
  cfg.lower_bounds = xl;
  cfg.upper_bounds = xu;
  cfg.initial_guess = x;          /* IN/OUT: holds the best point on return */
  cfg.ftol_rel = 1e-6;
  cfg.xtol_rel = 1e-6;
  cfg.max_evaluations = 200;
  double nmStep[n];               /* same VLA form as x/xl/xu above */
  if (initStepRad > 0.0) {
    /* nlopt_set_initial_step1 set ONE scalar for every dimension; the vendored
     * API takes a per-dimension array, so broadcast it. NULL leaves the NLopt
     * default-step heuristic, which is what the fine stage always used. */
    for (i = 0; i < (int)n; i++)
      nmStep[i] = initStepRad;
    cfg.step_sizes = nmStep;
  }
  run_nlopt_optimization(MIDAS_LN_NELDERMEAD, &cfg);
  minf = cfg.min_function_val;
  for (i = 0; i < 3; i++)
    eulerFit[i] = x[i];
  if (doCrystalFit != 0) {
    int cntr2 = 0;
    for (i = 0; i < 6; i++) {
      if (tol_LatC[i] != 0) {
        latCUpd[i] = x[3 + cntr2];
        cntr2++;
      } else {
        latCUpd[i] = latc[i];
      }
    }
    if (tol_c_over_a != 0) {
      // FIX: was pow(x, 1/3) — use cbrt()
      double aNew = cbrt(cellVol / (x[3] * phiVol));
      latCUpd[0] = aNew;
      latCUpd[1] = aNew;
      latCUpd[2] = x[3] * aNew;
    }
  }
  *minVal = -minf;
  return 0;
}

// ── Write overlap with optional spot info ───────────────────────────────
// Always buffers spot data. Caller decides whether to flush based on
// nrSps return value, or passes saveExtraInfo != 0 to auto-flush.

static inline int writeCalcOverlap(float *image, double euler[3], int *hkls,
                                   int nhkls, int nrPxX, int nrPxY,
                                   double recip[3][3], double *outArrThis,
                                   int maxNrSpots, double rotTranspose[3][3],
                                   double pArr[3], double pxX, double pxY,
                                   double Elo, double Ehi,
                                   double minSpotIntensity, FILE *ExtraInfo,
                                   int saveExtraInfo, int *simulNrSps,
                                   int imageNr) {
  int nrSps = 0;
  double OM[3][3], OMt[3][3];
  Euler2OrientMat(euler, OMt);
  double ki[3] = {0, 0, 1.0};
  MatrixMultF33(OMt, recip, OM);
  int hklnr, badSpot;
  double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp, px, py,
      sinTheta, E, result = 0;
  int spotNr = 0, iterNr, nrPos = 0;

  char *outputBuf = NULL;
  size_t currentOffset = 0;
  size_t outputBufCap = (size_t)maxNrSpots * 256;  // per-spot line budget
  if (saveExtraInfo != 0) {
    outputBuf = (char *)malloc(outputBufCap);
    if (outputBuf)
      outputBuf[0] = '\0';
  }

  for (hklnr = 0; hklnr < nhkls; hklnr++) {
    hkl[0] = hkls[hklnr * 3 + 0];
    hkl[1] = hkls[hklnr * 3 + 1];
    hkl[2] = hkls[hklnr * 3 + 2];
    MatrixMultF(OM, hkl, qvec);
    qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
    if (qlen == 0)
      continue;
    qhat[0] = qvec[0] / qlen;
    qhat[1] = qvec[1] / qlen;
    qhat[2] = qvec[2] / qlen;
    dot = qhat[2];
    kf[0] = ki[0] - 2 * dot * qhat[0];
    kf[1] = ki[1] - 2 * dot * qhat[1];
    kf[2] = ki[2] - 2 * dot * qhat[2];
    MatrixMultF(rotTranspose, kf, xyz);
    if (xyz[2] <= 0)
      continue;
    xyz[0] = xyz[0] * pArr[2] / xyz[2];
    xyz[1] = xyz[1] * pArr[2] / xyz[2];
    xyz[2] = pArr[2];
    xp = xyz[0] - pArr[0];
    yp = xyz[1] - pArr[1];
    px = (xp / pxX) + (0.5 * (nrPxX - 1));
    if (px < 0 || px > (nrPxX - 1))
      continue;
    py = (yp / pxY) + (0.5 * (nrPxY - 1));
    if (py < 0 || py > (nrPxY - 1))
      continue;
    sinTheta = -qhat[2];
    E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
    if (E < Elo || E > Ehi)
      continue;
    badSpot = 0;
    for (iterNr = 0; iterNr < spotNr; iterNr++) {
      if ((fabs(qhat[0] - outArrThis[3 * iterNr + 0]) * 100000 < 0.1) &&
          (fabs(qhat[1] - outArrThis[3 * iterNr + 1]) * 100000 < 0.1) &&
          (fabs(qhat[2] - outArrThis[3 * iterNr + 2]) * 100000 < 0.1)) {
        badSpot = 1;
        break;
      }
    }
    if (badSpot == 0) {
      outArrThis[3 * spotNr + 0] = qhat[0];
      outArrThis[3 * spotNr + 1] = qhat[1];
      outArrThis[3 * spotNr + 2] = qhat[2];
      if (image[(size_t)((size_t)py * nrPxX + (size_t)px)] >
          minSpotIntensity) {
        if (saveExtraInfo != 0) {
          if (outputBuf != NULL) {
            // snprintf with remaining space: never overrun outputBuf even if a
            // line is unexpectedly wide (large hkl / imageNr / intensity).
            size_t remBuf = (currentOffset < outputBufCap)
                                ? outputBufCap - currentOffset : 0;
            int wrote = 0;
            if (imageNr > 0)
              wrote = snprintf(
                  outputBuf + currentOffset, remBuf,
                  "%d\t%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                  imageNr, saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                  (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                  (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
            else
              wrote = snprintf(
                  outputBuf + currentOffset, remBuf,
                  "%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                  saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1], (int)hkl[2],
                  (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                  (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
            if (wrote > 0)
              currentOffset += ((size_t)wrote < remBuf) ? (size_t)wrote
                                                        : (remBuf ? remBuf - 1 : 0);
          } else {
#pragma omp critical
            {
              if (imageNr > 0)
                fprintf(
                    ExtraInfo,
                    "%d\t%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                    imageNr, saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                    (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1], qhat[2],
                    (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
              else
                fprintf(ExtraInfo,
                        "%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n",
                        saveExtraInfo, spotNr, (int)hkl[0], (int)hkl[1],
                        (int)hkl[2], (int)px, (int)py, qhat[0], qhat[1],
                        qhat[2],
                        (double)image[(size_t)((size_t)py * nrPxX + (size_t)px)]);
            }
          }
        }
        result += image[(size_t)((size_t)py * nrPxX + (size_t)px)];
        nrPos++;
      }
      spotNr++;
      if (spotNr == maxNrSpots)
        break;
    }
  }

  if (saveExtraInfo != 0 && outputBuf != NULL) {
    if (currentOffset > 0) {
#pragma omp critical
      {
        fputs(outputBuf, ExtraInfo);
      }
    }
    free(outputBuf);
  }

  *simulNrSps = spotNr;
  nrSps = nrPos;
  return nrSps;
}

// Cap on the number of coarse match results fed into the O(N^2) merge below.
// Far above any realistic per-frame count (hundreds to a few thousand), so it
// is a no-op for normal data; it only fires on a pathological frame to keep the
// pairwise misorientation matrix (nrResults^2) from exhausting memory.
#define MERGE_MAX_RESULTS 50000

typedef struct {
  double score;
  size_t row;
} ScoreRow_t;

static int cmpScoreRowDesc(const void *a, const void *b) {
  double sa = ((const ScoreRow_t *)a)->score;
  double sb = ((const ScoreRow_t *)b)->score;
  return (sa < sb) - (sa > sb); // descending by score
}

// ── Merge duplicate orientations (parallel) ──────────────────────────────
// Clusters match results by misorientation angle using OMP-parallel
// pairwise precomputation.  Returns number of unique solutions.
static inline int mergeDuplicateOrientations(double *orients, size_t *rowNrs,
                                             double *matchScores, int nrResults,
                                             double maxAngle, int numProcs,
                                             double *FinOrientArr, int *dArr,
                                             int *bsArr, double *bsScoreArr) {
  // Cap pathological result counts: keep the top MERGE_MAX_RESULTS by coarse
  // score before the O(N^2) clustering.  No-op for realistic frames.
  if (nrResults > MERGE_MAX_RESULTS) {
    fprintf(stderr,
            "WARNING: %d coarse matches exceed merge cap %d; keeping the "
            "top-scored before clustering.\n",
            nrResults, MERGE_MAX_RESULTS);
    ScoreRow_t *sr = (ScoreRow_t *)malloc((size_t)nrResults * sizeof(ScoreRow_t));
    if (sr == NULL) {
      fprintf(stderr, "FATAL: could not allocate score-sort buffer.\n");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nrResults; i++) {
      sr[i].score = matchScores[i];
      sr[i].row = rowNrs[i];
    }
    qsort(sr, (size_t)nrResults, sizeof(ScoreRow_t), cmpScoreRowDesc);
    for (int i = 0; i < MERGE_MAX_RESULTS; i++) {
      matchScores[i] = sr[i].score;
      rowNrs[i] = sr[i].row;
    }
    free(sr);
    nrResults = MERGE_MAX_RESULTS;
  }
  int *doneArr = (int *)calloc(nrResults, sizeof(int));
  // Step 1: Precompute quaternions for all matches (parallel)
  double *quats = (double *)malloc((size_t)nrResults * 4 * sizeof(double));
  if ((doneArr == NULL || quats == NULL) && nrResults > 0) {
    fprintf(stderr, "FATAL: mergeDuplicateOrientations could not allocate "
            "quaternion arrays (nrResults=%d).\n", nrResults);
    exit(EXIT_FAILURE);
  }
#pragma omp parallel for num_threads(numProcs)
  for (int qi = 0; qi < nrResults; qi++) {
    double or9[9];
    for (int k = 0; k < 9; k++)
      or9[k] = orients[rowNrs[qi] * 9 + k];
    OrientMat2Quat(or9, &quats[qi * 4]);
  }
  // Step 2: Precompute pairwise misorientations (parallel)
  size_t nPairs = (size_t)nrResults * (nrResults - 1) / 2;
  float *misoDist = (float *)malloc(nPairs * sizeof(float));
  // NOTE(perf): this pairwise matrix is O(nrResults^2) memory — for a
  // pathological frame with very many candidate matches it can be enormous.
  // Fail loudly here rather than segfault; bounding nrResults before the
  // merge is a tracked follow-up (see C_CUDA_HARDENING.md).
  if (misoDist == NULL && nPairs > 0) {
    fprintf(stderr, "FATAL: mergeDuplicateOrientations could not allocate the "
            "%zu-pair misorientation matrix (nrResults=%d). Too many matches.\n",
            nPairs, nrResults);
    exit(EXIT_FAILURE);
  }
#pragma omp parallel for num_threads(numProcs) schedule(dynamic)
  for (int i = 1; i < nrResults; i++) {
    for (int j = 0; j < i; j++) {
      size_t idx = (size_t)i * (i - 1) / 2 + j;
      misoDist[idx] = (float)GetMisOrientation(&quats[i * 4], &quats[j * 4]);
    }
  }
  // Step 3: Greedy cluster using precomputed distances (serial, fast)
  int iterNr = 0;
  for (int gi = 0; gi < nrResults; gi++) {
    if (doneArr[gi] != 0)
      continue;
    doneArr[gi] = 1;
    int bestSol = rowNrs[gi];
    double bestIntensity = matchScores[gi];
    for (int l = gi + 1; l < nrResults; l++) {
      if (doneArr[l] > 0)
        continue;
      size_t idx = (size_t)l * (l - 1) / 2 + gi;
      if (misoDist[idx] <= maxAngle) {
        doneArr[l] = 1;
        doneArr[gi]++;
        if (matchScores[l] > bestIntensity) {
          bestIntensity = matchScores[l];
          bestSol = rowNrs[l];
        }
      }
    }
    for (int k = 0; k < 9; k++)
      FinOrientArr[iterNr * 9 + k] = orients[(size_t)bestSol * 9 + k];
    dArr[iterNr] = doneArr[gi];
    bsArr[iterNr] = bestSol;
    bsScoreArr[iterNr] = bestIntensity;
    iterNr++;
  }
  free(quats);
  free(misoDist);
  free(doneArr);
  return iterNr;
}

// ── Fit and write orientations (parallel) ────────────────────────────────
// OMP-parallel fitting loop: prefilter HKLs, single-pass FitOrientation,
// writeCalcOverlap, fprintf results.
// imageNum > 0: prepend image number column (streaming mode)
// imageNum <= 0: no image column (batch mode)
// Geometry-scaled coarse-fit blur width (px).  Chosen so ~3 sigma covers the
// spot displacement of a worst-case orientation-grid seed error (~1.3x the
// grid spacing): displacement_px = 1.3 * gridDeg * (Lsd/pxSize).  Clamped to
// the empirically safe window [4,12] px -- below ~4 the blur cannot bridge the
// gap; above ~12 it over-smooths and merges neighbouring spots.  Callers may
// override with an explicit CoarseFitSigma.
static inline double autoCoarseSigma(double Lsd, double pxSize,
                                     double gridDeg) {
  if (gridDeg <= 0.0)
    gridDeg = 0.4; // default 100M-orientation DB spacing
  double sigma = 1.3 * gridDeg * deg2rad * (Lsd / pxSize) / 3.0;
  if (sigma < 4.0)
    sigma = 4.0;
  if (sigma > 12.0)
    sigma = 12.0;
  return sigma;
}

// Separable Gaussian blur of a float image (row-major, width nx).  Used to
// widen the match "capture radius" for the coarse stage of orientation
// refinement so a coarse-grid seed (up to ~one grid spacing off, i.e. tens of
// pixels of spot displacement) still has a smooth gradient path to the true
// peak.  Without this, a seed whose spots land outside the sharp-image spot
// blobs sits on a flat objective and the local optimizer cannot converge.
static inline void gaussianBlurImage(const float *in, float *out, int nx,
                                     int ny, double sigma, int nThreads) {
  int rad = (int)(3.0 * sigma + 0.5);
  if (rad < 1)
    rad = 1;
  int klen = 2 * rad + 1;
  double *kern = (double *)malloc((size_t)klen * sizeof(double));
  float *tmp = (float *)malloc((size_t)nx * ny * sizeof(float));
  if (kern == NULL || tmp == NULL) {
    fprintf(stderr, "FATAL: gaussianBlurImage allocation failed.\n");
    exit(EXIT_FAILURE);
  }
  double ksum = 0.0;
  for (int i = -rad; i <= rad; i++) {
    double w = exp(-(double)(i * i) / (2.0 * sigma * sigma));
    kern[i + rad] = w;
    ksum += w;
  }
  for (int i = 0; i < klen; i++)
    kern[i] /= ksum;
  // Both passes: parallel over rows, and the edge clamp hoisted out of the
  // innermost loop.
  //
  // EXACTNESS. Each output pixel still accumulates the same klen products in
  // the same k order, so every pixel is bit-identical to the serial version.
  // Splitting off the border columns/rows does not reorder anything either --
  // the interior branch is entered only where the clamp would never have
  // fired. Parallelising over y is safe because `out` rows are disjoint and
  // `in`/`tmp` are read-only within a pass.
  //
  // WHY. This was the whole streaming pipeline's bottleneck: one serial
  // 47-tap separable pass pair over 4.2M pixels, once per image, ~394M
  // multiply-adds with two clamp compares per tap, while the other 16 cores
  // idled waiting for it.
  if (nThreads < 1)
    nThreads = 1;
  // Horizontal pass (clamped edges).
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for (int y = 0; y < ny; y++) {
    const float *inRow = in + (size_t)y * nx;
    float *tmpRow = tmp + (size_t)y * nx;
    int xlo = rad < nx ? rad : nx;
    int xhi = nx - rad > xlo ? nx - rad : xlo;
    for (int x = 0; x < xlo; x++) {          // left border: clamp needed
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
    for (int x = xlo; x < xhi; x++) {        // interior: no clamp can fire
      double acc = 0.0;
      const float *w = inRow + x - rad;
      for (int k = 0; k < klen; k++)
        acc += kern[k] * w[k];
      tmpRow[x] = (float)acc;
    }
    for (int x = xhi; x < nx; x++) {         // right border: clamp needed
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
  // Vertical pass (clamped edges). Interior rows read a contiguous stack of
  // rows, so the row pointers are hoisted and the inner loop walks k with a
  // fixed stride instead of recomputing a clamped index per tap.
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

static inline void fitAndWriteOrientations(
    float *image, double *FinOrientArr, int *dArr, int *bsArr,
    double *bsScoreArr, int totalSols, int *hkls, int nhkls, int nrPxX,
    int nrPxY, double recip[3][3], double rotTranspose[3][3], double pArr[3],
    double pxX, double pxY, double Elo, double Ehi, double tol,
    double *LatticeParameter, int maxNrSpots, int minNrSpots, int numProcs,
    FILE *outF, FILE *ExtraInfo, int imageNum, double coarseFitSigma,
    double minSpotIntensity) {
  int iterNr;
  // Blur the match image once (shared, read-only across threads) to give the
  // coarse fit stage a wide capture radius.
  float *imageCoarse = (float *)malloc((size_t)nrPxX * nrPxY * sizeof(float));
  if (imageCoarse == NULL) {
    fprintf(stderr, "FATAL: could not allocate coarse-fit image buffer.\n");
    exit(EXIT_FAILURE);
  }
  double coarseSigma = (coarseFitSigma > 0.0)
                           ? coarseFitSigma
                           : autoCoarseSigma(pArr[2], pxX, 0.4);
  // Timed because it is a FIXED per-image cost, independent of totalSols, and
  // the streaming measurement put the whole pipeline's bottleneck here: of a
  // median 464 ms/image in this function, 449 ms was this one blur, against
  // 41 ms on the GPU (n=150, one run). The cost was flat from 1 to 93
  // orientations, and anything that does not scale with totalSols is upstream
  // of the parallel loop -- this is the only such work.
  double _wtBlur = omp_get_wtime();
  gaussianBlurImage(image, imageCoarse, nrPxX, nrPxY, coarseSigma, numProcs);
  double _blurMs = (omp_get_wtime() - _wtBlur) * 1000.0;
  if (imageNum > 0)
    printf("[Image %d]   coarse blur: %.0f ms (sigma %.2f px, radius %d)\n",
           imageNum, _blurMs, coarseSigma, (int)(3.0 * coarseSigma + 0.5));
  else
    printf("Coarse-fit blur: %.0f ms (sigma %.2f px, radius %d)\n",
           _blurMs, coarseSigma, (int)(3.0 * coarseSigma + 0.5));
#pragma omp parallel for num_threads(numProcs)
  for (iterNr = 0; iterNr < totalSols; iterNr++) {
    double orientBest[3][3], eulerBest[3], eulerFit[3], orientFit[3][3];
    double q1[4], q2[4];
    int iJ, iK;
    double *outArrThisFit = (double *)calloc(3 * maxNrSpots, sizeof(double));
    if (outArrThisFit == NULL) {
      fprintf(stderr, "FATAL: could not allocate fit spot buffer.\n");
      exit(EXIT_FAILURE);
    }
    for (iJ = 0; iJ < 3; iJ++)
      for (iK = 0; iK < 3; iK++)
        orientBest[iJ][iK] = FinOrientArr[iterNr * 9 + 3 * iJ + iK];
    OrientMat2Euler(orientBest, eulerBest);
    for (iJ = 0; iJ < 3; iJ++)
      eulerFit[iJ] = eulerBest[iJ];
    int doCrystalFit = 1;
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(double));
    double latCFit[6], recipFit[3][3], mv = 0;
    // Prefilter HKLs at coarse orientation
    int *validIdx = (int *)malloc(nhkls * sizeof(int));
    if (validIdx == NULL) {
      fprintf(stderr, "FATAL: could not allocate validIdx (nhkls=%d).\n", nhkls);
      exit(EXIT_FAILURE);
    }
    int nValid =
        prefilterHKLs(hkls, nhkls, eulerBest, recip, nrPxX, nrPxY, rotTranspose,
                      pArr, pxX, pxY, Elo, Ehi, validIdx, nhkls);
    // Coarse-to-fine fit.  Stage 1: orientation-only against the blurred
    // image (wide capture radius) to bridge a coarse-grid seed error that
    // would otherwise leave the true peak outside the sharp-image spot basin.
    double eulerCoarse[3], latCdummy[6], mvCoarse = 0;
    FitOrientation(imageCoarse, eulerBest, hkls, nhkls, nrPxX, nrPxY, recip,
                   outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo,
                   Ehi, tol, LatticeParameter, eulerCoarse, latCdummy, &mvCoarse,
                   0 /*doCrystalFit*/, NULL /*all HKLs*/, 0,
                   1 /*forceNelderMead*/, 0.2 * deg2rad /*initStep*/,
                   minSpotIntensity);
    // Re-prefilter HKLs at the coarse solution (spots can move on/off the
    // detector after a ~grid-spacing correction), then fine-fit on the sharp
    // image for full precision (orientation + crystal).
    free(validIdx);
    validIdx = (int *)malloc(nhkls * sizeof(int));
    nValid = prefilterHKLs(hkls, nhkls, eulerCoarse, recip, nrPxX, nrPxY,
                           rotTranspose, pArr, pxX, pxY, Elo, Ehi, validIdx,
                           nhkls);
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(double));
    // Stage 2: full-precision fit on the sharp image, seeded from the coarse
    // solution.
    FitOrientation(image, eulerCoarse, hkls, nhkls, nrPxX, nrPxY, recip,
                   outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo,
                   Ehi, tol, LatticeParameter, eulerFit, latCFit, &mv,
                   doCrystalFit, validIdx, nValid,
                   0 /*forceNelderMead: ignored, Nelder-Mead always*/, 0.0,
                   minSpotIntensity);
    free(validIdx);
    Euler2OrientMat(eulerFit, orientFit);
    OrientMat2Quat33(orientBest, q1);
    OrientMat2Quat33(orientFit, q2);
    int simulNrSps = 0;
    calcRecipArray(latCFit, sg_num, recipFit);
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(double));
    int saveExtraInfo = iterNr + 1;
    int nrSps = writeCalcOverlap(
        image, eulerFit, hkls, nhkls, nrPxX, nrPxY, recipFit, outArrThisFit,
        maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo, Ehi, minSpotIntensity,
        ExtraInfo, saveExtraInfo, &simulNrSps, imageNum);
    if (nrSps >= minNrSpots) {
      int bs = bsArr[iterNr];
      double miso = GetMisOrientation(q1, q2);
      double OF[3][3];
      MatrixMultF33(orientFit, recipFit, OF);
#pragma omp critical
      {
        if (imageNum > 0)
          fprintf(outF, "%d\t", imageNum);
        fprintf(outF, "%d\t%d\t", iterNr + 1, dArr[iterNr]);
        fprintf(outF, "%-13.4lf\t", (mv / nrSps) * (mv / nrSps));
        fprintf(outF, "%-13.4lf\t", nrSps * (mv / nrSps) * (mv / nrSps));
        fprintf(outF, "%-13.4lf\t", mv);
        fprintf(outF, "%d\t", nrSps);
        fprintf(outF, "%d\t", simulNrSps);
        for (int k = 0; k < 3; k++)
          for (int l = 0; l < 3; l++)
            fprintf(outF, "%-13.7lf\t\t", OF[k][l]);
        for (int k = 0; k < 6; k++)
          fprintf(outF, "%-13.7lf\t\t", latCFit[k]);
        for (int k = 0; k < 3; k++)
          for (int l = 0; l < 3; l++)
            fprintf(outF, "%-13.7lf\t\t", orientFit[k][l]);
        fprintf(outF, "%-13.4lf\t%-13.7lf\t%d\n", bsScoreArr[iterNr], miso, bs);
      }
    }
    free(outArrThisFit);
  }
  free(imageCoarse);
}

#endif /* LAUE_MATCHING_HEADERS_H */
