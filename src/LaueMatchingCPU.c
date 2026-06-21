//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//
// LaueMatching on the CPU.
// See LaueMatchingHeaders.h for all shared functions and data.
//

#include "LaueMatchingHeaders.h"
#include <errno.h>

// ── Global variable definitions (declared extern in header) ─────────────
double tol_LatC[6];
double tol_c_over_a;
double c_over_a_orig;
int sg_num;
double cellVol;
double phiVol;
int nSym;
double Symm[24][4];
int useBobyqa = 1; // default: BOBYQA

// ── Usage ───────────────────────────────────────────────────────────────
static void usageCPU() {
  puts("LaueMatching on the CPU\n"
       "Contact hsharma@anl.gov\n"
       "Arguments: \n"
       "* \tparameterFile (text)\n"
       "* \tbinary files for candidate orientation list [double]\n"
       "* \ttext of valid_hkls (preferably sorted on f2), space separated\n"
       "* \tbinary file for image [double]\n"
       "* \tnumber of CPU cores to use: \n"
       "* NOTE: some computers cannot read the full candidate orientation "
       "list, \n"
       "* must use multiple cores to distribute in that case\n\n"
       "Parameter file with the following parameters: \n"
       "\t\t* LatticeParameter (in nm and degrees),\n"
       "\t\t* tol_latC (in %%, 6 values),\n"
       "\t\t* tol_c_over_a (in %%, 1 value),\n"
       "\t\t* SpaceGroup,\n"
       "\t\t* P_Array, R_Array, PxX, PxY, NrPxX, NrPxY,\n"
       "\t\t* Elo, Ehi, MaxNrLaueSpots, ForwardFile, DoFwd,\n"
       "\t\t* MinNrSpots, MinIntensity, MaxAngle.\n");
}

// ── Main ────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  if (argc != 6) {
    usageCPU();
    return (0);
  }
  char *paramFN = argv[1];
  FILE *fileParam;
  fileParam = fopen(paramFN, "r");
  if (fileParam == NULL) {
    printf("Could not open parameter file %s.\n", paramFN);
    return 1;
  }
  char aline[1000], *str, dummy[1000], outfn[1000];
  int LowNr, nrPxX, nrPxY, maxNrSpots = 500, minNrSpots = 5, doFwd = 1;
  size_t batchSize = 1000000; // default 1M orientations per batch
  sg_num = 225;
  double pArr[3], rArr[3], pxX, pxY, Elo = 5, Ehi = 30;
  int iter;
  for (iter = 0; iter < 6; iter++)
    tol_LatC[iter] = 0;
  double minIntensity = 1000.0, maxAngle = 2.0;
  double LatticeParameter[6];
  puts("Reading parameter file");
  fflush(stdout);
  tol_c_over_a = 0;
  while (fgets(aline, 1000, fileParam) != NULL) {
    str = "LatticeParameter";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &LatticeParameter[0],
             &LatticeParameter[1], &LatticeParameter[2], &LatticeParameter[3],
             &LatticeParameter[4], &LatticeParameter[5]);
      calcV(LatticeParameter);
      continue;
    }
    str = "P_Array";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf", dummy, &pArr[0], &pArr[1], &pArr[2]);
      continue;
    }
    str = "R_Array";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf", dummy, &rArr[0], &rArr[1], &rArr[2]);
      continue;
    }
    str = "tol_c_over_a";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &tol_c_over_a);
      continue;
    }
    str = "PxX";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &pxX);
      continue;
    }
    str = "PxY";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &pxY);
      continue;
    }
    str = "Elo";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Elo);
      continue;
    }
    str = "Ehi";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &Ehi);
      continue;
    }
    str = "DoFwd";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &doFwd);
      continue;
    }
    str = "NrPxX";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nrPxX);
      continue;
    }
    str = "NrPxY";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &nrPxY);
      continue;
    }
    str = "MaxNrLaueSpots";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &maxNrSpots);
      continue;
    }
    str = "BatchSize";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %zu", dummy, &batchSize);
      if (batchSize == 0) batchSize = 1000000; // guard against 0
      continue;
    }
    str = "MinNrSpots";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &minNrSpots);
      continue;
    }
    str = "SpaceGroup";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &sg_num);
      continue;
    }
    str = "MinIntensity";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &minIntensity);
      continue;
    }
    str = "MaxAngle";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf", dummy, &maxAngle);
      continue;
    }
    str = "ForwardFile";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, outfn); // FIX: was &outfn
      continue;
    }
    str = "tol_LatC";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, &tol_LatC[0],
             &tol_LatC[1], &tol_LatC[2], &tol_LatC[3], &tol_LatC[4],
             &tol_LatC[5]);
      continue;
    }
    str = "Optimizer";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, dummy);
      if (strncmp(dummy, "NelderMead", 10) == 0)
        useBobyqa = 0;
      continue;
    }
  }
  if (tol_c_over_a != 0) {
    for (iter = 0; iter < 6; iter++)
      tol_LatC[iter] = 0;
  }
  c_over_a_orig = LatticeParameter[2] / LatticeParameter[0];
  fclose(fileParam);
  puts("Parameters read");

  // Rotation matrix (transpose = inverse for det=1 orthogonal matrix)
  double rotang = CalcLength(rArr[0], rArr[1], rArr[2]);
  double rotvect[3] = {rArr[0] / rotang, rArr[1] / rotang, rArr[2] / rotang};
  double rot[3][3] = {
      {cos(rotang) + (1 - cos(rotang)) * (rotvect[0] * rotvect[0]),
       (1 - cos(rotang)) * rotvect[0] * rotvect[1] - sin(rotang) * rotvect[2],
       (1 - cos(rotang)) * rotvect[0] * rotvect[2] + sin(rotang) * rotvect[1]},
      {(1 - cos(rotang)) * rotvect[1] * rotvect[0] + sin(rotang) * rotvect[2],
       cos(rotang) + (1 - cos(rotang)) * (rotvect[1] * rotvect[1]),
       (1 - cos(rotang)) * rotvect[1] * rotvect[2] - sin(rotang) * rotvect[0]},
      {(1 - cos(rotang)) * rotvect[2] * rotvect[0] - sin(rotang) * rotvect[1],
       (1 - cos(rotang)) * rotvect[2] * rotvect[1] + sin(rotang) * rotvect[0],
       cos(rotang) + (1 - cos(rotang)) * (rotvect[2] * rotvect[2])}};
  double rotTranspose[3][3] = {{rot[0][0], rot[1][0], rot[2][0]},
                               {rot[0][1], rot[1][1], rot[2][1]},
                               {rot[0][2], rot[1][2], rot[2][2]}};

  // Read orientations from a binary file
  puts("Reading orientations");
  double st_tm = omp_get_wtime();
  fflush(stdout);
  char *orientFN = argv[2];
  FILE *orientF = fopen(orientFN, "rb");
  if (orientF == NULL) {
    puts("Could not read orientation file, exiting.");
    return (1);
  }
  fseek(orientF, 0L, SEEK_END);
  size_t szFile = ftell(orientF);
  rewind(orientF);
  size_t nrOrients = (size_t)((double)szFile / (double)(9 * sizeof(double)));
  double *orients;
  int orientsMapped = 0;
  // If the file is located at /dev/shm, we can just mmap it:
  str = "/dev/shm";
  LowNr = strncmp(orientFN, str, strlen(str));
  if (LowNr == 0) {
    fclose(orientF);
    int fd = open(orientFN, O_RDONLY);
    orients = (double *)mmap(0, szFile, PROT_READ, MAP_SHARED, fd, 0);
    close(fd); // FIX: fd was never closed
    orientsMapped = 1;
    printf("%zu Orientations mapped into memory, took %lf seconds, now reading "
           "hkls\n",
           nrOrients, omp_get_wtime() - st_tm);
    fflush(stdout);
  } else {
    orients = (double *)malloc(szFile);
    size_t rc = fread(orients, 1, szFile, orientF);
    if (rc != szFile) {
      printf(
          "Error: Failed to read orientations. Expected %zu bytes, got %zu.\n",
          szFile, rc);
      if (ferror(orientF))
        perror("Error details");
      fclose(orientF);
      free(orients);
      return 1;
    }
    fclose(orientF);
    printf("%zu Orientations read, took %lf seconds, now reading hkls\n",
           nrOrients, omp_get_wtime() - st_tm);
    fflush(stdout);
  }

  // Read precomputed hkls from python
  char *hklfn = argv[3];
  FILE *hklf = fopen(hklfn, "r");
  if (hklf == NULL) {
    printf("Could not read hkl file %s.\n", hklfn);
    return 1;
  }
  int *hkls = calloc(MaxNHKLS * 3, sizeof(*hkls));
  if (hkls == NULL) {
    fprintf(stderr, "FATAL: could not allocate hkls (%zu bytes).\n",
            (size_t)MaxNHKLS * 3 * sizeof(*hkls));
    return 1;
  }
  int nhkls = 0;
  while (fgets(aline, 1000, hklf) != NULL) {
    if (nhkls >= MaxNHKLS) { // FIX: bounds check
      printf("Warning: more than %d HKLs in file, truncating.\n", MaxNHKLS);
      break;
    }
    sscanf(aline, "%d %d %d", &hkls[nhkls * 3 + 0], &hkls[nhkls * 3 + 1],
           &hkls[nhkls * 3 + 2]);
    nhkls++;
  }
  fclose(hklf);

  // Read image file
  printf("%d hkls read. \nNow reading image file %s, number of pixels %d. "
         "\nminNrSpots: %d, MinIntensity: %d\n",
         nhkls, argv[4], nrPxX * nrPxY, minNrSpots, (int)minIntensity);
  char *imageFN = argv[4];
  FILE *imageFile = fopen(imageFN, "rb");
  if (imageFile == NULL) {
    printf("Could not read image file %s.\n", imageFN);
    return 1;
  }
  double *image = malloc((size_t)nrPxX * nrPxY * sizeof(*image));
  if (image == NULL) {
    fprintf(stderr, "FATAL: could not allocate image buffer (%zu bytes).\n",
            (size_t)nrPxX * nrPxY * sizeof(*image));
    fclose(imageFile);
    return 1;
  }
  size_t read_cts = fread(image, nrPxX * nrPxY * sizeof(*image), 1, imageFile);
  if (read_cts != 1) {
    if (ferror(imageFile)) {
      perror("Error reading image file");
    } else if (feof(imageFile)) {
      printf("Error: Unexpected end of file while reading image.\n");
    } else {
      printf("Error: Failed to read full image data.\n");
    }
    free(image);
    fclose(imageFile);
    return 1;
  }
  int pxNr, nonZeroPx = 0;
  for (pxNr = 0; pxNr < nrPxX * nrPxY; pxNr++) {
    if (image[pxNr] > 0)
      nonZeroPx++;
  }
  fclose(imageFile);
  printf("Pixels with intensity: %d\n", nonZeroPx);

  // Create uint8 quantized image for cache-friendly matching
  size_t nPixels = (size_t)nrPxX * nrPxY;
  double maxImgVal = 0;
  for (pxNr = 0; pxNr < (int)nPixels; pxNr++)
    if (image[pxNr] > maxImgVal)
      maxImgVal = image[pxNr];
  double imScale = (maxImgVal > 0) ? maxImgVal / 255.0 : 1.0;
  uint8_t *image_u8 = (uint8_t *)malloc(nPixels);
  double imInvScale = (maxImgVal > 0) ? 255.0 / maxImgVal : 0.0;
  for (pxNr = 0; pxNr < (int)nPixels; pxNr++) {
    if (image[pxNr] > 0) {
      double v = image[pxNr] * imInvScale;
      image_u8[pxNr] = (v >= 255.0) ? 255 : (v < 1.0) ? 1 : (uint8_t)v;
    } else {
      image_u8[pxNr] = 0;
    }
  }
  printf("Quantized image: %.1f MB (double) -> %.1f MB (uint8)\n",
         nPixels * sizeof(double) / 1e6, nPixels / 1e6);

  // Create float image for CPU fitting (halves cache pressure vs double)
  float *imageF = (float *)malloc(nPixels * sizeof(float));
  for (pxNr = 0; pxNr < (int)nPixels; pxNr++)
    imageF[pxNr] = (float)image[pxNr];

  // Forward simulation & matching
  double *matchedArr = calloc(nrOrients, sizeof(*matchedArr));
  if (matchedArr == NULL) {
    printf(
        "Could not allocate matchedArr, requested %zu bytes. Please check.\n",
        (size_t)(nrOrients));
    return 1;
  }
  int numProcs = atoi(argv[5]);
  // Clamp BatchSize so we never over-allocate the per-thread batch buffer
  // beyond the thread's own slab of orientations.
  size_t nrOrientsThreadCap =
      (nrOrients + (size_t)numProcs - 1) / (size_t)numProcs;
  if (batchSize > nrOrientsThreadCap) batchSize = nrOrientsThreadCap;
  size_t batchBytes =
      batchSize * (size_t)(1 + 2 * maxNrSpots) * sizeof(uint16_t);
  printf("Now running using %d threads. BatchSize=%zu (per-thread batch buffer "
         "%.2f MB; %d threads => peak %.2f MB).\n",
         numProcs, batchSize, (double)batchBytes / (1024.0 * 1024.0), numProcs,
         (double)batchBytes * numProcs / (1024.0 * 1024.0));
  fflush(stdout);
  double start_time = omp_get_wtime();
  int global_iterator, k, l, nrResults = 0;
  double recip[3][3];
  calcRecipArray(LatticeParameter, sg_num, recip);

  // Check if forward file already exists
  if (doFwd == 0) {
    printf("Trying to see if the forward simulation exists. Looking for %s "
           "file.\n",
           outfn);
    int result = open(outfn, O_RDONLY, S_IRUSR | S_IWUSR);
    if (result < 0) { // FIX: was <1, but open returns -1 on error
      printf("Could not read the forward simulation file. Running in "
             "simulation mode!\n");
      doFwd = 1;
    } else {
      printf("%s file was found. Will not do forward simulation.\n",
             outfn); // FIX: missing outfn arg
      close(result);
    }
  } else {
    printf("Forward simulation was requested, will be saved to %s.\n", outfn);
  }

  uint16_t *outArr = NULL;
  LowNr = 1;
  bool *pxImgAll = NULL;

  // Open the forward file once before the parallel region (if writing)
  int fwdFd = -1;
  if (doFwd == 1) {
    pxImgAll = calloc((size_t)nrPxX * nrPxY * numProcs, sizeof(*pxImgAll));
    if (pxImgAll == NULL) {
      fprintf(stderr,
              "FATAL: could not allocate pxImgAll (%zu bytes for %d threads).\n",
              (size_t)nrPxX * nrPxY * numProcs * sizeof(*pxImgAll), numProcs);
      return 1;
    }
    // No O_SYNC: with batched pwrites we'd otherwise sync per-batch,
    // serializing writes against compute and inflating wallclock by ~5x.
    // We fsync once at the very end of the parallel region instead.
    fwdFd = open(outfn, O_CREAT | O_WRONLY,
                 S_IRUSR | S_IWUSR); // FIX: open once, not per-thread
    if (fwdFd < 0) {
      printf("Could not open forward output file %s.\n", outfn);
      return 1;
    }
  } else {
    // Read existing forward file
    str = "/dev/shm";
    LowNr = strncmp(outfn, str, strlen(str));
    size_t szArrFull = nrOrients * (1 + 2 * maxNrSpots);
    if (LowNr == 0) {
      int fd = open(outfn, O_RDONLY);
      outArr = (uint16_t *)mmap(0, szArrFull * sizeof(uint16_t), PROT_READ,
                                MAP_SHARED, fd, 0);
      close(fd); // FIX: fd was never closed
    }
  }

  // Per-orientation forward simulation table is uint16[1 + 2*maxNrSpots]
  // (entry 0 = spot count; remaining = spotNr × {ipx, ipy} pairs).  We
  // process orientations in batches of size `batchSize` per thread so that
  // the in-memory buffer is bounded regardless of how large the
  // (orientations × MaxNrLaueSpots) cross-product becomes.  Disk layout of
  // the cache file is unchanged: entry orientNr starts at byte offset
  // orientNr × (1 + 2*maxNrSpots) × sizeof(uint16_t).
  size_t entriesPerOrient = (size_t)(1 + 2 * maxNrSpots);
#pragma omp parallel num_threads(numProcs)
  {
    int procNr = omp_get_thread_num();
    size_t nrOrientsThread =
        (nrOrients + (size_t)numProcs - 1) / (size_t)numProcs;
    size_t startOrientNr = (size_t)procNr * nrOrientsThread;
    size_t endOrientNr = startOrientNr + nrOrientsThread;
    if (endOrientNr > nrOrients)
      endOrientNr = nrOrients;

    // Per-thread batch buffer (big enough for one batch only).
    size_t batchEntries = batchSize * entriesPerOrient;
    uint16_t *outArrBatch = calloc(batchEntries, sizeof(*outArrBatch));
    if (outArrBatch == NULL) {
      fprintf(stderr,
              "FATAL: thread %d failed to allocate %.2f MB batch buffer "
              "(BatchSize=%zu, MaxNrLaueSpots=%d). Reduce BatchSize and "
              "rerun.\n",
              procNr,
              (double)(batchEntries * sizeof(*outArrBatch)) / (1024.0 * 1024.0),
              batchSize, maxNrSpots);
      exit(EXIT_FAILURE);
    }
    double *qhatarr = calloc((size_t)maxNrSpots * 3, sizeof(*qhatarr));
    if (qhatarr == NULL) {
      fprintf(stderr, "FATAL: thread %d failed to allocate qhatarr.\n", procNr);
      exit(EXIT_FAILURE);
    }

    // Open the cache file for read once per thread (doFwd==0 + non-/dev/shm).
    int rfd = -1;
    if (doFwd == 0 && LowNr != 0) {
      rfd = open(outfn, O_RDONLY, S_IRUSR | S_IWUSR);
      if (rfd < 0) {
        fprintf(stderr,
                "FATAL: thread %d could not open cache for read: %s.\n",
                procNr, strerror(errno));
        exit(EXIT_FAILURE);
      }
    }

    // ── Batch loop ────────────────────────────────────────────────────────
    for (size_t batchStart = startOrientNr; batchStart < endOrientNr;
         batchStart += batchSize) {
      size_t batchEnd = batchStart + batchSize;
      if (batchEnd > endOrientNr)
        batchEnd = endOrientNr;
      size_t thisBatch = batchEnd - batchStart;
      size_t batchByteOffset =
          batchStart * entriesPerOrient * sizeof(*outArrBatch);
      size_t bytesThisBatch = thisBatch * entriesPerOrient * sizeof(*outArrBatch);

      // Populate the batch buffer for this batch.
      if (doFwd == 0) {
        if (LowNr == 0) {
          // /dev/shm fast path: copy from the read-only mmap region.
          size_t locOffset = batchStart * entriesPerOrient;
          memcpy(outArrBatch, &outArr[locOffset], bytesThisBatch);
        } else {
          // Regular file: pread (loop until satisfied).
          size_t bytesGot = 0;
          while (bytesGot < bytesThisBatch) {
            ssize_t rc =
                pread(rfd, (char *)outArrBatch + bytesGot,
                      bytesThisBatch - bytesGot, batchByteOffset + bytesGot);
            if (rc <= 0) {
              fprintf(stderr,
                      "FATAL: thread %d cache pread failed at offset %zu "
                      "(got %zu of %zu bytes): %s\n",
                      procNr, batchByteOffset + bytesGot, bytesGot,
                      bytesThisBatch,
                      (rc < 0) ? strerror(errno) : "unexpected EOF");
              exit(EXIT_FAILURE);
            }
            bytesGot += (size_t)rc;
          }
        }
      } else {
        // doFwd==1: clear the batch buffer to zero before fresh fill.
        memset(outArrBatch, 0, bytesThisBatch);
      }

      // ── Inner orientation loop (within the batch) ────────────────────
      int ipx, ipy;
      double thisInt;
      double tO[3][3], thisOrient[3][3];
      int i, j;
      int hklnr, badSpot;
      double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp,
          sinTheta, E;
      double ki[3] = {0, 0, 1.0};
      int spotNr, iterNr;
      int nSpots;
      double totInt;
      size_t loc;
      for (size_t orientNr = batchStart; orientNr < batchEnd; orientNr++) {
        nSpots = 0;
        totInt = 0;
        size_t local = orientNr - batchStart; // index within the batch buffer
        if (doFwd == 1) {
          bool *pxImg;
          size_t offstBoolImg = (size_t)nrPxX * nrPxY * (size_t)procNr;
          pxImg = &pxImgAll[offstBoolImg];
          spotNr = 0;
          for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
              tO[i][j] = orients[orientNr * 9 + (size_t)i * 3 + (size_t)j];
          MatrixMultF33(tO, recip, thisOrient);
          for (hklnr = 0; hklnr < nhkls; hklnr++) {
            hkl[0] = hkls[hklnr * 3 + 0];
            hkl[1] = hkls[hklnr * 3 + 1];
            hkl[2] = hkls[hklnr * 3 + 2];
            MatrixMultF(thisOrient, hkl, qvec);
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
            ipx = (int)((xp / pxX) + (0.5 * (nrPxX - 1)));
            if (ipx < 0 || ipx > (nrPxX - 1))
              continue;
            ipy = (int)((yp / pxY) + (0.5 * (nrPxY - 1)));
            if (ipy < 0 || ipy > (nrPxY - 1))
              continue;
            sinTheta = -qhat[2];
            E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
            if (E < Elo || E > Ehi)
              continue;
            badSpot = 0;
            if (pxImg[ipx * nrPxY + ipy])
              badSpot = 1;
            if (badSpot == 0) {
              pxImg[ipx * nrPxY + ipy] = true;
              qhatarr[3 * spotNr + 0] = qhat[0];
              qhatarr[3 * spotNr + 1] = qhat[1];
              qhatarr[3 * spotNr + 2] = qhat[2];
              outArrBatch[local * entriesPerOrient + 1 + 2 * spotNr + 0] =
                  (uint16_t)ipx;
              outArrBatch[local * entriesPerOrient + 1 + 2 * spotNr + 1] =
                  (uint16_t)ipy;
              thisInt = image[ipy * nrPxX + ipx];
              if (thisInt > 0) {
                totInt += thisInt;
                nSpots++;
              }
              spotNr++;
              if (spotNr == maxNrSpots)
                break;
            }
          }
          // Clear the per-thread pixel-occupancy mask for this orientation
          // so the next orientation starts clean.
          for (iterNr = 0; iterNr < spotNr; iterNr++) {
            pxImg[outArrBatch[local * entriesPerOrient + 1 + 2 * iterNr + 0] *
                      nrPxY +
                  outArrBatch[local * entriesPerOrient + 1 + 2 * iterNr + 1]] =
                false;
          }
          outArrBatch[local * entriesPerOrient + 0] = (uint16_t)spotNr;
        } else {
          loc = local * entriesPerOrient + 0;
          spotNr = (int)outArrBatch[loc];
          for (hklnr = 0; hklnr < spotNr; hklnr++) {
            loc++;
            ipx = outArrBatch[loc];
            loc++;
            ipy = outArrBatch[loc];
            uint8_t raw = image_u8[ipy * nrPxX + ipx];
            if (raw > 0) {
              totInt += (double)raw * imScale;
              nSpots++;
            }
          }
        }
        if (nSpots >= minNrSpots && totInt >= minIntensity) {
#pragma omp critical
          {
            nrResults++;
          }
          matchedArr[orientNr] = totInt * sqrt((double)nSpots);
        }
      }
      // ── End inner orientation loop ───────────────────────────────────

      // Write this batch back to the cache (doFwd==1).
      if (doFwd == 1) {
        size_t bytesWritten = 0;
        while (bytesWritten < bytesThisBatch) {
          ssize_t rc =
              pwrite(fwdFd, (char *)outArrBatch + bytesWritten,
                     bytesThisBatch - bytesWritten,
                     batchByteOffset + bytesWritten);
          if (rc < 0) {
            fprintf(stderr,
                    "FATAL: thread %d pwrite failed at offset %zu "
                    "(wrote %zu of %zu bytes): %s\n",
                    procNr, batchByteOffset + bytesWritten, bytesWritten,
                    bytesThisBatch, strerror(errno));
            exit(EXIT_FAILURE);
          }
          bytesWritten += (size_t)rc;
        }
      }
    } // end batch loop

    if (rfd >= 0)
      close(rfd);
    free(outArrBatch);
    free(qhatarr);
  }
  if (fwdFd >= 0) {
    // Single fsync at end (instead of per-batch O_SYNC) so the cache
    // is durable on disk before subsequent runs try to read it.
    if (fsync(fwdFd) < 0) {
      fprintf(stderr, "WARNING: fsync(fwdFd) failed: %s\n", strerror(errno));
    }
    close(fwdFd);
  }

  double time2 = omp_get_wtime() - start_time;
  printf("Finished comparing, time elapsed after comparing with forward "
         "simulation: %lf seconds.\n"
         "Searching for unique solutions.\n",
         time2);
  fflush(stdout);

  // Figure out the unique orientations (within maxAngle) and do optimization
  // for those.
  nSym = MakeSymmetries(sg_num, Symm);

  double tol = 3 * deg2rad;
  maxNrSpots *= 3;
  char outFN[1000];
  sprintf(outFN, "%s.solutions.txt", imageFN);
  FILE *outF = fopen(outFN, "w");
  if (outF == NULL) {
    printf("Could not open file for writing solutions. Exiting.\n");
    return (1);
  }
  sprintf(outFN, "%s.spots.txt", imageFN);
  FILE *ExtraInfo = fopen(outFN, "w");
  fprintf(ExtraInfo, "%%GrainNr\tSpotNr\th\tk\tl\tX\tY\tQhat[0]\tQhat[1]\tQhat["
                     "2]\tIntensity\n");
  fprintf(
      outF,
      "%%GrainNr\tNumberOfSolutions\tIntensity\tNMatches*Intensity\tNMatches*"
      "sqrt(Intensity)\t"
      "NMatches\tNSpotsCalc\t"
      "Recip1\tRecip2\tRecip3\tRecip4\tRecip5\tRecip6\tRecip7\tRecip8\tRecip9\t"
      "LatticeParameterFit[a]\tLatticeParameterFit[b]\tLatticeParameterFit[c]\t"
      "LatticeParameterFit[alpha]\tLatticeParameterFit[beta]"
      "\tLatticeParameterFit[gamma]\t"
      "OrientMatrix0\tOrientMatrix1\tOrientMatrix2\tOrientMatrix3\tOrientMatrix"
      "4\tOrientMatrix5\t"
      "OrientMatrix6\tOrientMatrix7\tOrientMatrix8\t"
      "CoarseNMatches*sqrt(Intensity)\t"
      "misOrientationPostRefinement[degrees]\torientationRowNr\n");

  // Collect results
  double *mA = calloc(nrResults, sizeof(*mA));
  size_t *rowNrs = calloc(nrResults, sizeof(*rowNrs));
  int resultNr = 0;
  for (global_iterator = 0; global_iterator < (int)nrOrients;
       global_iterator++) {
    if (matchedArr[global_iterator] != 0) {
      mA[resultNr] = matchedArr[global_iterator];
      rowNrs[resultNr] = global_iterator;
      resultNr++;
    }
  }

  double *FinOrientArr = calloc(nrResults * 9, sizeof(*FinOrientArr));
  int *dArr = calloc(nrResults, sizeof(*dArr));
  int *bsArr = calloc(nrResults, sizeof(*bsArr));
  double *bsScoreArr = calloc(nrResults, sizeof(*bsScoreArr));
  int totalSols = mergeDuplicateOrientations(orients, rowNrs, mA, nrResults,
                                             maxAngle, numProcs, FinOrientArr,
                                             dArr, bsArr, bsScoreArr);
  double time3 = omp_get_wtime() - start_time;
  printf("Finished finding unique solutions, took: %lf seconds.\n",
         time3 - time2);
  fitAndWriteOrientations(
      imageF, FinOrientArr, dArr, bsArr, bsScoreArr, totalSols, hkls, nhkls,
      nrPxX, nrPxY, recip, rotTranspose, pArr, pxX, pxY, Elo, Ehi, tol,
      LatticeParameter, maxNrSpots, minNrSpots, numProcs, outF, ExtraInfo, 0);
  fclose(ExtraInfo);
  fclose(outF);

  // Cleanup
  free(matchedArr);
  free(mA);
  free(rowNrs);
  free(FinOrientArr);
  free(dArr);
  free(bsArr);
  free(bsScoreArr);
  free(image_u8);
  free(imageF);
  free(image);
  free(hkls);
  if (orientsMapped)
    munmap(orients, szFile); // FIX: was never munmap'd
  else
    free(orients);
  if (pxImgAll)
    free(pxImgAll);

  double timef = omp_get_wtime() - start_time - time3;
  printf("Finished, time elapsed in fitting: %lf seconds.\n"
         "Initial solutions: %d Unique Orientations: %d\n"
         "Total time: %lf seconds.\n",
         timef, nrResults, totalSols, omp_get_wtime() - start_time);
  return 0;
}
