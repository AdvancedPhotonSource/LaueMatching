//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//
// LaueMatching on the GPU (CUDA).
// The CUDA kernel handles the forward-simulation comparison step.
// All shared math/utility code lives in LaueMatchingHeaders.h.
//

#define _XOPEN_SOURCE 500
#include <cuda.h>
#include <unistd.h>
extern "C" {
#include "LaueMatchingHeaders.h"
}

// ── Global variable definitions (declared extern in header) ─────────────
double tol_LatC[6];
double tol_c_over_a;
double c_over_a_orig;
int sg_num;
double cellVol;
double phiVol;
int nSym;
double Symm[24][4];

#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// ── CUDA kernel ─────────────────────────────────────────────────────────
__global__ void compare(size_t nrPxX, size_t nOr, size_t nrMaxSpots,
                        double minInt, size_t minSps, uint16_t *oA, double *im,
                        double *mA) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nOr) {
    size_t loc = i * (1 + 2 * nrMaxSpots);
    size_t nrSpots = (size_t)oA[loc];
    size_t hklnr;
    size_t px, py;
    double thisInt, totInt = 0;
    size_t nSps = 0;
    for (hklnr = 0; hklnr < nrSpots; hklnr++) {
      loc++;
      px = (size_t)oA[loc];
      loc++;
      py = (size_t)oA[loc];
      thisInt = im[py * nrPxX + px];
      if (thisInt > 0) {
        totInt += thisInt;
        nSps++;
      }
    }
    if (nSps >= minSps && totInt >= minInt) {
      mA[i] = totInt * sqrt((double)nSps);
    }
  }
}

// ── Usage ───────────────────────────────────────────────────────────────
static void usageGPU() {
  puts("LaueMatching on the GPU\n"
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
    usageGPU();
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
  }
  if (tol_c_over_a != 0) {
    for (iter = 0; iter < 6; iter++)
      tol_LatC[iter] = 0;
  }
  c_over_a_orig = LatticeParameter[2] / LatticeParameter[0];
  fclose(fileParam);
  puts("Parameters read");

  // Rotation matrix
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

  // Read orientations
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
  str = "/dev/shm";
  LowNr = strncmp(orientFN, str, strlen(str));
  if (LowNr == 0) {
    fclose(orientF);
    int fd = open(orientFN, O_RDONLY);
    orients = (double *)mmap(0, szFile, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
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

  // Read HKLs
  char *hklfn = argv[3];
  FILE *hklf = fopen(hklfn, "r");
  if (hklf == NULL) {
    printf("Could not read hkl file %s.\n", hklfn);
    return 1;
  }
  int *hkls = (int *)calloc(MaxNHKLS * 3, sizeof(*hkls));
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

  // Read image
  printf("%d hkls read. \nNow reading image file %s, number of pixels %d. "
         "\nminNrSpots: %d, MinIntensity: %d\n",
         nhkls, argv[4], nrPxX * nrPxY, minNrSpots, (int)minIntensity);
  char *imageFN = argv[4];
  FILE *imageFile = fopen(imageFN, "rb");
  if (imageFile == NULL) {
    printf("Could not read image file %s.\n", imageFN);
    return 1;
  }
  double *image = (double *)malloc(nrPxX * nrPxY * sizeof(*image));
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

  // Forward simulation & matching
  double *matchedArr = (double *)calloc(nrOrients, sizeof(*matchedArr));
  if (matchedArr == NULL) {
    printf("Could not allocate matchedArr (%zu bytes).\n",
           nrOrients * sizeof(*matchedArr));
    return 1;
  }
  int numProcs = atoi(argv[5]);
  printf("Now running using %d threads.\n", numProcs);
  fflush(stdout);
  double ki[3] = {0, 0, 1.0};
  double start_time = omp_get_wtime();
  int global_iterator, k, l, m;
  size_t nrResults = 0;
  double recip[3][3];
  calcRecipArray(LatticeParameter, sg_num, recip);

  if (doFwd == 0) {
    printf("Trying to see if the forward simulation exists. Looking for %s "
           "file.\n",
           outfn);
    int result = open(outfn, O_RDONLY, S_IRUSR | S_IWUSR);
    if (result < 0) {
      printf("Could not read the forward simulation file. Running in "
             "simulation mode!\n");
      doFwd = 1;
    } else {
      printf("%s file was found. Will not do forward simulation.\n",
             outfn); // FIX: missing arg
      close(result);
    }
  } else
    printf("Forward simulation was requested, will be saved to %s.\n", outfn);

  if (doFwd == 1) {
    // Forward simulation using OpenMP on CPUs, then save to file
    bool *pxImgAll =
        (bool *)calloc((size_t)nrPxX * nrPxY * numProcs, sizeof(*pxImgAll));
    int fwdFd = open(outfn, O_CREAT | O_WRONLY | O_SYNC,
                     S_IRUSR | S_IWUSR); // FIX: open once
    if (fwdFd < 0) {
      printf("Could not open forward output file %s.\n", outfn);
      return 1;
    }
#pragma omp parallel num_threads(numProcs)
    {
      int procNr = omp_get_thread_num();
      int nrOrientsThread = (int)ceil((double)nrOrients / (double)numProcs);
      uint16_t *outArrThis;
      size_t szArr = nrOrientsThread * (1 + 2 * maxNrSpots);
      size_t OffsetHere;
      OffsetHere = procNr;
      OffsetHere *= szArr;
      OffsetHere *= sizeof(*outArrThis);
      int startOrientNr = procNr * nrOrientsThread;
      int endOrientNr = startOrientNr + nrOrientsThread;
      if (endOrientNr > (int)nrOrients)
        endOrientNr = nrOrients;
      nrOrientsThread = endOrientNr - startOrientNr;
      szArr = nrOrientsThread * (1 + 2 * maxNrSpots);
      outArrThis = (uint16_t *)calloc(szArr, sizeof(*outArrThis));
      if (outArrThis == NULL) {
        printf("Could not allocate outArr per thread, needed %lldMB of RAM. "
               "Behavior unexpected now.\n",
               (long long int)nrOrientsThread * (10 + 5 * maxNrSpots) *
                   sizeof(double) / (1024 * 1024));
      }
      int orientNr;
      double *qhatarr = (double *)calloc(maxNrSpots * 3, sizeof(*qhatarr));
      int ipx, ipy; // FIX: int instead of uint16_t
      double thisInt;
      double tO[3][3], thisOrient[3][3];
      int i, j;
      int hklnr, badSpot;
      double hkl[3], qvec[3], qlen, qhat[3], dot, kf[3], xyz[3], xp, yp,
          sinTheta, E;
      int spotNr, iterNr;
      int nSpots;
      double totInt;
      bool *pxImg;
      size_t offstBoolImg;
      offstBoolImg = nrPxX;
      offstBoolImg *= nrPxY;
      offstBoolImg *= procNr;
      pxImg = &pxImgAll[offstBoolImg];
      for (orientNr = startOrientNr; orientNr < endOrientNr; orientNr++) {
        nSpots = 0;
        totInt = 0;
        spotNr = 0;
        for (i = 0; i < 3; i++)
          for (j = 0; j < 3; j++)
            tO[i][j] = orients[orientNr * 9 + i * 3 + j];
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
            outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) + 1 +
                       2 * spotNr + 0] = (uint16_t)ipx;
            outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) + 1 +
                       2 * spotNr + 1] = (uint16_t)ipy;
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
        for (iterNr = 0; iterNr < spotNr; iterNr++) {
          pxImg[outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) +
                           1 + 2 * iterNr + 0] *
                    nrPxY +
                outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) +
                           1 + 2 * iterNr + 1]] = false;
        }
        outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) + 0] =
            (uint16_t)spotNr;
        if (nSpots >= minNrSpots && totInt >= minIntensity) {
#pragma omp critical
          {
            nrResults++;
          }
          matchedArr[orientNr] = totInt * sqrt((double)nSpots);
        }
      }
      ssize_t rc =
          pwrite(fwdFd, outArrThis, szArr * sizeof(*outArrThis), OffsetHere);
      if (rc < 0)
        printf("Could not write to output file\n");
      else if (rc != (ssize_t)(szArr * sizeof(*outArrThis))) {
        size_t off2 = OffsetHere + rc;
        size_t offset_arr = rc / sizeof(*outArrThis);
        size_t bytesRemaining = szArr * sizeof(*outArrThis) - rc;
        rc = pwrite(fwdFd, outArrThis + offset_arr, bytesRemaining, off2);
        if (rc != (ssize_t)bytesRemaining)
          printf(
              "Second try didn't work either. Too big array. Update code.\n");
      }
      free(outArrThis);
      free(qhatarr);
    }
    close(fwdFd);
    free(pxImgAll);
  } else {
    // ── Read forward simulation ────────────────────────────────────────
    double wt0 = omp_get_wtime();
    size_t stride = (size_t)(1 + 2 * maxNrSpots);
    size_t szArr = nrOrients * stride;
    uint16_t *outArr;
    int outArrMapped = 0;
    str = "/dev/shm";
    LowNr = strncmp(outfn, str, strlen(str));
    if (LowNr == 0) {
      int fd = open(outfn, O_RDONLY);
      outArr = (uint16_t *)mmap(0, szArr * sizeof(uint16_t), PROT_READ,
                                MAP_SHARED, fd, 0);
      close(fd);
      outArrMapped = 1;
    } else {
      outArr = (uint16_t *)calloc(szArr, sizeof(uint16_t));
      if (outArr == NULL) {
        printf("Could not allocate outArr.\n");
        fflush(stdout);
        return 1;
      }
      FILE *fwdFN = fopen(outfn, "rb");
      if (fwdFN == NULL) {
        printf("Could not open forward file %s.\n", outfn);
        free(outArr);
        return 1;
      }
      size_t read_cts = fread(outArr, szArr * sizeof(uint16_t), 1, fwdFN);
      if (read_cts != 1) {
        printf("Error: Failed to read forward simulation file.\n");
        if (ferror(fwdFN))
          perror("Error details");
        free(outArr);
        fclose(fwdFN);
        return 1;
      }
      fclose(fwdFN);
    }
    double wt_read = omp_get_wtime() - wt0;
    printf("Time elapsed to read the forward simulation: %lf s\n", wt_read);

    // ── Chunked CUDA matching ─────────────────────────────────────────
    // Process orientations in chunks that fit in GPU memory.
    // Synchronous transfers avoid the massive cost of cudaMallocHost
    // page-locking (>10 s for large pinned regions on this system).
    wt0 = omp_get_wtime();
    size_t freeMem = 0, totalMem = 0;
    gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));
    size_t imageBytes = (size_t)nrPxX * nrPxY * sizeof(double);
    // Use up to 80% of free GPU memory for the outArr chunk buffer
    size_t usable = (freeMem > imageBytes + (freeMem / 5))
                        ? freeMem - imageBytes - (freeMem / 5)
                        : freeMem / 2;
    size_t chunkSize = usable / (stride * sizeof(uint16_t));
    if (chunkSize < 1024)
      chunkSize = 1024;
    if (chunkSize > nrOrients)
      chunkSize = nrOrients;
    size_t nChunks = (nrOrients + chunkSize - 1) / chunkSize;
    size_t chunkOutBytes = chunkSize * stride * sizeof(uint16_t);
    size_t chunkMatchBytes = chunkSize * sizeof(double);
    printf("GPU chunked matching: %zu orientations in %zu chunks of %zu "
           "(stride=%zu, GPU mem: %.1f/%.1f GB free)\n",
           nrOrients, nChunks, chunkSize, stride,
           freeMem / 1e9, totalMem / 1e9);
    fflush(stdout);

    // Allocate device buffers (single set — no double-buffering needed)
    uint16_t *d_outChunk;
    double *d_matchChunk, *d_image;
    gpuErrchk(cudaMalloc(&d_outChunk, chunkOutBytes));
    gpuErrchk(cudaMalloc(&d_matchChunk, chunkMatchBytes));
    gpuErrchk(cudaMalloc(&d_image, imageBytes));
    gpuErrchk(cudaMemcpy(d_image, image, imageBytes,
                         cudaMemcpyHostToDevice));

    printf("Time elapsed to prepare the GPU: %lf s\n",
           omp_get_wtime() - wt0);
    wt0 = omp_get_wtime();

    // ── Chunk loop ─────────────────────────────────────────────────────
    for (size_t c = 0; c < nChunks; c++) {
      size_t offset = c * chunkSize;
      size_t thisChunk = chunkSize;
      if (offset + thisChunk > nrOrients)
        thisChunk = nrOrients - offset;
      size_t thisOutBytes = thisChunk * stride * sizeof(uint16_t);
      size_t thisMatchBytes = thisChunk * sizeof(double);

      // Synchronous H2D directly from outArr (mmap'd or malloc'd)
      gpuErrchk(cudaMemcpy(d_outChunk, outArr + offset * stride,
                           thisOutBytes, cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemset(d_matchChunk, 0, thisMatchBytes));

      // Launch kernel
      int blocks = (int)((thisChunk + 1023) / 1024);
      compare<<<blocks, 1024>>>(nrPxX, thisChunk, maxNrSpots, minIntensity,
                                minNrSpots, d_outChunk, d_image, d_matchChunk);

      // Synchronous D2H into matchedArr
      gpuErrchk(cudaMemcpy(matchedArr + offset, d_matchChunk,
                           thisMatchBytes, cudaMemcpyDeviceToHost));
    }

    printf("Time elapsed to match on the GPU: %lf s\n",
           omp_get_wtime() - wt0);

    // Cleanup
    cudaFree(d_outChunk);
    cudaFree(d_matchChunk);
    cudaFree(d_image);
    if (outArrMapped)
      munmap(outArr, szArr * sizeof(uint16_t));
    else
      free(outArr);

    // Count results
    for (int i = 0; i < (int)nrOrients; i++) {
      if (matchedArr[i] > 0)
        nrResults++;
    }
    printf("NrMatches: %zu\n", nrResults);
  }

  double time2 = omp_get_wtime() - start_time;
  printf("Finished comparing, time elapsed after comparing with forward "
         "simulation: %lf seconds.\n"
         "Searching for unique solutions.\n",
         time2);
  fflush(stdout);

  // Unique solutions
  nSym = MakeSymmetries(sg_num, Symm);

  double orient1[9], orient2[9], quat1[4], quat2[4], misoAngle, bestIntensity;
  double tol = 3 * deg2rad;
  maxNrSpots *= 3;
  int bestSol;
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
  double *mA = (double *)calloc(nrResults, sizeof(*mA));
  size_t *rowNrs = (size_t *)calloc(nrResults, sizeof(*rowNrs));
  int resultNr = 0;
  for (global_iterator = 0; global_iterator < (int)nrOrients;
       global_iterator++) {
    if (matchedArr[global_iterator] != 0) {
      mA[resultNr] = matchedArr[global_iterator];
      rowNrs[resultNr] = global_iterator;
      resultNr++;
    }
  }

  int *doneArr = (int *)calloc(nrResults, sizeof(*doneArr));
  double *FinOrientArr = (double *)calloc(nrResults * 9, sizeof(*FinOrientArr));
  int iterNr = 0;
  int *dArr = (int *)calloc(nrResults, sizeof(*dArr));
  int *bsArr = (int *)calloc(nrResults, sizeof(*bsArr));
  for (global_iterator = 0; global_iterator < (int)nrResults;
       global_iterator++) {
    if (doneArr[global_iterator] != 0)
      continue;
    for (k = 0; k < 9; k++) {
      orient1[k] = orients[rowNrs[global_iterator] * 9 + k];
    }
    OrientMat2Quat(orient1, quat1);
    doneArr[global_iterator] = 1;
    bestSol = rowNrs[global_iterator];
    bestIntensity = mA[global_iterator];
    for (l = global_iterator + 1; l < (int)nrResults; l++) {
      if (doneArr[l] > 0)
        continue;
      for (m = 0; m < 9; m++) {
        orient2[m] =
            orients[rowNrs[l] * 9 +
                    m]; // FIX: was orients[l*9+m] (missing rowNrs indirection)
      }
      OrientMat2Quat(orient2, quat2);
      misoAngle = GetMisOrientation(quat1, quat2);
      if (misoAngle <= maxAngle) {
        doneArr[l] = 1;
        doneArr[global_iterator]++;
        if (mA[l] > bestIntensity) {
          bestIntensity = mA[l];
          bestSol = rowNrs[l];
        }
      }
    }
    for (k = 0; k < 9; k++) {
      FinOrientArr[iterNr * 9 + k] = orients[bestSol * 9 + k];
    }
    dArr[iterNr] = doneArr[global_iterator];
    bsArr[iterNr] = bestSol;
    iterNr++;
  }
  double time3 = omp_get_wtime() - start_time;
  printf("Finished finding unique solutions, took: %lf seconds.\n",
         time3 - time2);
  int totalSols = iterNr;
#pragma omp parallel for num_threads(numProcs)
  for (iterNr = 0; iterNr < totalSols; iterNr++) {
    double orientBest[3][3], eulerBest[3], eulerFit[3], orientFit[3][3], q1[4],
        q2[4];
    int iJ, iK;
    // Hoisted allocation
    double *outArrThisFit =
        (double *)calloc(3 * maxNrSpots, sizeof(*outArrThisFit));
    for (iJ = 0; iJ < 3; iJ++) {
      for (iK = 0; iK < 3; iK++) {
        orientBest[iJ][iK] = FinOrientArr[iterNr * 9 + 3 * iJ + iK];
      }
    }
    OrientMat2Euler(orientBest, eulerBest);
    for (iJ = 0; iJ < 3; iJ++)
      eulerFit[iJ] = eulerBest[iJ];
    int saveExtraInfo = 0;
    int doCrystalFit = 0;
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(*outArrThisFit));
    double latCFit[6], recipFit[3][3], mv = 0;
    // First fit: orientation only
    FitOrientation(image, eulerBest, hkls, nhkls, nrPxX, nrPxY, recip,
                   outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo,
                   Ehi, tol, LatticeParameter, eulerFit, latCFit, &mv,
                   doCrystalFit);
    // Second fit: orientation + crystal parameters
    doCrystalFit = 1;
    for (iK = 0; iK < 3; iK++)
      eulerBest[iK] = eulerFit[iK];
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(*outArrThisFit));
    FitOrientation(image, eulerBest, hkls, nhkls, nrPxX, nrPxY, recip,
                   outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX, pxY, Elo,
                   Ehi, tol, LatticeParameter, eulerFit, latCFit, &mv,
                   doCrystalFit);
    // Convert fitted Euler angles to orientation matrix and quaternions
    Euler2OrientMat(eulerFit, orientFit);
    OrientMat2Quat33(orientBest, q1);
    OrientMat2Quat33(orientFit, q2);
    int simulNrSps = 0;
    calcRecipArray(latCFit, sg_num, recipFit);
    memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(*outArrThisFit));
    int nrSps =
        writeCalcOverlap(image, eulerFit, hkls, nhkls, nrPxX, nrPxY, recipFit,
                         outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX,
                         pxY, Elo, Ehi, ExtraInfo, saveExtraInfo, &simulNrSps);
    if (nrSps >= minNrSpots) {
      int bs = bsArr[iterNr];
      double miso = GetMisOrientation(q1, q2);
      saveExtraInfo = iterNr + 1;
      calcRecipArray(latCFit, sg_num, recipFit);
      memset(outArrThisFit, 0, 3 * maxNrSpots * sizeof(*outArrThisFit));
      writeCalcOverlap(image, eulerFit, hkls, nhkls, nrPxX, nrPxY, recipFit,
                       outArrThisFit, maxNrSpots, rotTranspose, pArr, pxX, pxY,
                       Elo, Ehi, ExtraInfo, saveExtraInfo, &simulNrSps);
      double OF[3][3];
      MatrixMultF33(orientFit, recipFit, OF);
#pragma omp critical
      {
        fprintf(outF, "%d\t%d\t", iterNr + 1, dArr[iterNr]);
        fprintf(outF, "%-13.4lf\t", (mv / nrSps) * (mv / nrSps));
        fprintf(outF, "%-13.4lf\t", nrSps * (mv / nrSps) * (mv / nrSps));
        fprintf(outF, "%-13.4lf\t", mv);
        fprintf(outF, "%d\t", nrSps);
        fprintf(outF, "%d\t", simulNrSps);
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            fprintf(outF, "%-13.7lf\t\t", OF[k][l]);
          }
        }
        for (k = 0; k < 6; k++)
          fprintf(outF, "%-13.7lf\t\t", latCFit[k]);
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            fprintf(outF, "%-13.7lf\t\t", orientFit[k][l]);
          }
        }
        fprintf(outF, "%-13.4lf\t%-13.7lf\t%d\n", matchedArr[bs], miso, bs);
      }
    }
    free(outArrThisFit); // Final free per thread/iteration
  }
  fclose(ExtraInfo);
  fclose(outF);

  // Cleanup
  free(matchedArr);
  free(mA);
  free(rowNrs);
  free(doneArr);
  free(FinOrientArr);
  free(dArr);
  free(bsArr);
  free(hkls);
  free(image);
  if (orientsMapped)
    munmap(orients, szFile);
  else
    free(orients);

  double timef = omp_get_wtime() - start_time - time3;
  printf("Finished, time elapsed in fitting: %lf seconds.\n"
         "Initial solutions: %zu Unique Orientations: %d\n"
         "Total time: %lf seconds.\n",
         timef, nrResults, totalSols, omp_get_wtime() - start_time);
  return 0;
}
