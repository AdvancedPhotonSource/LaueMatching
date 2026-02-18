//
// Copyright (c) 2024, UChicago Argonne, LLC
// See LICENSE file.
// Hemant Sharma, hsharma@anl.gov
//
// LaueMatchingGPUStream — Persistent GPU daemon.
// Initializes CUDA context, forward simulation, and device buffers once,
// then accepts raw binary images over TCP and processes each one.
// Multiple CUDA streams overlap GPU matching with CPU fitting.
//

#define _XOPEN_SOURCE 500
#include <arpa/inet.h>
#include <cuda.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
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
int useBobyqa = 1;

// ── Constants ───────────────────────────────────────────────────────────
#define PORT 60517
#define MAX_CONNECTIONS 10
#define MAX_QUEUE_SIZE 32
#define MAX_STREAMS 4
#define HEADER_SIZE 2 // uint16_t image_num

volatile sig_atomic_t keep_running = 1;

// ── CUDA error handling ────────────────────────────────────────────────
#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__, true);                                \
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

// ── CUDA kernel (same as LaueMatchingGPU.cu) ───────────────────────────
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

// ── Data structures ────────────────────────────────────────────────────
typedef struct {
  uint16_t image_num;
  double *data; // raw image pixels
} ImageChunk;

typedef struct {
  ImageChunk chunks[MAX_QUEUE_SIZE];
  int front;
  int rear;
  int count;
  pthread_mutex_t mutex;
  pthread_cond_t not_empty;
  pthread_cond_t not_full;
} ProcessQueue;

// Per-stream GPU context
typedef struct {
  cudaStream_t stream;
  double *d_image;      // device image buffer
  double *d_matchChunk; // device match scores (per chunk)
  double *matchedArr;   // host match results (full nrOrients)
} StreamContext;

// ── Globals ────────────────────────────────────────────────────────────
static ProcessQueue process_queue;
static size_t g_imagePixels; // nrPxX * nrPxY

// ── Signal handler ─────────────────────────────────────────────────────
void sigint_handler(int signum) {
  if (keep_running) {
    printf("\nCaught signal %d, requesting shutdown...\n", signum);
    keep_running = 0;
  }
}

// ── Queue functions ────────────────────────────────────────────────────
void queue_init(ProcessQueue *q) {
  q->front = 0;
  q->rear = -1;
  q->count = 0;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
}

int queue_push(ProcessQueue *q, uint16_t image_num, double *data) {
  pthread_mutex_lock(&q->mutex);
  while (q->count >= MAX_QUEUE_SIZE && keep_running) {
    printf("Queue full, waiting...\n");
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 1;
    pthread_cond_timedwait(&q->not_full, &q->mutex, &ts);
  }
  if (!keep_running) {
    pthread_mutex_unlock(&q->mutex);
    return -1;
  }
  q->rear = (q->rear + 1) % MAX_QUEUE_SIZE;
  q->chunks[q->rear].image_num = image_num;
  q->chunks[q->rear].data = data;
  q->count++;
  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mutex);
  return 0;
}

int queue_pop(ProcessQueue *q, ImageChunk *chunk) {
  pthread_mutex_lock(&q->mutex);
  while (q->count == 0 && keep_running) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 1;
    pthread_cond_timedwait(&q->not_empty, &q->mutex, &ts);
  }
  if (q->count == 0 && !keep_running) {
    pthread_mutex_unlock(&q->mutex);
    return -1;
  }
  *chunk = q->chunks[q->front];
  q->front = (q->front + 1) % MAX_QUEUE_SIZE;
  q->count--;
  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->mutex);
  return 0;
}

void queue_destroy(ProcessQueue *q) {
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->not_empty);
  pthread_cond_destroy(&q->not_full);
  while (q->count > 0) {
    double *d = q->chunks[q->front].data;
    q->front = (q->front + 1) % MAX_QUEUE_SIZE;
    q->count--;
    free(d);
  }
}

// ── Socket handling ────────────────────────────────────────────────────
void *handle_client(void *arg) {
  int client_socket = *((int *)arg);
  free(arg);
  uint8_t header_buf[HEADER_SIZE];
  // Wire protocol: float pixels (4 bytes each), not double
  size_t wire_payload_size = g_imagePixels * sizeof(float);
  printf("Client handler started (socket %d), wire_payload=%zu bytes\n",
         client_socket, wire_payload_size);

  // Reusable receive buffer for float data
  float *recv_buf = (float *)malloc(wire_payload_size);
  if (!recv_buf) {
    perror("malloc recv_buf");
    close(client_socket);
    return NULL;
  }

  while (keep_running) {
    // Read header: 2 bytes (uint16_t image_num)
    int total_read = 0;
    while (total_read < HEADER_SIZE) {
      int n = recv(client_socket, header_buf + total_read,
                   HEADER_SIZE - total_read, 0);
      if (n <= 0)
        goto done;
      total_read += n;
    }
    uint16_t image_num;
    memcpy(&image_num, header_buf, 2);

    // Read payload: float image data
    size_t total_payload = 0;
    while (total_payload < wire_payload_size) {
      int n = recv(client_socket, (char *)recv_buf + total_payload,
                   wire_payload_size - total_payload, 0);
      if (n <= 0)
        goto done;
      total_payload += n;
    }

    // Convert float → double for GPU processing
    double *data = (double *)malloc(g_imagePixels * sizeof(double));
    if (!data) {
      perror("malloc image double");
      goto done;
    }
    for (size_t p = 0; p < g_imagePixels; p++)
      data[p] = (double)recv_buf[p];

    printf("Received image %u (%zu bytes wire, %zu pixels)\n", image_num,
           wire_payload_size, g_imagePixels);
    if (queue_push(&process_queue, image_num, data) < 0) {
      printf("Queue push failed for image %u\n", image_num);
      free(data);
      goto done;
    }
  }
done:
  free(recv_buf);
  close(client_socket);
  printf("Client handler finished (socket %d)\n", client_socket);
  return NULL;
}

void *accept_connections(void *server_fd_ptr) {
  int server_fd = *((int *)server_fd_ptr);
  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);
  printf("Accept thread started, listening on port %d\n", PORT);
  while (keep_running) {
    int *csock = (int *)malloc(sizeof(int));
    *csock = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
    if (!keep_running) {
      if (*csock >= 0)
        close(*csock);
      free(csock);
      break;
    }
    if (*csock < 0) {
      if (errno == EINTR)
        continue;
      perror("accept");
      free(csock);
      sleep(1);
      continue;
    }
    printf("Connection accepted from %s:%d\n", inet_ntoa(client_addr.sin_addr),
           ntohs(client_addr.sin_port));
    pthread_t tid;
    if (pthread_create(&tid, NULL, handle_client, (void *)csock) != 0) {
      perror("pthread_create");
      close(*csock);
      free(csock);
    } else {
      pthread_detach(tid);
    }
  }
  printf("Accept thread exiting.\n");
  return NULL;
}

// ── Usage ──────────────────────────────────────────────────────────────
static void usage(const char *prog) {
  printf("LaueMatchingGPUStream — persistent GPU daemon\n"
         "Contact hsharma@anl.gov\n\n"
         "Usage: %s parameterFile orientationFile hklFile nCPUs\n\n"
         "Listens on TCP port %d for binary images.\n"
         "Wire protocol: uint16_t image_num + double[NrPxX*NrPxY]\n",
         prog, PORT);
}

// ── Main ───────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  if (argc != 5) {
    usage(argv[0]);
    return 0;
  }

  signal(SIGINT, sigint_handler);
  signal(SIGTERM, sigint_handler);

  double st_tm = omp_get_wtime();

  // ── Parse parameters ──────────────────────────────────────────────
  char *paramFN = argv[1];
  FILE *fileParam = fopen(paramFN, "r");
  if (!fileParam) {
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
  tol_c_over_a = 0;
  char resultDir[1000] = "results_stream";
  int minGoodSpots = 5;
  puts("Reading parameter file");
  fflush(stdout);
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
      sscanf(aline, "%s %s", dummy, outfn);
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
    str = "ResultDir";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %s", dummy, resultDir);
      continue;
    }
    str = "MinGoodSpots";
    LowNr = strncmp(aline, str, strlen(str));
    if (LowNr == 0) {
      sscanf(aline, "%s %d", dummy, &minGoodSpots);
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

  // ── Read orientations ─────────────────────────────────────────────
  puts("Reading orientations");
  fflush(stdout);
  char *orientFN = argv[2];
  FILE *orientF = fopen(orientFN, "rb");
  if (!orientF) {
    puts("Could not read orientation file, exiting.");
    return 1;
  }
  fseek(orientF, 0L, SEEK_END);
  size_t szFile = ftell(orientF);
  rewind(orientF);
  size_t nrOrients = szFile / (9 * sizeof(double));
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
    printf("%zu orientations mapped into memory\n", nrOrients);
  } else {
    orients = (double *)malloc(szFile);
    size_t rc = fread(orients, 1, szFile, orientF);
    if (rc != szFile) {
      printf("Error reading orientations.\n");
      fclose(orientF);
      free(orients);
      return 1;
    }
    fclose(orientF);
    printf("%zu orientations read\n", nrOrients);
  }
  fflush(stdout);

  // ── Read HKLs ────────────────────────────────────────────────────
  char *hklfn = argv[3];
  FILE *hklf = fopen(hklfn, "r");
  if (!hklf) {
    printf("Could not read hkl file %s.\n", hklfn);
    return 1;
  }
  int *hkls = (int *)calloc(MaxNHKLS * 3, sizeof(*hkls));
  int nhkls = 0;
  while (fgets(aline, 1000, hklf) != NULL) {
    if (nhkls >= MaxNHKLS)
      break;
    sscanf(aline, "%d %d %d", &hkls[nhkls * 3 + 0], &hkls[nhkls * 3 + 1],
           &hkls[nhkls * 3 + 2]);
    nhkls++;
  }
  fclose(hklf);
  printf("%d hkls read\n", nhkls);

  int numProcs = atoi(argv[4]);
  printf("Will use %d CPU threads for fitting.\n", numProcs);
  g_imagePixels = (size_t)nrPxX * nrPxY;

  // ── Pre-compute ──────────────────────────────────────────────────
  nSym = MakeSymmetries(sg_num, Symm);
  double recip[3][3];
  calcRecipArray(LatticeParameter, sg_num, recip);

  // ── CUDA context + fwd sim ───────────────────────────────────────
  printf("Initializing CUDA...\n");
  fflush(stdout);
  double wt_cuda_init = omp_get_wtime();
  gpuErrchk(cudaFree(0));
  printf("CUDA context init: %lf s\n", omp_get_wtime() - wt_cuda_init);

  // Load or compute forward simulation (same logic as LaueMatchingGPU.cu)
  size_t stride = (size_t)(1 + 2 * maxNrSpots);
  size_t szArr = nrOrients * stride;
  uint16_t *outArr = NULL;
  int outArrMapped = 0;

  if (doFwd == 0) {
    int result = open(outfn, O_RDONLY, S_IRUSR | S_IWUSR);
    if (result < 0) {
      printf("Forward simulation file %s not found. Running fwd sim...\n",
             outfn);
      doFwd = 1;
    } else {
      printf("Forward simulation file %s found.\n", outfn);
      close(result);
    }
  }

  if (doFwd == 1) {
    // Forward simulation on CPU (same as LaueMatchingGPU.cu)
    printf("Running forward simulation (%zu orientations)...\n", nrOrients);
    fflush(stdout);
    double wt_fwd = omp_get_wtime();
    double *matchedArrFwd = (double *)calloc(nrOrients, sizeof(double));
    double ki[3] = {0, 0, 1.0};
    bool *pxImgAll =
        (bool *)calloc((size_t)nrPxX * nrPxY * numProcs, sizeof(bool));
    int fwdFd = open(outfn, O_CREAT | O_WRONLY | O_SYNC, S_IRUSR | S_IWUSR);
    if (fwdFd < 0) {
      printf("Could not open forward output file %s.\n", outfn);
      return 1;
    }
#pragma omp parallel num_threads(numProcs)
    {
      int procNr = omp_get_thread_num();
      int nrOrientsThread = (int)ceil((double)nrOrients / (double)numProcs);
      int startOrientNr = procNr * nrOrientsThread;
      int endOrientNr = startOrientNr + nrOrientsThread;
      if (endOrientNr > (int)nrOrients)
        endOrientNr = nrOrients;
      nrOrientsThread = endOrientNr - startOrientNr;
      size_t szArrThread = (size_t)nrOrientsThread * (1 + 2 * maxNrSpots);
      size_t OffsetHere = (size_t)procNr;
      OffsetHere *= (size_t)((int)ceil((double)nrOrients / (double)numProcs)) *
                    (1 + 2 * maxNrSpots);
      OffsetHere *= sizeof(uint16_t);
      uint16_t *outArrThis = (uint16_t *)calloc(szArrThread, sizeof(uint16_t));
      double *qhatarr = (double *)calloc(maxNrSpots * 3, sizeof(double));
      bool *pxImg = &pxImgAll[(size_t)nrPxX * nrPxY * procNr];
      int orientNr;
      for (orientNr = startOrientNr; orientNr < endOrientNr; orientNr++) {
        int nSpots = 0, spotNr = 0;
        double totInt = 0;
        double tO[3][3], thisOrient[3][3];
        int i, j;
        for (i = 0; i < 3; i++)
          for (j = 0; j < 3; j++)
            tO[i][j] = orients[orientNr * 9 + i * 3 + j];
        MatrixMultF33(tO, recip, thisOrient);
        for (int hklnr = 0; hklnr < nhkls; hklnr++) {
          double hkl[3] = {(double)hkls[hklnr * 3], (double)hkls[hklnr * 3 + 1],
                           (double)hkls[hklnr * 3 + 2]};
          double qvec[3], qhat[3], kf[3], xyz[3];
          MatrixMultF(thisOrient, hkl, qvec);
          double qlen = CalcLength(qvec[0], qvec[1], qvec[2]);
          if (qlen == 0)
            continue;
          qhat[0] = qvec[0] / qlen;
          qhat[1] = qvec[1] / qlen;
          qhat[2] = qvec[2] / qlen;
          double dot = qhat[2];
          kf[0] = ki[0] - 2 * dot * qhat[0];
          kf[1] = ki[1] - 2 * dot * qhat[1];
          kf[2] = ki[2] - 2 * dot * qhat[2];
          MatrixMultF(rotTranspose, kf, xyz);
          if (xyz[2] <= 0)
            continue;
          xyz[0] = xyz[0] * pArr[2] / xyz[2];
          xyz[1] = xyz[1] * pArr[2] / xyz[2];
          double xp = xyz[0] - pArr[0];
          double yp = xyz[1] - pArr[1];
          int ipx = (int)((xp / pxX) + (0.5 * (nrPxX - 1)));
          if (ipx < 0 || ipx > (nrPxX - 1))
            continue;
          int ipy = (int)((yp / pxY) + (0.5 * (nrPxY - 1)));
          if (ipy < 0 || ipy > (nrPxY - 1))
            continue;
          double sinTheta = -qhat[2];
          double E = hc_keVnm * qlen / (4 * M_PI * sinTheta);
          if (E < Elo || E > Ehi)
            continue;
          if (pxImg[ipx * nrPxY + ipy])
            continue;
          pxImg[ipx * nrPxY + ipy] = true;
          qhatarr[3 * spotNr + 0] = qhat[0];
          qhatarr[3 * spotNr + 1] = qhat[1];
          qhatarr[3 * spotNr + 2] = qhat[2];
          outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) + 1 +
                     2 * spotNr + 0] = (uint16_t)ipx;
          outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) + 1 +
                     2 * spotNr + 1] = (uint16_t)ipy;
          spotNr++;
          if (spotNr == maxNrSpots)
            break;
        }
        for (int it = 0; it < spotNr; it++)
          pxImg[outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) +
                           1 + 2 * it + 0] *
                    nrPxY +
                outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots) +
                           1 + 2 * it + 1]] = false;
        outArrThis[(orientNr - startOrientNr) * (1 + 2 * maxNrSpots)] =
            (uint16_t)spotNr;
      }
      pwrite(fwdFd, outArrThis, szArrThread * sizeof(uint16_t), OffsetHere);
      free(outArrThis);
      free(qhatarr);
    }
    close(fwdFd);
    free(pxImgAll);
    free(matchedArrFwd);
    printf("Forward simulation completed in %lf s\n", omp_get_wtime() - wt_fwd);
  }

  // ── Load forward simulation into host memory ─────────────────────
  double wt_read = omp_get_wtime();
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
    FILE *fwdFN = fopen(outfn, "rb");
    if (!fwdFN) {
      printf("Could not open forward file %s.\n", outfn);
      return 1;
    }
    fread(outArr, szArr * sizeof(uint16_t), 1, fwdFN);
    fclose(fwdFN);
  }
  printf("Forward simulation loaded: %lf s (%zu orientations, stride=%zu)\n",
         omp_get_wtime() - wt_read, nrOrients, stride);
  fflush(stdout);

  // ── GPU memory allocation ────────────────────────────────────────
  // Upload forward simulation to GPU once (shared, read-only)
  size_t outArrBytes = szArr * sizeof(uint16_t);
  size_t imageBytes = g_imagePixels * sizeof(double);

  size_t freeMem = 0, totalMem = 0;
  gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));
  printf("GPU memory: %.1f / %.1f GB free\n", freeMem / 1e9, totalMem / 1e9);

  // Check if entire outArr fits on GPU
  uint16_t *d_outArr = NULL;
  size_t usableForOutArr = freeMem * 8 / 10; // use 80% of free
  int outArrOnGPU = 0;                       // flag: entire outArr on device?
  size_t chunkSize = 0; // orientations per chunk (if chunking needed)

  if (outArrBytes <= usableForOutArr) {
    // Entire forward simulation fits on GPU — ideal case
    gpuErrchk(cudaMalloc(&d_outArr, outArrBytes));
    gpuErrchk(
        cudaMemcpy(d_outArr, outArr, outArrBytes, cudaMemcpyHostToDevice));
    outArrOnGPU = 1;
    chunkSize = nrOrients;
    printf("Forward simulation uploaded to GPU: %.1f GB\n", outArrBytes / 1e9);
  } else {
    // Need chunking — cap at 4 GB per chunk
    size_t margin = freeMem / 5;
    size_t usable = (freeMem > imageBytes * MAX_STREAMS + margin)
                        ? freeMem - imageBytes * MAX_STREAMS - margin
                        : freeMem / 2;
    const size_t DEVICE_CAP = 4ULL * 1024 * 1024 * 1024;
    if (usable > DEVICE_CAP)
      usable = DEVICE_CAP;
    chunkSize = usable / (stride * sizeof(uint16_t));
    if (chunkSize < 1024)
      chunkSize = 1024;
    if (chunkSize > nrOrients)
      chunkSize = nrOrients;
    gpuErrchk(cudaMalloc(&d_outArr, chunkSize * stride * sizeof(uint16_t)));
    printf("Chunked mode: %zu orientations per chunk (%.1f GB), outArr too "
           "large for GPU (%.1f GB)\n",
           chunkSize, chunkSize * stride * sizeof(uint16_t) / 1e9,
           outArrBytes / 1e9);
  }

  // Determine number of streams that fit
  gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));
  size_t perStreamCost =
      imageBytes + chunkSize * sizeof(double); // d_image + d_matchChunk
  int numStreams = (int)(freeMem / (perStreamCost + (1 << 20))); // +1MB margin
  if (numStreams > MAX_STREAMS)
    numStreams = MAX_STREAMS;
  if (numStreams < 1)
    numStreams = 1;
  printf("Allocating %d CUDA streams (%.1f MB each)\n", numStreams,
         perStreamCost / 1e6);

  // Allocate per-stream contexts
  StreamContext *streams =
      (StreamContext *)calloc(numStreams, sizeof(StreamContext));
  for (int s = 0; s < numStreams; s++) {
    gpuErrchk(cudaStreamCreate(&streams[s].stream));
    gpuErrchk(cudaMalloc(&streams[s].d_image, imageBytes));
    gpuErrchk(cudaMalloc(&streams[s].d_matchChunk, chunkSize * sizeof(double)));
    streams[s].matchedArr = (double *)calloc(nrOrients, sizeof(double));
  }

  // ── Open output files ────────────────────────────────────────────
  // Create result directory
  mkdir(resultDir, 0755);
  char solFN[2048], spotFN[2048];
  snprintf(solFN, sizeof(solFN), "%s/solutions.txt", resultDir);
  snprintf(spotFN, sizeof(spotFN), "%s/spots.txt", resultDir);
  FILE *outF = fopen(solFN, "w");
  FILE *ExtraInfo = fopen(spotFN, "w");
  if (!outF || !ExtraInfo) {
    printf("Could not open output files in %s.\n", resultDir);
    return 1;
  }
  fprintf(ExtraInfo, "%%ImageNr\tGrainNr\tSpotNr\th\tk\tl\tX\tY\tQhat[0]\t"
                     "Qhat[1]\tQhat[2]\tIntensity\n");
  fprintf(outF,
          "%%ImageNr\tGrainNr\tNumberOfSolutions\tIntensity\tNMatches*"
          "Intensity\t"
          "NMatches*sqrt(Intensity)\tNMatches\tNSpotsCalc\t"
          "Recip1\tRecip2\tRecip3\tRecip4\tRecip5\tRecip6\tRecip7\tRecip8\t"
          "Recip9\t"
          "LatticeParameterFit[a]\tLatticeParameterFit[b]\t"
          "LatticeParameterFit[c]\t"
          "LatticeParameterFit[alpha]\tLatticeParameterFit[beta]\t"
          "LatticeParameterFit[gamma]\t"
          "OrientMatrix0\tOrientMatrix1\tOrientMatrix2\tOrientMatrix3\t"
          "OrientMatrix4\tOrientMatrix5\t"
          "OrientMatrix6\tOrientMatrix7\tOrientMatrix8\t"
          "CoarseNMatches*sqrt(Intensity)\t"
          "misOrientationPostRefinement[degrees]\torientationRowNr\n");
  fflush(outF);
  fflush(ExtraInfo);

  // ── Setup TCP server ─────────────────────────────────────────────
  queue_init(&process_queue);
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    perror("socket");
    return 1;
  }
  int opt_val = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = htons(PORT);
  if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) <
      0) {
    perror("bind");
    return 1;
  }
  if (listen(server_fd, MAX_CONNECTIONS) < 0) {
    perror("listen");
    return 1;
  }

  pthread_t accept_tid;
  pthread_create(&accept_tid, NULL, accept_connections, &server_fd);

  double total_init = omp_get_wtime() - st_tm;
  printf("\n=== LaueMatchingGPUStream ready ===\n"
         "  Port:          %d\n"
         "  Orientations:  %zu\n"
         "  Image size:    %d x %d (%.1f MB)\n"
         "  Streams:       %d\n"
         "  Chunking:      %s (%zu orientations/chunk)\n"
         "  Optimizer:     %s\n"
         "  Init time:     %.2f s\n"
         "  Waiting for images...\n\n",
         PORT, nrOrients, nrPxX, nrPxY, imageBytes / 1e6, numStreams,
         outArrOnGPU ? "OFF (full outArr on GPU)" : "ON", chunkSize,
         useBobyqa ? "BOBYQA" : "NelderMead", total_init);
  fflush(stdout);

  // Use maxNrSpots*3 for fitting (same as batch mode)
  int maxNrSpotsFit = maxNrSpots * 3;
  double tol = 3 * deg2rad;

  // ═══════════════════════════════════════════════════════════════════
  // Main processing loop
  // ═══════════════════════════════════════════════════════════════════
  int streamId = 0;
  while (keep_running) {
    ImageChunk chunk;
    int rc = queue_pop(&process_queue, &chunk);
    if (rc < 0)
      break; // shutdown

    uint16_t image_num = chunk.image_num;
    double *image = chunk.data;
    StreamContext *ctx = &streams[streamId];

    printf("[Image %u] Processing on stream %d...\n", image_num, streamId);
    double wt_img_start = omp_get_wtime();

    // Reset matchedArr
    memset(ctx->matchedArr, 0, nrOrients * sizeof(double));

    // H2D: upload image
    gpuErrchk(cudaMemcpyAsync(ctx->d_image, image, imageBytes,
                              cudaMemcpyHostToDevice, ctx->stream));

    // ── GPU matching (chunked or full) ──────────────────────────────
    size_t nChunks = (nrOrients + chunkSize - 1) / chunkSize;
    for (size_t c = 0; c < nChunks; c++) {
      size_t offset = c * chunkSize;
      size_t thisChunk = chunkSize;
      if (offset + thisChunk > nrOrients)
        thisChunk = nrOrients - offset;
      size_t thisMatchBytes = thisChunk * sizeof(double);

      if (!outArrOnGPU) {
        // Need to upload this chunk of outArr
        size_t thisOutBytes = thisChunk * stride * sizeof(uint16_t);
        gpuErrchk(cudaMemcpyAsync(d_outArr, outArr + offset * stride,
                                  thisOutBytes, cudaMemcpyHostToDevice,
                                  ctx->stream));
      }

      gpuErrchk(
          cudaMemsetAsync(ctx->d_matchChunk, 0, thisMatchBytes, ctx->stream));

      // Launch kernel
      int blocks = (int)((thisChunk + 1023) / 1024);
      uint16_t *d_chunkPtr =
          outArrOnGPU ? d_outArr + offset * stride : d_outArr;
      compare<<<blocks, 1024, 0, ctx->stream>>>(
          nrPxX, thisChunk, maxNrSpots, minIntensity, minNrSpots, d_chunkPtr,
          ctx->d_image, ctx->d_matchChunk);

      // D2H: match results for this chunk
      gpuErrchk(cudaMemcpyAsync(ctx->matchedArr + offset, ctx->d_matchChunk,
                                thisMatchBytes, cudaMemcpyDeviceToHost,
                                ctx->stream));
    }

    // Synchronize stream before CPU work
    gpuErrchk(cudaStreamSynchronize(ctx->stream));
    double wt_gpu = omp_get_wtime() - wt_img_start;

    // ── Count results ──────────────────────────────────────────────
    size_t nrResults = 0;
    for (size_t i = 0; i < nrOrients; i++) {
      if (ctx->matchedArr[i] > 0)
        nrResults++;
    }
    printf("[Image %u] GPU matching: %lf s, %zu matches\n", image_num, wt_gpu,
           nrResults);

    if (nrResults == 0) {
      printf("[Image %u] No matches found, skipping fitting.\n", image_num);
      free(image);
      streamId = (streamId + 1) % numStreams;
      continue;
    }

    // ── Unique solutions search ────────────────────────────────────
    double *mA = (double *)calloc(nrResults, sizeof(double));
    size_t *rowNrs = (size_t *)calloc(nrResults, sizeof(size_t));
    int resultNr = 0;
    for (size_t gi = 0; gi < nrOrients; gi++) {
      if (ctx->matchedArr[gi] != 0) {
        mA[resultNr] = ctx->matchedArr[gi];
        rowNrs[resultNr] = gi;
        resultNr++;
      }
    }

    int *doneArr = (int *)calloc(nrResults, sizeof(int));
    double *FinOrientArr = (double *)calloc(nrResults * 9, sizeof(double));
    int *dArr = (int *)calloc(nrResults, sizeof(int));
    int *bsArr = (int *)calloc(nrResults, sizeof(int));
    int iterNr = 0;
    double orient1[9], orient2[9], quat1[4], quat2[4], misoAngle, bestIntensity;
    int k, l, m;
    int bestSol;

    for (int gi = 0; gi < (int)nrResults; gi++) {
      if (doneArr[gi] != 0)
        continue;
      for (k = 0; k < 9; k++)
        orient1[k] = orients[rowNrs[gi] * 9 + k];
      OrientMat2Quat(orient1, quat1);
      doneArr[gi] = 1;
      bestSol = rowNrs[gi];
      bestIntensity = mA[gi];
      for (l = gi + 1; l < (int)nrResults; l++) {
        if (doneArr[l] > 0)
          continue;
        for (m = 0; m < 9; m++)
          orient2[m] = orients[rowNrs[l] * 9 + m];
        OrientMat2Quat(orient2, quat2);
        misoAngle = GetMisOrientation(quat1, quat2);
        if (misoAngle <= maxAngle) {
          doneArr[l] = 1;
          doneArr[gi]++;
          if (mA[l] > bestIntensity) {
            bestIntensity = mA[l];
            bestSol = rowNrs[l];
          }
        }
      }
      for (k = 0; k < 9; k++)
        FinOrientArr[iterNr * 9 + k] = orients[bestSol * 9 + k];
      dArr[iterNr] = doneArr[gi];
      bsArr[iterNr] = bestSol;
      iterNr++;
    }
    int totalSols = iterNr;
    printf("[Image %u] %d unique orientations found\n", image_num, totalSols);

    // ── Parallel fitting ───────────────────────────────────────────
#pragma omp parallel for num_threads(numProcs)
    for (iterNr = 0; iterNr < totalSols; iterNr++) {
      double orientBest[3][3], eulerBest[3], eulerFit[3], orientFit[3][3],
          q1[4], q2[4];
      int iJ, iK;
      double *outArrThisFit =
          (double *)calloc(3 * maxNrSpotsFit, sizeof(double));
      for (iJ = 0; iJ < 3; iJ++)
        for (iK = 0; iK < 3; iK++)
          orientBest[iJ][iK] = FinOrientArr[iterNr * 9 + 3 * iJ + iK];
      OrientMat2Euler(orientBest, eulerBest);
      for (iJ = 0; iJ < 3; iJ++)
        eulerFit[iJ] = eulerBest[iJ];
      int saveExtraInfo = 0;
      int doCrystalFit = 0;
      memset(outArrThisFit, 0, 3 * maxNrSpotsFit * sizeof(double));
      double latCFit[6], recipFit[3][3], mv = 0;

      // First fit: orientation only
      FitOrientation(image, eulerBest, hkls, nhkls, nrPxX, nrPxY, recip,
                     outArrThisFit, maxNrSpotsFit, rotTranspose, pArr, pxX, pxY,
                     Elo, Ehi, tol, LatticeParameter, eulerFit, latCFit, &mv,
                     doCrystalFit);

      // Second fit: tighter bounds + crystal params
      doCrystalFit = 1;
      for (iK = 0; iK < 3; iK++)
        eulerBest[iK] = eulerFit[iK];
      memset(outArrThisFit, 0, 3 * maxNrSpotsFit * sizeof(double));
      double tol2 = tol / 3.0;
      FitOrientation(image, eulerBest, hkls, nhkls, nrPxX, nrPxY, recip,
                     outArrThisFit, maxNrSpotsFit, rotTranspose, pArr, pxX, pxY,
                     Elo, Ehi, tol2, LatticeParameter, eulerFit, latCFit, &mv,
                     doCrystalFit);

      Euler2OrientMat(eulerFit, orientFit);
      OrientMat2Quat33(orientBest, q1);
      OrientMat2Quat33(orientFit, q2);
      int simulNrSps = 0;
      calcRecipArray(latCFit, sg_num, recipFit);
      memset(outArrThisFit, 0, 3 * maxNrSpotsFit * sizeof(double));
      int nrSps = writeCalcOverlap(
          image, eulerFit, hkls, nhkls, nrPxX, nrPxY, recipFit, outArrThisFit,
          maxNrSpotsFit, rotTranspose, pArr, pxX, pxY, Elo, Ehi, ExtraInfo,
          saveExtraInfo, &simulNrSps, (int)image_num);
      if (nrSps >= minGoodSpots) {
        int bs = bsArr[iterNr];
        double miso = GetMisOrientation(q1, q2);
        saveExtraInfo = iterNr + 1;
        memset(outArrThisFit, 0, 3 * maxNrSpotsFit * sizeof(double));
        writeCalcOverlap(image, eulerFit, hkls, nhkls, nrPxX, nrPxY, recipFit,
                         outArrThisFit, maxNrSpotsFit, rotTranspose, pArr, pxX,
                         pxY, Elo, Ehi, ExtraInfo, saveExtraInfo, &simulNrSps,
                         (int)image_num);
        double OF[3][3];
        MatrixMultF33(orientFit, recipFit, OF);
#pragma omp critical
        {
          fprintf(outF, "%u\t%d\t%d\t", image_num, iterNr + 1, dArr[iterNr]);
          fprintf(outF, "%-13.4lf\t", (mv / nrSps) * (mv / nrSps));
          fprintf(outF, "%-13.4lf\t", nrSps * (mv / nrSps) * (mv / nrSps));
          fprintf(outF, "%-13.4lf\t", mv);
          fprintf(outF, "%d\t", nrSps);
          fprintf(outF, "%d\t", simulNrSps);
          for (k = 0; k < 3; k++)
            for (l = 0; l < 3; l++)
              fprintf(outF, "%-13.7lf\t\t", OF[k][l]);
          for (k = 0; k < 6; k++)
            fprintf(outF, "%-13.7lf\t\t", latCFit[k]);
          for (k = 0; k < 3; k++)
            for (l = 0; l < 3; l++)
              fprintf(outF, "%-13.7lf\t\t", orientFit[k][l]);
          fprintf(outF, "%-13.4lf\t%-13.7lf\t%d\n", ctx->matchedArr[bs], miso,
                  bs);
        }
      }
      free(outArrThisFit);
    }
    fflush(outF);
    fflush(ExtraInfo);

    double wt_total = omp_get_wtime() - wt_img_start;
    printf("[Image %u] Total: %.3f s (GPU: %.3f s, CPU fitting: %.3f s)\n",
           image_num, wt_total, wt_gpu, wt_total - wt_gpu);
    fflush(stdout);

    // Cleanup per-image allocations
    free(mA);
    free(rowNrs);
    free(doneArr);
    free(FinOrientArr);
    free(dArr);
    free(bsArr);
    free(image);

    streamId = (streamId + 1) % numStreams;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Cleanup
  // ═══════════════════════════════════════════════════════════════════
  printf("\nShutting down...\n");

  // Close server socket to unblock accept()
  close(server_fd);
  pthread_join(accept_tid, NULL);
  queue_destroy(&process_queue);

  // Close output files
  fclose(outF);
  fclose(ExtraInfo);

  // Free GPU
  for (int s = 0; s < numStreams; s++) {
    cudaFree(streams[s].d_image);
    cudaFree(streams[s].d_matchChunk);
    cudaStreamDestroy(streams[s].stream);
    free(streams[s].matchedArr);
  }
  free(streams);
  cudaFree(d_outArr);

  // Free host
  if (outArrMapped)
    munmap(outArr, szArr * sizeof(uint16_t));
  else
    free(outArr);
  free(hkls);
  if (orientsMapped)
    munmap(orients, szFile);
  else
    free(orients);

  printf("LaueMatchingGPUStream exited cleanly.\n");
  fflush(stdout);
  return 0;
}
