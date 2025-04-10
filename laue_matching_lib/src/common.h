/**
 * @file common.h
 * @brief Common definitions and utilities for internal use
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-09
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

 #ifndef LAUE_COMMON_H
 #define LAUE_COMMON_H
 
 /* Standard C headers */
 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <stddef.h>
 #include <string.h>
 #include <math.h>
 #include <stdbool.h>
 #include <time.h>
 
 /* System headers */
 #include <omp.h>
 #include <sys/mman.h>
 #include <fcntl.h>
 #include <nlopt.h>
 #include <unistd.h>  /* For read/write functions */
 
 #ifdef __linux__ 
 #include <malloc.h>
 #endif
 
 /* Project headers - for library build only */
#include "include/laue_matching.h"
 
 /* Constants */
 #define LAUE_DEG2RAD 0.0174532925199433
 #define LAUE_RAD2DEG 57.2957795130823
 #define LAUE_HC_KEVNM 1.2398419739
 #define LAUE_EPSILON 1E-12
 #define LAUE_MAX_HKLS 200000
 
 /* Macros */
 #define LAUE_CALC_LENGTH(x, y, z) sqrt((x)*(x) + (y)*(y) + (z)*(z))
 #define LAUE_CHECK_ALLOC(ptr, msg) do { if ((ptr) == NULL) { fprintf(stderr, "ERROR: Memory allocation failed: %s\n", (msg)); return -1; } } while(0)
 #define LAUE_CHECK_FILE(fp, filename) do { if ((fp) == NULL) { fprintf(stderr, "ERROR: Could not open file %s\n", (filename)); return -1; } } while(0)
 #define LAUE_MIN(a, b) (((a) < (b)) ? (a) : (b))
 #define LAUE_MAX(a, b) (((a) > (b)) ? (a) : (b))
 
 /**
  * @brief Lattice parameters structure
  * 
  * Contains the six parameters defining a crystal lattice:
  * a, b, c - lattice constants in nm
  * alpha, beta, gamma - lattice angles in degrees
  */
//  typedef struct {
//      double a, b, c;          /**< Lattice constants in nm */
//      double alpha, beta, gamma; /**< Lattice angles in degrees */
//  } LatticeParameters;
 
 /**
  * @brief Detector parameters structure
  * 
  * Contains parameters defining detector geometry and characteristics
  */
//  typedef struct {
//      double position[3];      /**< Detector center position [x,y,z] */
//      double rotation[3];      /**< Detector rotation angles [rx,ry,rz] */
//      double pixelSize[2];     /**< Pixel size [x,y] in mm */
//      int numPixels[2];        /**< Number of pixels [Nx,Ny] */
//      double energyRange[2];   /**< Energy range [min,max] in keV */
//  } DetectorParameters;
 
 /**
  * @brief Configuration parameters for matching
  */
//  typedef struct {
//      LatticeParameters lattice;   /**< Lattice parameters */
//      double latticeParamTol[6];   /**< Tolerance for each lattice parameter (%) */
//      double cOverATol;            /**< Tolerance for c/a ratio (%) */
//      int spaceGroup;              /**< Space group number */
//      DetectorParameters detectorParams; /**< Detector parameters */
//      int maxNumSpots;             /**< Maximum number of spots to simulate */
//      int minNumSpots;             /**< Minimum number of spots for a valid match */
//      double minIntensity;         /**< Minimum total intensity for a valid match */
//      double maxAngle;             /**< Maximum misorientation angle for merging (degrees) */
//      bool performForwardSimulation; /**< Whether to perform forward simulation */
//      char forwardSimulationFile[256]; /**< File name for forward simulation results */
//      int numThreads;              /**< Number of CPU cores to use */
//  } MatchingConfig;
 
 /**
  * @brief Results from matching
  */
//  typedef struct {
//      int numGrains;               /**< Number of grains found */
//      double *orientations;        /**< Array of orientation matrices [numGrains][9] */
//      double *eulerAngles;         /**< Array of Euler angles [numGrains][3] */
//      LatticeParameters *lattices; /**< Array of refined lattice parameters */
//      int *numSpots;               /**< Number of spots matched per grain */
//      double *intensities;         /**< Total intensity per grain */
//      int *numSolutions;           /**< Number of solutions merged into each grain */
//  } MatchingResults;
 
 /**
  * @brief Internal structure for optimization data
  */
 typedef struct {
     const double *image;
     const int *hkls;
     int nhkls;
     int nrPxX;
     int nrPxY;
     double latticeParamsOrig[6];
     double recipMatrix[3][3];
     double *outArray;
     int maxNrSpots;
     double rotTranspose[3][3];
     double detPos[3];
     double pixelSizeX;
     double pixelSizeY;
     double eMin;
     double eMax;
 } LaueOptimizationData;
 
 /* Error codes */
 enum LaueErrorCodes {
     LAUE_SUCCESS = 0,
     LAUE_ERROR_MEMORY_ALLOCATION = -1,
     LAUE_ERROR_FILE_IO = -2,
     LAUE_ERROR_FILE_NOT_FOUND = -2,
     LAUE_ERROR_INVALID_PARAMETER = -3,
     LAUE_ERROR_OPTIMIZATION_FAILED = -4,
     LAUE_ERROR_NO_GPU = -5,
     LAUE_ERROR_FORWARD_SIMULATION = -6,
     LAUE_ERROR_GPU = -7
  };
 
 /* Function prototypes for main API */
 int laue_init(void);
 int laue_gpu_init(void);
//  MatchingConfig laue_create_default_config(void);
//  int laue_perform_matching(
//      const MatchingConfig *config,
//      const char *orientationFile,
//      const char *hklFile,
//      const char *imageFile,
//      MatchingResults *results
//  );
//  void laue_free_results(MatchingResults *results);
 void laue_cleanup(void);
 
 /* Status and logging */
 extern int laue_verbose_level;
 void laue_set_verbose(int level);
 void laue_log(int level, const char *format, ...);
 
 #endif /* LAUE_COMMON_H */