/**
 * @file main_gpu.c
 * @brief GPU-accelerated main application for Laue pattern matching
 * 
 * This provides a GPU-accelerated version of the original LaueMatchingCPU program.
 * It uses CUDA to offload the matching part to the GPU, which is particularly
 * efficient when forward simulation is not needed.
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-10
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <omp.h>
 #include "src/common.h"
 #include "src/core/geometry.h"
 #include "src/core/crystallography.h"
 #include "src/core/diffraction.h"
 #include "src/core/optimization.h"
 #include "src/io/file_io.h"
 
 // Add CUDA-specific error code
 #define LAUE_ERROR_GPU -100
 
 // External function declaration for the GPU implementation
 extern int laue_perform_matching_gpu(
     const MatchingConfig *config,
     const char *orientationFile,
     const char *hklFile,
     const char *imageFile,
     MatchingResults *results
 );
 
 /**
  * @brief Print usage information
  */
 static void print_usage(void) {
     puts("LaueMatching on the GPU\n"
          "Contact hsharma@anl.gov\n"
          "Arguments: \n"
          "* \tparameterFile (text)\n"
          "* \tbinary files for candidate orientation list [double]\n"
          "* \ttext of valid_hkls (preferably sorted on f2), space separated\n"
          "* \tbinary file for image [double]\n"
          "* \tnumber of CPU cores to use for non-GPU parts: \n"
          "* NOTE: This GPU version requires a pre-computed forward simulation file.\n"
          "* \tuse the CPU version first with DoFwd=1 to generate this file.\n\n"
          "Parameter file with the following parameters: \n"
          "\t\t* LatticeParameter (in nm and degrees),\n"
          "\t\t* tol_latC (in %, 6 values),\n"
          "\t\t* tol_c_over_a (in %, 1 value), it will only change c, keep a constant,\n"
          "\t\t* SpaceGroup,\n"
          "\t\t* P_Array (3 array describing positioning on the detector),\n"
          "\t\t* R_Array (3 array describing tilts of the detector),\n"
          "\t\t* PxX,\n"
          "\t\t* PxY (PixelSize(x,y)),\n"
          "\t\t* NrPxX,\n"
          "\t\t* NrPxY (NrPixels(x,y)),\n"
          "\t\t* Elo (minimum energy for simulating diffraction spots),\n"
          "\t\t* Ehi (maximum energy for simulating diffraction spots),\n"
          "\t\t* MaxNrLaueSpots(maximum number of spots to simulate),\n"
          "\t\t* ForwardFile (file name to save forward simulation result),\n"
          "\t\t* DoFwd (must be 0 for GPU version),\n"
          "\t\t* MinNrSpots (minimum number of spots that qualify a grain, must\n"
          "\t\t\t\t  be smaller than MaxNrLaueSpots),\n"
          "\t\t* MinIntensity (minimum total intensity from the MinNrSpots that\n"
          "\t\t\t\twill qualify a match, usually 100 counts),\n"
          "\t\t* MaxAngle (maximum angle in degrees that defines a grain, \n"
          "\t\t\t\tif misorientation between two candidates is smaller \n"
          "\t\t\t\tthan this, the solutions will be merged).\n");
 }
 
 /**
  * @brief Main function
  */
 int main(int argc, char *argv[]) {
     if (argc != 6) {
         print_usage();
         return 0;
     }
     
     char *paramFile = argv[1];
     char *orientFile = argv[2];
     char *hklFile = argv[3];
     char *imageFile = argv[4];
     int numThreads = atoi(argv[5]);
     
     // Read configuration from parameter file
     MatchingConfig config;
     int ret = file_read_parameters(paramFile, &config);
     if (ret != LAUE_SUCCESS) {
         fprintf(stderr, "ERROR: Failed to read parameter file\n");
         return 1;
     }
     
     // Override number of threads with command line argument
     config.numThreads = numThreads;
     
     // Verify that forward simulation is disabled for the GPU version
     if (config.performForwardSimulation == 1) {
         fprintf(stderr, "WARNING: The GPU version does not support forward simulation.\n");
         fprintf(stderr, "Please set DoFwd=0 in the parameter file and ensure a forward simulation file exists.\n");
         fprintf(stderr, "Run the CPU version first with DoFwd=1 if needed.\n");
         return 1;
     }
     
     // Initialize the library
     ret = laue_init();
     if (ret != LAUE_SUCCESS) {
         fprintf(stderr, "ERROR: Failed to initialize library\n");
         return 1;
     }
     
     // Perform matching using the GPU implementation
     MatchingResults results;
     ret = laue_perform_matching_gpu(&config, orientFile, hklFile, imageFile, &results);
     if (ret != LAUE_SUCCESS) {
         if (ret == LAUE_ERROR_GPU) {
             fprintf(stderr, "ERROR: GPU operation failed\n");
         } else {
             fprintf(stderr, "ERROR: Failed to perform matching (error code: %d)\n", ret);
         }
         laue_cleanup();
         return 1;
     }
     
     // Print summary
     printf("Found %d grains\n", results.numGrains);
     
     // Clean up
     laue_free_results(&results);
     laue_cleanup();
     
     return 0;
 }