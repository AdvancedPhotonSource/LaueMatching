/**
 * @file laue_matching_gpu.h
 * @brief GPU-accelerated implementation of the Laue pattern matching API
 * 
 * This file provides GPU-accelerated functionality for matching Laue diffraction patterns
 * to determine grain orientations in polycrystalline materials.
 * 
 * @author Hemant Sharma
 * @date 2025-04-10
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

 #ifndef LAUE_MATCHING_GPU_H
 #define LAUE_MATCHING_GPU_H
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 #include "laue_matching.h"
 
 /**
  * @brief Initialize GPU resources for Laue pattern matching
  * 
  * This function initializes CUDA and selects the first available GPU device.
  * It should be called before any other GPU-related functions.
  * 
  * @return LAUE_SUCCESS on success, LAUE_ERROR_GPU on failure
  */
 int laue_init_gpu(void);
 
 /**
  * @brief Perform Laue pattern matching with GPU acceleration
  * 
  * This function performs Laue pattern matching using GPU acceleration where possible.
  * If forward simulation is required, it may fall back to CPU implementation.
  * 
  * @param config Configuration parameters
  * @param orientationFile Path to binary file with candidate orientations
  * @param hklFile Path to text file with hkl indices
  * @param imageFile Path to binary file with diffraction image
  * @param results Pointer to results structure that will be populated
  * @return LAUE_SUCCESS on success, error code on failure
  */
 int laue_perform_matching_gpu(
     const MatchingConfig *config,
     const char *orientationFile,
     const char *hklFile,
     const char *imageFile,
     MatchingResults *results
 );
 
 /**
  * @brief Set verbosity level for log messages
  * 
  * Controls the amount of information printed during processing:
  * 0 - Only errors
  * 1 - Errors and important information (default)
  * 2 - Verbose output including timing information
  * 3 - Debug level output
  * 
  * @param level Verbosity level (0-3)
  */
 void laue_set_verbose(int level);
 
 /**
  * @brief Log message with specified verbosity level
  * 
  * Prints a formatted message if the current verbosity level is
  * greater than or equal to the specified level.
  * 
  * @param level Minimum verbosity level for this message
  * @param format Printf-style format string
  * @param ... Variable arguments for format string
  */
 void laue_log(int level, const char *format, ...);
 
 /**
  * @brief GPU-accelerated version of the main matching function
  * 
  * This is a more complete implementation that includes GPU optimization
  * with fallback to CPU when needed.
  * 
  * @param config Configuration parameters
  * @param orientationFile Path to binary file with candidate orientations
  * @param hklFile Path to text file with hkl indices
  * @param imageFile Path to binary file with diffraction image
  * @param results Pointer to results structure that will be populated
  * @return LAUE_SUCCESS on success, error code on failure
  */
 int laue_gpu_perform_matching(
     const MatchingConfig *config,
     const char *orientationFile,
     const char *hklFile,
     const char *imageFile,
     MatchingResults *results
 );
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* LAUE_MATCHING_GPU_H */