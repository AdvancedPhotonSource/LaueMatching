/**
 * @file laue_matching_gpu.h
 * @brief Header for GPU-accelerated Laue pattern matching
 */

#ifndef LAUE_MATCHING_GPU_H
#define LAUE_MATCHING_GPU_H

#include "common.h"

/**
 * @brief Initialize GPU resources for Laue pattern matching
 * @return LAUE_SUCCESS on success, error code on failure
 */
int laue_gpu_init(void);

/**
 * @brief Clean up GPU resources
 */
void laue_gpu_cleanup(void);

/**
 * @brief Perform GPU-accelerated pattern matching
 * 
 * @param config Configuration parameters
 * @param orientationFile Path to file containing candidate orientations
 * @param hklFile Path to file containing valid HKL indices
 * @param imageFile Path to image file to match against
 * @param results Output structure to store matching results
 * @return LAUE_SUCCESS on success, error code on failure
 */
int laue_gpu_perform_matching(
    const MatchingConfig *config,
    const char *orientationFile,
    const char *hklFile,
    const char *imageFile,
    MatchingResults *results
);

#endif /* LAUE_MATCHING_GPU_H */