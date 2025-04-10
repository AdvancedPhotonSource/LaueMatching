/**
 * @file diffraction.h
 * @brief Diffraction simulation utilities for Laue pattern matching
 * 
 * Provides functions for simulating Laue diffraction patterns, including
 * spot position calculations, intensity calculations, and pattern matching.
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-09
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

#ifndef LAUE_DIFFRACTION_H
#define LAUE_DIFFRACTION_H

#include "../common.h"
#include "geometry.h"
#include "crystallography.h"

/**
 * @brief Calculate spot pattern for a given orientation
 * 
 * @param image Diffraction image data
 * @param euler Euler angles (radians)
 * @param hkls Array of hkl indices
 * @param nhkls Number of hkl indices
 * @param nrPxX Number of pixels in X direction
 * @param nrPxY Number of pixels in Y direction
 * @param recip Reciprocal lattice matrix
 * @param outArray Output array for calculated qhats
 * @param maxNrSpots Maximum number of spots to calculate
 * @param rotTranspose Rotation matrix transpose
 * @param detPos Detector position
 * @param pxX Pixel size in X direction
 * @param pxY Pixel size in Y direction
 * @param eMin Minimum energy
 * @param eMax Maximum energy
 * @return Score value representing match quality
 */
double diffraction_calculate_pattern(
    const double *image,
    const double euler[3],
    const int *hkls,
    int nhkls,
    int nrPxX,
    int nrPxY,
    const double recip[3][3],
    double *outArray,
    int maxNrSpots,
    const double rotTranspose[3][3],
    const double detPos[3],
    double pxX,
    double pxY,
    double eMin,
    double eMax
);

/**
 * @brief Write calculated pattern to file
 * 
 * @param image Diffraction image data
 * @param euler Euler angles (radians)
 * @param hkls Array of hkl indices
 * @param nhkls Number of hkl indices
 * @param nrPxX Number of pixels in X direction
 * @param nrPxY Number of pixels in Y direction
 * @param recip Reciprocal lattice matrix
 * @param outArray Output array for calculated qhats
 * @param maxNrSpots Maximum number of spots to calculate
 * @param rotTranspose Rotation matrix transpose
 * @param detPos Detector position
 * @param pxX Pixel size in X direction
 * @param pxY Pixel size in Y direction
 * @param eMin Minimum energy
 * @param eMax Maximum energy
 * @param outFile File to write results to
 * @param grainId Grain ID for output
 * @param simulSpotCount Pointer to store the number of simulated spots
 * @return Number of matched spots
 */
int diffraction_write_pattern(
    const double *image,
    const double euler[3],
    const int *hkls,
    int nhkls,
    int nrPxX,
    int nrPxY,
    const double recip[3][3],
    double *outArray,
    int maxNrSpots,
    const double rotTranspose[3][3],
    const double detPos[3],
    double pxX,
    double pxY,
    double eMin,
    double eMax,
    FILE *outFile,
    int grainId,
    int *simulSpotCount
);

/**
 * @brief Initialize diffraction module with precomputed data
 * 
 * @param config Configuration parameters
 * @return 0 on success, non-zero on failure
 */
int diffraction_init(const MatchingConfig *config);

/**
 * @brief Clean up diffraction module resources
 */
void diffraction_cleanup(void);

#endif /* LAUE_DIFFRACTION_H */