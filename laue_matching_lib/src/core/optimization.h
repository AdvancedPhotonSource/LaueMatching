/**
 * @file optimization.h
 * @brief Optimization utilities for Laue pattern matching
 * 
 * Provides functions for refining orientation and lattice parameters
 * to optimize the match between simulated and experimental patterns.
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-09
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

#ifndef LAUE_OPTIMIZATION_H
#define LAUE_OPTIMIZATION_H

#include "../common.h"

/**
 * @brief Fit orientation to optimize pattern matching
 * 
 * @param image Diffraction image data
 * @param initialEuler Initial Euler angles (radians)
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
 * @param tolerance Tolerance for optimization (radians)
 * @param latticeParams Original lattice parameters
 * @param optimizedEuler Output array for optimized Euler angles
 * @param optimizedLattice Output array for optimized lattice parameters (if doCrystalFit is true)
 * @param matchScore Pointer to store the match score
 * @param doCrystalFit Whether to optimize lattice parameters
 * @return 0 on success, non-zero on failure
 */
int optimization_fit_orientation(
    const double *image,
    const double initialEuler[3],
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
    double tolerance,
    const double latticeParams[6],
    double optimizedEuler[3],
    double optimizedLattice[6],
    double *matchScore,
    int doCrystalFit
);

/**
 * @brief Set tolerance parameters for lattice optimization
 * 
 * @param latticeTolerances Array of 6 tolerance values for lattice parameters (in %)
 * @param cOverATolerance Tolerance for c/a ratio (in %)
 */
void optimization_set_tolerances(
    const double latticeTolerances[6],
    double cOverATolerance
);

/**
 * @brief Objective function for optimization
 * 
 * @param n Number of parameters
 * @param x Parameter values
 * @param grad Gradient (not used, can be NULL)
 * @param data Optimization data
 * @return Objective function value (negative of match score)
 */
double optimization_objective_function(
    unsigned n,
    const double *x,
    double *grad,
    void *data
);

#endif /* LAUE_OPTIMIZATION_H */