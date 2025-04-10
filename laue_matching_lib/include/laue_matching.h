/**
 * @file laue_matching.h
 * @brief Main public API for Laue pattern matching library
 * 
 * This library provides functionality for matching Laue diffraction patterns
 * to determine grain orientations in polycrystalline materials.
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-09
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

#ifndef LAUE_MATCHING_H
#define LAUE_MATCHING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/** 
 * @brief Lattice parameters structure
 * 
 * Contains the six parameters defining a crystal lattice:
 * a, b, c - lattice constants in nm
 * alpha, beta, gamma - lattice angles in degrees
 */
typedef struct {
    double a, b, c;          /**< Lattice constants in nm */
    double alpha, beta, gamma; /**< Lattice angles in degrees */
} LatticeParameters;

/**
 * @brief Detector parameters structure
 * 
 * Contains parameters defining detector geometry and characteristics
 */
typedef struct {
    double position[3];      /**< Detector center position [x,y,z] */
    double rotation[3];      /**< Detector rotation angles [rx,ry,rz] */
    double pixelSize[2];     /**< Pixel size [x,y] in mm */
    int numPixels[2];        /**< Number of pixels [Nx,Ny] */
    double energyRange[2];   /**< Energy range [min,max] in keV */
} DetectorParameters;

/**
 * @brief Configuration parameters for matching
 */
typedef struct {
    LatticeParameters lattice;   /**< Lattice parameters */
    DetectorParameters detectorParams; /**< Detector parameters */
    double latticeParamTol[6];   /**< Tolerance for each lattice parameter (%) */
    double cOverATol;            /**< Tolerance for c/a ratio (%) */
    int spaceGroup;              /**< Space group number */
    int maxNumSpots;             /**< Maximum number of spots to simulate */
    int minNumSpots;             /**< Minimum number of spots for a valid match */
    double minIntensity;         /**< Minimum total intensity for a valid match */
    double maxAngle;             /**< Maximum misorientation angle for merging (degrees) */
    bool performForwardSimulation; /**< Whether to perform forward simulation */
    char forwardSimulationFile[256]; /**< File name for forward simulation results */
    int numThreads;              /**< Number of CPU threads to use */
} MatchingConfig;

/**
 * @brief Results from matching
 */
typedef struct {
    int numGrains;               /**< Number of grains found */
    double *orientations;        /**< Array of orientation matrices [numGrains][9] */
    double *eulerAngles;         /**< Array of Euler angles [numGrains][3] */
    LatticeParameters *lattices; /**< Array of refined lattice parameters */
    int *numSpots;               /**< Number of spots matched per grain */
    double *intensities;         /**< Total intensity per grain */
    int *numSolutions;           /**< Number of solutions merged into each grain */
} MatchingResults;

/**
 * @brief Initialize the library
 * 
 * @return 0 on success, non-zero on failure
 */
int laue_init(void);

/**
 * @brief Create default configuration
 * 
 * @return Default configuration structure
 */
MatchingConfig laue_create_default_config(void);

/**
 * @brief Perform Laue pattern matching
 * 
 * @param config Configuration parameters
 * @param orientationFile Path to binary file with candidate orientations
 * @param hklFile Path to text file with hkl indices
 * @param imageFile Path to binary file with diffraction image
 * @param results Pointer to results structure that will be populated
 * @return 0 on success, non-zero on failure
 */
int laue_perform_matching(
    const MatchingConfig *config,
    const char *orientationFile,
    const char *hklFile,
    const char *imageFile,
    MatchingResults *results
);

/**
 * @brief Free resources associated with results
 * 
 * @param results Results structure to free
 */
void laue_free_results(MatchingResults *results);

/**
 * @brief Clean up library resources
 */
void laue_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* LAUE_MATCHING_H */