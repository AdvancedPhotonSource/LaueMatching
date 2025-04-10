/**
 * @file file_io.h
 * @brief File input/output utilities for Laue pattern matching
 * 
 * Provides functions for reading and writing various file formats
 * used in Laue pattern matching.
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-09
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

#ifndef LAUE_FILE_IO_H
#define LAUE_FILE_IO_H

#include "../common.h"

/**
 * @brief Read parameter file
 * 
 * @param filename Parameter file path
 * @param config Configuration structure to populate
 * @return 0 on success, non-zero on failure
 */
int file_read_parameters(const char *filename, MatchingConfig *config);

/**
 * @brief Read orientation file
 * 
 * @param filename Orientation file path
 * @param orientations Pointer to array that will be allocated and populated
 * @param numOrientations Pointer to store number of orientations
 * @param numThreads Number of threads to distribute orientations
 * @return 0 on success, non-zero on failure
 */
int file_read_orientations(
    const char *filename,
    double **orientations,
    size_t *numOrientations,
    int numThreads
);

/**
 * @brief Read HKL file
 * 
 * @param filename HKL file path
 * @param hkls Pointer to array that will be allocated and populated
 * @param numHkls Pointer to store number of HKL indices
 * @return 0 on success, non-zero on failure
 */
int file_read_hkls(const char *filename, int **hkls, int *numHkls);

/**
 * @brief Read image file
 * 
 * @param filename Image file path
 * @param image Pointer to array that will be allocated and populated
 * @param nrPxX Number of pixels in X direction
 * @param nrPxY Number of pixels in Y direction
 * @param nonZeroPixels Pointer to store number of non-zero pixels
 * @return 0 on success, non-zero on failure
 */
int file_read_image(
    const char *filename,
    double **image,
    int nrPxX,
    int nrPxY,
    int *nonZeroPixels
);

/**
 * @brief Write forward simulation file
 * 
 * @param filename Output file path
 * @param outArray Output array to write
 * @param arraySize Size of output array
 * @param offset Offset in file to write
 * @return 0 on success, non-zero on failure
 */
int file_write_forward_simulation(
    const char *filename,
    const uint16_t *outArray,
    size_t arraySize,
    size_t offset
);

/**
 * @brief Read forward simulation file
 * 
 * @param filename Input file path
 * @param outArray Output array to populate
 * @param arraySize Size of output array
 * @param offset Offset in file to read from
 * @return 0 on success, non-zero on failure
 */
int file_read_forward_simulation(
    const char *filename,
    uint16_t *outArray,
    size_t arraySize,
    size_t offset
);

/**
 * @brief Open spot output file
 * 
 * @param inputImageFilename Image file path (used to generate output filename)
 * @param outFile Pointer to file handle that will be set
 * @return 0 on success, non-zero on failure
 */
int file_open_spot_output(const char *inputImageFilename, FILE **outFile);

/**
 * @brief Open solution output file
 * 
 * @param inputImageFilename Image file path (used to generate output filename)
 * @param outFile Pointer to file handle that will be set
 * @return 0 on success, non-zero on failure
 */
int file_open_solution_output(const char *inputImageFilename, FILE **outFile);

/**
 * @brief Write grain solution to file
 * 
 * @param file Output file handle
 * @param grainId Grain ID
 * @param numSolutions Number of merged solutions
 * @param intensity Total intensity
 * @param numMatches Number of matched spots
 * @param numSimulated Number of simulated spots
 * @param orientMatrix Orientation matrix
 * @param recipMatrix Reciprocal lattice matrix
 * @param latticeParams Lattice parameters
 * @param coarseMatchScore Coarse matching score
 * @param misorientation Misorientation angle
 * @param bestOrientation Row number of best orientation
 * @return 0 on success, non-zero on failure
 */
int file_write_grain_solution(
    FILE *file,
    int grainId,
    int numSolutions,
    double intensity,
    int numMatches,
    int numSimulated,
    const double orientMatrix[3][3],
    const double recipMatrix[3][3],
    const double latticeParams[6],
    double coarseMatchScore,
    double misorientation,
    int bestOrientation
);

#endif /* LAUE_FILE_IO_H */