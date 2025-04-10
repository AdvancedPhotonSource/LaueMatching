/**
 * @file crystallography.h
 * @brief Crystallography utilities for Laue pattern matching
 * 
 * Provides functions for handling crystal structures, symmetry operations,
 * lattice calculations, and related crystallographic operations.
 * 
 * @author Hemant Sharma (original code)
 * @date 2025-04-09
 * @copyright Copyright (c) 2025, UChicago Argonne, LLC
 */

#ifndef LAUE_CRYSTALLOGRAPHY_H
#define LAUE_CRYSTALLOGRAPHY_H

#include "../common.h"

/**
 * @brief Initialize crystallography module
 * 
 * @param spaceGroup Space group number
 * @return 0 on success, non-zero on failure
 */
int crystal_init(int spaceGroup);

/**
 * @brief Calculate cell volume and other parameters from lattice constants
 * 
 * @param latticeParams Array of 6 lattice parameters (a,b,c,alpha,beta,gamma)
 * @param cellVolume Pointer to store calculated cell volume
 * @param phiVolume Pointer to store calculated phi volume
 */
void crystal_calculate_volume(const double latticeParams[6], double *cellVolume, double *phiVolume);

/**
 * @brief Calculate reciprocal lattice matrix
 * 
 * @param latticeParams Array of 6 lattice parameters (a,b,c,alpha,beta,gamma)
 * @param spaceGroup Space group number
 * @param recipMatrix 3x3 matrix to store the reciprocal lattice
 */
void crystal_calculate_reciprocal_matrix(
    const double latticeParams[6], 
    int spaceGroup, 
    double recipMatrix[3][3]
);

/**
 * @brief Get symmetry operators for a space group
 * 
 * @param spaceGroupNumber Space group number
 * @param symmetryOperators Array to store symmetry operators (quaternions)
 * @return Number of symmetry operators
 */
int crystal_get_symmetry_operators(int spaceGroupNumber, double symmetryOperators[24][4]);

/**
 * @brief Clean up crystallography module resources
 */
void crystal_cleanup(void);

/* Global variables for symmetry operations */
extern int crystal_symmetry_count;
extern double crystal_symmetry_operators[24][4];

/* Define crystal system enumerations */
enum CrystalSystem {
    CRYSTAL_SYSTEM_TRICLINIC = 0,
    CRYSTAL_SYSTEM_MONOCLINIC,
    CRYSTAL_SYSTEM_ORTHORHOMBIC,
    CRYSTAL_SYSTEM_TETRAGONAL,
    CRYSTAL_SYSTEM_TRIGONAL,
    CRYSTAL_SYSTEM_HEXAGONAL,
    CRYSTAL_SYSTEM_CUBIC
};

/**
 * @brief Get crystal system from space group number
 * 
 * @param spaceGroupNumber Space group number
 * @return Corresponding crystal system
 */
enum CrystalSystem crystal_get_system(int spaceGroupNumber);

#endif /* LAUE_CRYSTALLOGRAPHY_H */