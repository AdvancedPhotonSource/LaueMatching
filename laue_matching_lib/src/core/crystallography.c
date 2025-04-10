/**
 * @file crystallography.c
 * @brief Implementation of crystallography utilities
 */

#include "../common.h"
#include "crystallography.h"
#include "geometry.h"

/* Global variables for symmetry operations */
int crystal_symmetry_count = 0;
double crystal_symmetry_operators[24][4];

/* Symmetry operator definitions for each crystal system */
static double CRYSTAL_TRICLINIC_SYM[1][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000}
};

static double CRYSTAL_MONOCLINIC_SYM[2][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},
    {0.00000, 1.00000, 0.00000, 0.00000}
};

static double CRYSTAL_ORTHORHOMBIC_SYM[4][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},
    {1.00000, 1.00000, 0.00000, 0.00000},
    {0.00000, 0.00000, 1.00000, 0.00000},
    {0.00000, 0.00000, 0.00000, 1.00000}
};

static double CRYSTAL_TETRAGONAL_SYM[8][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},
    {0.70711, 0.00000, 0.00000, 0.70711},
    {0.00000, 0.00000, 0.00000, 1.00000},
    {0.70711, -0.00000, -0.00000, -0.70711},
    {0.00000, 1.00000, 0.00000, 0.00000},
    {0.00000, 0.00000, 1.00000, 0.00000},
    {0.00000, 0.70711, 0.70711, 0.00000},
    {0.00000, -0.70711, 0.70711, 0.00000}
};

static double CRYSTAL_TRIGONAL_SYM[6][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},
    {0.50000, 0.00000, 0.00000, 0.86603},
    {0.50000, -0.00000, -0.00000, -0.86603},
    {0.00000, 0.50000, -0.86603, 0.00000},
    {0.00000, 1.00000, 0.00000, 0.00000},
    {0.00000, 0.50000, 0.86603, 0.00000}
};

static double CRYSTAL_HEXAGONAL_SYM[12][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},
    {0.86603, 0.00000, 0.00000, 0.50000},
    {0.50000, 0.00000, 0.00000, 0.86603},
    {0.00000, 0.00000, 0.00000, 1.00000},
    {0.50000, -0.00000, -0.00000, -0.86603},
    {0.86603, -0.00000, -0.00000, -0.50000},
    {0.00000, 1.00000, 0.00000, 0.00000},
    {0.00000, 0.86603, 0.50000, 0.00000},
    {0.00000, 0.50000, 0.86603, 0.00000},
    {0.00000, 0.00000, 1.00000, 0.00000},
    {0.00000, -0.50000, 0.86603, 0.00000},
    {0.00000, -0.86603, 0.50000, 0.00000}
};

static double CRYSTAL_CUBIC_SYM[24][4] = {
    {1.00000, 0.00000, 0.00000, 0.00000},
    {0.70711, 0.70711, 0.00000, 0.00000},
    {0.00000, 1.00000, 0.00000, 0.00000},
    {0.70711, -0.70711, 0.00000, 0.00000},
    {0.70711, 0.00000, 0.70711, 0.00000},
    {0.00000, 0.00000, 1.00000, 0.00000},
    {0.70711, 0.00000, -0.70711, 0.00000},
    {0.70711, 0.00000, 0.00000, 0.70711},
    {0.00000, 0.00000, 0.00000, 1.00000},
    {0.70711, 0.00000, 0.00000, -0.70711},
    {0.50000, 0.50000, 0.50000, 0.50000},
    {0.50000, -0.50000, -0.50000, -0.50000},
    {0.50000, -0.50000, 0.50000, 0.50000},
    {0.50000, 0.50000, -0.50000, -0.50000},
    {0.50000, 0.50000, -0.50000, 0.50000},
    {0.50000, -0.50000, 0.50000, -0.50000},
    {0.50000, -0.50000, -0.50000, 0.50000},
    {0.50000, 0.50000, 0.50000, -0.50000},
    {0.00000, 0.70711, 0.70711, 0.00000},
    {0.00000, -0.70711, 0.70711, 0.00000},
    {0.00000, 0.70711, 0.00000, 0.70711},
    {0.00000, 0.70711, 0.00000, -0.70711},
    {0.00000, 0.00000, 0.70711, 0.70711},
    {0.00000, 0.00000, 0.70711, -0.70711}
};

enum CrystalSystem crystal_get_system(int spaceGroupNumber) {
    if (spaceGroupNumber <= 2) {
        return CRYSTAL_SYSTEM_TRICLINIC;
    } else if (spaceGroupNumber <= 15) {
        return CRYSTAL_SYSTEM_MONOCLINIC;
    } else if (spaceGroupNumber <= 74) {
        return CRYSTAL_SYSTEM_ORTHORHOMBIC;
    } else if (spaceGroupNumber <= 142) {
        return CRYSTAL_SYSTEM_TETRAGONAL;
    } else if (spaceGroupNumber <= 167) {
        return CRYSTAL_SYSTEM_TRIGONAL;
    } else if (spaceGroupNumber <= 194) {
        return CRYSTAL_SYSTEM_HEXAGONAL;
    } else if (spaceGroupNumber <= 230) {
        return CRYSTAL_SYSTEM_CUBIC;
    }
    
    // Default to triclinic if unknown
    return CRYSTAL_SYSTEM_TRICLINIC;
}

int crystal_init(int spaceGroup) {
    crystal_symmetry_count = crystal_get_symmetry_operators(spaceGroup, crystal_symmetry_operators);
    return 0;
}

void crystal_cleanup(void) {
    crystal_symmetry_count = 0;
}

void crystal_calculate_volume(const double latticeParams[6], double *cellVolume, double *phiVolume) {
    double a = latticeParams[0];
    double b = latticeParams[1];
    double c = latticeParams[2];
    double alpha = latticeParams[3] * LAUE_DEG2RAD;
    double beta = latticeParams[4] * LAUE_DEG2RAD;
    double gamma = latticeParams[5] * LAUE_DEG2RAD;
    
    double ca = cos(alpha);
    double cb = cos(beta);
    double cg = cos(gamma);
    
    *phiVolume = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
    *cellVolume = a * b * c * (*phiVolume);
}

void crystal_calculate_reciprocal_matrix(
    const double latticeParams[6], 
    int spaceGroup, 
    double recipMatrix[3][3]
) {
    double a = latticeParams[0];
    double b = latticeParams[1];
    double c = latticeParams[2];
    double alpha = latticeParams[3];
    double beta = latticeParams[4];
    double gamma = latticeParams[5];
    
    // Check if rhombohedral setting for trigonal system
    int rhomb = 0;
    if (spaceGroup == 146 || spaceGroup == 148 || spaceGroup == 155 || 
        spaceGroup == 160 || spaceGroup == 161 || spaceGroup == 166 || 
        spaceGroup == 167) {
        rhomb = 1;
    }
    
    double ca = cos(alpha * LAUE_DEG2RAD);
    double cb = cos(beta * LAUE_DEG2RAD);
    double cg = cos(gamma * LAUE_DEG2RAD);
    double sg = sin(gamma * LAUE_DEG2RAD);
    
    double phi = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
    double Vc = a * b * c * phi;
    double pv = (2.0 * M_PI) / Vc;
    
    double a0, a1, a2, b0, b1, b2, c0, c1, c2;
    
    if (rhomb == 0) {
        a0 = a;
        a1 = 0.0;
        a2 = 0.0;
        
        b0 = b * cg;
        b1 = b * sg;
        b2 = 0.0;
        
        c0 = c * cb;
        c1 = c * (ca - cb * cg) / sg;
        c2 = c * phi / sg;
        
        a0 = geometry_zero_out(a0);
        a1 = geometry_zero_out(a1);
        a2 = geometry_zero_out(a2);
        b1 = geometry_zero_out(b1);
        b2 = geometry_zero_out(b2);
        c2 = geometry_zero_out(c2);
    } else {
        double p = sqrt(1.0 + 2 * ca);
        double q = sqrt(1.0 - ca);
        double pmq = (a / 3.0) * (p - q);
        double p2q = (a / 3.0) * (p + 2 * q);
        
        a0 = p2q; a1 = pmq; a2 = pmq;
        b0 = pmq; b1 = p2q; b2 = pmq;
        c0 = pmq; c1 = pmq; c2 = p2q;
    }
    
    recipMatrix[0][0] = geometry_zero_out((b1 * c2 - b2 * c1) * pv);
    recipMatrix[1][0] = geometry_zero_out((b2 * c0 - b0 * c2) * pv);
    recipMatrix[2][0] = geometry_zero_out((b0 * c1 - b1 * c0) * pv);
    
    recipMatrix[0][1] = geometry_zero_out((c1 * a2 - c2 * a1) * pv);
    recipMatrix[1][1] = geometry_zero_out((c2 * a0 - c0 * a2) * pv);
    recipMatrix[2][1] = geometry_zero_out((c0 * a1 - c1 * a0) * pv);
    
    recipMatrix[0][2] = geometry_zero_out((a1 * b2 - a2 * b1) * pv);
    recipMatrix[1][2] = geometry_zero_out((a2 * b0 - a0 * b2) * pv);
    recipMatrix[2][2] = geometry_zero_out((a0 * b1 - a1 * b0) * pv);
}

int crystal_get_symmetry_operators(int spaceGroupNumber, double symmetryOperators[24][4]) {
    int i, j, numOperators = 0;
    enum CrystalSystem system = crystal_get_system(spaceGroupNumber);
    
    switch (system) {
        case CRYSTAL_SYSTEM_TRICLINIC:
            numOperators = 1;
            for (i = 0; i < numOperators; i++) {
                for (j = 0; j < 4; j++) {
                    symmetryOperators[i][j] = CRYSTAL_TRICLINIC_SYM[i][j];
                }
            }
            break;
            
        case CRYSTAL_SYSTEM_MONOCLINIC:
            numOperators = 2;
            for (i = 0; i < numOperators; i++) {
                for (j = 0; j < 4; j++) {
                    symmetryOperators[i][j] = CRYSTAL_MONOCLINIC_SYM[i][j];
                }
            }
            break;
            
        case CRYSTAL_SYSTEM_ORTHORHOMBIC:
            numOperators = 4;
            for (i = 0; i < numOperators; i++) {
                for (j = 0; j < 4; j++) {
                    symmetryOperators[i][j] = CRYSTAL_ORTHORHOMBIC_SYM[i][j];
                }
            }
            break;
            
        case CRYSTAL_SYSTEM_TETRAGONAL:
            numOperators = 8;
            for (i = 0; i < numOperators; i++) {
                for (j = 0; j < 4; j++) {
                    symmetryOperators[i][j] = CRYSTAL_TETRAGONAL_SYM[i][j];
                }
            }
            break;
            
        case CRYSTAL_SYSTEM_TRIGONAL:
            numOperators = 6;
            for (i = 0; i < numOperators; i++) {
                for (j = 0; j < 4; j++) {
                    symmetryOperators[i][j] = CRYSTAL_TRIGONAL_SYM[i][j];
                }
            }
            break;
            
        case CRYSTAL_SYSTEM_HEXAGONAL:
            numOperators = 12;
            for (i = 0; i < numOperators; i++) {
                for (j = 0; j < 4; j++) {
                    symmetryOperators[i][j] = CRYSTAL_HEXAGONAL_SYM[i][j];
                }
            }
            break;
            
        case CRYSTAL_SYSTEM_CUBIC:
            numOperators = 24;
            for (i = 0; i < numOperators; i++) {
                for (j = 0; j < 4; j++) {
                    symmetryOperators[i][j] = CRYSTAL_CUBIC_SYM[i][j];
                }
            }
            break;
    }
    
    return numOperators;
}