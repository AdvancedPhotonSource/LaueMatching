/**
 * @file optimization.c
 * @brief Implementation of optimization utilities
 */

#include "../common.h"
#include "optimization.h"
#include "diffraction.h"
#include "crystallography.h"

/* Static variables for optimization */
static double lattice_tolerances[6] = {0, 0, 0, 0, 0, 0};  // Tolerances for a, b, c, alpha, beta, gamma
static double c_over_a_tolerance = 0;                     // Tolerance for c/a ratio
static double c_over_a_original = 0;                      // Original c/a ratio
static double cell_volume = 0;                           // Cell volume
static double phi_volume = 0;                            // Phi volume

void optimization_set_tolerances(
    const double latticeTolerances[6],
    double cOverATolerance
) {
    int i;
    
    for (i = 0; i < 6; i++) {
        lattice_tolerances[i] = latticeTolerances[i];
    }
    
    c_over_a_tolerance = cOverATolerance;
}

double optimization_objective_function(
    unsigned n,
    const double *x,
    double *grad,
    void *data
) {
    (void)grad;  // Unused parameter, suppress compiler warning
    // Cast data to our structure
    LaueOptimizationData *opt_data = (LaueOptimizationData *)data;
    
    double rotTranspose[3][3], detPos[3], recip[3][3];
    int i, j;
    
    // Copy detector and rotation parameters
    for (i = 0; i < 3; i++) {
        detPos[i] = opt_data->detPos[i];
        for (j = 0; j < 3; j++) {
            rotTranspose[i][j] = opt_data->rotTranspose[i][j];
            if (n == 3) {
                // If only optimizing orientation, use original reciprocal matrix
                recip[i][j] = opt_data->recipMatrix[i][j];
            }
        }
    }
    
    if (n > 3) {
        // We're also optimizing lattice parameters
        double latCNew[6];
        int counter = 0;
        
        // Determine which lattice parameters to update
        for (i = 0; i < 6; i++) {
            if (lattice_tolerances[i] != 0) {
                latCNew[i] = x[3 + counter];
                counter++;
            } else {
                latCNew[i] = opt_data->latticeParamsOrig[i];
            }
        }
        
        // Handle special case of c/a constraint
        if (c_over_a_tolerance != 0) {
            double a_new = pow((cell_volume / (x[3] * phi_volume)), 1.0/3.0);
            latCNew[0] = a_new;
            latCNew[1] = a_new;
            latCNew[2] = x[3] * a_new;
        }
        
        // Recalculate reciprocal matrix with updated lattice parameters
        crystal_calculate_reciprocal_matrix(latCNew, 0, recip);
    }
    
    // Extract Euler angles from optimization parameters
    double euler[3];
    for (i = 0; i < 3; i++) {
        euler[i] = x[i];
    }
    
    // Calculate pattern with current parameters
    double overlap = diffraction_calculate_pattern(
        opt_data->image,
        euler,
        opt_data->hkls,
        opt_data->nhkls,
        opt_data->nrPxX,
        opt_data->nrPxY,
        recip,
        opt_data->outArray,
        opt_data->maxNrSpots,
        rotTranspose,
        detPos,
        opt_data->pixelSizeX,
        opt_data->pixelSizeY,
        opt_data->eMin,
        opt_data->eMax
    );
    
    // We're minimizing, so return negative of overlap score
    return -overlap;
}

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
) {
    int i, j;
    unsigned n;
    
    // Determine number of parameters to optimize
    if (doCrystalFit == 0) {
        n = 3;  // Just orientation
    } else {
        // Count non-zero lattice tolerances
        int non_zero = 0;
        for (i = 0; i < 6; i++) {
            if (lattice_tolerances[i] != 0) {
                non_zero++;
            }
        }
        
        n = 3 + non_zero;
        
        // Special case for c/a ratio
        if (c_over_a_tolerance != 0) {
            n = 4;
            // Calculate c/a for initial value
            c_over_a_original = latticeParams[2] / latticeParams[0];
            // Calculate volume for volume conservation
            crystal_calculate_volume(latticeParams, &cell_volume, &phi_volume);
        }
    }
    
    // Setup optimization parameters
    double x[n], lower_bounds[n], upper_bounds[n];
    
    // Set initial values and bounds for Euler angles
    for (i = 0; i < 3; i++) {
        x[i] = initialEuler[i];
        lower_bounds[i] = initialEuler[i] - tolerance;
        upper_bounds[i] = initialEuler[i] + tolerance;
    }
    
    // If fitting lattice parameters, set their initial values and bounds
    if (doCrystalFit != 0) {
        if (c_over_a_tolerance != 0) {
            // Special case for c/a ratio
            x[3] = c_over_a_original;
            lower_bounds[3] = c_over_a_original * (1.0 - c_over_a_tolerance/100.0);
            upper_bounds[3] = c_over_a_original * (1.0 + c_over_a_tolerance/100.0);
        } else {
            // Regular case - fit specified lattice parameters
            int counter = 3;
            for (i = 0; i < 6; i++) {
                if (lattice_tolerances[i] != 0) {
                    x[counter] = latticeParams[i];
                    lower_bounds[counter] = latticeParams[i] * (1.0 - lattice_tolerances[i]/100.0);
                    upper_bounds[counter] = latticeParams[i] * (1.0 + lattice_tolerances[i]/100.0);
                    counter++;
                }
            }
        }
    }
    
    // Setup optimization data
    LaueOptimizationData opt_data;
    opt_data.image = image;
    opt_data.hkls = hkls;
    opt_data.nhkls = nhkls;
    opt_data.nrPxX = nrPxX;
    opt_data.nrPxY = nrPxY;
    opt_data.outArray = outArray;
    opt_data.maxNrSpots = maxNrSpots;
    
    for (i = 0; i < 3; i++) {
        opt_data.detPos[i] = detPos[i];
        for (j = 0; j < 3; j++) {
            opt_data.rotTranspose[i][j] = rotTranspose[i][j];
            opt_data.recipMatrix[i][j] = recip[i][j];
        }
    }
    
    for (i = 0; i < 6; i++) {
        opt_data.latticeParamsOrig[i] = latticeParams[i];
    }
    
    opt_data.pixelSizeX = pxX;
    opt_data.pixelSizeY = pxY;
    opt_data.eMin = eMin;
    opt_data.eMax = eMax;
    
    // Setup NLopt optimization
    nlopt_opt optimizer;
    double min_value;
    
    // Create optimizer - Nelder-Mead Simplex
    optimizer = nlopt_create(NLOPT_LN_NELDERMEAD, n);
    
    // Set bounds
    nlopt_set_lower_bounds(optimizer, lower_bounds);
    nlopt_set_upper_bounds(optimizer, upper_bounds);
    
    // Set objective function
    nlopt_set_min_objective(optimizer, optimization_objective_function, &opt_data);
    
    // Run optimization
    nlopt_result result = nlopt_optimize(optimizer, x, &min_value);
    
    // Cleanup
    nlopt_destroy(optimizer);
    
    // Check result
    if (result < 0) {
        // Optimization failed
        return LAUE_ERROR_OPTIMIZATION_FAILED;
    }
    
    // Copy optimized Euler angles
    for (i = 0; i < 3; i++) {
        optimizedEuler[i] = x[i];
    }
    
    // If fitting lattice parameters, copy optimized values
    if (doCrystalFit != 0) {
        if (c_over_a_tolerance != 0) {
            // Special case for c/a ratio
            double a_new = pow((cell_volume / (x[3] * phi_volume)), 1.0/3.0);
            optimizedLattice[0] = a_new;
            optimizedLattice[1] = a_new;
            optimizedLattice[2] = x[3] * a_new;
            optimizedLattice[3] = latticeParams[3];
            optimizedLattice[4] = latticeParams[4];
            optimizedLattice[5] = latticeParams[5];
        } else {
            // Regular case - individual lattice parameters
            int counter = 0;
            for (i = 0; i < 6; i++) {
                if (lattice_tolerances[i] != 0) {
                    optimizedLattice[i] = x[3 + counter];
                    counter++;
                } else {
                    optimizedLattice[i] = latticeParams[i];
                }
            }
        }
    } else {
        // If not fitting lattice parameters, just copy original values
        for (i = 0; i < 6; i++) {
            optimizedLattice[i] = latticeParams[i];
        }
    }
    
    // Return match score (negative of minimized value)
    *matchScore = -min_value;
    
    return LAUE_SUCCESS;
}