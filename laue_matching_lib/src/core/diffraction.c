/**
 * @file diffraction.c
 * @brief Implementation of diffraction simulation utilities
 */

#include "../common.h"
#include "diffraction.h"

/* Static global variables */
static double incident_beam[3] = {0, 0, 1.0}; // Incident beam direction

int diffraction_init(const MatchingConfig *config) {
    // Initialize crystallography module with space group
    return crystal_init(config->spaceGroup);
}

void diffraction_cleanup(void) {
    crystal_cleanup();
}

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
) {
    double orientMatrix[3][3], tempMatrix[3][3];
    
    // Convert Euler angles to orientation matrix
    geometry_euler_to_orientation_matrix(euler, tempMatrix);
    
    // Multiply orientation matrix with reciprocal lattice matrix
    geometry_matrix_multiply_3x3(tempMatrix, recip, orientMatrix);
    
    int hklIndex, badSpot;
    double hkl[3], qVector[3], qLength, qHat[3], dot;
    double scattered_beam[3], position[3];
    double xp, yp, pixelX, pixelY, sinTheta, energy;
    double result = 0;
    int spotCount = 0, matchedCount = 0;
    
    // Loop through all hkl indices
    for (hklIndex = 0; hklIndex < nhkls; hklIndex++) {
        // Get current hkl
        hkl[0] = hkls[hklIndex * 3 + 0];
        hkl[1] = hkls[hklIndex * 3 + 1];
        hkl[2] = hkls[hklIndex * 3 + 2];
        
        // Calculate q-vector for this hkl
        geometry_matrix_vector_multiply(orientMatrix, hkl, qVector);
        
        // Calculate length of q-vector
        qLength = LAUE_CALC_LENGTH(qVector[0], qVector[1], qVector[2]);
        if (qLength < LAUE_EPSILON) continue;
        
        // Normalize q-vector to get unit vector
        qHat[0] = qVector[0] / qLength;
        qHat[1] = qVector[1] / qLength;
        qHat[2] = qVector[2] / qLength;
        
        // Dot product of q-hat with incident beam direction
        dot = qHat[2]; // Simplified because incident_beam is [0,0,1]
        
        // Calculate scattered beam direction
        scattered_beam[0] = incident_beam[0] - 2 * dot * qHat[0];
        scattered_beam[1] = incident_beam[1] - 2 * dot * qHat[1];
        scattered_beam[2] = incident_beam[2] - 2 * dot * qHat[2];
        
        // Transform scattered beam to detector coordinates
        geometry_matrix_vector_multiply(rotTranspose, scattered_beam, position);
        
        // Check if beam hits detector
        if (position[2] <= 0) continue;
        
        // Project onto detector plane
        position[0] = position[0] * detPos[2] / position[2];
        position[1] = position[1] * detPos[2] / position[2];
        position[2] = detPos[2];
        
        // Calculate pixel coordinates
        xp = position[0] - detPos[0];
        yp = position[1] - detPos[1];
        
        pixelX = (xp / pxX) + (0.5 * (nrPxX - 1));
        if (pixelX < 0 || pixelX > (nrPxX - 1)) continue;
        
        pixelY = (yp / pxY) + (0.5 * (nrPxY - 1));
        if (pixelY < 0 || pixelY > (nrPxY - 1)) continue;
        
        // Calculate energy
        sinTheta = -qHat[2];
        energy = LAUE_HC_KEVNM * qLength / (4 * M_PI * sinTheta);
        
        // Check if energy is within range
        if (energy < eMin || energy > eMax) continue;
        
        // Check if spot overlaps with previously calculated spots
        badSpot = 0;
        for (int i = 0; i < spotCount; i++) {
            if ((fabs(qHat[0] - outArray[3 * i + 0]) * 100000 < 0.1) &&
                (fabs(qHat[1] - outArray[3 * i + 1]) * 100000 < 0.1) &&
                (fabs(qHat[2] - outArray[3 * i + 2]) * 100000 < 0.1)) {
                badSpot = 1;
                break;
            }
        }
        
        if (badSpot == 0) {
            // Store qHat in output array
            outArray[3 * spotCount + 0] = qHat[0];
            outArray[3 * spotCount + 1] = qHat[1];
            outArray[3 * spotCount + 2] = qHat[2];
            
            // Check if there's intensity at this pixel position in the image
            int pixelPos = (int)pixelY * nrPxX + (int)pixelX;
            if (image[pixelPos] > 0) {
                result += image[pixelPos];
                matchedCount++;
            }
            
            spotCount++;
            if (spotCount == maxNrSpots) {
                break;
            }
        }
    }
    
    // Apply scoring function: number of matched spots * sqrt of total intensity
    if (matchedCount > 0) {
        result = matchedCount * sqrt(result);
    } else {
        result = 0;
    }
    
    return result;
}

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
) {
    double orientMatrix[3][3], tempMatrix[3][3];
    
    // Convert Euler angles to orientation matrix
    geometry_euler_to_orientation_matrix(euler, tempMatrix);
    
    // Multiply orientation matrix with reciprocal lattice matrix
    geometry_matrix_multiply_3x3(tempMatrix, recip, orientMatrix);
    
    int hklIndex, badSpot;
    double hkl[3], qVector[3], qLength, qHat[3], dot;
    double scattered_beam[3], position[3];
    double xp, yp, pixelX, pixelY, sinTheta, energy;
    int spotCount = 0, matchedCount = 0;
    
    // Loop through all hkl indices
    for (hklIndex = 0; hklIndex < nhkls; hklIndex++) {
        // Get current hkl
        hkl[0] = hkls[hklIndex * 3 + 0];
        hkl[1] = hkls[hklIndex * 3 + 1];
        hkl[2] = hkls[hklIndex * 3 + 2];
        
        // Calculate q-vector for this hkl
        geometry_matrix_vector_multiply(orientMatrix, hkl, qVector);
        
        // Calculate length of q-vector
        qLength = LAUE_CALC_LENGTH(qVector[0], qVector[1], qVector[2]);
        if (qLength < LAUE_EPSILON) continue;
        
        // Normalize q-vector to get unit vector
        qHat[0] = qVector[0] / qLength;
        qHat[1] = qVector[1] / qLength;
        qHat[2] = qVector[2] / qLength;
        
        // Dot product of q-hat with incident beam direction
        dot = qHat[2]; // Simplified because incident_beam is [0,0,1]
        
        // Calculate scattered beam direction
        scattered_beam[0] = incident_beam[0] - 2 * dot * qHat[0];
        scattered_beam[1] = incident_beam[1] - 2 * dot * qHat[1];
        scattered_beam[2] = incident_beam[2] - 2 * dot * qHat[2];
        
        // Transform scattered beam to detector coordinates
        geometry_matrix_vector_multiply(rotTranspose, scattered_beam, position);
        
        // Check if beam hits detector
        if (position[2] <= 0) continue;
        
        // Project onto detector plane
        position[0] = position[0] * detPos[2] / position[2];
        position[1] = position[1] * detPos[2] / position[2];
        position[2] = detPos[2];
        
        // Calculate pixel coordinates
        xp = position[0] - detPos[0];
        yp = position[1] - detPos[1];
        
        pixelX = (xp / pxX) + (0.5 * (nrPxX - 1));
        if (pixelX < 0 || pixelX > (nrPxX - 1)) continue;
        
        pixelY = (yp / pxY) + (0.5 * (nrPxY - 1));
        if (pixelY < 0 || pixelY > (nrPxY - 1)) continue;
        
        // Calculate energy
        sinTheta = -qHat[2];
        energy = LAUE_HC_KEVNM * qLength / (4 * M_PI * sinTheta);
        
        // Check if energy is within range
        if (energy < eMin || energy > eMax) continue;
        
        // Check if spot overlaps with previously calculated spots
        badSpot = 0;
        for (int i = 0; i < spotCount; i++) {
            if ((fabs(qHat[0] - outArray[3 * i + 0]) * 100000 < 0.1) &&
                (fabs(qHat[1] - outArray[3 * i + 1]) * 100000 < 0.1) &&
                (fabs(qHat[2] - outArray[3 * i + 2]) * 100000 < 0.1)) {
                badSpot = 1;
                break;
            }
        }
        
        if (badSpot == 0) {
            // Store qHat in output array
            outArray[3 * spotCount + 0] = qHat[0];
            outArray[3 * spotCount + 1] = qHat[1];
            outArray[3 * spotCount + 2] = qHat[2];
            
            int pixelPos = (int)pixelY * nrPxX + (int)pixelX;
            double intensity = image[pixelPos];
            
            // If there's intensity at this pixel, write to output file and count
            if (intensity > 0) {
                if (outFile != NULL) {
                    fprintf(outFile, "%d\t%d\t%d\t%d\t%d\t%5d\t%5d\t%lf\t%lf\t%lf\t%lf\n", 
                        grainId, spotCount, (int)hkl[0], (int)hkl[1], (int)hkl[2], 
                        (int)pixelX, (int)pixelY, qHat[0], qHat[1], qHat[2], intensity);
                }
                matchedCount++;
            }
            
            spotCount++;
            if (spotCount == maxNrSpots) {
                break;
            }
        }
    }
    
    // Return number of simulated spots via pointer
    if (simulSpotCount != NULL) {
        *simulSpotCount = spotCount;
    }
    
    return matchedCount;
}