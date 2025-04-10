/**
 * @file laue_matching_gpu.c
 * @brief Implementation of GPU-accelerated Laue pattern matching
 */

 extern "C" {
#include "common.h"
#include "core/geometry.h"
#include "core/crystallography.h"
#include "core/diffraction.h"
#include "core/optimization.h"
#include "io/file_io.h"
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdarg.h>

/* Global variables */
static int laue_gpu_initialized = 0;
extern int laue_initialized;
extern int laue_verbose_level;

/**
 * CUDA kernel to compare simulated patterns with image data
 */
__global__ void laue_gpu_compare_patterns(
    size_t nrPxX,
    size_t nOr,
    size_t nrMaxSpots,
    double minInt,
    size_t minSps,
    uint16_t *outArray,
    double *image,
    double *matchedArray
) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nOr) {
        size_t loc = i*(1+2*nrMaxSpots);
        size_t nrSpots = (size_t) outArray[loc];
        size_t hklnr;
        size_t px, py;
        double thisInt, totInt = 0.0;
        size_t nSps = 0;
        
        for (hklnr = 0; hklnr < nrSpots; hklnr++) {
            loc++;
            px = (size_t) outArray[loc];
            loc++;
            py = (size_t) outArray[loc];
            thisInt = image[py*nrPxX+px];
            if (thisInt > 0) {
                totInt += thisInt;
                nSps++;
            }
        }
        
        if (nSps >= minSps && totInt >= minInt) {
            matchedArray[i] = totInt * sqrt((double)nSps);
        }
    }
}

/**
 * Check for and report CUDA errors
 */
static void laue_gpu_check_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        laue_log(0, "CUDA Error: %s at %s:%d", cudaGetErrorString(err), file, line);
    }
}

#define CUDA_CHECK(err) laue_gpu_check_error(err, __FILE__, __LINE__)

/**
 * Initialize GPU resources
 */
int laue_gpu_init(void) {
    if (laue_gpu_initialized) {
        return LAUE_SUCCESS;
    }
    
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        laue_log(0, "ERROR: No CUDA-capable devices found");
        return LAUE_ERROR_NO_GPU;
    }
    
    // Select first device by default
    CUDA_CHECK(cudaSetDevice(0));
    
    // Print device information
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
    laue_log(1, "Using GPU device: %s", deviceProp.name);
    laue_log(1, "Compute capability: %d.%d", deviceProp.major, deviceProp.minor);
    laue_log(1, "Total global memory: %.2f GB", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    laue_gpu_initialized = 1;
    return LAUE_SUCCESS;
}

/**
 * Clean up GPU resources
 */
void laue_gpu_cleanup(void) {
    if (!laue_gpu_initialized) {
        return;
    }
    
    CUDA_CHECK(cudaDeviceReset());
    laue_gpu_initialized = 0;
}

/**
 * Perform GPU-accelerated pattern matching
 */
int laue_gpu_perform_matching(
    const MatchingConfig *config,
    const char *orientationFile,
    const char *hklFile,
    const char *imageFile,
    MatchingResults *results
) {
    int laue_initialized = 0;
    int ret;
    double start_time, time_checkpoint;
    double *image = NULL;
    double *orientations = NULL;
    int *hkls = NULL;
    size_t numOrientations;
    int numHkls, nonZeroPixels;
    int nrPxX, nrPxY;
    uint16_t *outArray = NULL;
    double *matchedScores = NULL;
    double *d_image = NULL;
    uint16_t *d_outArray = NULL;
    double *d_matchedScores = NULL;
    
    // Initialize result structure
    memset(results, 0, sizeof(MatchingResults));
    
    // Check if GPU is initialized
    if (!laue_gpu_initialized) {
        ret = laue_gpu_init();
        if (ret != LAUE_SUCCESS) {
            return ret;
        }
    }
    
    // Initialize standard library components
    if (!laue_initialized) {
        ret = laue_init();
        if (ret != LAUE_SUCCESS) {
            return ret;
        }
    }
    
    start_time = omp_get_wtime();
    
    // Initialize core modules
    ret = crystal_init(config->spaceGroup);
    if (ret != LAUE_SUCCESS) {
        return ret;
    }
    
    ret = diffraction_init(config);
    if (ret != LAUE_SUCCESS) {
        crystal_cleanup();
        return ret;
    }
    
    // Set optimization tolerances
    optimization_set_tolerances(config->latticeParamTol, config->cOverATol);
    
    // Read input files
    nrPxX = config->detectorParams.numPixels[0];
    nrPxY = config->detectorParams.numPixels[1];
    
    laue_log(1, "Reading orientations from %s", orientationFile);
    ret = file_read_orientations(orientationFile, &orientations, &numOrientations, config->numThreads);
    if (ret != LAUE_SUCCESS) {
        laue_gpu_cleanup();
        return ret;
    }
    
    time_checkpoint = omp_get_wtime();
    laue_log(1, "%zu orientations read in %.2f seconds", numOrientations, time_checkpoint - start_time);
    
    laue_log(1, "Reading HKLs from %s", hklFile);
    ret = file_read_hkls(hklFile, &hkls, &numHkls);
    if (ret != LAUE_SUCCESS) {
        // Free previous allocations
        if (orientations != NULL) {
            free(orientations);
        }
        laue_gpu_cleanup();
        return ret;
    }
    
    laue_log(1, "%d HKLs read", numHkls);
    
    laue_log(1, "Reading image from %s", imageFile);
    ret = file_read_image(imageFile, &image, nrPxX, nrPxY, &nonZeroPixels);
    if (ret != LAUE_SUCCESS) {
        // Free previous allocations
        if (orientations != NULL) {
            free(orientations);
        }
        if (hkls != NULL) {
            free(hkls);
        }
        laue_gpu_cleanup();
        return ret;
    }
    
    laue_log(1, "Image read with %d non-zero pixels", nonZeroPixels);
    
    // Setup rotation matrix for detector
    double rotTranspose[3][3];
    double rotangle = LAUE_CALC_LENGTH(
        config->detectorParams.rotation[0],
        config->detectorParams.rotation[1],
        config->detectorParams.rotation[2]
    );
    
    if (rotangle > LAUE_EPSILON) {
        double rotvect[3] = {
            config->detectorParams.rotation[0] / rotangle,
            config->detectorParams.rotation[1] / rotangle,
            config->detectorParams.rotation[2] / rotangle
        };
        
        geometry_axis_angle_to_matrix_transpose(rotvect, rotangle, rotTranspose);
    } else {
        // Identity matrix if no rotation
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotTranspose[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    
    // Calculate reciprocal lattice matrix
    double recipMatrix[3][3];
    double latticeParams[6] = {
        config->lattice.a,
        config->lattice.b,
        config->lattice.c,
        config->lattice.alpha,
        config->lattice.beta,
        config->lattice.gamma
    };
    
    crystal_calculate_reciprocal_matrix(latticeParams, config->spaceGroup, recipMatrix);
    
    // Allocate array to store matching scores
    matchedScores = (double *)calloc(numOrientations, sizeof(double));
    if (matchedScores == NULL) {
        laue_log(0, "ERROR: Failed to allocate memory for matching scores");
        if (orientations != NULL) {
            free(orientations);
        }
        if (hkls != NULL) {
            free(hkls);
        }
        if (image != NULL) {
            free(image);
        }
        laue_gpu_cleanup();
        return LAUE_ERROR_MEMORY_ALLOCATION;
    }
    
    int nrPxTotal = nrPxX * nrPxY;
    
    // Create forward simulation file if needed
    int doForwardSimulation = config->performForwardSimulation;
    if (doForwardSimulation == 0) {
        // Check if forward simulation file exists
        int result = open(config->forwardSimulationFile, O_RDONLY, S_IRUSR | S_IWUSR);
        if (result < 0) {
            laue_log(1, "Forward simulation file not found. Running in simulation mode.");
            doForwardSimulation = 1;
        } else {
            laue_log(1, "Using existing forward simulation file.");
            close(result);
        }
    }
    
    // Determine number of orientations per thread for CPU forward sim
    int numThreads = config->numThreads;
    int maxNumSpots = config->maxNumSpots;
    
    // Track number of matched patterns
    int numResults = 0;
    
    // Perform forward simulation on CPU if needed
    if (doForwardSimulation) {
        laue_log(1, "Performing forward simulation on CPU with %d threads", numThreads);
        
        bool *pxImgAll = (bool *)calloc(nrPxX * nrPxY * numThreads, sizeof(bool));
        if (pxImgAll == NULL) {
            laue_log(0, "ERROR: Failed to allocate memory for pixel mask");
            free(matchedScores);
            free(orientations);
            free(hkls);
            free(image);
            laue_gpu_cleanup();
            return LAUE_ERROR_MEMORY_ALLOCATION;
        }
        
        // Create the output array structure for the forward simulation
        size_t totalArraySize = numOrientations * (1 + 2 * maxNumSpots);
        outArray = (uint16_t *)calloc(totalArraySize, sizeof(uint16_t));
        if (outArray == NULL) {
            laue_log(0, "ERROR: Failed to allocate memory for forward simulation array");
            free(pxImgAll);
            free(matchedScores);
            free(orientations);
            free(hkls);
            free(image);
            laue_gpu_cleanup();
            return LAUE_ERROR_MEMORY_ALLOCATION;
        }
        
        int anyThreadError = 0;
        
        #pragma omp parallel num_threads(numThreads) reduction(+:numResults) shared(anyThreadError)
        {
            int threadId = omp_get_thread_num();
            int threadError = 0;
            int orientationsPerThread = (int)ceil((double)numOrientations / (double)numThreads);
            int startOrientNr = threadId * orientationsPerThread;
            int endOrientNr = startOrientNr + orientationsPerThread;
            
            if (endOrientNr > (int)numOrientations) {
                endOrientNr = (int)numOrientations;
            }
            
            // Get pointer to thread-specific pixel mask
            bool *pxImg = &pxImgAll[nrPxX * nrPxY * threadId];
            
            // Allocate memory for qHat array
            double *qHatArray = (double *)calloc(3 * maxNumSpots, sizeof(double));
            if (qHatArray == NULL) {
                threadError = 1;
                #pragma omp atomic write
                anyThreadError = 1;
            }
            
            if (!threadError) {
                // Process orientations assigned to this thread
                for (int orientNr = startOrientNr; orientNr < endOrientNr; orientNr++) {
                    int spotCount = 0;
                    int matchCount = 0;
                    double totalIntensity = 0.0;
                    
                    // Generate forward simulation
                    // Extract orientation matrix
                    double orientMatrix[3][3];
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            orientMatrix[i][j] = orientations[orientNr * 9 + i * 3 + j];
                        }
                    }
                    
                    // Convert to Euler angles
                    double euler[3];
                    geometry_orientation_matrix_to_euler(orientMatrix, euler);
                    
                    // Reset pixel mask for this orientation
                    for (int i = 0; i < nrPxX * nrPxY; i++) {
                        pxImg[i] = false;
                    }
                    
                    // Calculate Laue pattern
                    double orientMatrix2[3][3], tempMatrix[3][3];
                    
                    // Convert Euler angles to orientation matrix
                    geometry_euler_to_orientation_matrix(euler, tempMatrix);
                    
                    // Multiply orientation matrix with reciprocal lattice matrix
                    geometry_matrix_multiply_3x3(tempMatrix, recipMatrix, orientMatrix2);
                    
                    // Loop through all hkl indices
                    for (int hklIndex = 0; hklIndex < numHkls; hklIndex++) {
                        double hkl[3], qVector[3], qLength, qHat[3], dot;
                        double scattered_beam[3], position[3];
                        double xp, yp, pixelX, pixelY, sinTheta, energy;
                        int badSpot;
                        
                        // Get current hkl
                        hkl[0] = hkls[hklIndex * 3 + 0];
                        hkl[1] = hkls[hklIndex * 3 + 1];
                        hkl[2] = hkls[hklIndex * 3 + 2];
                        
                        // Calculate q-vector for this hkl
                        geometry_matrix_vector_multiply(orientMatrix2, hkl, qVector);
                        
                        // Calculate length of q-vector
                        qLength = LAUE_CALC_LENGTH(qVector[0], qVector[1], qVector[2]);
                        if (qLength < LAUE_EPSILON) continue;
                        
                        // Normalize q-vector to get unit vector
                        qHat[0] = qVector[0] / qLength;
                        qHat[1] = qVector[1] / qLength;
                        qHat[2] = qVector[2] / qLength;
                        
                        // Dot product with incident beam (0,0,1)
                        dot = qHat[2];
                        
                        // Calculate scattered beam direction
                        double ki[3] = {0.0, 0.0, 1.0}; // Incident beam direction
                        scattered_beam[0] = ki[0] - 2 * dot * qHat[0];
                        scattered_beam[1] = ki[1] - 2 * dot * qHat[1];
                        scattered_beam[2] = ki[2] - 2 * dot * qHat[2];
                        
                        // Transform scattered beam to detector coordinates
                        geometry_matrix_vector_multiply(rotTranspose, scattered_beam, position);
                        
                        // Check if beam hits detector
                        if (position[2] <= 0) continue;
                        
                        // Project onto detector plane
                        position[0] = position[0] * config->detectorParams.position[2] / position[2];
                        position[1] = position[1] * config->detectorParams.position[2] / position[2];
                        position[2] = config->detectorParams.position[2];
                        
                        // Calculate pixel coordinates
                        xp = position[0] - config->detectorParams.position[0];
                        yp = position[1] - config->detectorParams.position[1];
                        
                        pixelX = (xp / config->detectorParams.pixelSize[0]) + (0.5 * (nrPxX - 1));
                        if (pixelX < 0 || pixelX > (nrPxX - 1)) continue;
                        
                        pixelY = (yp / config->detectorParams.pixelSize[1]) + (0.5 * (nrPxY - 1));
                        if (pixelY < 0 || pixelY > (nrPxY - 1)) continue;
                        
                        // Calculate energy
                        sinTheta = -qHat[2];
                        energy = LAUE_HC_KEVNM * qLength / (4 * M_PI * sinTheta);
                        
                        // Check if energy is within range
                        if (energy < config->detectorParams.energyRange[0] || 
                            energy > config->detectorParams.energyRange[1]) continue;
                        
                        // Check if pixel has already been used
                        int pixelIndex = (int)pixelX * nrPxY + (int)pixelY;
                        if (pxImg[pixelIndex]) continue;
                        
                        // Check if spot overlaps with previously calculated spots
                        badSpot = 0;
                        for (int i = 0; i < spotCount; i++) {
                            if ((fabs(qHat[0] - qHatArray[3 * i + 0]) * 100000 < 0.1) &&
                                (fabs(qHat[1] - qHatArray[3 * i + 1]) * 100000 < 0.1) &&
                                (fabs(qHat[2] - qHatArray[3 * i + 2]) * 100000 < 0.1)) {
                                badSpot = 1;
                                break;
                            }
                        }
                        
                        if (badSpot == 0) {
                            // Store qHat in output array
                            qHatArray[3 * spotCount + 0] = qHat[0];
                            qHatArray[3 * spotCount + 1] = qHat[1];
                            qHatArray[3 * spotCount + 2] = qHat[2];
                            
                            // Mark pixel as used
                            pxImg[pixelIndex] = true;
                            
                            // Store pixel coordinates in output array
                            size_t outArrayIdx = orientNr * (1 + 2 * maxNumSpots);
                            outArray[outArrayIdx + 1 + 2 * spotCount + 0] = (uint16_t)pixelX;
                            outArray[outArrayIdx + 1 + 2 * spotCount + 1] = (uint16_t)pixelY;
                            
                            // Check if there's intensity at this pixel position in the image
                            if (image[pixelIndex] > 0) {
                                totalIntensity += image[pixelIndex];
                                matchCount++;
                            }
                            
                            spotCount++;
                            if (spotCount >= maxNumSpots) {
                                break;
                            }
                        }
                    }
                    
                    // Store number of spots in output array
                    size_t outArrayIdx = orientNr * (1 + 2 * maxNumSpots);
                    outArray[outArrayIdx] = (uint16_t)spotCount;
                    
                    // Clean up used pixels
                    for (int i = 0; i < spotCount; i++) {
                        uint16_t px = outArray[outArrayIdx + 1 + 2 * i + 0];
                        uint16_t py = outArray[outArrayIdx + 1 + 2 * i + 1];
                        int pixelIndex = px * nrPxY + py;
                        pxImg[pixelIndex] = false;
                    }
                    
                    // Store matching score
                    if (matchCount >= config->minNumSpots && totalIntensity >= config->minIntensity) {
                        matchedScores[orientNr] = totalIntensity * sqrt((double)matchCount);
                        numResults++;
                    }
                }
            }
            
            // Clean up thread resources
            if (qHatArray != NULL) {
                free(qHatArray);
            }
        } // End of parallel region
        
        if (anyThreadError) {
            laue_log(0, "ERROR: One or more threads encountered errors during forward simulation");
            free(pxImgAll);
            free(outArray);
            free(matchedScores);
            free(orientations);
            free(hkls);
            free(image);
            laue_gpu_cleanup();
            return LAUE_ERROR_FORWARD_SIMULATION;
        }
        
        // Write forward simulation to file
        laue_log(1, "Writing forward simulation to %s", config->forwardSimulationFile);
        ret = file_write_forward_simulation_full(
            config->forwardSimulationFile,
            outArray,
            numOrientations * (1 + 2 * maxNumSpots)
        );
        
        if (ret != LAUE_SUCCESS) {
            laue_log(0, "WARNING: Failed to write forward simulation file");
            // Continue despite write error
        }
        
        free(pxImgAll);
    } else {
        // Load existing forward simulation from file
        laue_log(1, "Loading forward simulation from %s", config->forwardSimulationFile);
        
        size_t totalArraySize = numOrientations * (1 + 2 * maxNumSpots);
        outArray = (uint16_t *)calloc(totalArraySize, sizeof(uint16_t));
        if (outArray == NULL) {
            laue_log(0, "ERROR: Failed to allocate memory for forward simulation array");
            free(matchedScores);
            free(orientations);
            free(hkls);
            free(image);
            laue_gpu_cleanup();
            return LAUE_ERROR_MEMORY_ALLOCATION;
        }
        
        ret = file_read_forward_simulation_full(
            config->forwardSimulationFile,
            outArray,
            totalArraySize
        );
        
        if (ret != LAUE_SUCCESS) {
            laue_log(0, "ERROR: Failed to read forward simulation file");
            free(outArray);
            free(matchedScores);
            free(orientations);
            free(hkls);
            free(image);
            laue_gpu_cleanup();
            return ret;
        }
        
        // Now run the pattern matching on GPU
        laue_log(1, "Running pattern matching on GPU");
        
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc((void**)&d_outArray, totalArraySize * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_image, nrPxX * nrPxY * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_matchedScores, numOrientations * sizeof(double)));
        
        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(d_outArray, outArray, totalArraySize * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_image, image, nrPxX * nrPxY * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_matchedScores, 0, numOrientations * sizeof(double)));
        
        // Launch kernel
        int blockSize = 1024;
        int numBlocks = (numOrientations + blockSize - 1) / blockSize;
        
        laue_gpu_compare_patterns<<<numBlocks, blockSize>>>(
            nrPxX,
            numOrientations,
            maxNumSpots,
            config->minIntensity,
            config->minNumSpots,
            d_outArray,
            d_image,
            d_matchedScores
        );
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(matchedScores, d_matchedScores, numOrientations * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Cleanup GPU resources
        CUDA_CHECK(cudaFree(d_outArray));
        CUDA_CHECK(cudaFree(d_image));
        CUDA_CHECK(cudaFree(d_matchedScores));
        
        // Count matches
        for (size_t i = 0; i < numOrientations; i++) {
            if (matchedScores[i] > 0) {
                numResults++;
            }
        }
    }
    
    time_checkpoint = omp_get_wtime();
    laue_log(1, "Pattern matching completed in %.2f seconds", time_checkpoint - start_time);
    laue_log(1, "Initial matches found: %d", numResults);
    
    // Find unique grains (merge orientations within maxAngle)
    laue_log(1, "Finding unique grains...");
    
    // Sort matchedScores to find best matches
    double *sortedScores = (double *)calloc(numResults, sizeof(double));
    size_t *rowIndices = (size_t *)calloc(numResults, sizeof(size_t));
    
    int resultIndex = 0;
    for (size_t i = 0; i < numOrientations; i++) {
        if (matchedScores[i] > 0) {
            sortedScores[resultIndex] = matchedScores[i];
            rowIndices[resultIndex] = i;
            resultIndex++;
        }
    }
    
    // Mark processed orientations
    int *processedFlags = (int *)calloc(numResults, sizeof(int));
    
    // Prepare arrays for final orientations
    double *finalOrientations = (double *)calloc(numResults * 9, sizeof(double));
    int *solutionCounts = (int *)calloc(numResults, sizeof(int));
    int *bestSolutionIndices = (int *)calloc(numResults, sizeof(int));
    
    int uniqueCount = 0;
    for (int i = 0; i < numResults; i++) {
        if (processedFlags[i] != 0) continue;
        
        double orient1[9];
        for (int k = 0; k < 9; k++) {
            orient1[k] = orientations[rowIndices[i] * 9 + k];
        }
        
        double quat1[4];
        geometry_orientation_matrix_to_quaternion(orient1, quat1);
        
        processedFlags[i] = 1;
        int bestSolution = rowIndices[i];
        double bestScore = sortedScores[i];
        
        // Find all orientations within maxAngle of this one
        for (int j = i + 1; j < numResults; j++) {
            if (processedFlags[j] > 0) continue;
            
            double orient2[9];
            for (int k = 0; k < 9; k++) {
                orient2[k] = orientations[rowIndices[j] * 9 + k];
            }
            
            double quat2[4];
            geometry_orientation_matrix_to_quaternion(orient2, quat2);
            
            double misoAngle = geometry_get_misorientation(quat1, quat2);
            if (misoAngle <= config->maxAngle) {
                processedFlags[j] = 1;
                processedFlags[i]++;
                
                if (sortedScores[j] > bestScore) {
                    bestScore = sortedScores[j];
                    bestSolution = rowIndices[j];
                }
            }
        }
        
        // Store the best orientation for this grain
        for (int k = 0; k < 9; k++) {
            finalOrientations[uniqueCount * 9 + k] = orientations[bestSolution * 9 + k];
        }
        
        solutionCounts[uniqueCount] = processedFlags[i];
        bestSolutionIndices[uniqueCount] = bestSolution;
        uniqueCount++;
    }
    
    time_checkpoint = omp_get_wtime();
    laue_log(1, "Found %d unique grains in %.2f seconds", uniqueCount, time_checkpoint - start_time);
    
    // Optimize each unique grain orientation
    laue_log(1, "Refining grain orientations...");
    
    // Open output files
    FILE *spotFile = NULL, *solutionFile = NULL;
    ret = file_open_spot_output(imageFile, &spotFile);
    if (ret != LAUE_SUCCESS) {
        laue_log(0, "WARNING: Failed to open spot output file");
    }
    
    ret = file_open_solution_output(imageFile, &solutionFile);
    if (ret != LAUE_SUCCESS) {
        laue_log(0, "ERROR: Failed to open solution output file");
        if (spotFile != NULL) {
            fclose(spotFile);
        }
        
        // Clean up
        free(matchedScores);
        free(sortedScores);
        free(rowIndices);
        free(processedFlags);
        free(finalOrientations);
        free(solutionCounts);
        free(bestSolutionIndices);
        free(outArray);
        
        if (orientations != NULL) {
            free(orientations);
        }
        if (hkls != NULL) {
            free(hkls);
        }
        if (image != NULL) {
            free(image);
        }
        
        laue_gpu_cleanup();
        return ret;
    }
    
    // Allocate memory for results
    results->numGrains = uniqueCount;
    results->orientations = (double *)malloc(uniqueCount * 9 * sizeof(double));
    results->eulerAngles = (double *)malloc(uniqueCount * 3 * sizeof(double));
    results->lattices = (LatticeParameters *)malloc(uniqueCount * sizeof(LatticeParameters));
    results->numSpots = (int *)malloc(uniqueCount * sizeof(int));
    results->intensities = (double *)malloc(uniqueCount * sizeof(double));
    results->numSolutions = (int *)malloc(uniqueCount * sizeof(int));
    
    if (results->orientations == NULL || results->eulerAngles == NULL || 
        results->lattices == NULL || results->numSpots == NULL || 
        results->intensities == NULL || results->numSolutions == NULL) {
        
        laue_log(0, "ERROR: Failed to allocate memory for results");
        
        // Clean up
        laue_free_results(results);
        
        free(matchedScores);
        free(sortedScores);
        free(rowIndices);
        free(processedFlags);
        free(finalOrientations);
        free(solutionCounts);
        free(bestSolutionIndices);
        free(outArray);
        
        if (orientations != NULL) {
            free(orientations);
        }
        if (hkls != NULL) {
            free(hkls);
        }
        if (image != NULL) {
            free(image);
        }
        
        if (spotFile != NULL) {
            fclose(spotFile);
        }
        if (solutionFile != NULL) {
            fclose(solutionFile);
        }
        
        laue_gpu_cleanup();
        return LAUE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Use larger max spots for refinement
    int refinementMaxSpots = config->maxNumSpots * 3;
    
    // Process each unique grain
    // Local array to track thread errors in the next parallel section
    int threadRefinementErrors[uniqueCount];
    for (int i = 0; i < uniqueCount; i++) {
        threadRefinementErrors[i] = 0;
    }
    
    #pragma omp parallel for num_threads(numThreads)
    for (int grainIndex = 0; grainIndex < uniqueCount; grainIndex++) {
        double orientMatrix[3][3];
        double euler[3], eulerRefined[3];
        double latticeParamsRefined[6];
        double matchScore;
        
        // Extract orientation matrix
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                orientMatrix[i][j] = finalOrientations[grainIndex * 9 + i * 3 + j];
            }
        }
        
        // Convert to Euler angles
        geometry_orientation_matrix_to_euler(orientMatrix, euler);
        
        // Allocate space for qHat arrays
        double *qHatArray = (double *)calloc(3 * refinementMaxSpots, sizeof(double));
        if (qHatArray == NULL) {
            laue_log(0, "ERROR: Failed to allocate memory for qHat array in grain %d", grainIndex + 1);
            threadRefinementErrors[grainIndex] = 1;
            continue; // Skip this grain
        }
        
        // First optimize orientation only
        int doCrystalFit = 0;
        ret = optimization_fit_orientation(
            image,
            euler,
            hkls,
            numHkls,
            nrPxX,
            nrPxY,
            recipMatrix,
            qHatArray,
            refinementMaxSpots,
            rotTranspose,
            config->detectorParams.position,
            config->detectorParams.pixelSize[0],
            config->detectorParams.pixelSize[1],
            config->detectorParams.energyRange[0],
            config->detectorParams.energyRange[1],
            3 * LAUE_DEG2RAD,  // 3 degrees tolerance
            latticeParams,
            eulerRefined,
            latticeParamsRefined,
            &matchScore,
            doCrystalFit
        );
        
        if (ret != LAUE_SUCCESS) {
            laue_log(0, "WARNING: Orientation optimization failed for grain %d", grainIndex + 1);
            free(qHatArray);
            threadRefinementErrors[grainIndex] = 1;
            continue; // Skip to next grain
        }
        
        // Copy optimized Euler angles for lattice optimization
        for (int i = 0; i < 3; i++) {
            euler[i] = eulerRefined[i];
        }
        
        // Now optimize lattice parameters if requested
        doCrystalFit = (config->latticeParamTol[0] != 0 || config->latticeParamTol[1] != 0 || 
                          config->latticeParamTol[2] != 0 || config->latticeParamTol[3] != 0 || 
                          config->latticeParamTol[4] != 0 || config->latticeParamTol[5] != 0 || 
                          config->cOverATol != 0);
                          
        if (doCrystalFit) {
            ret = optimization_fit_orientation(
                image,
                euler,
                hkls,
                numHkls,
                nrPxX,
                nrPxY,
                recipMatrix,
                qHatArray,
                refinementMaxSpots,
                rotTranspose,
                config->detectorParams.position,
                config->detectorParams.pixelSize[0],
                config->detectorParams.pixelSize[1],
                config->detectorParams.energyRange[0],
                config->detectorParams.energyRange[1],
                3 * LAUE_DEG2RAD,  // 3 degrees tolerance
                latticeParams,
                eulerRefined,
                latticeParamsRefined,
                &matchScore,
                doCrystalFit
            );
            
            if (ret != LAUE_SUCCESS) {
                laue_log(0, "WARNING: Lattice parameter optimization failed for grain %d", grainIndex + 1);
                // Continue with orientation-only optimization results
            }
        } else {
            // If no lattice parameter fitting, just copy original values
            for (int i = 0; i < 6; i++) {
                latticeParamsRefined[i] = latticeParams[i];
            }
        }
        
        // Calculate final reciprocal matrix with optimized lattice parameters
        double recipRefined[3][3];
        crystal_calculate_reciprocal_matrix(latticeParamsRefined, config->spaceGroup, recipRefined);
        
        // Calculate final orientation matrix
        double orientRefined[3][3];
        geometry_euler_to_orientation_matrix(eulerRefined, orientRefined);
        
        // Write results to spot file
        int simulatedSpotCount = 0;
        int matchedSpotCount = 0;
        
        // Allocate fresh qHat array for final calculation
        free(qHatArray);
        qHatArray = (double *)calloc(3 * refinementMaxSpots, sizeof(double));
        if (qHatArray == NULL) {
            laue_log(0, "ERROR: Failed to allocate memory for final qHat array in grain %d", grainIndex + 1);
            threadRefinementErrors[grainIndex] = 1;
            continue; // Skip to next grain
        }
        
        matchedSpotCount = diffraction_write_pattern(
            image,
            eulerRefined,
            hkls,
            numHkls,
            nrPxX,
            nrPxY,
            recipRefined,
            qHatArray,
            refinementMaxSpots,
            rotTranspose,
            config->detectorParams.position,
            config->detectorParams.pixelSize[0],
            config->detectorParams.pixelSize[1],
            config->detectorParams.energyRange[0],
            config->detectorParams.energyRange[1],
            spotFile,
            grainIndex + 1,
            &simulatedSpotCount
        );
        
        // Calculate misorientation between original and refined orientation
        double quat1[4], quat2[4];
        geometry_orientation_matrix_3x3_to_quaternion(orientMatrix, quat1);
        geometry_orientation_matrix_3x3_to_quaternion(orientRefined, quat2);
        double misoAngle = geometry_get_misorientation(quat1, quat2);
        
        // Write solution to file
        #pragma omp critical
        {
            file_write_grain_solution(
                solutionFile,
                grainIndex + 1,
                solutionCounts[grainIndex],
                matchScore,
                matchedSpotCount,
                simulatedSpotCount,
                orientRefined,
                recipRefined,
                latticeParamsRefined,
                sortedScores[grainIndex],
                misoAngle,
                bestSolutionIndices[grainIndex]
            );
        }
        
        // Store results
        for (int i = 0; i < 3; i++) {
            results->eulerAngles[grainIndex * 3 + i] = eulerRefined[i];
            for (int j = 0; j < 3; j++) {
                results->orientations[grainIndex * 9 + i * 3 + j] = orientRefined[i][j];
            }
        }
        
        results->lattices[grainIndex].a = latticeParamsRefined[0];
        results->lattices[grainIndex].b = latticeParamsRefined[1];
        results->lattices[grainIndex].c = latticeParamsRefined[2];
        results->lattices[grainIndex].alpha = latticeParamsRefined[3];
        results->lattices[grainIndex].beta = latticeParamsRefined[4];
        results->lattices[grainIndex].gamma = latticeParamsRefined[5];
        
        results->numSpots[grainIndex] = matchedSpotCount;
        results->intensities[grainIndex] = matchScore;
        results->numSolutions[grainIndex] = solutionCounts[grainIndex];
        
        // Clean up
        free(qHatArray);
    }
    
    // Check if any thread errors occurred during refinement
    int refinementErrorCount = 0;
    for (int i = 0; i < uniqueCount; i++) {
        if (threadRefinementErrors[i]) {
            refinementErrorCount++;
        }
    }
    
    if (refinementErrorCount > 0) {
        laue_log(0, "WARNING: %d grains failed during refinement", refinementErrorCount);
    }
    
    // Close output files
    if (spotFile != NULL) {
        fclose(spotFile);
    }
    if (solutionFile != NULL) {
        fclose(solutionFile);
    }
    
    // Clean up
    free(matchedScores);
    free(sortedScores);
    free(rowIndices);
    free(processedFlags);
    free(finalOrientations);
    free(solutionCounts);
    free(bestSolutionIndices);
    free(outArray);
    
    if (orientations != NULL) {
        free(orientations);
    }
    if (hkls != NULL) {
        free(hkls);
    }
    if (image != NULL) {
        free(image);
    }
    
    double totalTime = omp_get_wtime() - start_time;
    laue_log(1, "Pattern matching completed in %.2f seconds", totalTime);
    laue_log(1, "Found %d grains", uniqueCount);
    
    return LAUE_SUCCESS;
}