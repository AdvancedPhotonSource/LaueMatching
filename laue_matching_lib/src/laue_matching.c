/**
 * @file laue_matching.c
 * @brief Implementation of the main Laue pattern matching API
 */

 #include "common.h"
 #include "core/geometry.h"
 #include "core/crystallography.h"
 #include "core/diffraction.h"
 #include "core/optimization.h"
 #include "io/file_io.h"
 #include <stdarg.h>  // Added for va_start, va_end
 
 /* Global variables */
 static int laue_initialized = 0;
 int laue_verbose_level = 1;  // Changed from static to match extern declaration
 
 /* Verbose logging function */
 void laue_log(int level, const char *format, ...) {
     if (level <= laue_verbose_level) {
         va_list args;
         va_start(args, format);
         vprintf(format, args);
         va_end(args);
         printf("\n");
     }
 }
 
 void laue_set_verbose(int level) {
     laue_verbose_level = level;
 }
 
 int laue_init(void) {
     if (laue_initialized) {
         return LAUE_SUCCESS;
     }
     
     // Nothing specific to initialize yet, but this function
     // can be expanded in the future
     
     laue_initialized = 1;
     return LAUE_SUCCESS;
 }
 
 MatchingConfig laue_create_default_config(void) {
     MatchingConfig config;
     
     // Lattice parameters
     config.lattice.a = 0.4;
     config.lattice.b = 0.4;
     config.lattice.c = 0.4;
     config.lattice.alpha = 90.0;
     config.lattice.beta = 90.0;
     config.lattice.gamma = 90.0;
     
     // Tolerances
     for (int i = 0; i < 6; i++) {
         config.latticeParamTol[i] = 0.0;
     }
     config.cOverATol = 0.0;
     
     // Crystallography
     config.spaceGroup = 225;  // Default to cubic
     
     // Detector parameters
     config.detectorParams.position[0] = 0.0;
     config.detectorParams.position[1] = 0.0;
     config.detectorParams.position[2] = 0.0;
     config.detectorParams.rotation[0] = 0.0;
     config.detectorParams.rotation[1] = 0.0;
     config.detectorParams.rotation[2] = 0.0;
     config.detectorParams.pixelSize[0] = 0.1;
     config.detectorParams.pixelSize[1] = 0.1;
     config.detectorParams.numPixels[0] = 2048;
     config.detectorParams.numPixels[1] = 2048;
     config.detectorParams.energyRange[0] = 5.0;
     config.detectorParams.energyRange[1] = 30.0;
     
     // Matching parameters
     config.maxNumSpots = 500;
     config.minNumSpots = 5;
     config.minIntensity = 1000.0;
     config.maxAngle = 2.0;
     
     // Forward simulation
     config.performForwardSimulation = 1;
     strcpy(config.forwardSimulationFile, "forward_sim.bin");
     
     // Performance
     config.numThreads = omp_get_max_threads();
     
     return config;
 }
 
 void laue_free_results(MatchingResults *results) {
     if (results == NULL) {
         return;
     }
     
     if (results->orientations != NULL) {
         free(results->orientations);
         results->orientations = NULL;
     }
     
     if (results->eulerAngles != NULL) {
         free(results->eulerAngles);
         results->eulerAngles = NULL;
     }
     
     if (results->lattices != NULL) {
         free(results->lattices);
         results->lattices = NULL;
     }
     
     if (results->numSpots != NULL) {
         free(results->numSpots);
         results->numSpots = NULL;
     }
     
     if (results->intensities != NULL) {
         free(results->intensities);
         results->intensities = NULL;
     }
     
     if (results->numSolutions != NULL) {
         free(results->numSolutions);
         results->numSolutions = NULL;
     }
     
     results->numGrains = 0;
 }
 
 void laue_cleanup(void) {
     if (!laue_initialized) {
         return;
     }
     
     crystal_cleanup();
     diffraction_cleanup();
     
     laue_initialized = 0;
 }
 
 int laue_perform_matching(
     const MatchingConfig *config,
     const char *orientationFile,
     const char *hklFile,
     const char *imageFile,
     MatchingResults *results
 ) {
    printf("%s\n",&config->forwardSimulationFile);
    int ret;
     double start_time, time_checkpoint;
     double *image = NULL;
     double *orientations = NULL;
     int *hkls = NULL;
     size_t numOrientations;
     int numHkls, nonZeroPixels;
     int nrPxX, nrPxY;
     
     // Initialize result structure
     memset(results, 0, sizeof(MatchingResults));
     
     // Check if library is initialized
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
         laue_cleanup();
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
         laue_cleanup();
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
         laue_cleanup();
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
     
     // Perform forward simulation and matching
     laue_log(1, "Performing pattern matching with %d threads", config->numThreads);
     
     // Allocate array to store matching scores
     double *matchedScores = (double *)calloc(numOrientations, sizeof(double));
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
         laue_cleanup();
         return LAUE_ERROR_MEMORY_ALLOCATION;
     }
     
     int nrPxTotal = nrPxX * nrPxY;
     
     // Create forward simulation file if needed
     int doForwardSimulation = config->performForwardSimulation;
     printf("%s\n",config->forwardSimulationFile);
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
     
     // Determine number of orientations per thread
     int numThreads = config->numThreads;
     int maxNumSpots = config->maxNumSpots;
     
     // Track number of matched patterns
     int numResults = 0;
     int anyThreadError = 0; // Flag to track if any thread encountered an error
     printf("%s\n",config->forwardSimulationFile);
     
     #pragma omp parallel num_threads(numThreads) reduction(+:numResults) shared(anyThreadError)
     {
         int threadId = omp_get_thread_num();
         int threadError = 0; // Local thread error flag
         int orientationsPerThread = (int)ceil((double)numOrientations / (double)numThreads);
         int startOrientNr = threadId * orientationsPerThread;
         int endOrientNr = startOrientNr + orientationsPerThread;
         
         if (endOrientNr > (int)numOrientations) { // Fixed sign comparison warning
             endOrientNr = (int)numOrientations;
         }
         
         // Allocate arrays for forward simulation
         uint16_t *outArray = NULL;
         bool *pixelMask = NULL;
         
         size_t arraySize = orientationsPerThread * (1 + 2 * maxNumSpots);
         size_t arrayOffset = threadId * arraySize * sizeof(uint16_t);
         
         outArray = (uint16_t *)calloc(arraySize, sizeof(uint16_t));
         if (outArray == NULL) {
             laue_log(0, "ERROR: Failed to allocate memory for forward simulation array in thread %d", threadId);
             threadError = 1;
             #pragma omp atomic write
             anyThreadError = 1;
         }
         
         if (!threadError && doForwardSimulation) {
             pixelMask = (bool *)calloc(nrPxTotal, sizeof(bool));
             if (pixelMask == NULL) {
                 laue_log(0, "ERROR: Failed to allocate memory for pixel mask in thread %d", threadId);
                 threadError = 1;
                 #pragma omp atomic write
                 anyThreadError = 1;
             }
         } else if (!threadError) {
             // Read existing forward simulation data
             int ret = file_read_forward_simulation(
                 config->forwardSimulationFile,
                 outArray,
                 arraySize,
                 arrayOffset
             );
             
             if (ret != LAUE_SUCCESS) {
                 laue_log(0, "ERROR: Failed to read forward simulation data in thread %d", threadId);
                 threadError = 1;
                 #pragma omp atomic write
                 anyThreadError = 1;
             }
         }
         
         // Process orientations assigned to this thread if no errors
         if (!threadError) {
             // Process orientations assigned to this thread
             for (int orientNr = startOrientNr; orientNr < endOrientNr; orientNr++) {
                 int spotCount = 0;
                 int matchCount = 0;
                 double totalIntensity = 0.0;
                 
                 if (doForwardSimulation) {
                    // printf("%s\n",config->forwardSimulationFile);
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
                     
                     // Calculate pattern
                     double *qHatArray = (double *)calloc(3 * maxNumSpots, sizeof(double));
                     if (qHatArray == NULL) {
                         laue_log(0, "ERROR: Failed to allocate memory for qHat array in thread %d", threadId);
                         continue;  // Skip this orientation but continue with others
                     }
                     
                     // Store spot positions
                     for (int i = 0; i < nrPxTotal; i++) {
                         pixelMask[i] = false;
                     }
                     
                     spotCount = 0;
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
                         scattered_beam[0] = 0.0 - 2 * dot * qHat[0];
                         scattered_beam[1] = 0.0 - 2 * dot * qHat[1];
                         scattered_beam[2] = 1.0 - 2 * dot * qHat[2];
                         
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
                         
                         // Check for pixel already used
                         int pixelIndex = (int)pixelX * nrPxY + (int)pixelY;
                         if (pixelMask[pixelIndex]) continue;
                         
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
                             pixelMask[pixelIndex] = true;
                             
                             // Store pixel coordinates in output array
                             int outArrayIdx = (orientNr - startOrientNr) * (1 + 2 * maxNumSpots);
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
                     int outArrayIdx = (orientNr - startOrientNr) * (1 + 2 * maxNumSpots);
                     outArray[outArrayIdx] = (uint16_t)spotCount;
                     
                     // Clean up
                     for (int i = 0; i < spotCount; i++) {
                         int pixelX = outArray[outArrayIdx + 1 + 2 * i + 0];
                         int pixelY = outArray[outArrayIdx + 1 + 2 * i + 1];
                         int pixelIndex = pixelX * nrPxY + pixelY;
                         pixelMask[pixelIndex] = false;
                     }
                     
                     free(qHatArray);
                 } else {
                     // Use existing forward simulation data
                     int outArrayIdx = (orientNr - startOrientNr) * (1 + 2 * maxNumSpots);
                     spotCount = (int)outArray[outArrayIdx];
                     
                     // Count matches
                     for (int i = 0; i < spotCount; i++) {
                         int pixelX = outArray[outArrayIdx + 1 + 2 * i + 0];
                         int pixelY = outArray[outArrayIdx + 1 + 2 * i + 1];
                         int pixelIndex = pixelX * nrPxY + pixelY;
                         
                         if (pixelIndex >= 0 && pixelIndex < nrPxTotal && image[pixelIndex] > 0) {
                             totalIntensity += image[pixelIndex];
                             matchCount++;
                         }
                     }
                 }
                 
                 // Store match score if matches are found
                 if (matchCount >= config->minNumSpots && totalIntensity >= config->minIntensity) {
                     matchedScores[orientNr] = totalIntensity * sqrt((double)matchCount);
                     numResults++; // Using reduction(+:numResults)
                 }
             }
             
             // Write forward simulation data if needed
             if (doForwardSimulation) {
                // printf("%s\n",config->forwardSimulationFile);
                 int ret = file_write_forward_simulation(
                     config->forwardSimulationFile,
                     outArray,
                     arraySize,
                     arrayOffset
                 );
                 
                 if (ret != LAUE_SUCCESS) {
                     laue_log(0, "ERROR: Failed to write forward simulation data in thread %d", threadId);
                     // Continue despite write error
                 }
             }
         }
         
         // Clean up thread resources
         if (outArray != NULL) {
             free(outArray);
         }
         if (pixelMask != NULL) {
             free(pixelMask);
         }
     } // End of parallel region
     
     // Check if any thread had an error
     if (anyThreadError) {
         laue_log(0, "ERROR: One or more threads encountered errors");
         // We can continue with partial results or return an error
         // Depends on the application requirements
     }
     
     time_checkpoint = omp_get_wtime();
     laue_log(1, "Forward simulation and matching completed in %.2f seconds", time_checkpoint - start_time);
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
         
         if (orientations != NULL) {
             free(orientations);
         }
         if (hkls != NULL) {
             free(hkls);
         }
         if (image != NULL) {
             free(image);
         }
         
         laue_cleanup();
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
         
         laue_cleanup();
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