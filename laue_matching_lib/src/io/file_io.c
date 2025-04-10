/**
 * @file file_io.c
 * @brief Implementation of file input/output utilities
 */

#include "../common.h"
#include "file_io.h"

int file_read_parameters(const char *filename, MatchingConfig *config) {
    FILE *fileParam;
    char aline[1000], *str, dummy[1000];
    int lowNr;
    
    // Initialize config with default values
    config->lattice.a = 0.4;
    config->lattice.b = 0.4;
    config->lattice.c = 0.4;
    config->lattice.alpha = 90.0;
    config->lattice.beta = 90.0;
    config->lattice.gamma = 90.0;
    
    for (int i = 0; i < 6; i++) {
        config->latticeParamTol[i] = 0.0;
    }
    
    config->cOverATol = 0.0;
    config->spaceGroup = 225;
    config->maxNumSpots = 500;
    config->minNumSpots = 5;
    config->minIntensity = 1000.0;
    config->maxAngle = 2.0;
    config->performForwardSimulation = 1;
    strcpy(config->forwardSimulationFile, "forward_sim.bin");
    config->numThreads = omp_get_max_threads();
    
    fileParam = fopen(filename, "r");
    if (fileParam == NULL) {
        fprintf(stderr, "ERROR: Could not open parameter file: %s\n", filename);
        return LAUE_ERROR_FILE_IO;
    }
    
    while (fgets(aline, 1000, fileParam) != NULL) {
        str = "LatticeParameter";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy,
                &config->lattice.a, &config->lattice.b, &config->lattice.c,
                &config->lattice.alpha, &config->lattice.beta, &config->lattice.gamma);
            continue;
        }
        
        str = "P_Array";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf %lf %lf", dummy,
                &config->detectorParams.position[0],
                &config->detectorParams.position[1],
                &config->detectorParams.position[2]);
            continue;
        }
        
        str = "R_Array";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf %lf %lf", dummy,
                &config->detectorParams.rotation[0],
                &config->detectorParams.rotation[1],
                &config->detectorParams.rotation[2]);
            continue;
        }
        
        str = "tol_c_over_a";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf", dummy, &config->cOverATol);
            continue;
        }
        
        str = "PxX";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf", dummy, &config->detectorParams.pixelSize[0]);
            continue;
        }
        
        str = "PxY";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf", dummy, &config->detectorParams.pixelSize[1]);
            continue;
        }
        
        str = "Elo";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf", dummy, &config->detectorParams.energyRange[0]);
            continue;
        }
        
        str = "Ehi";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf", dummy, &config->detectorParams.energyRange[1]);
            continue;
        }
        
        str = "DoFwd";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %d", dummy, (int*) &config->performForwardSimulation);
            continue;
        }
        
        str = "NrPxX";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %d", dummy, &config->detectorParams.numPixels[0]);
            continue;
        }
        
        str = "NrPxY";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %d", dummy, &config->detectorParams.numPixels[1]);
            continue;
        }
        
        str = "MaxNrLaueSpots";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %d", dummy, &config->maxNumSpots);
            continue;
        }
        
        str = "MinNrSpots";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %d", dummy, &config->minNumSpots);
            continue;
        }
        
        str = "SpaceGroup";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %d", dummy, &config->spaceGroup);
            continue;
        }
        
        str = "MinIntensity";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf", dummy, &config->minIntensity);
            continue;
        }
        
        str = "MaxAngle";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf", dummy, &config->maxAngle);
            continue;
        }
        
        str = "ForwardFile";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %s", dummy, config->forwardSimulationFile);
            printf("While reading %s\n",config->forwardSimulationFile);
            continue;
        }
        
        str = "tol_LatC";
        lowNr = strncmp(aline, str, strlen(str));
        if (lowNr == 0) {
            sscanf(aline, "%s %lf %lf %lf %lf %lf %lf", dummy, 
                &config->latticeParamTol[0], &config->latticeParamTol[1],
                &config->latticeParamTol[2], &config->latticeParamTol[3],
                &config->latticeParamTol[4], &config->latticeParamTol[5]);
            continue;
        }
    }
    
    fclose(fileParam);
    
    // Handle special case for c/a tolerance
    if (config->cOverATol != 0) {
        // If c/a tolerance is set, zero out all individual lattice tolerances
        for (int i = 0; i < 6; i++) {
            config->latticeParamTol[i] = 0.0;
        }
    }
    printf("After reading %s\n",config->forwardSimulationFile);
    return LAUE_SUCCESS;
}

int file_read_orientations(
    const char *filename,
    double **orientations,
    size_t *numOrientations,
    int numThreads
) {
    (void)numThreads;  // Unused parameter
    FILE *orientFile;
    size_t fileSize;
    int useMemoryMapping = 0;
    
    // Check if file is in shared memory for potential memory mapping
    if (strncmp(filename, "/dev/shm", 8) == 0) {
        useMemoryMapping = 1;
    }
    
    orientFile = fopen(filename, "rb");
    if (orientFile == NULL) {
        fprintf(stderr, "ERROR: Could not open orientation file: %s\n", filename);
        return LAUE_ERROR_FILE_IO;
    }
    
    // Get file size
    fseek(orientFile, 0L, SEEK_END);
    fileSize = ftell(orientFile);
    rewind(orientFile);
    
    // Calculate number of orientations (each is 9 doubles)
    *numOrientations = (size_t)((double)fileSize / (double)(9 * sizeof(double)));
    
    if (useMemoryMapping) {
        // Use memory mapping for shared memory files
        fclose(orientFile);
        int fd = open(filename, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "ERROR: Could not open orientation file for memory mapping: %s\n", filename);
            return LAUE_ERROR_FILE_IO;
        }
        
        *orientations = (double *)mmap(0, fileSize, PROT_READ, MAP_SHARED, fd, 0);
        if (*orientations == MAP_FAILED) {
            fprintf(stderr, "ERROR: Memory mapping failed for: %s\n", filename);
            close(fd);
            return LAUE_ERROR_MEMORY_ALLOCATION;
        }
    } else {
        // Use regular file reading
        *orientations = (double *)malloc(fileSize);
        if (*orientations == NULL) {
            fprintf(stderr, "ERROR: Memory allocation failed for orientations\n");
            fclose(orientFile);
            return LAUE_ERROR_MEMORY_ALLOCATION;
        }
        
        size_t bytesRead = fread(*orientations, 1, fileSize, orientFile);
        if (bytesRead != fileSize) {
            fprintf(stderr, "ERROR: Failed to read complete orientation file\n");
            free(*orientations);
            fclose(orientFile);
            return LAUE_ERROR_FILE_IO;
        }
        
        fclose(orientFile);
    }
    
    return LAUE_SUCCESS;
}

int file_read_hkls(const char *filename, int **hkls, int *numHkls) {
    FILE *hklFile;
    char aline[1000];
    int h, k, l;
    
    hklFile = fopen(filename, "r");
    if (hklFile == NULL) {
        fprintf(stderr, "ERROR: Could not open HKL file: %s\n", filename);
        return LAUE_ERROR_FILE_IO;
    }
    
    // Allocate memory for HKLs
    *hkls = (int *)calloc(LAUE_MAX_HKLS * 3, sizeof(int));
    if (*hkls == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed for HKLs\n");
        fclose(hklFile);
        return LAUE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Read HKLs
    *numHkls = 0;
    while (fgets(aline, 1000, hklFile) != NULL) {
        if (sscanf(aline, "%d %d %d", &h, &k, &l) == 3) {
            (*hkls)[(*numHkls) * 3 + 0] = h;
            (*hkls)[(*numHkls) * 3 + 1] = k;
            (*hkls)[(*numHkls) * 3 + 2] = l;
            (*numHkls)++;
            
            if (*numHkls >= LAUE_MAX_HKLS) {
                fprintf(stderr, "WARNING: Reached maximum number of HKLs (%d)\n", LAUE_MAX_HKLS);
                break;
            }
        }
    }
    
    fclose(hklFile);
    return LAUE_SUCCESS;
}

int file_read_image(
    const char *filename,
    double **image,
    int nrPxX,
    int nrPxY,
    int *nonZeroPixels
) {
    FILE *imageFile;
    size_t imageSize = nrPxX * nrPxY * sizeof(double);
    
    imageFile = fopen(filename, "rb");
    if (imageFile == NULL) {
        fprintf(stderr, "ERROR: Could not open image file: %s\n", filename);
        return LAUE_ERROR_FILE_IO;
    }
    
    // Allocate memory for image
    *image = (double *)malloc(imageSize);
    if (*image == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed for image\n");
        fclose(imageFile);
        return LAUE_ERROR_MEMORY_ALLOCATION;
    }
    
    // Read image data
    size_t bytesRead = fread(*image, 1, imageSize, imageFile);
    if (bytesRead != imageSize) {
        fprintf(stderr, "ERROR: Failed to read complete image file\n");
        free(*image);
        fclose(imageFile);
        return LAUE_ERROR_FILE_IO;
    }
    
    fclose(imageFile);
    
    // Count non-zero pixels
    *nonZeroPixels = 0;
    for (int i = 0; i < nrPxX * nrPxY; i++) {
        if ((*image)[i] > 0) {
            (*nonZeroPixels)++;
        }
    }
    
    return LAUE_SUCCESS;
}

int file_write_forward_simulation(
    const char *filename,
    const uint16_t *outArray,
    size_t arraySize,
    size_t offset
) {
    int result = open(filename, O_CREAT | O_WRONLY | O_SYNC, S_IRUSR | S_IWUSR);
    if (result <= 0) {
        fprintf(stderr, "ERROR: Could not open forward simulation file for writing: %s\n", filename);
        return LAUE_ERROR_FILE_IO;
    }
    
    ssize_t bytesWritten = pwrite(result, outArray, arraySize * sizeof(uint16_t), offset);
    if (bytesWritten < 0) {
        fprintf(stderr, "ERROR: Failed to write to forward simulation file\n");
        close(result);
        return LAUE_ERROR_FILE_IO;
    } else if ((size_t)bytesWritten != arraySize * sizeof(uint16_t)) {
        // Try writing remaining data
        size_t bytesRemaining = arraySize * sizeof(uint16_t) - bytesWritten;
        size_t newOffset = offset + bytesWritten;
        size_t offset_arr = bytesWritten / sizeof(uint16_t);
        
        bytesWritten = pwrite(result, outArray + offset_arr, bytesRemaining, newOffset);
        if ((size_t)bytesWritten != bytesRemaining) {
            fprintf(stderr, "ERROR: Failed to write complete data to forward simulation file\n");
            close(result);
            return LAUE_ERROR_FILE_IO;
        }
    }
    
    close(result);
    return LAUE_SUCCESS;
}

int file_read_forward_simulation(
    const char *filename,
    uint16_t *outArray,
    size_t arraySize,
    size_t offset
) {
    int useMemoryMapping = 0;
    
    // Check if file is in shared memory for potential memory mapping
    if (strncmp(filename, "/dev/shm", 8) == 0) {
        useMemoryMapping = 1;
    }
    
    if (useMemoryMapping) {
        // Use memory mapping for shared memory files
        // Note: This assumes outArray is already allocated and will be filled by mapping
        int fd = open(filename, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "ERROR: Could not open forward simulation file for reading: %s\n", filename);
            return LAUE_ERROR_FILE_IO;
        }
        
        // Map just this part of the file
        uint16_t *mappedData = (uint16_t *)mmap(0, arraySize * sizeof(uint16_t), 
                                               PROT_READ, MAP_SHARED, fd, offset);
        if (mappedData == MAP_FAILED) {
            fprintf(stderr, "ERROR: Memory mapping failed for: %s\n", filename);
            close(fd);
            return LAUE_ERROR_MEMORY_ALLOCATION;
        }
        
        // Copy mapped data to outArray
        memcpy(outArray, mappedData, arraySize * sizeof(uint16_t));
        
        // Unmap the data
        munmap(mappedData, arraySize * sizeof(uint16_t));
        close(fd);
    } else {
        // Use regular file reading
        int result = open(filename, O_RDONLY | O_SYNC, S_IRUSR | S_IWUSR);
        if (result < 0) {
            fprintf(stderr, "ERROR: Could not open forward simulation file for reading: %s\n", filename);
            return LAUE_ERROR_FILE_IO;
        }
        
        ssize_t bytesRead = pread(result, outArray, arraySize * sizeof(uint16_t), offset);
        if (bytesRead < 0) {
            fprintf(stderr, "ERROR: Failed to read from forward simulation file\n");
            close(result);
            return LAUE_ERROR_FILE_IO;
        } else if ((size_t)bytesRead != arraySize * sizeof(uint16_t)) {
            // Try reading remaining data
            size_t bytesRemaining = arraySize * sizeof(uint16_t) - bytesRead;
            size_t newOffset = offset + bytesRead;
            size_t offset_arr = bytesRead / sizeof(uint16_t);
            
            bytesRead = pread(result, outArray + offset_arr, bytesRemaining, newOffset);
            if ((size_t)bytesRead != bytesRemaining) {
                fprintf(stderr, "ERROR: Failed to read complete data from forward simulation file\n");
                close(result);
                return LAUE_ERROR_FILE_IO;
            }
        }
        
        close(result);
    }
    printf("%s\n",filename);
    return LAUE_SUCCESS;
}

int file_open_spot_output(const char *inputImageFilename, FILE **outFile) {
    char outFilename[1000];
    
    sprintf(outFilename, "%s.spots.txt", inputImageFilename);
    *outFile = fopen(outFilename, "w");
    if (*outFile == NULL) {
        fprintf(stderr, "ERROR: Could not open spot output file: %s\n", outFilename);
        return LAUE_ERROR_FILE_IO;
    }
    
    // Write header
    fprintf(*outFile, "%%GrainNr\tSpotNr\th\tk\tl\tX\tY\tQhat[0]\tQhat[1]\tQhat[2]\tIntensity\n");
    
    return LAUE_SUCCESS;
}

int file_open_solution_output(const char *inputImageFilename, FILE **outFile) {
    char outFilename[1000];
    
    sprintf(outFilename, "%s.solutions.txt", inputImageFilename);
    *outFile = fopen(outFilename, "w");
    if (*outFile == NULL) {
        fprintf(stderr, "ERROR: Could not open solution output file: %s\n", outFilename);
        return LAUE_ERROR_FILE_IO;
    }
    
    // Write header
    fprintf(*outFile, "%%GrainNr\tNumberOfSolutions\tIntensity\tNMatches*Intensity\t"
           "NMatches*sqrt(Intensity)\tNMatches\tNSpotsCalc\t"
           "Recip1\tRecip2\tRecip3\tRecip4\tRecip5\tRecip6\tRecip7\tRecip8\tRecip9\t"
           "LatticeParameterFit[a]\tLatticeParameterFit[b]\tLatticeParameterFit[c]\t"
           "LatticeParameterFit[alpha]\tLatticeParameterFit[beta]\tLatticeParameterFit[gamma]\t"
           "OrientMatrix0\tOrientMatrix1\tOrientMatrix2\tOrientMatrix3\tOrientMatrix4\tOrientMatrix5\t"
           "OrientMatrix6\tOrientMatrix7\tOrientMatrix8\t"
           "CoarseNMatches*sqrt(Intensity)\t""misOrientationPostRefinement[degrees]\torientationRowNr\n");
    
    return LAUE_SUCCESS;
}

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
) {
    if (file == NULL) {
        return LAUE_ERROR_FILE_IO;
    }
    
    // Calculate intensity metrics
    double intensityPerSpot = intensity / numMatches;
    double nMatchesTimesIntensity = numMatches * intensityPerSpot * intensityPerSpot;
    double nMatchesSqrtIntensity = numMatches * intensityPerSpot;
    
    // Write grain information
    fprintf(file, "%d\t%d\t", grainId, numSolutions);
    fprintf(file, "%-13.4lf\t", intensityPerSpot * intensityPerSpot);
    fprintf(file, "%-13.4lf\t", nMatchesTimesIntensity);
    fprintf(file, "%-13.4lf\t", nMatchesSqrtIntensity);
    fprintf(file, "%d\t", numMatches);
    fprintf(file, "%d\t", numSimulated);
    
    // Write reciprocal matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            fprintf(file, "%-13.7lf\t\t", recipMatrix[i][j]);
        }
    }
    
    // Write lattice parameters
    for (int i = 0; i < 6; i++) {
        fprintf(file, "%-13.7lf\t\t", latticeParams[i]);
    }
    
    // Write orientation matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            fprintf(file, "%-13.7lf\t\t", orientMatrix[i][j]);
        }
    }
    
    // Write matching scores and information
    fprintf(file, "%-13.4lf\t%-13.7lf\t%d\n", 
            coarseMatchScore, misorientation, bestOrientation);
    
    return LAUE_SUCCESS;
}

int file_write_forward_simulation_full(
    const char *filename,
    const uint16_t *data,
    size_t size
) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        return LAUE_ERROR_FILE_IO;
    }
    
    size_t written = fwrite(data, sizeof(uint16_t), size, file);
    fclose(file);
    
    if (written != size) {
        return LAUE_ERROR_FILE_IO;
    }
    
    return LAUE_SUCCESS;
}

int file_read_forward_simulation_full(
    const char *filename,
    uint16_t *data,
    size_t size
) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return LAUE_ERROR_FILE_IO;
    }
    
    size_t read = fread(data, sizeof(uint16_t), size, file);
    fclose(file);
    
    if (read != size) {
        return LAUE_ERROR_FILE_IO;
    }
    
    return LAUE_SUCCESS;
}