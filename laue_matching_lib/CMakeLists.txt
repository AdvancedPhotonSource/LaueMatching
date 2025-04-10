cmake_minimum_required(VERSION 3.10)
# Use the project settings from parent but don't redefine project
# project(LaueMatching VERSION 1.0.0 LANGUAGES C)

# Include check for header files
include(CheckIncludeFile)
CHECK_INCLUDE_FILE("stdio.h" HAVE_STDIO_H)
CHECK_INCLUDE_FILE("stdlib.h" HAVE_STDLIB_H)
CHECK_INCLUDE_FILE("string.h" HAVE_STRING_H)
CHECK_INCLUDE_FILE("stdint.h" HAVE_STDINT_H)
CHECK_INCLUDE_FILE("stddef.h" HAVE_STDDEF_H)
CHECK_INCLUDE_FILE("stdbool.h" HAVE_STDBOOL_H)
CHECK_INCLUDE_FILE("math.h" HAVE_MATH_H)
CHECK_INCLUDE_FILE("time.h" HAVE_TIME_H)
CHECK_INCLUDE_FILE("sys/mman.h" HAVE_SYS_MMAN_H)
CHECK_INCLUDE_FILE("fcntl.h" HAVE_FCNTL_H)
CHECK_INCLUDE_FILE("unistd.h" HAVE_UNISTD_H)

# Ensure we have all required headers
if(NOT HAVE_STDIO_H OR NOT HAVE_STDLIB_H OR NOT HAVE_STRING_H OR 
   NOT HAVE_STDINT_H OR NOT HAVE_STDDEF_H OR NOT HAVE_MATH_H)
  message(FATAL_ERROR "Missing required standard headers")
endif()

# OpenMP is found in parent

# NLopt is handled in parent

# Define source files for CPU version - excluding main.c as it's used directly by the CPU executable
set(CPU_SOURCES
  src/laue_matching.c
  src/core/crystallography.c
  src/core/diffraction.c
  src/core/geometry.c
  src/core/optimization.c
  src/io/file_io.c
)

# If the parent wants to build a library
if(BUILD_LIBRARY)
    # Create shared library for CPU version
    add_library(laue_matching SHARED ${CPU_SOURCES})
    
    # Add library build definition
    target_compile_definitions(laue_matching PRIVATE BUILD_LIBRARY)
    
    # Include directories
    target_include_directories(laue_matching 
        PUBLIC 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Link dependencies - Use parent-provided NLopt
    target_link_libraries(laue_matching PRIVATE OpenMP::OpenMP_C ${NLOPT_LIBRARIES} m)
    
    # Set compiler flags for C
    target_compile_options(laue_matching PRIVATE 
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -O3>
    )
    
    # If CUDA is enabled, build the GPU library
    if(USE_CUDA)
        # Define source files for GPU version - shared core files plus GPU-specific code
        set(GPU_SOURCES
          src/laue_matching_gpu.cu
          src/core/crystallography.c
          src/core/diffraction.c
          src/core/geometry.c
          src/core/optimization.c
          src/io/file_io.c
        )
        
        # Create shared library for GPU version
        add_library(laue_matching_gpu SHARED ${GPU_SOURCES})
        
        # Add library build definition
        target_compile_definitions(laue_matching_gpu PRIVATE BUILD_LIBRARY USE_CUDA)
        
        # Include directories
        target_include_directories(laue_matching_gpu 
            PUBLIC 
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
        # Link dependencies - Use parent-provided NLopt
        target_link_libraries(laue_matching_gpu PRIVATE OpenMP::OpenMP_C ${NLOPT_LIBRARIES} m)
        
        # Set compiler flags for C and CUDA
        target_compile_options(laue_matching_gpu PRIVATE 
            $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 -G -g -w>
        )
        
        # CUDA-specific settings
        set_target_properties(laue_matching_gpu PROPERTIES 
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
    
    # Let the parent handle installation
endif()