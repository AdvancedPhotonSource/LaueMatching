cmake_minimum_required(VERSION 3.15)

# Define options first, before the project command
option(USE_CUDA "Build with CUDA support" ON)

# Only include CUDA language if needed
if(USE_CUDA)
  project(LaueMatching LANGUAGES C CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 90)
else()
  project(LaueMatching LANGUAGES C)
endif()

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add our CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options (continued)
option(DOWNLOAD_ORIENTATION_FILE "Download 100MilOrients.bin file" ON)
option(USE_SYSTEM_NLOPT "Use system-installed NLopt library" OFF)
option(BUILD_LIBRARY "Build shared library" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Handle CUDA
if(USE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    # Custom CUDA finder logic
    find_program(CMAKE_CUDA_COMPILER
      NAMES nvcc
      PATHS
        /usr/local/cuda/bin
        /opt/cuda/bin
        ~/opt/localcuda/bin
        $ENV{CUDA_PATH}/bin
        ${CUDA_BIN_PATH}
      DOC "NVIDIA CUDA Compiler"
    )
    
    if(NOT CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "CUDA compiler not found. Set CUDA_BIN_PATH or CMAKE_CUDA_COMPILER to the directory containing nvcc.")
    endif()
  endif()
  
  # Set CUDA flags
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lgomp -O3 -G -g -w \
     -gencode=arch=compute_90,code=sm_90 \
     -lm -rdc=true")
endif()

# NLopt handling
if(USE_SYSTEM_NLOPT)
  # Try to find system-installed NLopt
  find_package(NLOPT REQUIRED)
  set(NLOPT_INCLUDE_DIRS ${NLOPT_INCLUDE_DIRS})
  set(NLOPT_LIBRARIES ${NLOPT_LIBRARIES})
else()
  # Build NLopt from source with C++ interfaces disabled
  include(ExternalProject)
  ExternalProject_Add(
    nlopt_external
    GIT_REPOSITORY https://github.com/stevengj/nlopt.git
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/nlopt_install
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
              -DBUILD_SHARED_LIBS=OFF
              -DNLOPT_PYTHON=OFF
              -DNLOPT_OCTAVE=OFF
              -DNLOPT_MATLAB=OFF
              -DNLOPT_GUILE=OFF
              -DNLOPT_SWIG=OFF
              -DNLOPT_CXX=OFF
  )
  
  # Set include and library paths for built NLopt
  set(NLOPT_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/nlopt_install/include)
  if(WIN32)
    set(NLOPT_LIBRARIES ${CMAKE_BINARY_DIR}/nlopt_install/lib/nlopt.lib)
  else()
    set(NLOPT_LIBRARIES nlopt)
    link_directories(${CMAKE_BINARY_DIR}/nlopt_install/lib ${CMAKE_BINARY_DIR}/nlopt_install/lib64)
  endif()
endif()

# Include NLopt headers
include_directories(${NLOPT_INCLUDE_DIRS})

# Create necessary directories
add_custom_target(make_dirs ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

# Add the library subdirectory
add_subdirectory(laue_matching_lib)

# CPU executable from parent
add_executable(LaueMatchingCPU src/LaueMatchingCPU.c)
target_include_directories(LaueMatchingCPU PRIVATE 
    ${NLOPT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/laue_matching_lib/include
)
target_link_libraries(LaueMatchingCPU 
    ${NLOPT_LIBRARIES}
    m
    dl
    OpenMP::OpenMP_C
)
if(BUILD_LIBRARY)
    target_link_libraries(LaueMatchingCPU laue_matching)
endif()
target_compile_options(LaueMatchingCPU PRIVATE -fPIC -g -O3 -w -Wall -fgnu89-inline)
if(NOT USE_SYSTEM_NLOPT)
  add_dependencies(LaueMatchingCPU nlopt_external)
endif()

# CUDA executable
if(USE_CUDA)
  cuda_add_executable(LaueMatchingGPU src/LaueMatchingGPU.cu)
  target_include_directories(LaueMatchingGPU PRIVATE 
      ${NLOPT_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/laue_matching_lib/include
  )
  target_link_libraries(LaueMatchingGPU 
      ${NLOPT_LIBRARIES}
      m
      dl
      OpenMP::OpenMP_C
  )
  if(BUILD_LIBRARY)
      target_link_libraries(LaueMatchingGPU laue_matching)
  endif()
  if(NOT USE_SYSTEM_NLOPT)
    add_dependencies(LaueMatchingGPU nlopt_external)
  endif()
endif()

# Download orientation file if needed and if it doesn't exist
if(DOWNLOAD_ORIENTATION_FILE)
  # Create the CMake script that checks for the file
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DownloadOrientFile.cmake 
"# Script to check if 100MilOrients.bin exists and download if not

# Check both in the binary directory and in the parent directory
if(EXISTS \"\${CMAKE_CURRENT_BINARY_DIR}/100MilOrients.bin\")
  message(STATUS \"Orientation file already exists in build directory.\")
elseif(EXISTS \"\${CMAKE_CURRENT_SOURCE_DIR}/100MilOrients.bin\")
  message(STATUS \"Orientation file found in source directory. Copying to build directory...\")
  file(COPY \"\${CMAKE_CURRENT_SOURCE_DIR}/100MilOrients.bin\" DESTINATION \"\${CMAKE_CURRENT_BINARY_DIR}\")
else()
  message(STATUS \"Orientation file not found. Downloading orientation file (~7GB, might take long)...\")
  file(DOWNLOAD
    \"https://anl.box.com/shared/static/qhao454ub2nh5t89zymj1bhlxw1q4obu.bin\"
    \"\${CMAKE_CURRENT_BINARY_DIR}/100MilOrients.bin\"
    SHOW_PROGRESS
    STATUS download_status
  )
  
  list(GET download_status 0 status_code)
  if(NOT status_code EQUAL 0)
    list(GET download_status 1 error_message)
    message(FATAL_ERROR \"Error downloading orientation file: \${error_message}\")
  else()
    message(STATUS \"Orientation file downloaded successfully.\")
  endif()
endif()"
  )

  # Add the target to run the download script
  add_custom_target(download_orientation_file
    COMMAND ${CMAKE_COMMAND} -E echo "Checking for orientation file..."
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DownloadOrientFile.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  # Also add the orientation file to installation
  install(FILES 
      ${CMAKE_BINARY_DIR}/100MilOrients.bin
      DESTINATION bin
      OPTIONAL
  )
endif()

# Copy executables to bin directory
add_custom_command(TARGET LaueMatchingCPU POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:LaueMatchingCPU> ${CMAKE_BINARY_DIR}/bin/
)

if(USE_CUDA)
  add_custom_command(TARGET LaueMatchingGPU POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:LaueMatchingGPU> ${CMAKE_BINARY_DIR}/bin/
  )
endif()

# Installation targets
install(TARGETS LaueMatchingCPU DESTINATION bin)
if(USE_CUDA)
  install(TARGETS LaueMatchingGPU DESTINATION bin)
endif()

# Python components
install(FILES 
    RunImage.py
    requirements.txt
    DESTINATION .
)

# Install library if built
if(BUILD_LIBRARY)
  install(TARGETS laue_matching
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
  )
  
  # Install library headers
  install(DIRECTORY laue_matching_lib/include/
      DESTINATION include/laue_matching
      FILES_MATCHING PATTERN "*.h"
  )
endif()