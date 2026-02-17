#
# CMakeLists.txt for LaueMatching
#
# Copyright (c) 2024, UChicago Argonne, LLC
# See LICENSE file.
# Hemant Sharma, hsharma@anl.gov
#

cmake_minimum_required(VERSION 3.18)

# CMP0135: Use download timestamps for ExternalProject
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(USE_CUDA  "Build LaueMatchingGPU (requires CUDA toolkit)" OFF)
option(BUILD_OMP "Build with OpenMP support"                      ON)

# Project definition
if(USE_CUDA)
  project(LaueMatching LANGUAGES C CUDA)
else()
  project(LaueMatching LANGUAGES C)
endif()

# C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# RPATH for installed binaries
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

include(GNUInstallDirs)
include(ExternalProject)

# ---------------------------------------------------------------------------
# NLOPT Dependency
# ---------------------------------------------------------------------------
# Install into LIBS/NLOPT/ in the source tree so the layout matches the
# existing Makefile convention.  If a pre-built NLOPT is already there
# (nlopt.h exists) we skip the download entirely.
# ---------------------------------------------------------------------------
set(NLOPT_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/LIBS/NLOPT")
set(NLOPT_INCLUDE_DIR "${NLOPT_INSTALL_DIR}/include")
set(NLOPT_LIB_DIR     "${NLOPT_INSTALL_DIR}/lib")
set(NLOPT_LIB64_DIR   "${NLOPT_INSTALL_DIR}/lib64")

if(EXISTS "${NLOPT_INCLUDE_DIR}/nlopt.h")
  message(STATUS "NLOPT already present in ${NLOPT_INSTALL_DIR} — skipping download")
else()
  message(STATUS "NLOPT not found — downloading and building into ${NLOPT_INSTALL_DIR}")
  ExternalProject_Add(nlopt_ext
    GIT_REPOSITORY https://github.com/stevengj/nlopt.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${NLOPT_INSTALL_DIR}
      -DCMAKE_BUILD_TYPE=Release
      -DNLOPT_CXX=OFF
      -DNLOPT_PYTHON=OFF
      -DNLOPT_OCTAVE=OFF
      -DNLOPT_MATLAB=OFF
      -DNLOPT_GUILE=OFF
      -DNLOPT_SWIG=OFF
      -DNLOPT_TESTS=OFF
      -DBUILD_SHARED_LIBS=OFF
    BUILD_BYPRODUCTS
      "${NLOPT_LIB_DIR}/libnlopt.a"
      "${NLOPT_LIB64_DIR}/libnlopt.a"
  )
endif()

# Ensure the include directory exists at configure time so CMake's
# INTERFACE_INCLUDE_DIRECTORIES validation passes (ExternalProject
# only populates it at build time).
file(MAKE_DIRECTORY "${NLOPT_INCLUDE_DIR}")

# Create an imported target for NLOPT so we can use target_link_libraries()
add_library(NLOPT::NLOPT STATIC IMPORTED GLOBAL)
# On some distros the lib goes to lib64/ instead of lib/
if(EXISTS "${NLOPT_LIB64_DIR}/libnlopt.a")
  set(_NLOPT_LIB "${NLOPT_LIB64_DIR}/libnlopt.a")
elseif(EXISTS "${NLOPT_LIB_DIR}/libnlopt.a")
  set(_NLOPT_LIB "${NLOPT_LIB_DIR}/libnlopt.a")
else()
  # Library will exist after build; set the expected path
  set(_NLOPT_LIB "${NLOPT_LIB_DIR}/libnlopt.a")
endif()
set_target_properties(NLOPT::NLOPT PROPERTIES
  IMPORTED_LOCATION             "${_NLOPT_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${NLOPT_INCLUDE_DIR}"
)

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
if(BUILD_OMP)
  find_package(OpenMP QUIET)
  if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_C_FLAGS}")
  else()
    message(WARNING "OpenMP not found — CPU target may lack parallelism")
  endif()
endif()

# ---------------------------------------------------------------------------
# Common compiler flags
# ---------------------------------------------------------------------------
set(LAUE_C_FLAGS -fPIC -O3 -Wall)

# ---------------------------------------------------------------------------
# LaueMatchingCPU
# ---------------------------------------------------------------------------
add_executable(LaueMatchingCPU src/LaueMatchingCPU.c)

target_compile_options(LaueMatchingCPU PRIVATE ${LAUE_C_FLAGS})
target_link_libraries(LaueMatchingCPU PRIVATE NLOPT::NLOPT m)

if(OpenMP_C_FOUND AND BUILD_OMP)
  target_link_libraries(LaueMatchingCPU PRIVATE OpenMP::OpenMP_C)
endif()

# If NLOPT was just downloaded, the CPU target must wait for the build
if(TARGET nlopt_ext)
  add_dependencies(LaueMatchingCPU nlopt_ext)
endif()

# Install
install(TARGETS LaueMatchingCPU RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Copy to source bin/ (matches existing Makefile behaviour)
add_custom_command(TARGET LaueMatchingCPU POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:LaueMatchingCPU>
          "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  COMMENT "Copied LaueMatchingCPU → ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
  VERBATIM
)

# ---------------------------------------------------------------------------
# LaueMatchingGPU  (optional — requires USE_CUDA=ON)
# ---------------------------------------------------------------------------
if(USE_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, using: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()

  add_executable(LaueMatchingGPU src/LaueMatchingGPU.cu)

  set_target_properties(LaueMatchingGPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON     # -rdc=true
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_compile_options(LaueMatchingGPU PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -w>
  )

  target_link_libraries(LaueMatchingGPU PRIVATE NLOPT::NLOPT m)

  if(OpenMP_C_FOUND AND BUILD_OMP)
    target_compile_options(LaueMatchingGPU PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
    target_link_libraries(LaueMatchingGPU PRIVATE gomp)
  endif()

  if(TARGET nlopt_ext)
    add_dependencies(LaueMatchingGPU nlopt_ext)
  endif()

  install(TARGETS LaueMatchingGPU RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_custom_command(TARGET LaueMatchingGPU POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:LaueMatchingGPU>
            "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMENT "Copied LaueMatchingGPU → ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    VERBATIM
  )
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "LaueMatching Configuration")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  OpenMP:         ${BUILD_OMP}")
message(STATUS "  CUDA (GPU):     ${USE_CUDA}")
message(STATUS "  NLOPT dir:      ${NLOPT_INSTALL_DIR}")
if(USE_CUDA)
  message(STATUS "  CUDA archs:     ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
