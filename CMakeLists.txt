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

# Default CUDA architectures.
#
# This used to be a hardcoded `70 80 86 90`, which CUDA 13 refuses outright --
# it dropped Volta, so configuring fails before a line is compiled:
#     nvcc fatal : Unsupported gpu architecture 'compute_70'
# and it silently covered no card newer than Hopper (a Blackwell needs 120).
# Ask the machine what it has; failing that, ask the toolkit what it supports.
# Set CMAKE_CUDA_ARCHITECTURES yourself to override either.
if(USE_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES)
  set(_laue_archs "")
  find_program(_laue_nvidia_smi nvidia-smi)
  if(_laue_nvidia_smi)
    execute_process(
      COMMAND "${_laue_nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _laue_caps RESULT_VARIABLE _laue_caps_rc
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_laue_caps_rc EQUAL 0 AND _laue_caps)
      string(REPLACE "\n" ";" _laue_caps "${_laue_caps}")
      foreach(_cap IN LISTS _laue_caps)
        string(STRIP "${_cap}" _cap)
        string(REPLACE "." "" _cap "${_cap}")   # 9.0 -> 90, 12.0 -> 120
        if(_cap MATCHES "^[0-9]+$")
          list(APPEND _laue_archs "${_cap}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES _laue_archs)
    endif()
  endif()
  if(_laue_archs)
    set(CMAKE_CUDA_ARCHITECTURES "${_laue_archs}")
  elseif(NOT CMAKE_VERSION VERSION_LESS 3.23)
    # No GPU visible (a login node, a container image being built). `all-major`
    # asks the toolkit, so it can never name an architecture it has dropped.
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
  else()
    set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
  endif()
  message(STATUS "CUDA architectures (auto): ${CMAKE_CUDA_ARCHITECTURES}")
endif()

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

# ---------------------------------------------------------------------------
# Optimizer: vendored, no external dependency
# ---------------------------------------------------------------------------
# NLopt used to be fetched and built here by ExternalProject. It is gone.
#
# The refiner only ever needed a bounded simplex: BOBYQA was measured WORSE
# than Nelder-Mead on this objective (median 0.0054 vs 0.0041 deg over 198
# paired synthetic seeds, p95 3.3x looser, max 25x looser, same wall-clock),
# which is unsurprising -- the objective samples intensity at integer pixels
# and is piecewise-constant, so a smooth quadratic model has nothing to fit.
#
# src/nelder_mead.{c,h} is an exact reimplementation of NLopt's LN_NELDERMEAD
# (nldrmd.c), vendored from MIDAS packages/midas_ckernel/c_src. Removing the
# dependency means: no network fetch during a build, no lib/lib64 install-path
# divergence across distros, and the C can be compiled at `pip install` time.
set(LAUE_OPTIMIZER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/nelder_mead.c)

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
add_executable(LaueMatchingCPU src/LaueMatchingCPU.c ${LAUE_OPTIMIZER_SRC})

target_compile_options(LaueMatchingCPU PRIVATE ${LAUE_C_FLAGS})
target_link_libraries(LaueMatchingCPU PRIVATE m)

if(OpenMP_C_FOUND AND BUILD_OMP)
  target_link_libraries(LaueMatchingCPU PRIVATE OpenMP::OpenMP_C)
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


  add_executable(LaueMatchingGPU src/LaueMatchingGPU.cu ${LAUE_OPTIMIZER_SRC})

  set_target_properties(LaueMatchingGPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON     # -rdc=true
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_compile_options(LaueMatchingGPU PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -w>
  )

  target_link_libraries(LaueMatchingGPU PRIVATE m)

  if(OpenMP_C_FOUND AND BUILD_OMP)
    target_compile_options(LaueMatchingGPU PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
    target_link_libraries(LaueMatchingGPU PRIVATE gomp)
  endif()

  install(TARGETS LaueMatchingGPU RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_custom_command(TARGET LaueMatchingGPU POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:LaueMatchingGPU>
            "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMENT "Copied LaueMatchingGPU → ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    VERBATIM
  )

  # -------------------------------------------------------------------------
  # LaueMatchingGPUStream  (persistent GPU daemon — socket-based streaming)
  # -------------------------------------------------------------------------
  add_executable(LaueMatchingGPUStream src/LaueMatchingGPUStream.cu ${LAUE_OPTIMIZER_SRC})

  set_target_properties(LaueMatchingGPUStream PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_compile_options(LaueMatchingGPUStream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -w>
  )

  target_link_libraries(LaueMatchingGPUStream PRIVATE m pthread)

  if(OpenMP_C_FOUND AND BUILD_OMP)
    target_compile_options(LaueMatchingGPUStream PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
    target_link_libraries(LaueMatchingGPUStream PRIVATE gomp)
  endif()

  install(TARGETS LaueMatchingGPUStream RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_custom_command(TARGET LaueMatchingGPUStream POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:LaueMatchingGPUStream>
            "${CMAKE_CURRENT_SOURCE_DIR}/bin/"
    COMMENT "Copied LaueMatchingGPUStream → ${CMAKE_CURRENT_SOURCE_DIR}/bin/"
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
message(STATUS "  Optimizer:      vendored Nelder-Mead (no NLopt)")
if(USE_CUDA)
  message(STATUS "  CUDA archs:     ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
