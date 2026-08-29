# Builds the LaueMatching C indexer at `pip install` time and installs it under
# <site-packages>/laue_index/bin/. laue_index.indexer.binary_path() finds it
# there. The CUDA binaries are built too, but only on request -- see the CUDA
# section at the bottom of this file.
#
# NEVER FAIL THE INSTALL. If there is no C compiler, or no OpenMP, this returns
# early and `pip install laue-index` still succeeds with the Python-only path —
# indexer.available() then reports False at runtime and the error names every
# location it looked in. Same contract as MIDAS packages/midas_index.
#
# There is no external dependency to find: the optimiser is vendored
# (src/nelder_mead.c), so nothing is downloaded during a pip install. That is
# what makes this viable at all — the previous NLopt ExternalProject would have
# meant a network fetch and a full NLopt build inside pip.
cmake_minimum_required(VERSION 3.20)
project(laue_index_c LANGUAGES NONE)

# The C is VENDORED into this package's own c_src/, not read from the repo-root
# src/. That duplication is deliberate: pip builds from this package's sdist,
# which contains only what is under packages/laue_index/, so a CMakeLists
# reaching up to ../../src works in a checkout and silently produces no binary
# for every pip user. Keep the copy honest with utils/sync_vendored_c.py
# (--check in CI).
set(LAUE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/c_src")

if(NOT EXISTS "${LAUE_SRC}/LaueMatchingCPU.c")
  message(WARNING
    "Vendored C sources not found at ${LAUE_SRC} — installing the Python "
    "package only. laue_index.indexer.available() will report False; set "
    "LAUEMATCHING_BIN to an existing binary if you have one.")
  return()
endif()

include(CheckLanguage)
check_language(C)
if(NOT CMAKE_C_COMPILER)
  message(WARNING
    "No C compiler found — the indexer will not be built. The Python-only "
    "path remains available; laue_index.indexer.available() will report False.")
  return()
endif()
enable_language(C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP COMPONENTS C)
if(NOT OpenMP_C_FOUND)
  message(WARNING
    "OpenMP not found — the indexer will not be built. On macOS install libomp "
    "via `brew install libomp`; on Linux libgomp usually comes with gcc. The "
    "Python-only path remains available.")
  return()
endif()

add_executable(LaueMatchingCPU
  "${LAUE_SRC}/LaueMatchingCPU.c"
  "${LAUE_SRC}/nelder_mead.c"
)
target_include_directories(LaueMatchingCPU PRIVATE "${LAUE_SRC}")
target_compile_options(LaueMatchingCPU PRIVATE -fPIC -O3)
target_link_libraries(LaueMatchingCPU PRIVATE m OpenMP::OpenMP_C)

# Lands at <site-packages>/laue_index/bin/LaueMatchingCPU.
install(TARGETS LaueMatchingCPU RUNTIME DESTINATION laue_index/bin)

message(STATUS "laue_index: building LaueMatchingCPU from ${LAUE_SRC}")

# ── CUDA: opt-in, never automatic ───────────────────────────────────────────
#
#   LAUEMATCHING_CUDA=1 pip install laue-index
#
# Auto-detecting nvcc was considered and rejected. A toolkit that cannot compile
# these sources -- a host-compiler mismatch, an architecture the toolkit dropped
# -- fails at BUILD time, and once a target is added CMake cannot try-and-
# continue: the whole `pip install` dies and takes the working CPU binary with
# it. Opt-in confines that risk to people who deliberately asked for the GPU.
#
# Accepts either the environment variable or -Dcmake.define.LAUE_CUDA=ON via
# pip's --config-settings.
option(LAUE_CUDA "Build the CUDA binaries (LaueMatchingGPU, LaueMatchingGPUStream)" OFF)
set(_laue_cuda_env "$ENV{LAUEMATCHING_CUDA}")
if(_laue_cuda_env AND NOT _laue_cuda_env STREQUAL "0")
  set(LAUE_CUDA ON)
endif()

if(NOT LAUE_CUDA)
  return()
endif()

# NEVER hardcode an architecture list. The repo-root build did (70;80;86;90) and
# CUDA 13 -- which dropped Volta -- refuses it outright, before compiling a line:
#     nvcc fatal : Unsupported gpu architecture 'compute_70'
# Ask the machine what it has; failing that, ask the toolkit what it supports.
# This must happen BEFORE enable_language(CUDA), which compiles a test program
# with these architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
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
    message(STATUS "laue_index: building for the GPUs in this machine: ${CMAKE_CUDA_ARCHITECTURES}")
  elseif(NOT CMAKE_VERSION VERSION_LESS 3.23)
    # No GPU visible (a login node, a container build). `all-major` asks the
    # toolkit itself, so it can never name an architecture that toolkit dropped.
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
    message(STATUS "laue_index: no GPU visible; building for all-major")
  else()
    message(WARNING
      "LAUEMATCHING_CUDA is set, but no GPU is visible and CMake ${CMAKE_VERSION} "
      "is too old for `all-major` (needs 3.23). Skipping the CUDA binaries. Pass "
      "the architectures yourself, e.g. -Dcmake.define.CMAKE_CUDA_ARCHITECTURES=90.")
    return()
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING
    "LAUEMATCHING_CUDA is set, but no CUDA compiler was found -- the GPU binaries "
    "will not be built. Install the CUDA toolkit (nvcc), or point at it with "
    "-Dcmake.define.CMAKE_CUDA_COMPILER=/path/to/nvcc. The CPU binary is unaffected.")
  return()
endif()
enable_language(CUDA)

# The .cu wrap LaueMatchingHeaders.h in extern "C", so the vendored simplex --
# compiled here as C -- links into a translation unit nvcc compiles as C++.
# Mirrors the repo-root build, which CI compiles on every push.
find_library(LAUE_OMP_LIB NAMES gomp omp)

foreach(_gpu_target LaueMatchingGPU LaueMatchingGPUStream)
  add_executable(${_gpu_target}
    "${LAUE_SRC}/${_gpu_target}.cu"
    "${LAUE_SRC}/nelder_mead.c"
  )
  set_target_properties(${_gpu_target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON      # -rdc=true
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_include_directories(${_gpu_target} PRIVATE "${LAUE_SRC}")
  target_compile_options(${_gpu_target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -w -Xcompiler=${OpenMP_C_FLAGS}>
    $<$<COMPILE_LANGUAGE:C>:-fPIC -O3>
  )
  target_link_libraries(${_gpu_target} PRIVATE m)
  if(LAUE_OMP_LIB)
    target_link_libraries(${_gpu_target} PRIVATE "${LAUE_OMP_LIB}")
  endif()
  install(TARGETS ${_gpu_target} RUNTIME DESTINATION laue_index/bin)
endforeach()

# The streaming daemon serves results over TCP from worker threads.
target_link_libraries(LaueMatchingGPUStream PRIVATE pthread)

message(STATUS "laue_index: building the CUDA binaries for ${CMAKE_CUDA_ARCHITECTURES}")
