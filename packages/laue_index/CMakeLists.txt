# Builds the LaueMatching C indexer at `pip install` time and installs it under
# <site-packages>/laue_index/bin/. laue_index.indexer.binary_path() finds it
# there.
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
