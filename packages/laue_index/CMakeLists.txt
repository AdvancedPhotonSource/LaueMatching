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

# c_src/ is where the C LIVES -- the only copy in the repository. It has to be
# inside the package because pip builds from this package's sdist, which
# contains only what is under packages/laue_index/: a CMakeLists reaching up to
# a repo-root ../../src works in a checkout and silently produces no binary for
# every pip user. The repo-root build reaches DOWN into this same directory, so
# a checkout and a pip install compile identical sources by construction rather
# than by a sync check.
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

# ── CUDA: attempted BY DEFAULT, in an isolated sub-build ────────────────────
#
#   (unset)                   attempt if nvcc is present; warn and continue if
#                             it cannot build. This is the default.
#   LAUEMATCHING_CUDA=0       skip entirely -- fast installs, CI.
#   LAUEMATCHING_CUDA=1       attempt, same as unset.
#   LAUEMATCHING_CUDA=require FAIL the install if the GPU binaries cannot be
#                             built. What a beamline deployment wants.
#
# `=1` MEANS ATTEMPT, NOT REQUIRE. 0.6.0 briefly redefined it to "require" and
# that was a gratuitous break: every README and script since the opt-in era says
# `LAUEMATCHING_CUDA=1 pip install laue-index`, so on any machine without nvcc
# the install started failing where it used to succeed CPU-only. The new
# capability gets a new name instead.
#
# Auto-detect used to be rejected because "once a target is added CMake cannot
# try-and-continue: the whole pip install dies and takes the working CPU binary
# with it". That is true of a target in THIS project. cmake/cuda/ is a separate
# project, configured and built below through execute_process(), so a failure
# is an exit code we catch -- and the CPU binary above is already installed and
# cannot be affected.
# A STALE CACHE ENTRY MUST NOT DECIDE THIS. Versions before 0.6.0 declared
# `option(LAUE_CUDA ... OFF)`, so every build directory made by one of them has
# LAUE_CUDA:BOOL=OFF cached. Reading `if(DEFINED LAUE_CUDA)` then silently
# forces the CUDA build off forever in that tree, with the manifest cheerfully
# reporting "skipped: LAUEMATCHING_CUDA=0" for a variable nobody set -- caught
# on the first local build of this change. Clear it, and read a name that has
# never been cached.
unset(LAUE_CUDA CACHE)

set(LAUE_CUDA_MODE "auto")
set(_laue_cuda_env "$ENV{LAUEMATCHING_CUDA}")
if(DEFINED LAUE_CUDA_REQUIRE)  # -Dcmake.define.LAUE_CUDA_REQUIRE=ON/OFF via pip
  if(LAUE_CUDA_REQUIRE)
    set(LAUE_CUDA_MODE "required")
  else()
    set(LAUE_CUDA_MODE "off")
  endif()
elseif(NOT _laue_cuda_env STREQUAL "")
  string(TOLOWER "${_laue_cuda_env}" _cuda_env_lc)
  if(_cuda_env_lc STREQUAL "0" OR _cuda_env_lc STREQUAL "off"
     OR _cuda_env_lc STREQUAL "false" OR _cuda_env_lc STREQUAL "no")
    set(LAUE_CUDA_MODE "off")
  elseif(_cuda_env_lc STREQUAL "require" OR _cuda_env_lc STREQUAL "required")
    set(LAUE_CUDA_MODE "required")
  else()
    set(LAUE_CUDA_MODE "auto")   # includes the historical `=1`
  endif()
endif()

set(_cuda_built FALSE)
set(_cuda_reason "")
set(_cuda_archs "")
set(_cuda_nvcc "")
set(_cuda_nvcc_version "")

if(LAUE_CUDA_MODE STREQUAL "off")
  set(_cuda_reason "skipped: LAUEMATCHING_CUDA=0")
  message(STATUS "laue_index: CUDA build disabled (LAUEMATCHING_CUDA=0)")
else()
  # check_language adds no targets, so this cannot break the build.
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    set(_cuda_reason "skipped: no CUDA compiler (nvcc) found")
    if(LAUE_CUDA_MODE STREQUAL "required")
      message(FATAL_ERROR
        "LAUEMATCHING_CUDA=require was set but no CUDA compiler was found.\n"
        "Install the CUDA toolkit (nvcc) and reinstall, or drop the variable to\n"
        "build whatever this machine can (CPU only, here).")
    endif()
    message(STATUS "laue_index: no nvcc; building the CPU binary only")
  else()
    set(_cuda_src   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")
    set(_cuda_bld   "${CMAKE_CURRENT_BINARY_DIR}/cuda-subbuild")
    set(_cuda_stage "${CMAKE_CURRENT_BINARY_DIR}/cuda-stage")
    file(MAKE_DIRECTORY "${_cuda_bld}" "${_cuda_stage}")

    execute_process(
      COMMAND "${CMAKE_COMMAND}" -S "${_cuda_src}" -B "${_cuda_bld}"
              "-DLAUE_SRC=${LAUE_SRC}"
              "-DCMAKE_INSTALL_PREFIX=${_cuda_stage}"
              "-DCMAKE_BUILD_TYPE=Release"
      RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
    if(_rc EQUAL 0)
      execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${_cuda_bld}" --target install --parallel
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out2 ERROR_VARIABLE _err2)
      set(_err "${_err}${_err2}")
    endif()

    if(_rc EQUAL 0 AND EXISTS "${_cuda_stage}/LaueMatchingGPU")
      set(_cuda_built TRUE)
      set(_cuda_reason "built")
      if(EXISTS "${_cuda_bld}/cuda_report.txt")
        file(STRINGS "${_cuda_bld}/cuda_report.txt" _rep)
        foreach(_line IN LISTS _rep)
          if(_line MATCHES "^architectures=(.*)$")
            set(_cuda_archs "${CMAKE_MATCH_1}")
          elseif(_line MATCHES "^nvcc_version=(.*)$")
            set(_cuda_nvcc_version "${CMAKE_MATCH_1}")
          elseif(_line MATCHES "^nvcc=(.*)$")
            set(_cuda_nvcc "${CMAKE_MATCH_1}")
          endif()
        endforeach()
      endif()
      install(DIRECTORY "${_cuda_stage}/" DESTINATION laue_index/bin
              USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "LaueMatching*")
      message(STATUS "laue_index: CUDA binaries built for ${_cuda_archs}")
    else()
      # Keep the ERROR lines, not the first 400 characters of stderr: an nvcc
      # run emits pages of "support for offline compilation for architectures
      # prior to ..." deprecation warnings, and a blind head of the stream
      # records those instead of the thing that actually failed.
      set(_err_lines "")
      string(REPLACE "\n" ";" _err_list "${_err}")
      foreach(_l IN LISTS _err_list)
        if(_l MATCHES "[Ee]rror|FAILED|fatal")
          string(APPEND _err_lines "${_l} | ")
        endif()
      endforeach()
      if(NOT _err_lines)
        string(REPLACE "\n" " | " _err_lines "${_err}")
      endif()
      string(SUBSTRING "${_err_lines}" 0 400 _err_short)
      set(_cuda_reason "failed: ${_err_short}")
      if(LAUE_CUDA_MODE STREQUAL "required")
        message(FATAL_ERROR
          "LAUEMATCHING_CUDA=require was set but the CUDA build FAILED.\n${_err}")
      endif()
      message(WARNING
        "laue_index: the CUDA build failed; installing the CPU binary only.\n"
        "The CPU path is unaffected. Run `laue-index doctor` for the diagnosis.\n"
        "${_err_short}")
    endif()
  endif()
endif()

# ── Build manifest ──────────────────────────────────────────────────────────
#
# What was built, by which toolkit, for which architectures -- and if the CUDA
# binaries were NOT built, why not. Without this a `pip install --upgrade` that
# silently drops the GPU binaries is indistinguishable from one that keeps
# them: measured 2026-09-06, an upgrade removed them from two beamline
# environments and nothing in the install output said so.
#
# c_src_sha256 is the hash of the compiled sources, so a restored binary can be
# PROVEN equivalent to what this install would have produced rather than
# assumed to be. `laue-index doctor` uses it.
set(_hash_input "")
file(GLOB _csrc_files "${LAUE_SRC}/*.c" "${LAUE_SRC}/*.h" "${LAUE_SRC}/*.cu")
list(SORT _csrc_files)
foreach(_f IN LISTS _csrc_files)
  file(SHA256 "${_f}" _h)
  get_filename_component(_n "${_f}" NAME)
  string(APPEND _hash_input "${_n}:${_h}\n")
endforeach()
string(SHA256 _csrc_hash "${_hash_input}")

string(TIMESTAMP _now "%Y-%m-%dT%H:%M:%SZ" UTC)
cmake_host_system_information(RESULT _host QUERY HOSTNAME)
set(_ver "${SKBUILD_PROJECT_VERSION}")
if(NOT _ver)
  set(_ver "unknown")
endif()
if(_cuda_built)
  set(_cuda_built_json "true")
else()
  set(_cuda_built_json "false")
endif()
string(REPLACE "\\" "/" _cuda_reason "${_cuda_reason}")
string(REPLACE "\"" "'" _cuda_reason "${_cuda_reason}")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json"
"{
  \"schema\": 1,
  \"version\": \"${_ver}\",
  \"built_at\": \"${_now}\",
  \"build_host\": \"${_host}\",
  \"c_src_sha256\": \"${_csrc_hash}\",
  \"cpu\": {\"built\": true, \"binary\": \"LaueMatchingCPU\"},
  \"cuda\": {
    \"mode\": \"${LAUE_CUDA_MODE}\",
    \"built\": ${_cuda_built_json},
    \"reason\": \"${_cuda_reason}\",
    \"architectures\": \"${_cuda_archs}\",
    \"nvcc\": \"${_cuda_nvcc}\",
    \"nvcc_version\": \"${_cuda_nvcc_version}\"
  }
}
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json"
        DESTINATION laue_index)
