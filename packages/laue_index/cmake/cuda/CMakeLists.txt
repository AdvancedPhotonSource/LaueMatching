# Standalone CUDA sub-project for LaueMatchingGPU / LaueMatchingGPUStream.
#
# WHY THIS IS A SEPARATE PROJECT, not a section of the parent CMakeLists.
# Once a CUDA target is added to a project, CMake cannot try-and-continue: a
# toolkit that cannot compile these sources -- a host-compiler mismatch, an
# architecture the toolkit dropped -- kills the whole configure/build and takes
# the working CPU binary down with it. That is why the CUDA build used to be
# opt-in.
#
# Configured and built by the parent through execute_process(), in its own
# process, so a failure here is a non-zero exit code the parent catches and
# reports. The CPU binary cannot be affected, by construction. That is what
# makes attempting CUDA BY DEFAULT safe.
#
# Inputs from the parent, all required:
#   LAUE_SRC              the c_src directory (the only copy in the repo)
#   CMAKE_INSTALL_PREFIX  staging dir the parent installs FROM
cmake_minimum_required(VERSION 3.20)
project(laue_index_cuda LANGUAGES NONE)

if(NOT LAUE_SRC OR NOT EXISTS "${LAUE_SRC}/LaueMatchingGPU.cu")
  message(FATAL_ERROR "LAUE_SRC must point at c_src (got '${LAUE_SRC}')")
endif()

enable_language(C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP COMPONENTS C REQUIRED)

# Architectures BEFORE enable_language(CUDA), which compiles a test program with
# them. Never a hardcoded list -- see cmake/LaueCudaArch.cmake for why, and for
# why it is neither the local GPU nor `all-major`.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../LaueCudaArch.cmake")
  laue_cuda_default_architectures(_laue_archs)
  set(CMAKE_CUDA_ARCHITECTURES "${_laue_archs}")
endif()
message(STATUS "laue_index[cuda]: architectures = ${CMAKE_CUDA_ARCHITECTURES}")

enable_language(CUDA)

# Report what we actually used, so the parent can put it in the build manifest
# and `laue-index doctor` can compare it against the card in front of it.
file(WRITE "${CMAKE_BINARY_DIR}/cuda_report.txt"
     "architectures=${CMAKE_CUDA_ARCHITECTURES}\n"
     "nvcc=${CMAKE_CUDA_COMPILER}\n"
     "nvcc_version=${CMAKE_CUDA_COMPILER_VERSION}\n")

# The .cu wrap LaueMatchingHeaders.h in extern "C", so the vendored simplex --
# compiled here as C -- links into a translation unit nvcc compiles as C++.
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
  # Link the OpenMP runtime BY NAME and let the compiler driver locate it.
  # Two other spellings were tried and both failed, in opposite places:
  #   find_library(NAMES gomp) resolves on RHEL (/usr/lib64) but not on Ubuntu,
  #     where the linker symlink lives inside GCC's own directory -- the CUDA
  #     link then died with undefined GOMP_* and omp_get_wtime;
  #   -Xcompiler=-fopenmp as a link option also reaches the DEVICE link step,
  #     which invokes the host compiler directly: "gcc: error: unrecognized
  #     command-line option '-Xcompiler=-fopenmp'".
  foreach(_omp_lib IN LISTS OpenMP_C_LIB_NAMES)
    target_link_libraries(${_gpu_target} PRIVATE ${_omp_lib})
  endforeach()
  if(NOT OpenMP_C_LIB_NAMES)
    target_link_libraries(${_gpu_target} PRIVATE gomp)
  endif()
  target_link_libraries(${_gpu_target} PRIVATE m)
  install(TARGETS ${_gpu_target} RUNTIME DESTINATION .)
endforeach()

# The streaming daemon serves results over TCP from worker threads.
target_link_libraries(LaueMatchingGPUStream PRIVATE pthread)
