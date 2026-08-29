# laue_cuda_default_architectures(<out_var>)
#
# Pick CUDA architectures that RUN EVERYWHERE, by asking the toolkit which ones
# it supports rather than hardcoding a list.
#
# One module, included by both builds: the repo-root CMakeLists reaches down
# into this package, and the package's own CMakeLists includes it from here, so
# it ships in the sdist. The reverse (package reaching up to the repo root) is
# the trap that once published a laue-index with no binary at all.
#
# WHY NOT THE LOCAL GPU. An earlier version asked `nvidia-smi` and compiled for
# the card in the build machine. That is wrong wherever the binary outlives the
# machine, which here is everywhere: /home/beams is shared, so a binary built
# on a host with an A100 (sm_80) was being handed to hosts with H200 (sm_90) and
# Blackwell (sm_120) cards, where it cannot launch at all. Verified: the 0.3.0
# sdist installed on such a host produced an sm_80-only binary with no PTX.
#
# WHY NOT `all-major` OR `all` ALONE. Measured on CUDA 13.3, neither emits PTX:
#   all-major -> sm_75 80 90 100 110 120                (6 cubins, no PTX)
#   all       -> sm_75 80 86 87 88 89 90 100 103 120 121 (12 cubins, no PTX)
# Without PTX there is no JIT path, so a card NEWER than the build toolkit
# cannot run the binary at all. This builds real code for every supported
# architecture and virtual (PTX) for the newest, which is the JIT fallback.
#
# Override at any time with -DCMAKE_CUDA_ARCHITECTURES=... (e.g. `native` for a
# fast local build, or an explicit list).

function(laue_cuda_default_architectures out_var)
  set(_archs "")

  # `nvcc --list-gpu-arch` is the toolkit's own answer, so a dropped
  # architecture (CUDA 13 removed Volta) can never end up in the list.
  #
  # Find nvcc ourselves when CMAKE_CUDA_ARCHITECTURES has to be decided BEFORE
  # project(... CUDA) -- which is the repo-root build's situation, where
  # CMAKE_CUDA_COMPILER is not set yet.
  # NOTE the two different names. find_program writes a CACHE entry, and a
  # normal variable of the same name shadows it -- so reusing `_nvcc` here left
  # the result invisible and silently took the fallback in exactly the build
  # that needed the lookup (the repo root, where CMAKE_CUDA_COMPILER is unset).
  set(_nvcc "${CMAKE_CUDA_COMPILER}")
  if(NOT _nvcc)
    find_program(LAUE_NVCC_EXECUTABLE NAMES nvcc HINTS ENV CUDACXX ENV CUDA_PATH
                 PATH_SUFFIXES bin PATHS /usr/local/cuda /opt/cuda)
    if(LAUE_NVCC_EXECUTABLE)
      set(_nvcc "${LAUE_NVCC_EXECUTABLE}")
    endif()
  endif()

  if(_nvcc)
    execute_process(
      COMMAND "${_nvcc}" --list-gpu-arch
      OUTPUT_VARIABLE _raw RESULT_VARIABLE _rc
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_rc EQUAL 0 AND _raw)
      string(REGEX MATCHALL "compute_([0-9]+)" _matches "${_raw}")
      set(_nums "")
      foreach(_m IN LISTS _matches)
        string(REGEX REPLACE "compute_" "" _n "${_m}")
        list(APPEND _nums "${_n}")
      endforeach()
      if(_nums)
        list(REMOVE_DUPLICATES _nums)
        list(SORT _nums COMPARE NATURAL)
        list(GET _nums -1 _highest)
        foreach(_n IN LISTS _nums)
          list(APPEND _archs "${_n}-real")
        endforeach()
        # PTX for the newest: lets a future card JIT rather than fail to launch.
        list(APPEND _archs "${_highest}-virtual")
      endif()
    endif()
  endif()

  if(NOT _archs)
    # No usable answer from the toolkit. `all-major` still covers every
    # architecture that toolkit supports (cubins are compatible within a major
    # family); it just has no forward-compatibility PTX.
    if(NOT CMAKE_VERSION VERSION_LESS 3.23)
      set(_archs "all-major")
    else()
      set(_archs "80-real;86-real;89-real;90-real;90-virtual")
    endif()
  endif()

  set(${out_var} "${_archs}" PARENT_SCOPE)
endfunction()
