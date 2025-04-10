# FindNLOPT.cmake
# Finds the NLopt library
#
# This module defines:
#  NLOPT_FOUND        - True if NLopt is found
#  NLOPT_INCLUDE_DIRS - The NLopt include directories
#  NLOPT_LIBRARIES    - The NLopt libraries
#  NLOPT_VERSION      - The NLopt version string (if available)
#
# The following variables can be set as arguments:
#  NLOPT_ROOT_DIR     - Root directory to search for NLopt

# Find include directory
find_path(NLOPT_INCLUDE_DIR
  NAMES nlopt.h
  HINTS ${NLOPT_ROOT_DIR}
  PATH_SUFFIXES include
  DOC "NLopt include directory"
)

# Find library
find_library(NLOPT_LIBRARY
  NAMES nlopt nlopt_cxx
  HINTS ${NLOPT_ROOT_DIR}
  PATH_SUFFIXES lib lib64
  DOC "NLopt library"
)

# Try to extract version from header
if(NLOPT_INCLUDE_DIR)
  if(EXISTS "${NLOPT_INCLUDE_DIR}/nlopt.h")
    file(STRINGS "${NLOPT_INCLUDE_DIR}/nlopt.h" nlopt_version_str
      REGEX "^#define[\t ]+NLOPT_VERSION_(MAJOR|MINOR|BUGFIX)[\t ]+[0-9]+.*")

    string(REGEX REPLACE ".*NLOPT_VERSION_MAJOR[\t ]+([0-9]+).*" "\\1"
      NLOPT_VERSION_MAJOR "${nlopt_version_str}")
    string(REGEX REPLACE ".*NLOPT_VERSION_MINOR[\t ]+([0-9]+).*" "\\1"
      NLOPT_VERSION_MINOR "${nlopt_version_str}")
    string(REGEX REPLACE ".*NLOPT_VERSION_BUGFIX[\t ]+([0-9]+).*" "\\1"
      NLOPT_VERSION_PATCH "${nlopt_version_str}")

    if(NLOPT_VERSION_MAJOR AND NLOPT_VERSION_MINOR AND NLOPT_VERSION_PATCH)
      set(NLOPT_VERSION "${NLOPT_VERSION_MAJOR}.${NLOPT_VERSION_MINOR}.${NLOPT_VERSION_PATCH}")
    endif()
  endif()
endif()

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NLOPT
  REQUIRED_VARS NLOPT_LIBRARY NLOPT_INCLUDE_DIR
  VERSION_VAR NLOPT_VERSION
)

# Set the output variables
if(NLOPT_FOUND)
  set(NLOPT_INCLUDE_DIRS ${NLOPT_INCLUDE_DIR})
  set(NLOPT_LIBRARIES ${NLOPT_LIBRARY})
endif()

# Hide internal variables
mark_as_advanced(NLOPT_INCLUDE_DIR NLOPT_LIBRARY)