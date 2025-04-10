# Script to check if 100MilOrients.bin exists and download if not

# Check both in the binary directory and in the parent directory
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/100MilOrients.bin")
  message(STATUS "Orientation file already exists in build directory.")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/100MilOrients.bin")
  message(STATUS "Orientation file found in source directory. Copying to build directory...")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/100MilOrients.bin" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
else()
  message(STATUS "Orientation file not found. Downloading orientation file (~7GB, might take long)...")
  file(DOWNLOAD
    "https://anl.box.com/shared/static/qhao454ub2nh5t89zymj1bhlxw1q4obu.bin"
    "${CMAKE_CURRENT_BINARY_DIR}/100MilOrients.bin"
    SHOW_PROGRESS
    STATUS download_status
  )
  
  list(GET download_status 0 status_code)
  if(NOT status_code EQUAL 0)
    list(GET download_status 1 error_message)
    message(FATAL_ERROR "Error downloading orientation file: ${error_message}")
  else()
    message(STATUS "Orientation file downloaded successfully.")
  endif()
endif()