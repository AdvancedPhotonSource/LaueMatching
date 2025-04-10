# CMake Guide for LaueMatching Developers

This guide covers how to extend and modify the CMake build system for the LaueMatching project.

## CMake Structure

The build system is organized in the following way:

- `CMakeLists.txt` - Main CMake configuration file
- `laue_matching_lib/CMakeLists.txt` - Library CMake configuration
- `cmake/` - Directory with CMake modules (like FindNLOPT.cmake)
- `build/` - Generated build directory (not in version control)

## Adding New Source Files

If you add new C or CUDA source files to the project, you need to update the appropriate CMakeLists.txt file:

### For Main Executables

Add new source files to the LaueMatchingCPU target in the main CMakeLists.txt:

```cmake
# Original line
add_executable(LaueMatchingCPU src/LaueMatchingCPU.c)

# Modified to include new files
add_executable(LaueMatchingCPU 
    src/LaueMatchingCPU.c
    src/your_new_file.c
    src/another_file.c
)
```

For GPU executable:

```cmake
# Original line
cuda_add_executable(LaueMatchingGPU src/LaueMatchingGPU.cu)

# Modified to include new files
cuda_add_executable(LaueMatchingGPU 
    src/LaueMatchingGPU.cu
    src/your_new_file.cu
    src/another_file.cu
)
```

### For Library Source Files

Add new source files to the library in laue_matching_lib/CMakeLists.txt:

1. Place your new C files in the `laue_matching_lib/src/` directory:
   ```
   laue_matching_lib/src/your_new_lib_file.c
   ```

2. The build system uses `file(GLOB_RECURSE SOURCES "src/*.c")` to automatically find all source files, so new files will be included automatically.

3. If you prefer explicit listing, you can modify the CMakeLists.txt:
   ```cmake
   # Replace the GLOB_RECURSE with explicit listing
   set(SOURCES
       src/file1.c
       src/file2.c
       src/your_new_lib_file.c
   )
   ```

### For Library Headers

Place new public header files in the `laue_matching_lib/include/` directory:
```
laue_matching_lib/include/your_new_header.h
```

## Adding New Dependencies

If your changes require new external libraries, you need to:

1. Find the package
2. Include its headers
3. Link against its libraries

Example for adding a new dependency:

```cmake
# Find the package
find_package(NewDependency REQUIRED)

# Include headers
include_directories(${NEWDEPENDENCY_INCLUDE_DIRS})

# Link against libraries (in main CMakeLists.txt)
target_link_libraries(LaueMatchingCPU 
    ${NLOPT_LIBRARIES}
    ${NEWDEPENDENCY_LIBRARIES}
    m
    dl
    OpenMP::OpenMP_C
)

# And for the library in laue_matching_lib/CMakeLists.txt
target_link_libraries(laue_matching 
    PRIVATE 
    OpenMP::OpenMP_C 
    ${NLOPT_LIBRARIES} 
    ${NEWDEPENDENCY_LIBRARIES}
    m
)
```

## Adding New Build Options

To add a new configurable option:

```cmake
# Add the option with a default value
option(NEW_FEATURE "Enable the new feature" OFF)

# Use the option in your code
if(NEW_FEATURE)
    add_definitions(-DHAS_NEW_FEATURE)
    # Additional setup for the feature
endif()
```

## Debugging the Build System

If you encounter issues with CMake:

1. Enable verbose output during compilation:
   ```
   make VERBOSE=1
   ```

2. Print CMake variables for debugging:
   ```cmake
   message(STATUS "NLOPT_LIBRARIES: ${NLOPT_LIBRARIES}")
   message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
   ```

3. Clear the CMake cache if you make significant changes:
   ```
   rm -rf build
   mkdir build
   cd build
   cmake ..
   ```

## Working with the Orientation File

The orientation file (100MilOrients.bin) is handled automatically:

1. If the file exists in the build directory, it will be used
2. If the file exists in the source directory, it will be copied to the build directory
3. If the file doesn't exist in either location, it will be downloaded

You can also manually run:
```
make download_orientation_file
```

## Cross-Platform Considerations

The LaueMatching project primarily targets Linux, but some parts can work on macOS:

```cmake
if(APPLE)
    # macOS specific configuration
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -framework Accelerate")
    # Disable CUDA on macOS as it's not supported
    set(USE_CUDA OFF)
elseif(UNIX AND NOT APPLE)
    # Linux specific configuration
endif()
```

## Best Practices

1. Keep the build system modular and clean
2. Document all options and non-obvious choices
3. Test your changes on multiple platforms if possible
4. Use CMake's built-in features rather than custom scripts when possible
5. Keep compatibility with older CMake versions (3.15+)

## Extending the Installation Process

To add new files to the installation:

```cmake
install(FILES 
    your_new_file.txt
    another_file.py
    DESTINATION .
)

install(DIRECTORY data/
    DESTINATION data
    FILES_MATCHING PATTERN "*.txt"
)
```

## Additional Resources

- [CMake Documentation](https://cmake.org/documentation/)
- [Modern CMake Guide](https://cliutils.gitlab.io/modern-cmake/)
- [CMake Cookbook](https://github.com/dev-cafe/cmake-cookbook)