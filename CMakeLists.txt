cmake_minimum_required(VERSION 3.10)

project(kharm CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

# PREREQUISITES
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# SOURCES
### kokkos: Performance Portability
add_subdirectory(external/kokkos)
# pass Kokkos include directories to our target application
include_directories(${Kokkos_INCLUDE_DIRS_RET})

### HighFive: HDF5 for C++
# We don't need to compile this, it's header-only
#add_subdirectory(external/HighFive)
# include dirs
include_directories(${PROJECT_SOURCE_DIR}/external/HighFive/include)

### K/HARM source
add_subdirectory(kharm)

##################### PRINT CONFIGURE STATUS ######################
message("===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                         "${CMAKE_CXX_COMPILER_VERSION} " 
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("                 ${CMAKE_CXX_COMPILER}")
message("  Kokkos OpenMP enabled : ${KOKKOS_ENABLE_OPENMP}")
message("  Kokkos CUDA   enabled : ${KOKKOS_ENABLE_CUDA}")
if (Kokkos_ENABLE_CUDA)
  message("  Kokkos CUDA   Lambda  : ${KOKKOS_ENABLE_CUDA_LAMBDA}")
  message("  Kokkos CUDA   flags   : ${KOKKOS_CUDA_OPTIONS}")
endif(Kokkos_ENABLE_CUDA)
message("  Kokkos HWLOC  enabled : ${KOKKOS_ENABLE_HWLOC}")
message("===================================================")
