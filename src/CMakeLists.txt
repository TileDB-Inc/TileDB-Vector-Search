#
# CMakeLists.txt
#
#
# The MIT License
#
# Copyright (c) 2023 TileDB, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

cmake_minimum_required(VERSION 3.22)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# The cmake options don't seem to really work
if ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR "In-source builds are disabled.
    Please create a subfolder and use `cmake ..` inside it.
    IMPORTANT: cmake will still have created CMakeCache.txt and CMakeFiles/*.
               You must delete them, or cmake will refuse to work.")
endif() # yes, end-markers and even else() need empty parens

# -----------------------------------------------------------------------------
# Build options and defaults
# -----------------------------------------------------------------------------

option(SUPERBUILD "If true, perform a superbuild (builds all missing dependencies)." ON)
option(CMAKE_IDE "(Used for CLion builds). Disables superbuild and sets the EP install dir." OFF)
option(FORCE_EXTERNAL_TILEDB "Forces a local build of TileDB instead of searching system paths." OFF)
option(DOWNLOAD_TILEDB_PREBUILT "If tiledb is being super built, this controls downloading prebuilt artifacts or building from source" ON)
option(TILEDB_S3 "Enables S3/minio support using aws-cpp-sdk" ON)
option(OVERRIDE_INSTALL_PREFIX "Ignores the setting of CMAKE_INSTALL_PREFIX and sets a default prefix" OFF)
option(USE_MKL_CBLAS "Try to use mkl_cblas.h (requires additional CMake configuration)" OFF)
option(TILEDB_VS_PYTHON "Build Python wrapping of C++ API" OFF)
option(TILEDB_VS_ENABLE_BLAS "Build components and demos which require BLAS" OFF)
option(BUILD_TESTS "Build the tests" ON)
option(BUILD_CLI "Build the index and query command-line interface programs" OFF)

set(CMAKE_CXX_STANDARD 20)

if (TILEDB_PATH)
    set(CMAKE_PREFIX_PATH  "${TILEDB_PATH};${CMAKE_PREFIX_PATH}")
    message(STATUS "Adding TILEDB_PATH to CMAKE_PREFIX_PATH")
endif()


# -----------------------------------------------------------------------------
# Extend the module path so we can find our custom modules
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Search the externals install directory for dependencies.
list(APPEND CMAKE_PREFIX_PATH "${EP_INSTALL_PREFIX}")

# If this is an in-IDE build, we need to disable the superbuild and explicitly
# set the EP base dir. The normal 'cmake && make' process won't need this step,
# it is for better CLion support of the superbuild architecture.
if (CMAKE_IDE)
  set(SUPERBUILD OFF)
  set(EP_BASE "${CMAKE_CURRENT_BINARY_DIR}/externals")
endif()

if (SUPERBUILD)
  project(TileDB-Vectorsearch-Superbuild)
  message(STATUS "Starting TileDB-Vectorsearch superbuild.")
  include("cmake/Superbuild.cmake")
  return()
endif()

project(TileDB-Vectorsearch)
message(STATUS "Starting TileDB-Vectorsearch regular build.")

# Paths to locate the installed external projects.
set(EP_SOURCE_DIR "${EP_BASE}/src")
set(EP_INSTALL_PREFIX "${EP_BASE}/install")

############################################################
# Regular build
############################################################

# Enable testing
enable_testing()

# -----------------------------------------------------------------------------
# Build types and default flags
# -----------------------------------------------------------------------------

message("Architecture is ${CMAKE_SYSTEM_PROCESSOR}")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(FCONCEPTS_DIAGNOSTICS_DEPTH "-fconcepts-diagnostics-depth=3")
endif()

# Control compiler-specific flags.
include(CompilerFlags)

if (MSVC)
    add_compile_options(/bigobj)
    add_compile_definitions("_ITERATOR_DEBUG_LEVEL=0")
endif()

if (NOT $ENV{CIBUILDWHEEL} EQUAL 1 AND NOT $ENV{CONDA_BUILD} EQUAL 1)
  if(MSVC)
      set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /EHsc /RTC1" CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_RELEASE "/O2" CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Zi" CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_MINSIZEREL "/O1" CACHE STRING "" FORCE)
      add_compile_definitions("$<$<NOT:$<CONFIG:Debug>>:NDEBUG>")
  elseif (CMAKE_OSX_ARCHITECTURES STREQUAL arm64 OR CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
      set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-elide-constructors ${FCONCEPTS_DIAGNOSTICS_DEPTH} " CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -g -DNDEBUG " CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -UNDEBUG" CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG " CACHE STRING "" FORCE)
  else()
      set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-elide-constructors ${FCONCEPTS_DIAGNOSTICS_DEPTH} " CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -g -march=native -DNDEBUG " CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -march=native -UNDEBUG" CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -march=native -DNDEBUG " CACHE STRING "" FORCE)
  endif()
endif()

# Default to Release build
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

# -----------------------------------------------------------------------------
# Interface library for multi-threading
# -----------------------------------------------------------------------------
# Control compiler-specific flags.
include(MultiThreading)

# -----------------------------------------------------------------------------
# Interface library for BLAS
# -----------------------------------------------------------------------------
# Control compiler-specific flags.
if (TILEDB_VS_ENABLE_BLAS)
  include(BLAS)
endif()

if (DEFINED TileDB_DIR)
    set(CMAKE_PREFIX_PATH "${TileDB_DIR};${CMAKE_PREFIX_PATH}")
    message(STATUS "TileDB_DIR is set to ${TileDB_DIR} -- find_package will search there first.")
else()
    message(STATUS "TileDB_DIR is not set -- find_package will use default search path.")
endif()

find_package(TileDB_EP REQUIRED)

if (TileDB_EP_FOUND)
    message(STATUS "Found TileDB")
    get_target_property(_TDB_INCLUDES TileDB::tiledb_shared INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "TileDB include directories are ${_TDB_INCLUDES}")
endif()

include(Docopt)
include(mdspan)
include(nlohmann_json)


# -----------------------------------------------------------------------------
# Config header
# -----------------------------------------------------------------------------
# Get git info, etc
include (logging)

# -----------------------------------------------------------------------------
# Interface libraries
# -----------------------------------------------------------------------------
add_library(kmeans_lib INTERFACE)
add_library(kmeans_linalg INTERFACE)
if (TILEDB_VS_ENABLE_BLAS)
    target_link_Libraries(kmeans_linalg INTERFACE
            flat_blas)
endif()
add_library(kmeans_queries INTERFACE)
target_include_directories(kmeans_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(kmeans_lib INTERFACE
        kmeans_linalg
        kmeans_queries
)
target_link_libraries(kmeans_lib INTERFACE
        nlohmann_json::nlohmann_json
        docopt
        flat_mt
        TileDB::tiledb_shared
        mdspan
)


# -----------------------------------------------------------------------------
# Command-line interface programs
# -----------------------------------------------------------------------------
if (BUILD_CLI)
    add_subdirectory(src)
endif()

if(BUILD_TESTS)
    # include(Catch2)
    add_subdirectory(include/test)
    # enable_testing()
endif()

if (TILEDB_VS_PYTHON)
    # HACK: make this work properly
    add_subdirectory(../apis/python ${CMAKE_CURRENT_BINARY_DIR}/python)
endif()
