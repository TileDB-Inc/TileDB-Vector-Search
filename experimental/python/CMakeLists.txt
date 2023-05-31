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

cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)

project(TileDB-VS-Py)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ----------------------------------------------------------------------------
# Build scenarios:
# 1. add_subdirectory from parent tree CMake build
#   -> build python extension only, no package install
# 2. Run python build (ie `pip install`) within this directory
#   -> do a relative include for the parent tree to get libtiledbvs
# 3. (TODO) Run python build from a *python source distribution*
#   -> use FetchContent to get parent library

# This option forces FetchContent as in scenario (3) above.
option(TILEDB_VS_PYTHON_STANDALONE "Build TileDB-VS-Py using FetchContent to retrieve the TileDB-VS library" OFF)

if (TILEDB_VS_PYTHON_STANDALONE)
  # TODO -- include(BuildTileDBVS_FetchContent)
elseif(NOT DEFINED TILEDB_VS_PYTHON)
  # TODO -- If we are building standalone, then include the top-level project
  add_subdirectory("../" "${CMAKE_BINARY_DIR}/tiledb_vs")
endif()
# ----------------------------------------------------------------------------

find_package(TileDB REQUIRED)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory("tiledb/vector_search")