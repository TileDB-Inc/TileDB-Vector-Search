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

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)
include(FetchContent)

find_package(TileDB REQUIRED)

# Currently, Scikit-build does not support FindPython, so we convert the
# provided hints ourselves.
# Otherwise, the extension will use system Python instead of cibuildwheel's venv
if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARY "${PYTHON_LIBRARY}")
endif()

set(Python_FIND_VIRTUALENV FIRST)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

#set(VSPY_TARGET_NAME _vector_search)
set(VSPY_TARGET_NAME _tiledbvspy)

python_add_library(${VSPY_TARGET_NAME} MODULE "src/tiledb/vector_search/module.cc" WITH_SOABI)

target_link_libraries(${VSPY_TARGET_NAME}
  PRIVATE
    pybind11::headers
    mdspan
    kmeans_lib
    TileDB::tiledb_shared)

target_compile_definitions(${VSPY_TARGET_NAME} PRIVATE TILEDBVS_ENABLE_STATS VERSION_INFO=${PROJECT_VERSION})

if (APPLE)
  set_target_properties(${VSPY_TARGET_NAME} PROPERTIES INSTALL_RPATH "@loader_path/lib")
elseif(UNIX)
  set_target_properties(${VSPY_TARGET_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
endif()

# Install the extension module
install(TARGETS ${VSPY_TARGET_NAME} DESTINATION .)

# Install libtiledb so that we can repackage it with the extension
install(IMPORTED_RUNTIME_ARTIFACTS TileDB::tiledb_shared DESTINATION lib)

# Quiet a warning, since this project is only valid with SKBUILD
set(ignoreMe "${SKBUILD}")
