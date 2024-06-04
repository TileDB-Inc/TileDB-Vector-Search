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

find_package(TileDB REQUIRED)

set(PYBIND11_NEWPYTHON ON)
# NOTE: If you remove the line below the Windows build will fail
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(VSPY_TARGET_NAME _tiledbvspy)

pybind11_add_module(
        ${VSPY_TARGET_NAME}
        src/tiledb/vector_search/module.cc
        src/tiledb/vector_search/kmeans.cc
        src/tiledb/vector_search/type_erased_module.cc
)

target_compile_features(
        ${VSPY_TARGET_NAME}
        PRIVATE
        cxx_std_20
)

target_link_libraries(
        ${VSPY_TARGET_NAME}
        PRIVATE
        mdspan
        kmeans_lib
        TileDB::tiledb_shared
)

target_compile_definitions(
        ${VSPY_TARGET_NAME}
        PRIVATE
        TILEDBVS_ENABLE_STATS
        VERSION_INFO=${PROJECT_VERSION}
)

# Install the extension module
install(TARGETS ${VSPY_TARGET_NAME} LIBRARY DESTINATION .)

if (TILEDB_DOWNLOADED)
  message(STATUS "Adding libtiledb into install group")
  # Install libtiledb so that we can repackage it with the extension
  install(IMPORTED_RUNTIME_ARTIFACTS TileDB::tiledb_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})

  if (APPLE)
    set_target_properties(${VSPY_TARGET_NAME} PROPERTIES INSTALL_RPATH "@loader_path/lib")
  elseif(UNIX)
    set_target_properties(${VSPY_TARGET_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
  endif()
else()
  # If using external TileDB core library force it to be linked at runtime using RPATH
  get_property(TILEDB_LOCATION TARGET TileDB::tiledb_shared PROPERTY LOCATION)
  get_filename_component(TILEDB_LOCATION ${TILEDB_LOCATION} DIRECTORY)
  message(STATUS "Setting RPATH to ${TILEDB_LOCATION}")
  set_target_properties(${VSPY_TARGET_NAME} PROPERTIES INSTALL_RPATH ${TILEDB_LOCATION})
endif()
