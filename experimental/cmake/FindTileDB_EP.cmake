#
# FindTileDB_EP.cmake
#
#
# The MIT License
#
# Copyright (c) 2022 TileDB, Inc.
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
# Finds the TileDB library, downloading it with FetchContent is necessary.

include(FetchContent)

# Try to find TileDB from the system
find_package(TileDB CONFIG)

if (TILEDB_FOUND)
    if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
        get_target_property(TILEDB_LIB TileDB::tiledb_shared IMPORTED_LOCATION_DEBUG)
    else()
        get_target_property(TILEDB_LIB TileDB::tiledb_shared IMPORTED_LOCATION_RELEASE)
    endif()
  message(STATUS "Found TileDB: ${TILEDB_LIB}")
else()
  message(STATUS "Adding TileDB as an external project")

  # Try to download prebuilt artifacts unless the user specifies to build from source
  if (WIN32) # Windows
    SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.15.1/tiledb-windows-x86_64-2.15.1-432d4c2.zip")
    SET(DOWNLOAD_SHA1 "7235a3bc0b5675ed83762a4290d99c6ff9db285f")
  elseif(APPLE) # macOS
    if (CMAKE_OSX_ARCHITECTURES STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
      SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.15.1/tiledb-macos-x86_64-2.15.1-432d4c2.tar.gz")
      SET(DOWNLOAD_SHA1 "f5986f5a85912147e2b839e0968963df6c540688")
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL arm64 OR CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
      SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.15.1/tiledb-macos-arm64-2.15.1-432d4c2.tar.gz")
      SET(DOWNLOAD_SHA1 "6abb30f9dd7371d5a06a273008179bd57d691e36")
    endif()
  else() # Linux
    SET(DOWNLOAD_URL "https://github.com/TileDB-Inc/TileDB/releases/download/2.15.1/tiledb-linux-x86_64-2.15.1-432d4c2.tar.gz")
    SET(DOWNLOAD_SHA1 "9b705af26007b193800f0382c343d421a8b59003")
  endif()

  FetchContent_Declare(
    ep_tiledb
    URL ${DOWNLOAD_URL}
    URL_HASH SHA1=${DOWNLOAD_SHA1}
  )
  FetchContent_GetProperties(ep_tiledb)
  if(NOT ep_tiledb_POPULATED)
    FetchContent_Populate(ep_tiledb)
  endif()
  set(TileDB_DIR ${ep_tiledb_SOURCE_DIR}/lib/cmake/TileDB)
  find_package(TileDB CONFIG REQUIRED)
endif()
