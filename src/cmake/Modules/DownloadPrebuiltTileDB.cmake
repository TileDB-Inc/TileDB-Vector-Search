#
# FindTileDB_EP.cmake
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

include(FetchContent)

function(fetch_tiledb_release_list VERSION)
        # Local constants
        set(UPSTREAM_URL "https://github.com/TileDB-Inc/TileDB/releases/download")
        list(LENGTH ARGV COUNT)
        if (${COUNT} GREATER 1)
                list(GET ARGV 1 EXPECTED_HASH)
        endif()

        if(NOT VERSION)
                set(VERSION latest)
        endif()

        if(EXPECTED_HASH)
                file(DOWNLOAD
                        ${UPSTREAM_URL}/${VERSION}/releases.csv
                        ${CMAKE_CURRENT_BINARY_DIR}/releases.csv
                        SHOW_PROGRESS
                        EXPECTED_HASH ${EXPECTED_HASH}
                )
        else()
                message(WARNING "Downloading release list without SHA checksum!")
                file(DOWNLOAD
                        ${UPSTREAM_URL}/${VERSION}/releases.csv
                        ${CMAKE_CURRENT_BINARY_DIR}/releases.csv
                        SHOW_PROGRESS
                )
        endif()

        file(STRINGS
                ${CMAKE_CURRENT_BINARY_DIR}/releases.csv
                RELLIST
        )

        # Remove csv table headers
        list(POP_FRONT RELLIST)

        foreach(LINE ${RELLIST})
                string(REPLACE "," ";" LINE ${LINE})
                list(LENGTH LINE LENGTH)

                list(GET LINE 0 PLATFORM)
                list(GET LINE 1 URL)
                list(GET LINE 2 SHA)

                set(RELEASE_VAR TILEDB_${PLATFORM})
                set(URL_${RELEASE_VAR} ${URL} PARENT_SCOPE)
                set(HASH_${RELEASE_VAR} ${SHA} PARENT_SCOPE)
        endforeach()
endfunction()

function(detect_artifact_name OUT_VAR)
        if (WIN32) # Windows
                SET(${OUT_VAR} TILEDB_WINDOWS-X86_64 PARENT_SCOPE)
        elseif(APPLE) # OSX
                if (DEFINED CMAKE_OSX_ARCHITECTURES)
                        set(ACTUAL_TARGET ${CMAKE_OSX_ARCHITECTURES})
                else()
                        set(ACTUAL_TARGET ${CMAKE_SYSTEM_PROCESSOR})
                endif()


                if (ACTUAL_TARGET MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
                        SET(${OUT_VAR} TILEDB_MACOS-X86_64 PARENT_SCOPE)
                elseif (ACTUAL_TARGET STREQUAL arm64 OR ACTUAL_TARGET MATCHES "^aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
                        SET(${OUT_VAR} TILEDB_MACOS-ARM64 PARENT_SCOPE)
                endif()
        else() # Linux
                SET(${OUT_VAR} TILEDB_LINUX-X86_64 PARENT_SCOPE)
        endif()
endfunction()

function(fetch_prebuilt_tiledb)
        # Arguments
        set(oneValueArgs VERSION ARTIFACT_NAME RELLIST_HASH)
        set(multiValueArgs)
        cmake_parse_arguments(
                FETCH_PREBUILT_TILEDB
                "${options}"
                "${oneValueArgs}"
                "${multiValueArgs}"
                ${ARGN}
        )

        fetch_tiledb_release_list(${FETCH_PREBUILT_TILEDB_VERSION} ${FETCH_PREBUILT_TILEDB_RELLIST_HASH})

        if(NOT FETCH_PREBUILT_TILEDB_ARTIFACT_NAME)
                detect_artifact_name(FETCH_PREBUILT_TILEDB_ARTIFACT_NAME)
        endif()

        string(STRIP ${HASH_${FETCH_PREBUILT_TILEDB_ARTIFACT_NAME}} HASH_${FETCH_PREBUILT_TILEDB_ARTIFACT_NAME})
        ExternalProject_Add(ep_tiledb
                PREFIX "externals"
                URL ${URL_${FETCH_PREBUILT_TILEDB_ARTIFACT_NAME}}
                URL_HASH SHA256=${HASH_${FETCH_PREBUILT_TILEDB_ARTIFACT_NAME}}
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                UPDATE_COMMAND ""
                PATCH_COMMAND ""
                TEST_COMMAND ""
                INSTALL_COMMAND
                ${CMAKE_COMMAND} -E copy_directory ${EP_BASE}/src/ep_tiledb ${EP_INSTALL_PREFIX}
                LOG_DOWNLOAD TRUE
                LOG_CONFIGURE FALSE
                LOG_BUILD FALSE
                LOG_INSTALL FALSE
        )
endfunction()

function(fetch_source_tiledb)
        # Arguments
        set(oneValueArgs VERSION ARTIFACT_NAME RELLIST_HASH)
        set(multiValueArgs)
        cmake_parse_arguments(
                FETCH_PREBUILT_TILEDB
                "${options}"
                "${oneValueArgs}"
                "${multiValueArgs}"
                ${ARGN}
        )

        fetch_tiledb_release_list(${FETCH_PREBUILT_TILEDB_VERSION} ${FETCH_PREBUILT_TILEDB_RELLIST_HASH})

        string(STRIP ${HASH_TILEDB_source} HASH_TILEDB_source)
        ExternalProject_Add(ep_tiledb
                PREFIX "externals"
                URL ${URL_TILEDB_source}
                URL_HASH SHA256=${HASH_TILEDB_source}
                DOWNLOAD_NAME "tiledb.tar.gz"
                CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=${EP_INSTALL_PREFIX}
                -DCMAKE_PREFIX_PATH=${EP_INSTALL_PREFIX}
                -DTILEDB_S3=${TILEDB_S3}
                -DTILEDB_SKIP_S3AWSSDK_DIR_LENGTH_CHECK=ON # for windows build
                -DTILEDB_VERBOSE=ON
                -DTILEDB_SERIALIZATION=ON
                -DTILEDB_TESTS=OFF
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DTILEDB_WERROR=OFF #avoid the pointer p use after free issue...
                UPDATE_COMMAND ""
                INSTALL_COMMAND
                ${CMAKE_COMMAND} --build . --target install-tiledb
                LOG_DOWNLOAD TRUE
                LOG_CONFIGURE TRUE
                LOG_BUILD TRUE
                LOG_INSTALL TRUE
        )
endfunction()
