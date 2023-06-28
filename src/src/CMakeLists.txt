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

# Get the current date and time
string(TIMESTAMP CURRENT_DATETIME "%Y-%m-%d %H:%M:%S")

# Execute "git rev-parse --abbrev-ref HEAD" command to get the current branch name
execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the Git repository URL
execute_process(
        COMMAND git config --get remote.origin.url
        OUTPUT_VARIABLE GIT_REPO_URL
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the Git repository name
execute_process(
        COMMAND git rev-parse --show-toplevel
        OUTPUT_VARIABLE GIT_REPO_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
get_filename_component(GIT_REPO_NAME ${GIT_REPO_DIR} NAME)

# Get the short commit name
execute_process(
        COMMAND git rev-parse --short HEAD
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

execute_process(
        COMMAND git show -s --format=%cd --date=short HEAD
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

execute_process(
        COMMAND git log -1 --format=%cd --date=format:%H:%M:%S
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_TIME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

execute_process(
    COMMAND git describe --tags --abbrev=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

get_filename_component(IVF_HACK_CXX_COMPILER ${CMAKE_CXX_COMPILER} NAME)

configure_file(../include/config.h.in config.h)

#---[ Dependencies ]------------------------------------------------------------
add_library(kmeans_linalg INTERFACE)
target_sources(kmeans_linalg INTERFACE
        ../include/linalg.h ../include/detail/linalg/tdb_matrix.h ../include/detail/linalg/tdb_partitioned_matrix.h ../include/detail/linalg/matrix.h
        ../include/detail/linalg/vector.h ../include/detail/linalg/linalg_defs.h
        ../include/detail/linalg/tdb_io.h
        )

add_library(kmeans_queries INTERFACE)
target_sources(kmeans_queries INTERFACE
        ../include/detail/flat/qv.h ../include/detail/flat/vq.h ../include/detail/flat/gemm.h
        ../include/detail/ivf/qv.h ../include/detail/ivf/vq.h ../include/detail/ivf/gemm.h
        )

add_library(kmeans_lib INTERFACE)
target_sources(kmeans_lib INTERFACE
        ../include/flat_query.h ../include/ivf_query.h ../include/scoring.h ../include/utils/fixed_min_queues.h
        ../include/defs.h ../include/algorithm.h ../include/concepts.h ../include/stats.h
        ../include/utils/timer.h ../include/utils/logging.h ../include/utils/print_types.h
        ../include/flat_index.h ../include/ivf_index.h
        )

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
        flat_blas
        TileDB::tiledb_shared
        mdspan
)


#---[ Demos ]------------------------------------------------------------
add_executable(flat_l2 flat_l2.cc)
target_link_libraries(flat_l2 PUBLIC kmeans_lib nlohmann_json::nlohmann_json)
target_compile_definitions(flat_l2 PUBLIC TILEDBVS_ENABLE_STATS)

add_executable(ivf_flat ivf_flat.cc)
target_link_libraries(ivf_flat PUBLIC kmeans_lib)
target_compile_definitions(ivf_flat PUBLIC TILEDBVS_ENABLE_STATS)

add_executable(index index.cc)
target_link_libraries(index PUBLIC kmeans_lib)
target_compile_definitions(index PUBLIC TILEDBVS_ENABLE_STATS)

#---[ TBD ]------------------------------------------------------------
if (FALSE)

add_executable(ingest ingest.cc)
target_link_libraries(ingest PUBLIC kmeans_lib)

add_executable(kmeans kmeans.cc)
target_link_libraries(kmeans PUBLIC kmeans_lib)
endif()

# Quick and dirty program to assess latency of one-byte array access
# add_executable(latency latency.cc sift_array.h timer.h)

# add_executable(ingest ingest.cc sift_db.h timer.h)
# target_include_directories(ingest PUBLIC
#     /Users/lums/Contrib/dist/include
# )
# target_link_libraries(ingest PUBLIC docopt TileDB::tiledb_shared)

#
# Some debugging code
#
# Get the compile options for flat
# get_target_property(TargetCompileOptions flat COMPILE_OPTIONS)

# Print the compile options
# message("Compile options for flat:")
# foreach(Option ${TargetCompileOptions})
#     message("  ${Option}")
# endforeach()

# message (STATUS ${MKL_THREAD_LIB})
# message (STATUS ${INTERFACE_COMPILE_OPTIONS})
# message (STATUS ${INTERFACE_INCLUDE_DIRECTORIES})
