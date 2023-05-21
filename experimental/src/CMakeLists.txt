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
# Print the Git branch name
# message("Current Git branch: ${GIT_BRANCH}")



configure_file(config.h.in config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(flat flat.cc flat_query.h defs.h sift_db.h timer.h sift_array.h linalg.h)

target_link_libraries(flat PUBLIC docopt)
target_link_libraries(flat PUBLIC flat_mt)
target_link_libraries(flat PUBLIC flat_blas)
target_link_libraries(flat PUBLIC TileDB::tiledb_shared)
target_link_libraries(flat PUBLIC mdspan)

add_executable(kmeans flat_query.h flat.cc defs.h sift_db.h timer.h sift_array.h linalg.h)

target_link_libraries(kmeans PUBLIC docopt)
target_link_libraries(kmeans PUBLIC flat_mt)
target_link_libraries(kmeans PUBLIC flat_blas)
target_link_libraries(kmeans PUBLIC TileDB::tiledb_shared)
target_link_libraries(kmeans PUBLIC mdspan)


add_executable(ivf_hack ivf_hack.cc flat_query.h defs.h sift_db.h timer.h sift_array.h linalg.h)

target_link_libraries(ivf_hack PUBLIC docopt)
target_link_libraries(ivf_hack PUBLIC flat_mt)
target_link_libraries(ivf_hack PUBLIC flat_blas)
target_link_libraries(ivf_hack PUBLIC TileDB::tiledb_shared)
target_link_libraries(ivf_hack PUBLIC mdspan)

add_executable(ingest ingest.cc flat_query.h defs.h sift_db.h timer.h sift_array.h linalg.h)

target_link_libraries(ingest PUBLIC docopt)
target_link_libraries(ingest PUBLIC flat_mt)
target_link_libraries(ingest PUBLIC flat_blas)
target_link_libraries(ingest PUBLIC TileDB::tiledb_shared)
target_link_libraries(ingest PUBLIC mdspan)
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
