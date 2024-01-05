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
        ../include/detail/ivf/qv.h ../include/detail/ivf/vq.h ../include/detail/ivf/gemm.h ../include/detail/ivf/index.h
        )

add_library(kmeans_lib INTERFACE)
target_sources(kmeans_lib INTERFACE
        ../include/scoring.h ../include/utils/fixed_min_heap.h
        ../include/scoring.h ../include/algorithm.h ../include/stats.h
        ../include/utils/timer.h ../include/utils/logging.h ../include/utils/print_types.h
        ../include/index/flat_l2_index.h ../include/index/ivf_flat_index.h
        )

target_include_directories(kmeans_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/src/include
        ${CMAKE_BINARY_DIR}/src/include
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
if (TILEDB_VS_ENABLE_BLAS)
  target_link_Libraries(kmeans_lib INTERFACE
      flat_blas)
endif()


#---[ Demos ]------------------------------------------------------------
if (TILEDB_VS_ENABLE_BLAS OR TRUE)
  # these don't really requires BLAS any longer

  add_subdirectory(flat)
  add_subdirectory(ivf)
  add_subdirectory(pq)
  add_subdirectory(vamana)

else()
  message(STATUS "Disabling demo executables build because BLAS is disabled (TILEDBVS_ENABLE_BLAS).")
endif()

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
