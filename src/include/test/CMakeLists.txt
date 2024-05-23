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

include(Catch2)
include(CTest)

if ($ENV{CI})
  set(CISKIP "~[ci-skip]")
endif()

macro (kmeans_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cc)
    target_link_libraries(${EXENAME} kmeans_lib)
endmacro (kmeans_add_exe)

macro (kmeans_add_test TESTNAME)
    add_executable(${TESTNAME} ${TESTNAME}.cc)
    target_link_libraries(${TESTNAME} PRIVATE kmeans_lib Catch2::Catch2WithMain)
    # target_compile_definitions(${TESTNAME} PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/data/")
    # target_compile_definitions(${TESTNAME} PRIVATE DATA_FILE="${CMAKE_SOURCE_DIR}/data/karate.mtx")
    add_test(NAME ${TESTNAME}_test
      COMMAND ${TESTNAME} "${CISKIP}")
    # catch_discover_tests(${TESTNAME})
endmacro (kmeans_add_test)

# To be updated
# kmeans_add_test(time_abstraction_penalty)
# kmeans_add_test(time_insert)

# kmeans_add_test(time_open)
# kmeans_add_exe(time_l2)

kmeans_add_exe(time_scoring)

kmeans_add_test(unit_vamana_index)

kmeans_add_test(unit_vamana_group)

kmeans_add_test(unit_vamana_metadata)

kmeans_add_test(unit_adj_list)

kmeans_add_test(unit_algorithm)

kmeans_add_test(unit_api_feature_vector)

kmeans_add_test(unit_api_feature_vector_array)

kmeans_add_test(unit_api_flat_l2_index)

kmeans_add_test(unit_api_ivf_flat_index)

kmeans_add_test(unit_api_ivf_pq_index)

kmeans_add_test(unit_api_vamana_index)

kmeans_add_test(unit_array_defs)

kmeans_add_test(unit_backwards_compatibility)

kmeans_add_test(unit_concepts)

kmeans_add_test(unit_concepts_vs)

kmeans_add_test(unit_cpos)

kmeans_add_test(unit_execution_policy)

kmeans_add_test(unit_fixed_min_heap)

kmeans_add_test(unit_flat_l2_index)

kmeans_add_test(unit_flat_pq_index)

kmeans_add_test(unit_flat_qv)

kmeans_add_test(unit_flat_vq)

kmeans_add_test(unit_gen_graphs)

kmeans_add_test(unit_inner_product_distance)

kmeans_add_test(unit_ivf_flat_group)

kmeans_add_test(unit_ivf_flat_index)

kmeans_add_test(unit_ivf_flat_metadata)

kmeans_add_test(unit_ivf_pq_group)

kmeans_add_test(unit_ivf_pq_index)

kmeans_add_test(unit_ivf_pq_metadata)

kmeans_add_test(unit_index_defs)

kmeans_add_test(unit_ivf_qv)

# Needs to be updated to new partitioned matrix API
# kmeans_add_test(unit_ivf_vq)

add_executable(unit_linalg unit_linalg.cc)
target_link_libraries(unit_linalg PRIVATE kmeans_lib Catch2::Catch2WithMain)
# Skip the tdbmatrix tests because the TileDB arrays do not exist in CI.
add_test(NAME unit_linalg_test COMMAND unit_linalg ~[tdbmatrix])

kmeans_add_test(unit_logging)

kmeans_add_test(unit_l2_distance)

kmeans_add_test(unit_matrix)

kmeans_add_test(unit_partitioned_matrix)

kmeans_add_test(unit_matrix_with_ids)

kmeans_add_test(unit_mdspan)

kmeans_add_test(unit_memory)

kmeans_add_test(unit_nn-descent)

kmeans_add_test(unit_nn-graph)

kmeans_add_test(unit_partition)

kmeans_add_test(unit_scoring)

kmeans_add_test(unit_slicing)

kmeans_add_test(unit_stats)

kmeans_add_test(unit_stl)

kmeans_add_test(unit_tdb_io)

kmeans_add_test(unit_tdb_matrix)

kmeans_add_test(unit_tdb_matrix_with_ids)

kmeans_add_test(unit_tdb_partitioned_matrix)

kmeans_add_test(unit_utils)

kmeans_add_test(unit_vector)

# Temporarily deprecated
if (FALSE)
  if (TILEDB_VS_ENABLE_BLAS)
    kmeans_add_test(unit_queries)
    kmeans_add_test(unit_gemm)
    target_link_libraries(unit_gemm PRIVATE flat_blas)
  endif()
endif()
