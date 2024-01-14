/**
 * @file   unit_flat_qv.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 */

#include <catch2/catch_all.hpp>
#include "array_defs.h"
#include "detail/flat/qv.h"
#include "detail/linalg/tdb_matrix.h"
#include "query_common.h"

bool global_debug = false;

TEST_CASE("qv test test", "[flat qv]") {
  REQUIRE(true);
}

TEST_CASE("flat qv compare arrays and files", "[flat qv]") {
  tiledb::Context ctx;

  auto array_inputs = tdbColMajorPreLoadMatrix<siftsmall_feature_type>(
      ctx, siftsmall_inputs_uri);
  auto array_queries = tdbColMajorPreLoadMatrix<siftsmall_feature_type>(
      ctx, siftsmall_query_uri);
  auto array_groundtruth = tdbColMajorPreLoadMatrix<siftsmall_groundtruth_type>(
      ctx, siftsmall_groundtruth_uri);

  auto file_inputs =
      read_bin_local<siftsmall_feature_type>(siftsmall_inputs_file);
  auto file_queries =
      read_bin_local<siftsmall_feature_type>(siftsmall_query_file);
  auto file_groundtruth = read_bin_local<uint32_t>(siftsmall_groundtruth_file);

  auto file_groundtruth_64 = ColMajorMatrix<siftsmall_groundtruth_type>(
      file_groundtruth.num_rows(), file_groundtruth.num_cols());

  std::copy(
      file_groundtruth.raveled().begin(),
      file_groundtruth.raveled().end(),
      file_groundtruth_64.raveled().begin());

  CHECK(file_groundtruth_64 == file_groundtruth);

  CHECK(array_inputs == file_inputs);
  CHECK(array_queries == file_queries);

  auto intersections00 =
      (long)count_intersections(file_groundtruth_64, array_groundtruth, 100);
  CHECK(intersections00 != 0);
  auto expected00 = array_groundtruth.num_cols() * 100;
  CHECK(intersections00 == expected00);
}

TEST_CASE(
    "flat qv all or nothing, tdbMatrix with siftsmall arrays", "[flat qv]") {
  tiledb::Context ctx;
  size_t k_nn = 10;

  auto array_inputs = tdbColMajorPreLoadMatrix<siftsmall_feature_type>(
      ctx, siftsmall_inputs_uri);
  auto array_queries = tdbColMajorPreLoadMatrix<siftsmall_feature_type>(
      ctx, siftsmall_query_uri);
  auto array_groundtruth = tdbColMajorPreLoadMatrix<siftsmall_groundtruth_type>(
      ctx, siftsmall_groundtruth_uri);

  auto&& [D00, I00] =
      detail::flat::qv_query_heap(array_inputs, array_queries, k_nn, 1);

  auto intersections00 =
      (long)count_intersections(I00, array_groundtruth, k_nn);
  CHECK(intersections00 != 0);
  auto expected00 = array_groundtruth.num_cols() * k_nn;
  CHECK(intersections00 == expected00);
}

TEST_CASE(
    "flat qv all or nothing, tdbMatrix with siftsmall files", "[flat qv]") {
  tiledb::Context ctx;
  size_t k_nn = 10;

  auto array_inputs =
      read_bin_local<siftsmall_feature_type>(siftsmall_inputs_file);
  auto array_queries =
      read_bin_local<siftsmall_feature_type>(siftsmall_query_file);
  auto array_groundtruth = read_bin_local<uint32_t>(siftsmall_groundtruth_file);

  auto&& [D00, I00] =
      detail::flat::qv_query_heap(array_inputs, array_queries, k_nn, 1);

  auto intersections00 =
      (long)count_intersections(I00, array_groundtruth, k_nn);
  CHECK(intersections00 != 0);
  auto expected00 = array_groundtruth.num_cols() * k_nn;
  CHECK(intersections00 == expected00);
}

// @todo: test with tdbMatrix
TEST_CASE("flat qv all or nothing", "[flat qv]") {
  auto ids = std::vector<size_t>(sift_base.num_cols());
  std::iota(ids.rbegin(), ids.rend(), 9);

  auto&& [D00, I00] = detail::flat::qv_query_heap(sift_base, sift_query, 3, 1);
  auto&& [D01, I01] =
      detail::flat::qv_query_heap_tiled(sift_base, sift_query, 3, 1);

  CHECK(std::equal(D00.data(), D00.data() + D00.size(), D01.data()));
  CHECK(std::equal(I00.data(), I00.data() + I00.size(), I01.data()));

  auto&& [D10, I10] =
      detail::flat::qv_query_heap(sift_base, sift_query, ids, 3, 1);
  auto&& [D11, I11] =
      detail::flat::qv_query_heap_tiled(sift_base, sift_query, ids, 3, 1);

  CHECK(!std::equal(I00.data(), I00.data() + I00.size(), I10.data()));

  CHECK(std::equal(D10.data(), D10.data() + D10.size(), D11.data()));
  CHECK(std::equal(I10.data(), I10.data() + I10.size(), I11.data()));
}

TEST_CASE("flat qv: qv_partition vs qv_partition_with_scores", "[flat_qv]") {
  CHECK(true);
}
