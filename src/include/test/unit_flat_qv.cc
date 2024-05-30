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
#include "detail/flat/qv.h"
#include "detail/linalg/tdb_matrix.h"
#include "test/utils/array_defs.h"
#include "test/utils/query_common.h"

TEST_CASE("flat qv simple case", "[flat qv]") {
  auto db = ColMajorMatrix<float>{
      {
          1,
          1,
          0,
          9,
          4,
          7,
      },
      {
          11,
          254,
          7,
          10,
          250,
          0,
      },
      {
          249,
          0,
          0,
          6,
          2,
          251,
      },
      {
          253,
          8,
          252,
          6,
          3,
          0,
      },
      {
          248,
          249,
          1,
          255,
          7,
          7,
      },
      {
          249,
          255,
          251,
          7,
          2,
          4,
      },
      {
          249,
          4,
          254,
          247,
          250,
          4,
      },
      {
          250,
          0,
          1,
          251,
          252,
          247,
      },
      {
          247,
          4,
          254,
          2,
          5,
          9,
      },
      {
          4,
          251,
          5,
          3,
          6,
          252,
      },
      {
          250,
          251,
          4,
          254,
          245,
          0,
      },
      {
          2,
          6,
          0,
          7,
          0,
          5,
      },
      {
          0,
          254,
          7,
          251,
          255,
          250,
      },
      {
          3,
          247,
          0,
          8,
          1,
          6,
      },
      {
          254,
          253,
          5,
          254,
          248,
          248,
      },
      {
          0,
          254,
          0,
          255,
          4,
          255,
      },
      {
          3,
          253,
          0,
          1,
          255,
          7,
      },
      {
          5,
          0,
          252,
          248,
          1,
          0,
      },
  };

  auto centroids = ColMajorMatrix<float>{
      {38.9571, 109.171, 155.557, 180.971, 149.514, 80.2286}};

  auto parts = detail::flat::qv_partition(centroids, db, 8);
  auto z = 0;
  /*
1,  11, 249, 253, 248, 249, 249, 250, 247,   4, 250, 2,   0,   3, 254,   0,   3,
5 1, 254,   0,   8, 249, 255,   4,   0,   4, 251, 251, 6, 254, 247, 253, 254,
253,   0 0,   7,   0, 252,   1, 251, 254,   1, 254,   5,   4, 0,   7,   0,   5,
0,   0, 252 9,  10,   6,   6, 255,   7, 247, 251,   2,   3, 254, 7, 251,   8,
254, 255,   1, 248, 4, 250,   2,   3,   7,   2, 250, 252,   5,   6, 245, 0, 255,
1, 248,   4, 255,   1 7,   0, 251,   0,   7,   4,   4, 247,   9, 252,   0, 5,
250,   0, 248, 255,   7,   0

   # >>> centroids # 38.9571 # 109.171 # 155.557 # 180.971 #
  149.514 # 80.2286
  */
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

  SECTION("Invoke with default distance function") {
    auto&& [D00, I00] =
        detail::flat::qv_query_heap(array_inputs, array_queries, k_nn, 1);

    size_t intersections00 =
        (long)count_intersections(I00, array_groundtruth, k_nn);
    CHECK(intersections00 != 0);
    size_t expected00 = array_groundtruth.num_cols() * k_nn;
    CHECK(intersections00 == expected00);
  }

  SECTION("Invoke with l2_distance function") {
    auto&& [D00, I00] = detail::flat::qv_query_heap(
        array_inputs, array_queries, k_nn, 1, l2_distance);

    CHECK(counting_l2_distance.num_comps_ == 0);

    size_t intersections00 =
        (long)count_intersections(I00, array_groundtruth, k_nn);
    CHECK(intersections00 != 0);
    size_t expected00 = array_groundtruth.num_cols() * k_nn;
    CHECK(intersections00 == expected00);
  }

  SECTION("Invoke with counting_l2_distance function") {
    counting_l2_distance.reset();
    counting_l2_distance.num_comps_ = 99;
    auto&& [D00, I00] = detail::flat::qv_query_heap(
        array_inputs, array_queries, k_nn, 1, counting_l2_distance);

    CHECK(counting_l2_distance.num_comps_ == 100 * 10'000 + 99);

    size_t intersections00 =
        (long)count_intersections(I00, array_groundtruth, k_nn);
    CHECK(intersections00 != 0);
    size_t expected00 = array_groundtruth.num_cols() * k_nn;
    CHECK(intersections00 == expected00);
  }

  SECTION("Invoke with qv_query_heal_0 counting_l2_distance function") {
    counting_l2_distance.reset();
    counting_l2_distance.num_comps_ = 99;
    auto&& [D00, I00] = detail::flat::qv_query_heap_tiled(
        array_inputs, array_queries, k_nn, 1, counting_l2_distance);

    CHECK(counting_l2_distance.num_comps_ == 100 * 10'000 + 99);

    size_t intersections00 =
        (long)count_intersections(I00, array_groundtruth, k_nn);
    CHECK(intersections00 != 0);
    size_t expected00 = array_groundtruth.num_cols() * k_nn;
    CHECK(intersections00 == expected00);
  }
}

TEST_CASE(
    "flat qv all or nothing, tdbMatrix with siftsmall files", "[flat qv]") {
  tiledb::Context ctx;
  size_t k_nn = 10;

  auto array_inputs =
      read_bin_local<siftsmall_feature_type>(ctx, siftsmall_inputs_file);
  auto array_queries =
      read_bin_local<siftsmall_feature_type>(ctx, siftsmall_query_file);
  auto array_groundtruth =
      read_bin_local<uint32_t>(ctx, siftsmall_groundtruth_file);

  SECTION("Invoke with default distance function") {
    auto&& [D00, I00] =
        detail::flat::qv_query_heap(array_inputs, array_queries, k_nn, 1);

    size_t intersections00 =
        (long)count_intersections(I00, array_groundtruth, k_nn);
    CHECK(intersections00 != 0);
    size_t expected00 = array_groundtruth.num_cols() * k_nn;
    CHECK(intersections00 == expected00);
  }
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
