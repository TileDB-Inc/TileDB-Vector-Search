/**
 * @file   unit_partition.cc
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
#include "detail/ivf/partition.h"
#include "test/utils/array_defs.h"
#include "test/utils/query_common.h"

TEST_CASE("top_centroids", "[partition]") {
  auto parts = ColMajorMatrix<float>{
      {
          1,
          1,
          1,
          1,
      },
      {
          1,
          1,
          1,
          1,
      },
      {2, 2, 2, 2},
      {2, 2, 2, 2},
      {
          3,
          3,
          3,
          3,
      },
  };
  auto centroids = ColMajorMatrix<float>{
      {
          1,
          1,
          1,
          1,
      },
      {2, 2, 2, 2},
  };
  auto top_centroids = detail::ivf::ivf_top_centroids(centroids, parts, 1, 1);

  CHECK(top_centroids.num_cols() == 5);
  CHECK(top_centroids.num_rows() == 1);
  CHECK(top_centroids(0, 0) == 0);
  CHECK(top_centroids(0, 1) == 0);
  CHECK(top_centroids(0, 2) == 1);
  CHECK(top_centroids(0, 3) == 1);
  CHECK(top_centroids(0, 4) == 1);
}

TEST_CASE("partition_ivf_index", "[partition]") {
  // auto partition_ivf_index(
  //      auto&& centroids, auto&& query, size_t nprobe, size_t nthreads)

  auto nprobe = GENERATE(1, 2, 3);
  auto nthreads = GENERATE(3);

  auto&& [active_partitions, active_queries] =
      detail::ivf::partition_ivf_flat_index<uint32_t>(
          centroids, query, nprobe, nthreads);

  auto x = active_partitions;
  auto y = active_queries;

  CHECK(size(active_partitions) == size(active_queries));
  auto sum = 0;
  for (size_t i = 0; i < size(active_queries); ++i) {
    sum += size(active_queries[i]);
  }
  CHECK(sum == 5 * nprobe);
  std::vector<size_t> counts(5);
  for (auto&& i : active_queries) {
    for (auto&& j : i) {
      counts[j] += 1;
    }
  }
  CHECK(std::equal(
      begin(counts), end(counts), begin(std::vector<size_t>(5, nprobe))));

  if (nprobe == 1) {
    std::vector<size_t> expected_active_partitions = {3, 10, 12, 14, 18};
    std::vector<std::vector<size_t>> expected_active_queries = {
        {0}, {4}, {3}, {1}, {2}};
    CHECK(size(active_partitions) == 5);
    CHECK(std::equal(
        begin(active_partitions),
        end(active_partitions),
        begin(expected_active_partitions)));
  }
  if (nprobe == 2) {
    std::vector<size_t> expected_active_partitions = {
        3, 6, 7, 10, 12, 14, 15, 18};
    std::vector<std::vector<size_t>> expected_active_queries = {
        {0}, {0}, {4}, {4}, {2, 3}, {1}, {1}, {2, 3}};
    CHECK(size(active_partitions) == 8);

    CHECK(std::equal(
        begin(active_partitions),
        end(active_partitions),
        begin(expected_active_partitions)));
  }
}
