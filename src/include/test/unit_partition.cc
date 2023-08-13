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

#include "detail/ivf/partition.h"
#include <catch2/catch_all.hpp>

TEST_CASE("partition: test test", "[partition]") {
  REQUIRE(true);
}

TEST_CASE("partition: partition_ivf_index", "[partition]") {
  //auto partition_ivf_index(
//      auto&& centroids, auto&& query, size_t nprobe, size_t nthreads)

  auto nprobe = GENERATE(1, 2, 3);
  auto nthreads = GENERATE(3);


  // clang-format off

  auto centroids = ColMajorMatrix<float> {
      {
          {8, 6, 7},
          {5, 3, 0},
          {9, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
          {4, 5, 6},
          {7, 8, 9},
          {3.14, 1.59, 2.65},
          {35, 89, 793},
          {2, 384, 6.26},
          {4, 33, 8},
          {32.7, 9.502, 8},
          {84, 1, 97},
          {3, 1, 4},
          {1, 5, 9},
          {9, 0, 3,},
          {5, 7, 6},
      }
  };
  auto query = ColMajorMatrix<float>{
      {
          {3, 4, 5},
          {2, 300, 8},
          { 3, 1, 3.5},
          {3, 1, 3},
          {4, 5, 6},
      }
  };
  // clang-format on

  auto && [active_partitions, active_queries] =
      detail::ivf::partition_ivf_index(centroids, query, nprobe, nthreads);

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
    for (auto && j: i) {
      counts[j] += 1;
    }
  }
  CHECK(std::equal(
      begin(counts),
      end(counts),
      begin(std::vector<size_t>(5, nprobe))));

  if (nprobe == 1) {
    std::vector<size_t> expected_active_partitions = {3, 10, 12, 14, 18};
    std::vector<std::vector<size_t>> expected_active_queries = {{0}, {4}, {3}, {1}, {2}};
    CHECK(size(active_partitions) == 5);
    CHECK(std::equal(
        begin(active_partitions),
        end(active_partitions),
        begin(expected_active_partitions)));
  } if (nprobe == 2) {
    std::vector<size_t> expected_active_partitions = {3, 6, 7, 10, 12, 14, 15, 18};
    std::vector<std::vector<size_t>> expected_active_queries = {{0}, {0}, {4}, {4}, {2, 3}, {1}, {1}, {2, 3}};
    CHECK(size(active_partitions) == 8);

    CHECK(std::equal(
        begin(active_partitions),
        end(active_partitions),
        begin(expected_active_partitions)));

  }
}
