/**
 * @file   unit_partitioned_matrix.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <vector>
#include "cpos.h"
#include "detail/linalg/partitioned_matrix.h"
#include "mdspan/mdspan.hpp"

TEST_CASE("partitioned_matrix: sizes constructor", "[partitioned_matrix]") {
  using feature_type = int;
  using id_type = int;
  using part_index_type = int;
  size_t dimensions = 3;
  size_t max_num_vectors = 5;
  size_t max_num_partitions = 2;

  auto partitioned_matrix =
      ColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
          dimensions, max_num_vectors, max_num_partitions);
  CHECK(partitioned_matrix.num_vectors() == 0);
  CHECK(partitioned_matrix.num_partitions() == 0);
  CHECK(std::equal(
      partitioned_matrix.ids().begin(),
      partitioned_matrix.ids().end(),
      std::vector<part_index_type>{0, 0, 0, 0, 0}.begin()));
  CHECK(std::equal(
      partitioned_matrix.indices().begin(),
      partitioned_matrix.indices().end(),
      std::vector<part_index_type>{0, 0, 0}.begin()));

  CHECK(partitioned_matrix.load() == false);
  CHECK(partitioned_matrix.num_vectors() == 0);
  CHECK(partitioned_matrix.num_partitions() == 0);
  CHECK(std::equal(
      partitioned_matrix.ids().begin(),
      partitioned_matrix.ids().end(),
      std::vector<part_index_type>{0, 0, 0, 0, 0}.begin()));
  CHECK(std::equal(
      partitioned_matrix.indices().begin(),
      partitioned_matrix.indices().end(),
      std::vector<part_index_type>{0, 0, 0}.begin()));
}

TEST_CASE("partitioned_matrix: vectors constructor", "[partitioned_matrix]") {
  using feature_type = float;
  using id_type = float;
  using part_index_type = float;

  auto parts =
      ColMajorMatrix<feature_type>{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
  std::vector<id_type> ids = {1, 2, 3, 4};
  std::vector<part_index_type> part_index = {0, 1, 4};

  auto partitioned_matrix =
      ColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
          parts, ids, part_index);

  CHECK(partitioned_matrix.num_vectors() == 4);
  CHECK(partitioned_matrix.num_partitions() == 2);
  CHECK(std::equal(
      partitioned_matrix.ids().begin(),
      partitioned_matrix.ids().end(),
      std::vector<part_index_type>{1, 2, 3, 4}.begin()));
  CHECK(std::equal(
      partitioned_matrix.indices().begin(),
      partitioned_matrix.indices().end(),
      std::vector<part_index_type>{0, 1, 4}.begin()));

  CHECK(partitioned_matrix.load() == false);
  CHECK(partitioned_matrix.num_vectors() == 4);
  CHECK(partitioned_matrix.num_partitions() == 2);
  CHECK(std::equal(
      partitioned_matrix.ids().begin(),
      partitioned_matrix.ids().end(),
      std::vector<part_index_type>{1, 2, 3, 4}.begin()));
  CHECK(std::equal(
      partitioned_matrix.indices().begin(),
      partitioned_matrix.indices().end(),
      std::vector<part_index_type>{0, 1, 4}.begin()));
}

TEST_CASE("partitioned_matrix: training constructor", "[partitioned_matrix]") {
  using feature_type = uint64_t;
  using id_type = uint64_t;
  using part_index_type = uint64_t;

  auto training_set =
      ColMajorMatrix<feature_type>{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
  std::vector<id_type> part_labels = {1, 0, 1, 0, 1};
  size_t num_parts = 2;

  auto partitioned_matrix =
      ColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
          training_set, part_labels, num_parts);
  CHECK(partitioned_matrix.num_vectors() == _cpo::num_vectors(training_set));
  CHECK(partitioned_matrix.num_partitions() == num_parts);
  CHECK(std::equal(
      partitioned_matrix.data(),
      partitioned_matrix.data() + partitioned_matrix.num_vectors() *
                                      _cpo::dimensions(partitioned_matrix),
      std::vector<feature_type>{2, 2, 4, 4, 1, 1, 3, 3, 5, 5}.begin()));
  CHECK(std::equal(
      partitioned_matrix.ids().begin(),
      partitioned_matrix.ids().end(),
      std::vector<part_index_type>{1, 3, 0, 2, 4}.begin()));
  CHECK(std::equal(
      partitioned_matrix.indices().begin(),
      partitioned_matrix.indices().end(),
      std::vector<part_index_type>{0, 2, 5}.begin()));
}
