/**
 * @file   unit_tdb_partitioned_matrix.cc
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

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <vector>
#include "cpos.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/tdb_matrix_with_ids.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "mdspan/mdspan.hpp"

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

/**
 * Generates all non-empty subsets of the set {0, 1, ..., num_parts-1}.
 *
 * @param num_parts The number of elements in the set {0, 1, ..., num_parts-1}.
 * @return A vector of vectors containing all non-empty subsets.
 */
template <typename T>
std::vector<std::vector<T>> generateSubsets(int num_parts) {
  auto max_subsets = 100;
  std::vector<T> elements(num_parts);
  std::vector<std::vector<T>> all_subsets;

  // Initialize the elements vector with values from 0 to num_parts-1
  for (int i = 0; i < num_parts; ++i) {
    elements[i] = static_cast<T>(i);
  }

  // Generate all non-empty subsets
  for (int subset_size = 1; subset_size <= num_parts; ++subset_size) {
    std::vector<T> subset(subset_size);
    std::vector<bool> selection(num_parts, false);
    for (int i = 0; i < subset_size; ++i) {
      selection[i] = true;
    }

    do {
      int index = 0;
      for (int i = 0; i < num_parts; ++i) {
        if (selection[i]) {
          subset[index++] = elements[i];
        }
      }
      all_subsets.push_back(subset);

      if (all_subsets.size() >= max_subsets) {
        return all_subsets;
      }

    } while (std::prev_permutation(selection.begin(), selection.end()));
  }

  return all_subsets;
}

TEST_CASE("can load correctly", "[tdb_partitioned_matrix]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  using feature_type = int;
  using id_type = int;
  using part_index_type = int;

  std::string partitioned_vectors_uri =
      (std::filesystem::temp_directory_path() / "partitioned_vectors").string();
  std::string ids_uri =
      (std::filesystem::temp_directory_path() / "ids").string();
  // Setup data.
  {
    if (vfs.is_dir(partitioned_vectors_uri)) {
      vfs.remove_dir(partitioned_vectors_uri);
    }
    if (vfs.is_dir(ids_uri)) {
      vfs.remove_dir(ids_uri);
    }

    auto partitioned_vectors =
        ColMajorMatrix<feature_type>{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
    write_matrix(ctx, partitioned_vectors, partitioned_vectors_uri);
    std::vector<id_type> ids = {1, 2, 3, 4, 5};
    write_vector(ctx, ids, ids_uri);
  }

  // Test if we have two parts we can load them both.
  {
    std::vector<part_index_type> indices = {0, 3, 5};
    std::vector<part_index_type> relevant_parts = {0, 1};

    auto matrix =
        tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
            ctx, partitioned_vectors_uri, indices, ids_uri, relevant_parts, 0);
    matrix.load();

    CHECK(matrix.num_vectors() == 5);
    CHECK(matrix.num_partitions() == 2);
    CHECK(std::equal(
        matrix.data(),
        matrix.data() + matrix.num_vectors() * _cpo::dimensions(matrix),
        std::vector<feature_type>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5}.begin()));
    CHECK(std::equal(
        matrix.ids().begin(),
        matrix.ids().end(),
        std::vector<part_index_type>{1, 2, 3, 4, 5}.begin()));
    CHECK(std::equal(
        matrix.indices().begin(), matrix.indices().end(), indices.begin()));
  }

  // Test if we have two parts we can load just the first.
  {
    std::vector<part_index_type> indices = {0, 3, 5};
    std::vector<part_index_type> relevant_parts = {0};

    auto matrix =
        tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
            ctx, partitioned_vectors_uri, indices, ids_uri, relevant_parts, 0);
    matrix.load();

    CHECK(matrix.num_vectors() == 3);
    CHECK(matrix.num_partitions() == 1);
    CHECK(std::equal(
        matrix.data(),
        matrix.data() + matrix.num_vectors() * _cpo::dimensions(matrix),
        std::vector<feature_type>{1, 1, 2, 2, 3, 3}.begin()));
    CHECK(std::equal(
        matrix.ids().begin(),
        matrix.ids().end(),
        std::vector<part_index_type>{1, 2, 3}.begin()));
    CHECK(std::equal(
        matrix.indices().begin(), matrix.indices().end(), indices.begin()));
  }

  // Test if we have two parts we can load none.
  {
    std::vector<part_index_type> indices = {0, 3, 5};
    std::vector<part_index_type> relevant_parts = {};

    auto matrix =
        tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
            ctx, partitioned_vectors_uri, indices, ids_uri, relevant_parts, 0);
    matrix.load();

    CHECK(matrix.num_vectors() == 0);
    CHECK(matrix.num_partitions() == 0);
    CHECK(std::equal(
        matrix.data(),
        matrix.data() + matrix.num_vectors() * _cpo::dimensions(matrix),
        std::vector<feature_type>{}.begin()));
    CHECK(std::equal(
        matrix.ids().begin(),
        matrix.ids().end(),
        std::vector<part_index_type>{}.begin()));
    CHECK(std::equal(
        matrix.indices().begin(), matrix.indices().end(), indices.begin()));
  }
}

TEST_CASE("generateSubsets", "[tdb_partitioned_matrix]") {
  SECTION("generateSubsets with num_parts = 0") {
    auto subsets = generateSubsets<int>(0);
    REQUIRE(subsets.empty());
  }

  SECTION("generateSubsets with num_parts = 1") {
    auto subsets = generateSubsets<int>(1);
    std::vector<std::vector<int>> expected = {{0}};
    REQUIRE(subsets == expected);
  }

  SECTION("generateSubsets with num_parts = 2") {
    auto subsets = generateSubsets<int>(2);
    std::vector<std::vector<int>> expected = {{0}, {1}, {0, 1}};
    REQUIRE(subsets == expected);
  }

  SECTION("generateSubsets with num_parts = 3") {
    auto subsets = generateSubsets<int>(3);
    std::vector<std::vector<int>> expected = {
        {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}};
    REQUIRE(subsets == expected);
  }

  SECTION("generateSubsets with num_parts = 4") {
    auto subsets = generateSubsets<int>(4);
    std::vector<std::vector<int>> expected = {
        {0},
        {1},
        {2},
        {3},
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3},
        {0, 1, 2, 3}};
    REQUIRE(subsets == expected);
  }
}

TEST_CASE("test different combinations", "[tdb_partitioned_matrix]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  using feature_type = uint64_t;
  using id_type = uint64_t;
  using part_index_type = uint64_t;

  std::string partitioned_vectors_uri =
      (std::filesystem::temp_directory_path() / "partitioned_vectors").string();
  std::string ids_uri =
      (std::filesystem::temp_directory_path() / "ids").string();

  std::vector<int> num_vectors_options = {9, 199, 2143};
  std::vector<int> dimensions_options = {3, 77};
  std::vector<int> num_parts_options = {1, 3, 99, 989, 2143};
  for (auto num_vectors : num_vectors_options) {
    for (auto dimensions : dimensions_options) {
      for (auto num_parts : num_parts_options) {
        auto training_set =
            ColMajorMatrix<feature_type>(dimensions, num_vectors);
        for (size_t i = 0; i < dimensions; ++i) {
          for (size_t j = 0; j < num_vectors; ++j) {
            training_set(i, j) = j;
          }
        }

        std::vector<id_type> part_labels(num_vectors, 0);
        for (size_t i = 0; i < num_vectors; ++i) {
          part_labels[i] = i % num_parts;
        }

        auto partitioned_matrix =
            ColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
                training_set, part_labels, num_parts);

        if (vfs.is_dir(partitioned_vectors_uri)) {
          vfs.remove_dir(partitioned_vectors_uri);
        }
        if (vfs.is_dir(ids_uri)) {
          vfs.remove_dir(ids_uri);
        }
        write_matrix(ctx, partitioned_matrix, partitioned_vectors_uri);
        write_vector(ctx, partitioned_matrix.ids(), ids_uri);

        // We have num_parts partitions. Create combinations of them. i.e. for
        // num_parts = 3: [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2].
        auto relevant_parts_combinations =
            generateSubsets<part_index_type>(num_parts);
        for (auto relevant_parts : relevant_parts_combinations) {
          auto tdb_partitioned_matrix = tdbColMajorPartitionedMatrix<
              feature_type,
              id_type,
              part_index_type>(
              ctx,
              partitioned_vectors_uri,
              partitioned_matrix.indices(),
              ids_uri,
              relevant_parts,
              0);
          tdb_partitioned_matrix.load();

          auto expected_num_vectors = 0;
          auto expected_num_partitions = 0;
          for (auto part : relevant_parts) {
            auto expected_num_vectors_for_partition =
                partitioned_matrix.indices()[part + 1] -
                partitioned_matrix.indices()[part];
            expected_num_vectors += expected_num_vectors_for_partition;
            if (expected_num_vectors_for_partition > 0) {
              expected_num_partitions++;
            }
          }

          CHECK(tdb_partitioned_matrix.num_vectors() == expected_num_vectors);
          CHECK(
              tdb_partitioned_matrix.num_partitions() ==
              expected_num_partitions);
        }
      }
    }
  }
}

TEST_CASE(
    "tdb_partitioned_matrix: empty partition", "[tdb_partitioned_matrix]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  using feature_type = uint64_t;
  using id_type = uint64_t;
  using part_index_type = uint64_t;

  std::string partitioned_vectors_uri =
      (std::filesystem::temp_directory_path() / "partitioned_vectors").string();
  std::string ids_uri =
      (std::filesystem::temp_directory_path() / "ids").string();

  size_t num_vectors = 10000;
  size_t dimensions = 128;

  // Setup data.
  {
    if (vfs.is_dir(partitioned_vectors_uri)) {
      vfs.remove_dir(partitioned_vectors_uri);
    }
    if (vfs.is_dir(ids_uri)) {
      vfs.remove_dir(ids_uri);
    }

    auto partitioned_vectors =
        ColMajorMatrix<feature_type>(dimensions, num_vectors);
    for (size_t i = 0; i < dimensions; ++i) {
      for (size_t j = 0; j < num_vectors; ++j) {
        partitioned_vectors(i, j) = j;
      }
    }
    write_matrix(ctx, partitioned_vectors, partitioned_vectors_uri);
    std::vector<id_type> ids(num_vectors, 0);
    for (size_t i = 0; i < num_vectors; ++i) {
      ids[i] = i;
    }
    write_vector(ctx, ids, ids_uri);
  }

  // Test that we do not crash if we have an empty part (i.e. two elements in
  // indices with the same value). These values were taken from running
  // `api_ivf_flat_index: read index and query infinite and finite - finite out
  // of core, 1000, nprobe: 32, max_iter: 8` which used to crash with these
  // values.
  std::vector<part_index_type> relevant_parts = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
      51, 52, 53, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68,
      69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
      85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
  std::vector<part_index_type> indices = {
      0,    1,    116,  215,  318,  418,  600,  662,  862,  1041, 1176, 1248,
      1349, 1488, 1612, 1754, 1877, 1878, 1880, 2028, 2135, 2228, 2328, 2330,
      2464, 2526, 2682, 2785, 2911, 3059, 3191, 3192, 3266, 3395, 3516, 3607,
      3757, 3758, 3861, 3998, 4100, 4306, 4446, 4618, 4733, 4838, 4958, 5112,
      5169, 5277, 5372, 5466, 5653, 5729, 5810, 5811, 5977, 6056, 6057, 6266,
      6269, 6337, 6338, 6338, 6437, 6570, 6660, 6727, 6820, 6900, 7004, 7138,
      7139, 7220, 7227, 7339, 7414, 7539, 7695, 7781, 8004, 8095, 8161, 8235,
      8320, 8389, 8495, 8619, 8769, 8840, 9043, 9088, 9183, 9241, 9293, 9425,
      9548, 9625, 9743, 9880, 10000};
  auto matrix =
      tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
          ctx, partitioned_vectors_uri, indices, ids_uri, relevant_parts, 1000);
  while (matrix.load()) {
    CHECK(matrix.num_vectors() > 0);
    CHECK(matrix.num_partitions() > 0);
    CHECK(_cpo::dimensions(matrix) == dimensions);
  }
}
