/**
 * @file unit_kmeans.cc
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

#include <fstream>
#include <iostream>
#include <vector>

#include "index/ivf_flat_index.h"
#include "test/utils/array_defs.h"
#include "test/utils/gen_graphs.h"
#include "test/utils/query_common.h"

TEST_CASE("test kmeans random initialization", "[kmeans][init]") {
  const bool debug = false;

  // Sample data: 4-dimensional data points, 8 data points total
  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 
                             0, 5, 1, 2, 9, 9, 5, 9, 2, 0, 2, 7, 
                             7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);  // 4 rows, 8 columns
  std::copy(begin(data), end(data), training_data.data());

  // Number of partitions (centroids) to initialize
  size_t num_partitions = 3;

  // Create an empty matrix for centroids with dimensions matching the number of partitions
  ColMajorMatrix<float> centroids(4, num_partitions);

  // Perform random initialization of centroids
  kmeans_random_init(training_data, centroids, num_partitions);

  {
    // Verify Centroid Dimensions
    CHECK(centroids.num_cols() == num_partitions);
    CHECK(centroids.num_rows() == 4);
  }

  {
    // Verify Centroid Uniqueness
    for (size_t i = 0; i < centroids.num_cols() - 1; ++i) {
      for (size_t j = i + 1; j < centroids.num_cols(); ++j) {
        CHECK_FALSE(std::equal(
          centroids[i].begin(), centroids[i].end(),
          centroids[j].begin()));
      }
    }
  }

  {
    // Centroids Match Training Data Points
    size_t outer_counts = 0;
    for (size_t i = 0; i < centroids.num_cols(); ++i) {
      size_t inner_counts = 0;
      for (size_t j = 0; j < training_data.num_cols(); ++j) {
        inner_counts += std::equal(
          centroids[i].begin(), centroids[i].end(),
          training_data[j].begin());
      }
      CHECK(inner_counts == 1); // Each centroid should match exactly one training data point
      outer_counts += inner_counts;
    }
    CHECK(outer_counts == centroids.num_cols()); // Total matches should equal the number of centroids
  }
}

TEST_CASE("test kmeans random initialization edge cases", "[kmeans][init][edge]") {
  {
    // Case: num_partitions is 0
    std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 
                               0, 5, 1, 2, 9, 9, 5, 9, 2, 0, 2, 7, 
                               7, 9, 8, 6, 7, 9, 6, 6};
    ColMajorMatrix<float> training_data(4, 8);
    std::copy(begin(data), end(data), training_data.data());
    size_t num_partitions = 0;
    ColMajorMatrix<float> centroids(4, num_partitions);  // Centroids matrix with zero partitions

    kmeans_random_init(training_data, centroids, num_partitions);

    // Expect centroids to have zero columns
    CHECK(centroids.num_cols() == 0);
    CHECK(centroids.num_rows() == 4);
  }

  {
    // Case: Empty centroids and num_partitions is 0
    std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 
                               0, 5, 1, 2, 9, 9, 5, 9, 2, 0, 2, 7, 
                               7, 9, 8, 6, 7, 9, 6, 6};
    ColMajorMatrix<float> training_data(4, 8);
    std::copy(begin(data), end(data), training_data.data());
    size_t num_partitions = 0;
    ColMajorMatrix<float> centroids(0, 0);  // Empty centroids matrix

    kmeans_random_init(training_data, centroids, num_partitions);

    // Expect centroids to remain empty
    CHECK(centroids.num_cols() == 0);
    CHECK(centroids.num_rows() == 0);
  }

  {
    // Case: All empty
    ColMajorMatrix<float> training_data(0, 0);  // No rows, no columns
    size_t num_partitions = 0;
    ColMajorMatrix<float> centroids(0, 0);  // Empty centroids matrix

    kmeans_random_init(training_data, centroids, num_partitions);

    // Expect centroids to remain empty
    CHECK(centroids.num_cols() == 0);
    CHECK(centroids.num_rows() == 0);
  }
}