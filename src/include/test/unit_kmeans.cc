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

TEST_CASE("test kmeans random initialization", "[kmeans]") {
  const bool debug = false;

  // Sample data: 4-dimensional data points, 8 data points total
  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);  // 4 rows, 8 columns
  std::copy(begin(data), end(data), training_data.data());

  // Number of partitions (centroids) to initialize
  size_t num_partitions = 3;

  // Create an empty matrix for centroids with dimensions matching the number of
  // partitions
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
            centroids[i].begin(), centroids[i].end(), centroids[j].begin()));
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
            centroids[i].begin(), centroids[i].end(), training_data[j].begin());
      }
      CHECK(
          inner_counts ==
          1);  // Each centroid should match exactly one training data point
      outer_counts += inner_counts;
    }
    CHECK(outer_counts == centroids.num_cols());  // Total matches should equal
                                                  // the number of centroids
  }
}

TEST_CASE("test kmeans++ initialization", "[kmeans]") {
  const bool debug = false;

  // Sample data: 4-dimensional data points, 8 data points total
  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);  // 4 rows, 8 columns
  std::copy(begin(data), end(data), training_data.data());

  // Number of partitions (centroids) to initialize
  size_t num_partitions = 3;
  size_t num_threads = 2;

  // Create an empty matrix for centroids with dimensions matching the number of
  // partitions
  ColMajorMatrix<float> centroids(4, num_partitions);

  // Perform kmeans++ initialization of centroids
  kmeans_pp(training_data, centroids, num_partitions, num_threads);

  {
    // Verify Centroid Dimensions
    CHECK(centroids.num_cols() == num_partitions);
    CHECK(centroids.num_rows() == 4);
  }

  {
    // Centroids Match Training Data Points
    size_t outer_counts = 0;
    for (size_t i = 0; i < centroids.num_cols(); ++i) {
      size_t inner_counts = 0;
      for (size_t j = 0; j < training_data.num_cols(); ++j) {
        inner_counts += std::equal(
            centroids[i].begin(), centroids[i].end(), training_data[j].begin());
      }
      CHECK(
          inner_counts ==
          1);  // Each centroid should match exactly one training data point
      outer_counts += inner_counts;
    }
    // Total matches should equal the number of centroids
    CHECK(outer_counts == centroids.num_cols());
  }
}

TEST_CASE("test kmeans and kmeans++ edge cases and exceptions", "[kmeans]") {
  {
    // Case: num_partitions is 0
    std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                               9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};
    ColMajorMatrix<float> training_data(4, 8);
    std::copy(begin(data), end(data), training_data.data());
    size_t num_partitions = 0;
    size_t num_threads = 2;
    // Centroids matrix with zero partitions
    ColMajorMatrix<float> centroids(4, num_partitions);

    kmeans_random_init(training_data, centroids, num_partitions);
    kmeans_pp(training_data, centroids, num_partitions, num_threads);

    // Expect centroids to have zero columns
    CHECK(centroids.num_cols() == 0);
    CHECK(centroids.num_rows() == 4);
  }

  {
    // Case: Empty centroids and num_partitions is 0
    std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                               9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};
    ColMajorMatrix<float> training_data(4, 8);
    std::copy(begin(data), end(data), training_data.data());
    size_t num_partitions = 0;
    size_t num_threads = 2;
    // Empty centroids matrix
    ColMajorMatrix<float> centroids(0, 0);

    kmeans_random_init(training_data, centroids, num_partitions);
    kmeans_pp(training_data, centroids, num_partitions, num_threads);

    // Expect centroids to remain empty
    CHECK(centroids.num_cols() == 0);
    CHECK(centroids.num_rows() == 0);
  }

  {
    // Case: All empty
    // No rows, no columns
    ColMajorMatrix<float> training_data(0, 0);
    size_t num_partitions = 0;
    size_t num_threads = 2;
    // Empty centroids matrix
    ColMajorMatrix<float> centroids(0, 0);

    kmeans_random_init(training_data, centroids, num_partitions);
    kmeans_pp(training_data, centroids, num_partitions, num_threads);

    // Expect centroids to remain empty
    CHECK(centroids.num_cols() == 0);
    CHECK(centroids.num_rows() == 0);
  }

  {
    // Invalid Case: num_partitions does not match the number of centroids
    std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                               9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};
    ColMajorMatrix<float> training_data(4, 8);
    std::copy(begin(data), end(data), training_data.data());
    size_t num_partitions = 3;
    size_t num_threads = 2;
    // Mismatch: 2 centroids for 3 partitions
    ColMajorMatrix<float> centroids(4, 2);

    CHECK_THROWS_AS(
        kmeans_random_init(training_data, centroids, num_partitions),
        std::runtime_error);
    CHECK_THROWS_AS(
        kmeans_pp(training_data, centroids, num_partitions, num_threads),
        std::runtime_error);
  }
}

TEST_CASE(
    "test kmeans initialization with more partitions than data points",
    "[kmeans]") {
  const bool debug = false;

  // Sample data: 4-dimensional data points, 3 data points total
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  ColMajorMatrix<float> training_data(4, 3);  // 4 rows, 3 columns
  std::copy(begin(data), end(data), training_data.data());

  // Number of partitions (centroids) to initialize, which is more than the
  // number of data points
  size_t num_partitions = 5;
  size_t num_threads = 2;

  // Create an empty matrix for centroids with dimensions matching the number of
  // partitions
  ColMajorMatrix<float> centroids_random(4, num_partitions);
  ColMajorMatrix<float> centroids_pp(4, num_partitions);

  // Perform random initialization of centroids
  kmeans_random_init(training_data, centroids_random, num_partitions);

  // Perform kmeans++ initialization of centroids
  kmeans_pp(training_data, centroids_pp, num_partitions, num_threads);

  auto verify_centroids = [&](const ColMajorMatrix<float>& centroids) {
    // Verify Centroid Dimensions
    CHECK(centroids.num_cols() == num_partitions);
    CHECK(centroids.num_rows() == 4);

    // Centroids Match Training Data Points or are Zero
    size_t matched_vectors = 0;
    size_t zero_vectors = 0;
    for (size_t i = 0; i < centroids.num_cols(); ++i) {
      bool matched = false;
      for (size_t j = 0; j < training_data.num_cols(); ++j) {
        if (std::equal(
                centroids[i].begin(),
                centroids[i].end(),
                training_data[j].begin())) {
          matched = true;
          break;
        }
      }
      if (matched) {
        matched_vectors++;
      } else {
        // Check if this centroid is all zeros
        bool is_zero = std::all_of(
            centroids[i].begin(), centroids[i].end(), [](float val) {
              return val == 0.0f;
            });
        if (is_zero) {
          zero_vectors++;
        }
      }
    }
    // Check that all training data points are used as centroids
    CHECK(matched_vectors == training_data.num_cols());
    // Check that the remaining centroids are zero-filled
    CHECK(zero_vectors == num_partitions - training_data.num_cols());
  };

  // Verify results for kmeans_random_init
  verify_centroids(centroids_random);

  // Verify results for kmeans_pp
  verify_centroids(centroids_pp);
}

TEST_CASE("test kmeans train_no_init random data", "[kmeans]") {
  // Sample data: 6-dimensional vectors, 10 vectors total (column major)
  std::vector<float> data = {
      7, 6,   249, 3,   2,   2,   254, 249, 7,   0,  9,  3,   248, 255, 4,
      0, 249, 0,   251, 249, 245, 3,   250, 252, 6,  7,  5,   252, 4,   5,
      9, 9,   248, 254, 7,   1,   4,   1,   253, 5,  2,  255, 250, 6,   3,
      0, 2,   249, 0,   250, 5,   4,   5,   2,   99, 30, 3,   1,   55,  88};

  ColMajorMatrix<float> training_set(6, 10);  // 6 rows, 10 columns
  std::copy(data.begin(), data.end(), training_set.data());

  // Initial centroids: 6-dimensional vectors, 3 centroids total
  std::vector<float> centroids_data = {
      3,
      5,
      250,
      245,
      249,
      0,
      249,
      248,
      250,
      0,
      5,
      251,
      251,
      249,
      245,
      3,
      250,
      252};

  ColMajorMatrix<float> centroids(6, 3);
  std::copy(centroids_data.begin(), centroids_data.end(), centroids.data());

  size_t dimension_ = 6;
  size_t num_partitions_ = 3;
  uint32_t max_iterations = 2;
  float tol_ = 2.5e-05;
  size_t num_threads_ = 12;
  float reassign_ratio_ = 0.075;

  CHECK(centroids.num_rows() == dimension_);
  CHECK(centroids.num_cols() == num_partitions_);

  train_no_init(
      training_set,
      centroids,
      dimension_,
      num_partitions_,
      max_iterations,
      tol_,
      num_threads_,
      reassign_ratio_);

  CHECK(centroids.num_rows() == dimension_);
  CHECK(centroids.num_cols() == num_partitions_);

  {
    ColMajorMatrix<float> original_centroids(6, 3);
    std::copy(
        centroids_data.begin(),
        centroids_data.end(),
        original_centroids.data());
    float max_diff = 0.0;
    for (size_t i = 0; i < centroids.num_cols(); ++i) {
      float diff =
          sum_of_squares_distance{}(centroids[i], original_centroids[i]);
      max_diff = std::max(max_diff, diff);
    }
    REQUIRE_THAT(max_diff, Catch::Matchers::WithinAbs(91858.75f, 1e-2));
  }
}

TEST_CASE("test kmeans train_no_init training_set is empty", "[kmeans]") {
  ColMajorMatrix<float> training_set(0, 0);  // Empty training set
  ColMajorMatrix<float> centroids(0, 0);     // Empty centroids

  train_no_init(training_set, centroids, 0, 0, 2, 0.00001, 12, 0.075);

  CHECK(centroids.num_cols() == 0);  // Expect centroids to remain empty
  CHECK(centroids.num_rows() == 0);
}

TEST_CASE(
    "test kmeans train_no_init number of centroids exceeds data points",
    "[kmeans]") {
  std::vector<float> small_data = {1, 2, 3, 4, 5, 6};
  ColMajorMatrix<float> small_training_set(6, 1);  // 6 rows, 1 column
  std::copy(small_data.begin(), small_data.end(), small_training_set.data());

  ColMajorMatrix<float> more_centroids(
      6, 3);  // More centroids than data points

  train_no_init(
      small_training_set, more_centroids, 6, 3, 2, 0.00001, 12, 0.075);

  CHECK(more_centroids.num_cols() == 3);  // Verify centroids were generated
  for (size_t i = 0; i < more_centroids.num_cols(); ++i) {
    // Ensure some centroids match the data point and the rest are zeros
    bool is_zero = std::all_of(
        more_centroids[i].begin(), more_centroids[i].end(), [](float val) {
          return val == 0.0f;
        });
    if (!is_zero) {
      CHECK(std::equal(
          more_centroids[i].begin(),
          more_centroids[i].end(),
          small_data.begin()));
    }
  }
}
