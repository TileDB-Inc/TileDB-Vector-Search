/**
 * @file   flatpq_index.h
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
 * Header-only library of class that implements a flat index that used product
 * quantization.
 *
 */

#ifndef TILEDB_FLATPQ_INDEX_H
#define TILEDB_FLATPQ_INDEX_H

#include <cstddef>

#include "detail/flat/qv.h"

/**
 * @brief Run kmeans on a subspace
 * @tparam T Data type of the input vectors
 * @param sub_begin Beginning of the subspace
 * @param sub_end End of the subspace
 * @param num_clusters Number of clusters to find
 * @return
 *
 * @todo update with concepts
 * @todo fix up zero sized partitions
 * @todo this would be more cache friendly to do sub_begin and sub_end
 * in an inner loop
 */
template <class T>
auto sub_kmeans(
    const ColMajorMatrix<T>& training_set,
    ColMajorMatrix<T>& centroids,
    size_t sub_begin,
    size_t sub_end,
    size_t num_clusters,
    double tol,
    size_t max_iter,
    size_t num_threads) {
  size_t sub_dimension_ = sub_end - sub_begin;

  std::vector<size_t> degrees(num_clusters, 0);

  // Copy centroids to new centroids -- note only one subspace will be changing
  // @todo Keep new_centroids outside function so we don't need to copy all
  ColMajorMatrix<T> new_centroids(dimension(centroids), num_vectors(centroids));
  for (size_t i = 0; i < num_vectors(new_centroids); ++i) {
    for (size_t j = 0; j < dimension(new_centroids); ++j) {
      new_centroids(j, i) = centroids(j, i);
    }
  }

  for (size_t iter = 0; iter < max_iter; ++iter) {
    auto parts = detail::flat::qv_partition(
        centroids,
        training_set,
        num_threads,
        sub_sum_of_squares_distance{sub_begin, sub_end});

    for (size_t j = 0; j < num_vectors(new_centroids); ++j) {
      for (size_t i = sub_begin; i < sub_end; ++i) {
        new_centroids(i, j) = 0;
      }
    }
    std::fill(begin(degrees), end(degrees), 0);

    for (size_t i = 0; i < num_vectors(training_set); ++i) {
      auto part = parts[i];
      auto centroid = new_centroids[part];
      auto vector = training_set[i];

      for (size_t j = sub_begin; j < sub_end; ++j) {
        centroid[j] += vector[j];
      }
      ++degrees[part];
    }

    double max_diff = 0.0;
    double total_weight = 0.0;
    for (size_t j = 0; j < num_vectors(centroids); ++j) {
      if (degrees[j] != 0) {
        auto centroid = new_centroids[j];
        for (size_t k = sub_begin; k < sub_end; ++k) {
          centroid[k] /= degrees[j];
          total_weight += centroid[k] * centroid[k];
        }
      }
      auto diff = sub_sum_of_squares(
          centroids[j], new_centroids[j], sub_begin, sub_end);
      max_diff = std::max<double>(max_diff, diff);
    }
    centroids.swap(new_centroids);

    if (max_diff < tol * total_weight) {
      break;
    }
  }
}

/**
 * @brief Flat index that uses product quantization
 *
 * @tparam T Data type of the input vectors
 * @tparam shuffled_ids_type Data type of the shuffled ids
 * @tparam indices_type Data type of the indices
 */
template <
    class T,
    class shuffled_ids_type = size_t,
    class indices_type = size_t>
class flat_index {
  size_t dimension_{0};
  size_t num_subspaces_{0};
  size_t bits_per_subspace_{8};

 public:
  /**
   * @brief Construct a new flat index object
   * @param dimension Dimensionality of the input vectors
   * @param num_subspaces Number of subspaces (number of sections of the
   *       vector to quantize)
   * @param bits_per_subspace Number of bits per section (per subspace)
   *
   * @todo We don't really need dimension as an argument for any of our indexes
   */
  flat_index(
      size_t dimension, size_t num_subspaces, size_t bits_per_subspace = 8)
      : dimension_(dimension)
      , num_subspaces_(num_subspaces)
      , bits_per_subspace_(bits_per_subspace) {
    // Number of subspaces must evenly divide dimension of vector
    assert(dimension_ % num_subspaces_ == 0);
  }

  /**
   * @brief Train the index on a training set.  Run kmeans on each subspace and
   * create codewords from the centroids.
   *
   * @param training_set Training set
   */
  auto train(const ColMajorMatrix<T>& training_set) {
  }

  auto add() {
  }

  auto query() {
  }

  auto encode() {
  }

  auto decode() {
  }
};

#endif  // TILEDB_FLATPQ_INDEX_
