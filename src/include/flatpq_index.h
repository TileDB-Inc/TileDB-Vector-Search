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
#include <unordered_set>

#include "detail/flat/qv.h"

template <class T, class U>
void sub_kmeans_random_init(
    const ColMajorMatrix<T>& training_set,
    ColMajorMatrix<U>& centroids,
    size_t sub_begin,
    size_t sub_end,
    size_t seed = 0) {
  scoped_timer _{__FUNCTION__};

  if (num_vectors(training_set) < num_vectors(centroids)) {
    throw std::invalid_argument(
        "Number of vectors in training set must be greater than or equal to "
        "number of centroids");
  }

  std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_int_distribution<> dis(0, num_vectors(training_set) - 1);

  size_t num_clusters = num_vectors(centroids);

  std::vector<size_t> indices(num_clusters);
  std::unordered_set<size_t> visited;
  for (size_t i = 0; i < num_clusters; ++i) {
    size_t index;
    do {
      index = dis(gen);
    } while (visited.contains(index));
    indices[i] = index;
    visited.insert(index);
  }
  // std::iota(begin(indices), end(indices), 0);
  // std::shuffle(begin(indices), end(indices), gen);

  for (size_t i = 0; i < num_clusters; ++i) {
    for (size_t j = sub_begin; j < sub_end; ++j) {
      centroids(j, i) = training_set(j, indices[i]);
    }
  }
}

/**
 * @brief Run kmeans on a subspace
 * @tparam T Data type of the input vectors
 * @param training_set Training set
 * @param centroids Initial locations of centroids.  Will be updated in
 * locations [sub_begin, sub_end).
 * @param sub_begin Beginning of the subspace
 * @param sub_end End of the subspace
 * @param num_clusters Number of clusters to find
 * @return
 *
 * @todo update with concepts
 * @todo fix up zero sized partitions
 * @todo this would be more cache friendly to do sub_begin and sub_end
 * in an inner loop
 * @todo We can probably just reuse plain kmeans at some point.
 */
template <class T, class U>
auto sub_kmeans(
    const ColMajorMatrix<T>& training_set,
    ColMajorMatrix<U>& centroids,
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
  ColMajorMatrix<U> new_centroids(dimension(centroids), num_vectors(centroids));
  for (size_t i = 0; i < num_vectors(new_centroids); ++i) {
    for (size_t j = 0; j < dimension(new_centroids); ++j) {
      new_centroids(j, i) = centroids(j, i);
    }
  }

  size_t iter = 0;
  double max_diff = 0.0;
  double total_weight = 0.0;
  for (iter = 0; iter < max_iter; ++iter) {
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

    max_diff = 0;
    total_weight = 0;
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
  return std::make_tuple(iter, max_diff/ total_weight);
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
class flatpq_index {

  // using feature_type = typename
  // std::remove_reference_t<decltype(partitioned_db)>::value_type;
  using feature_type = T;
  using id_type = shuffled_ids_type;
  using score_type = float;
  using centroid_feature_type = float;
  using code_type = uint8_t;

 // @todo Temporary only!!
 public:

  size_t dimension_{0};
  size_t num_subspaces_{0};
  size_t sub_dimension_{0};
  size_t bits_per_subspace_{8};
  size_t num_clusters_{256};
  double tol_ = 0.001;
  double max_iter_ = 16;
  size_t num_threads_ = std::thread::hardware_concurrency();
  ColMajorMatrix<centroid_feature_type> centroids_;
  std::vector<ColMajorMatrix<score_type>> distance_tables_;
  ColMajorMatrix<code_type> pq_vectors_;

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
  flatpq_index(
      size_t dimension,
      size_t num_subspaces,
      size_t bits_per_subspace = 8,
      size_t num_clusters = 256)
      : dimension_(dimension)
      , num_subspaces_(num_subspaces)
      , bits_per_subspace_(bits_per_subspace)
      , num_clusters_(num_clusters) {
    // Number of subspaces must evenly divide dimension of vector
    if ((dimension_ % num_subspaces_) != 0) {
      throw std::invalid_argument(
          "Number of subspaces must evenly divide dimension of vector");
    }
    sub_dimension_ = dimension_ / num_subspaces_;

#if 0
    switch (bits_per_subspace) {
      case 8: {
        num_clusters_ = 256;
        break;
      }
      case 16: {
        // @todo -- allow setting to smaller value
        // num_clusters_ = 65536;
        num_clusters_ = 2048;
        break;
      }
        throw std::invalid_argument(
            "bits_per_subspace must be equal to 8 or 16");
    }
#endif
  }

  /**
   * @brief Train the index on a training set.  Run kmeans on each subspace and
   * create codewords from the centroids.
   *
   * @param training_set Training set
   */
  auto train(const ColMajorMatrix<feature_type>& training_set) {
    centroids_ = std::move(ColMajorMatrix<centroid_feature_type>(dimension_, num_clusters_));
    distance_tables_ = std::vector<ColMajorMatrix<score_type>>(num_subspaces_);
    for (size_t i = 0; i < num_subspaces_; ++i) {
      distance_tables_[i] =
          std::move(ColMajorMatrix<centroid_feature_type>(num_clusters_, num_clusters_));
    }

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * dimension_ / num_subspaces_;
      auto sub_end = (subspace + 1) * dimension_ / num_subspaces_;

      sub_kmeans_random_init(
          training_set, centroids_, sub_begin, sub_end, 0xdeadbeef);
      size_t iters;
      double conv;
      std::tie(iters, conv) = sub_kmeans(
          training_set,
          centroids_,
          sub_begin,
          sub_end,
          num_clusters_,
          tol_,
          max_iter_,
          num_threads_);

      auto x = 0;
    }

    // Create table
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      for (size_t i = 0; i < num_clusters_; ++i) {
        for (size_t j = 0; j < num_clusters_; ++j) {
          auto sub_begin = subspace * dimension_ / num_subspaces_;
          auto sub_end = (subspace + 1) * dimension_ / num_subspaces_;
          auto sub_distance = sub_sum_of_squares(
              centroids_[i], centroids_[j], sub_begin, sub_end);
          distance_tables_[subspace](i, j) = sub_distance;
        }
      }
    }
  }

  auto add(const ColMajorMatrix<feature_type>& feature_vectors) {
    pq_vectors_ = std::move(
        ColMajorMatrix<code_type>(num_subspaces_, num_vectors(feature_vectors)));

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = sub_dimension_ * subspace;
      auto sub_end = sub_dimension_ * (subspace + 1);

      auto x = detail::flat::qv_partition(
          centroids_,
          feature_vectors,
          num_threads_,
          sub_sum_of_squares_distance{sub_begin, sub_end});

      for (size_t i = 0; i < num_vectors(feature_vectors); ++i) {
        pq_vectors_(subspace, i) = x[i];

        // Debugging
        std::vector<float> a(
            begin(feature_vectors[i]), end(feature_vectors[i]));
        std::vector<float> b{begin(pq_vectors_[i]), end(pq_vectors_[i])};
        // std::vector<float> c{begin(centroids_[i]), end(centroids_[i])};
        auto foo = 0;
      }
    }
  }

  auto verify_pq_encoding(const ColMajorMatrix<feature_type>& feature_vectors) {
    double total_distance = 0.0;
    double total_normalizer = 0.0;

    debug_slice(centroids_, "verify pq encoding centroids");
    debug_slice(feature_vectors, "verify pq encoding feature vectors");

    auto debug_vectors =
        ColMajorMatrix<float>(dimension_, num_vectors(feature_vectors));

    for (size_t i = 0; i < num_vectors(feature_vectors); ++i) {
      auto re = std::vector<float>(dimension_);
      for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
        auto sub_begin = sub_dimension_ * subspace;
        auto sub_end = sub_dimension_ * (subspace + 1);
        auto centroid = centroids_[pq_vectors_(subspace, i)];

        std::vector<float> x(
            begin(feature_vectors[i]), end(feature_vectors[i]));
        std::vector<float> y{begin(pq_vectors_[i]), end(pq_vectors_[i])};
        std::vector<float> z{begin(centroid), end(centroid)};

        for (size_t j = sub_begin; j < sub_end; ++j) {
          re[j] = centroid[j];
        }
      }
      for (size_t j = 0; j < dimension_; ++j) {
        debug_vectors(j, i) = re[j];
      }

      std::vector<float> x(begin(feature_vectors[i]), end(feature_vectors[i]));

      auto distance = sum_of_squares(feature_vectors[i], re);
      total_distance += distance;
      total_normalizer += sum_of_squares(feature_vectors[i]);
    }
    debug_slice(debug_vectors, "verify pq encoding re");

    return total_distance / total_normalizer;
  }

  auto verify_pq_distances(const ColMajorMatrix<feature_type>& feature_vectors) {
    double total_diff = 0.0;
    double total_normalizer = 0.0;

    for (size_t i = 0; i < num_vectors(feature_vectors); ++i) {
      for (size_t j = i + 1; j < num_vectors(feature_vectors); ++j) {
        auto real_distance =
            sum_of_squares(feature_vectors[i], feature_vectors[j]);

        auto pq_distance = 0.0;
        auto pq_distance_too = 0.0;
        auto re_i = std::vector<float>(dimension_);
        auto re_j = std::vector<float>(dimension_);

        for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
          auto sub_distance = distance_tables_[subspace](i, j);
          pq_distance += sub_distance;

          auto sub_begin = subspace * sub_dimension_;
          auto sub_end = (subspace + 1) * sub_dimension_;
          auto sub_distance_too = sub_sum_of_squares(
              centroids_[i], centroids_[j], sub_begin, sub_end);
          pq_distance_too += sub_distance_too;

          auto centroid = centroids_[pq_vectors_(subspace, i)];

          std::vector<float> x(
              begin(feature_vectors[i]), end(feature_vectors[i]));
          std::vector<float> y{begin(pq_vectors_[i]), end(pq_vectors_[i])};
          std::vector<float> z{begin(centroid), end(centroid)};

          for (size_t k = sub_begin; k < sub_end; ++k) {
            re_i[k] = centroids_[pq_vectors_(subspace, i)][k];
            re_j[k] = centroids_[pq_vectors_(subspace, j)][k];
          }
        }

        std::vector<float> x_i(
            begin(feature_vectors[i]), end(feature_vectors[i]));
        std::vector<float> x_j(
            begin(feature_vectors[j]), end(feature_vectors[j]));
        std::vector<float> y_i{begin(pq_vectors_[i]), end(pq_vectors_[i])};
        std::vector<float> y_j{begin(pq_vectors_[j]), end(pq_vectors_[j])};
        std::vector<float> z_i{begin(centroids_[i]), end(centroids_[i])};
        std::vector<float> z_j{begin(centroids_[j]), end(centroids_[j])};
        auto ydiff_i = sum_of_squares(feature_vectors[i], re_i);
        auto ydiff_j = sum_of_squares(feature_vectors[j], re_j);

        auto zdiff_i = sum_of_squares(feature_vectors[i], feature_vectors[j]);
        auto zdiff_j = sum_of_squares(re_i, re_j);

        auto xdiff = std::abs(pq_distance - pq_distance_too);

        auto diff = std::abs(real_distance - pq_distance);
        total_diff += diff;
      }
    }
    return total_diff / (num_vectors(feature_vectors) *
                         (num_vectors(feature_vectors) - 1) / 2);
  }

  float sub_distance_symmetric(auto&& a, auto&& b) {
    float pq_distance = 0.0;
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto i = a[subspace];
      auto j = b[subspace];

      auto sub_distance = distance_tables_[subspace](i, j);
      pq_distance += sub_distance;
    }
    return pq_distance;
  }

  auto make_pq_distance_symmetric() {
    using A = decltype(centroids_[0]);
    using B = decltype(pq_vectors_[0]);
    struct pq_distance {
      flatpq_index* outer_;
      float operator()(const A& a, const B& b) {
        return outer_->sub_distance_symmetric(a, b);
      }
    };
    return pq_distance{this};
  }

  /**
   *
   * @param a The uncompressed vector
   * @param b The compressed vector
   * @return
   */
  float sub_distance_asymmetric(auto&& a, auto&& b) {
    float pq_distance = 0.0;

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimension_;
      auto sub_end = (subspace + 1) * sub_dimension_;
      auto i = b[subspace];

      pq_distance += sub_sum_of_squares(
          a, centroids_[i], sub_begin, sub_end);
    }

    return pq_distance;
  }

  auto make_pq_distance_asymmetric() {
    // using A = decltype(centroids_[0]);
    using A = std::span<feature_type>;  // @todo: Don't hardcode span
    using B = decltype(pq_vectors_[0]);

    struct pq_distance {
      flatpq_index* outer_;
      float operator()(const A& a, const B& b) {
        return outer_->sub_distance_asymmetric(a, b);
      }
    };
    return pq_distance{this};
  }

  template <feature_vector_array Q>
  auto query(const Q& query_vectors, size_t k_nn) {
    return detail::flat::qv_query_heap(
        pq_vectors_,
        query_vectors,
        k_nn,
        num_threads_,
        make_pq_distance_asymmetric());
  }

  template <feature_vector V>
  auto encode(const V& v) {
    using value_type = typename V::value_type;
    auto pq = std::vector<score_type>(num_subspaces_);

    debug_slice(centroids_, "centroids");

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = sub_dimension_ * subspace;
      auto sub_end = sub_dimension_ * (subspace + 1);

      auto min_score = std::numeric_limits<value_type>::max();
      size_t idx{0};
      for (size_t i = 0; i < num_vectors(centroids_); ++i) {
        auto score = sub_sum_of_squares(v, centroids_[i], sub_begin, sub_end);
        if (score < min_score) {
          min_score = score;
          idx = i;
        }
      }
      pq[subspace] = idx;
    }
    return pq;
  }

  template <class F = feature_type, feature_vector PQ>
  auto decode(const PQ& pq) {
    using local_feature_type = F;
    auto un_pq = std::vector<local_feature_type>(dimension_);

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimension_;
      auto sub_end = (subspace + 1) * sub_dimension_;
      auto i = pq[subspace];
      for (size_t j = sub_begin; j < sub_end; ++j) {
        un_pq[j] = centroids_(j, i);
      }
    }
    return un_pq;
  }

  auto write_index(const std::string& uri, bool overwrite) {

  }
};

#endif  // TILEDB_FLATPQ_INDEX_
