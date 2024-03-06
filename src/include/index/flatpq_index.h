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
 * @todo Measure any overheads of applying pq distance function objects.
 * @todo More thorough documentation
 */

#ifndef TILEDB_FLATPQ_INDEX_H
#define TILEDB_FLATPQ_INDEX_H

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

#include "detail/flat/qv.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"

#include <tiledb/group_experimental.h>
#include <tiledb/tiledb>

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

  for (size_t i = 0; i < num_clusters; ++i) {
    for (size_t j = sub_begin; j < sub_end; ++j) {
      centroids(j, i) = training_set(j, indices[i]);
    }
  }
}

// #define REASSIGN

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
 *
 * @note We use a distance function with cached view because we need to call
 * qv_partition with a plaing distance function.  We lose CTAD here since
 * SubDistance is not on the function parameter list.
 */
template <
    class T,
    class U,
    class SubDistance = cached_sub_sum_of_squares_distance>
  requires cached_sub_distance_function<
      SubDistance,
      typename ColMajorMatrix<T>::span_type,
      typename ColMajorMatrix<U>::span_type>
auto sub_kmeans(
    const ColMajorMatrix<T>& training_set,
    ColMajorMatrix<U>& centroids,
    size_t sub_begin,
    size_t sub_end,
    size_t num_clusters,
    double tol,
    size_t max_iter,
    size_t num_threads,
    float reassign_ratio = 0.05) {
  size_t sub_dimension_ = sub_end - sub_begin;
  auto local_sub_distance = SubDistance{sub_begin, sub_end};

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
    // @todo Get rid of pre-processor for this
#ifdef REASSIGN
    auto [scores, parts] = detail::flat::qv_partition_with_scores(
        centroids, training_set, num_threads, local_sub_distance);
#else
    auto parts = detail::flat::qv_partition(
        centroids, training_set, num_threads, local_sub_distance);
#endif

    for (size_t j = 0; j < num_vectors(new_centroids); ++j) {
      for (size_t i = sub_begin; i < sub_end; ++i) {
        new_centroids(i, j) = 0;
      }
    }
    std::fill(begin(degrees), end(degrees), 0);

#ifdef REASSIGN
    // How many centroids should we try to fix up
    size_t heap_size = std::ceil(reassign_ratio * num_clusters) + 5;
    auto high_scores = fixed_min_pair_heap<
        feature_type,
        index_type,
        std::greater<feature_type>>(heap_size, std::greater<feature_type>());
    auto low_degrees = fixed_min_pair_heap<index_type, index_type>(heap_size);
#endif

    for (size_t i = 0; i < num_vectors(training_set); ++i) {
      auto part = parts[i];
      auto centroid = new_centroids[part];
      auto vector = training_set[i];

      for (size_t j = sub_begin; j < sub_end; ++j) {
        centroid[j] += vector[j];
      }
      ++degrees[part];
#ifdef REASSIGN
      high_scores.insert(scores[i], i);
#endif
    }

#ifdef REASSIGN
    size_t max_degree = 0;

    for (size_t i = 0; i < num_clusters; ++i) {
      auto degree = degrees[i];
      max_degree = std::max<size_t>(max_degree, degree);
      low_degrees.insert(degree, i);
    }
    size_t lower_degree_bound = std::ceil(max_degree * reassign_ratio);

    if (iter != max_iter - 1) {
      // Move vectors with high scores to replace zero-degree partitions
      std::sort_heap(begin(low_degrees), end(low_degrees));
      std::sort_heap(begin(high_scores), end(high_scores), [](auto a, auto b) {
        return std::get<0>(a) > std::get<0>(b);
      });
      for (size_t i = 0; i < size(low_degrees) &&
                         std::get<0>(low_degrees[i]) <= lower_degree_bound;
           ++i) {
        // std::cout << "i: " << i << " low_degrees: ("
        //           << std::get<1>(low_degrees[i]) << " "
        //           << std::get<0>(low_degrees[i]) << ") high_scores: ("
        //           << parts[std::get<1>(high_scores[i])] << " "
        //          << std::get<1>(high_scores[i]) << " "
        //          << std::get<0>(high_scores[i]) << ")" << std::endl;

        auto [degree, zero_part] = low_degrees[i];
        auto [score, high_vector_id] = high_scores[i];
        auto low_centroid = new_centroids[zero_part];
        auto high_vector = training_set[high_vector_id];

        for (size_t i = sub_begin; i < sub_end; ++i) {
          low_centroid[i] = high_vector[i];
        }
        for (size_t i = sub_begin; i < sub_end; ++i) {
          new_centroids[parts[high_vector_id]][i] -= high_vector[i];
        }

        ++degrees[zero_part];
        --degrees[parts[high_vector_id]];
      }
    }
#endif

    // Check for convergence
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
      auto diff = local_sub_distance(centroids[j], new_centroids[j]);
      max_diff = std::max<double>(max_diff, diff);
    }
    centroids.swap(new_centroids);

    if (max_diff < tol * total_weight) {
      break;
    }
  }
  return std::make_tuple(iter, max_diff / total_weight);
}

/**
 * @brief Flat index that uses product quantization
 *
 * @tparam T Data type of the input vectors
 * @tparam shuffled_ids_type Data type of the shuffled ids
 * @tparam indices_type Data type of the indices
 *
 * @todo IMPORTANT: I am thinking that Distance should be a template parameter
 * for the class.  We can't really use different distance functions for
 * queries once the index is built.
 */
template <
    class T,
    class shuffled_ids_type = uint64_t,
    class indices_type = uint64_t>
class flatpq_index {
  using feature_type = T;
  using id_type = shuffled_ids_type;

  // Some hardcoded types (for now) -- maybe make these template parameters?
  using score_type = float;
  using centroid_feature_type = float;
  using code_type = uint8_t;

  // @todo Temporarily public to facilitate early testing
 public:
  // metadata
  size_t dimension_{0};
  size_t num_subspaces_{0};
  size_t sub_dimension_{0};
  size_t bits_per_subspace_{8};
  size_t num_clusters_{256};
  float tol_ = 0.001;
  size_t max_iter_ = 16;
  size_t num_threads_ = std::thread::hardware_concurrency();

  using metadata_element = std::tuple<std::string, void*, tiledb_datatype_t>;
  std::vector<metadata_element> metadata{
      {"dimension", &dimension_, TILEDB_UINT64},
      {"num_subspaces", &num_subspaces_, TILEDB_UINT64},
      {"sub_dimension", &sub_dimension_, TILEDB_UINT64},
      {"bits_per_subspace", &bits_per_subspace_, TILEDB_UINT64},
      {"num_clusters", &num_clusters_, TILEDB_UINT64},
      {"tol", &tol_, TILEDB_FLOAT32},
      {"max_iter", &max_iter_, TILEDB_UINT64},
      {"num_threads", &num_threads_, TILEDB_UINT64},
  };

  // array data
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
    if (dimension_ == 0 || num_subspaces_ == 0 || bits_per_subspace_ == 0) {
      throw std::invalid_argument(
          "dimension, num_subspaces, and bits_per_subspace must be greater "
          "than zero");
    }
    if ((dimension_ % num_subspaces_) != 0) {
      throw std::invalid_argument(
          "Number of subspaces must evenly divide dimension of vector");
    }
    sub_dimension_ = dimension_ / num_subspaces_;
  }

  /**
   * Load constructor
   */
  flatpq_index(tiledb::Context ctx, const std::string& group_uri) {
    tiledb::Config cfg;
    auto read_group = tiledb::Group(ctx, group_uri, TILEDB_READ, cfg);

    for (auto& [name, value, datatype] : metadata) {
      if (!read_group.has_metadata(name, &datatype)) {
        throw std::runtime_error("Missing metadata: " + name);
      }
      uint32_t count;
      void* addr;
      read_group.get_metadata(name, &datatype, &count, (const void**)&addr);
      if (datatype == TILEDB_UINT64) {
        *reinterpret_cast<uint64_t*>(value) =
            *reinterpret_cast<uint64_t*>(addr);
      } else if (datatype == TILEDB_FLOAT32) {
        *reinterpret_cast<float*>(value) = *reinterpret_cast<float*>(addr);
      } else {
        throw std::runtime_error("Unsupported datatype");
      }
    }

    centroids_ =
        std::move(tdbPreLoadMatrix<centroid_feature_type, stdx::layout_left>(
            ctx, group_uri + "/centroids"));
    pq_vectors_ = std::move(tdbPreLoadMatrix<code_type, stdx::layout_left>(
        ctx, group_uri + "/pq_vectors"));
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      std::ostringstream oss;
      oss << std::setw(2) << std::setfill('0') << subspace;
      std::string number = oss.str();
      distance_tables_.emplace_back(
          tdbPreLoadMatrix<score_type, stdx::layout_left>(
              ctx, group_uri + "/distance_table_" + number));
    }
  }

  /**
   * @brief Train the index on a training set.  Run kmeans on each subspace and
   * create codewords from the centroids.
   *
   * @param training_set Training set
   */
  template <class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        typename ColMajorMatrix<feature_type>::span_type,
        typename ColMajorMatrix<centroid_feature_type>::span_type>
  auto train(const ColMajorMatrix<feature_type>& training_set) {
    centroids_ =
        ColMajorMatrix<centroid_feature_type>(dimension_, num_clusters_);
    distance_tables_ = std::vector<ColMajorMatrix<score_type>>(num_subspaces_);
    for (size_t i = 0; i < num_subspaces_; ++i) {
      distance_tables_[i] =
          ColMajorMatrix<centroid_feature_type>(num_clusters_, num_clusters_);
    }

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * dimension_ / num_subspaces_;
      auto sub_end = (subspace + 1) * dimension_ / num_subspaces_;

      auto local_sub_distance = SubDistance{sub_begin, sub_end};

      sub_kmeans_random_init(
          training_set, centroids_, sub_begin, sub_end, 0xdeadbeef);
      size_t iters;
      double conv;
      std::tie(iters, conv) =
          sub_kmeans<feature_type, centroids_type, SubDistance>(
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
      auto sub_begin = subspace * sub_dimension_;
      auto sub_end = (subspace + 1) * sub_dimension_;
      auto local_sub_distance = SubDistance{sub_begin, sub_end};

      for (size_t i = 0; i < num_clusters_; ++i) {
        for (size_t j = 0; j < num_clusters_; ++j) {
          auto sub_distance = local_sub_distance(centroids_[i], centroids_[j]);
          distance_tables_[subspace](i, j) = sub_distance;
        }
      }
    }
  }

  template <class SubDistance = cached_sub_sum_of_squares_distance>
  auto add(const ColMajorMatrix<feature_type>& feature_vectors) {
    pq_vectors_ =
        ColMajorMatrix<code_type>(num_subspaces_, num_vectors(feature_vectors));

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = sub_dimension_ * subspace;
      auto sub_end = sub_dimension_ * (subspace + 1);
      auto local_sub_distance = SubDistance{sub_begin, sub_end};

      // For each vector, find the closest centroid
      // We use sub_sum_of_squares_distance, which will find the closest
      // centroid for the current subspace
      auto x = detail::flat::qv_partition(
          centroids_,
          feature_vectors,
          num_threads_,
          local_sub_distance);  // @todo Measure abstraction penalty

      // Save the index (code) of the closest centroid
      // @todo Avoid this copy.  Do "in-place" or return 8-bit result from
      // qv_partition and move to pq_vectors_
      for (size_t i = 0; i < num_vectors(feature_vectors); ++i) {
        pq_vectors_(subspace, i) = x[i];

        // Debugging -- copy to std::vector so the debugger can inspect
        std::vector<float> a(
            begin(feature_vectors[i]), end(feature_vectors[i]));
        std::vector<float> b{begin(pq_vectors_[i]), end(pq_vectors_[i])};
        // std::vector<float> c{begin(centroids_[i]), end(centroids_[i])};
        auto foo = 0;
      }
    }
  }

  // @todo IMPORTANT: We need to do some abstraction penalty tests to make sure
  // that the distance functions are inlined.
  float sub_distance_symmetric(auto&& a, auto&& b) const {
    float pq_distance = 0.0;
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto i = a[subspace];
      auto j = b[subspace];

      auto sub_distance = distance_tables_[subspace](i, j);
      pq_distance += sub_distance;
    }
    return pq_distance;
  }

  auto make_pq_distance_symmetric() const {
    using A = decltype(pq_vectors_[0]);
    using B = decltype(pq_vectors_[0]);
    struct pq_distance {
      const flatpq_index* outer_;
      inline float operator()(const A& a, const B& b) {
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
  template <feature_vector U, feature_vector V>
  float sub_distance_asymmetric(const U& a, const V& b) const {
    float pq_distance = 0.0;

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimension_;
      auto sub_end = (subspace + 1) * sub_dimension_;
      auto i = b[subspace];

      pq_distance += sub_l2_distance(a, centroids_[i], sub_begin, sub_end);
    }

    return pq_distance;
  }

  auto make_pq_distance_asymmetric() {
    using A = std::span<feature_type>;  // @todo: Don't hardcode span
    using B = decltype(pq_vectors_[0]);

    // @todo Do we need to worry about function overhead here?
    struct pq_distance {
      const flatpq_index* outer_;
      inline float operator()(const A& a, const B& b) {
        return outer_->sub_distance_asymmetric(a, b);
      }
    };
    return pq_distance{this};
  }

  template <feature_vector_array Q>
  auto asymmetric_query(const Q& query_vectors, size_t k_nn) {
    return detail::flat::qv_query_heap(
        pq_vectors_,
        query_vectors,
        k_nn,
        num_threads_,
        make_pq_distance_asymmetric());
  }

  template <
      feature_vector_array Q,
      class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        typename Q::span_type,
        decltype(centroids_[0])>
  auto symmetric_query(const Q& query_vectors, size_t k_nn) {
    auto encoded_query = encode<Q, SubDistance>(query_vectors);
    return detail::flat::qv_query_heap(
        pq_vectors_,
        encoded_query,
        k_nn,
        num_threads_,
        make_pq_distance_symmetric());
  }

  template <
      feature_vector V,
      feature_vector W,
      class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        V,
        decltype(centroids_[0])>
  auto encode(const V& v, W& pq) const {
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = sub_dimension_ * subspace;
      auto sub_end = sub_begin + sub_dimension_;
      auto local_sub_distance = SubDistance{sub_begin, sub_end};

      auto min_score = std::numeric_limits<score_type>::max();
      code_type idx{0};
      for (size_t i = 0; i < num_vectors(centroids_); ++i) {
        auto score = local_sub_distance(v, centroids_[i]);
        if (score < min_score) {
          min_score = score;
          idx = i;
        }
      }
      pq[subspace] = idx;
    }
  }

  // @todo Make an in-place variant so we don't construct a vector
  template <
      feature_vector V,
      class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        V,
        decltype(centroids_[0])>
  auto encode(const V& v) {
    // @todo Use Vector instead of std::vector
    auto pq = std::vector<code_type>(num_subspaces_);
    encode<V, decltype(pq), SubDistance>(v, pq);
    return pq;
  }

  // @todo Make an in-place variant so we don't construct a matrix
  template <
      feature_vector_array V,
      class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        typename V::span_type,
        decltype(centroids_[0])>
  auto encode(const V& v) {
    auto pq = ColMajorMatrix<code_type>(num_subspaces_, num_vectors(v));
    for (size_t i = 0; i < num_vectors(pq); ++i) {
      auto x = pq[i];
      encode<typename V::span_type, decltype(pq[0]), SubDistance>(v[i], x);
    }

    return pq;
  }

  template <class F = feature_type, feature_vector PQ>
  auto decode(const PQ& pq) {
    using local_feature_type = F;
    // @todo Use Vector instead of std::vector
    auto un_pq = std::vector<local_feature_type>(dimension_);

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimension_;
      auto sub_end = (subspace + 1) * sub_dimension_;
      auto i = pq[subspace];

      // Copy the centroid into the appropriate subspace portion of the
      // decoded vector
      for (size_t j = sub_begin; j < sub_end; ++j) {
        un_pq[j] = centroids_(j, i);
      }
    }
    return un_pq;
  }

  auto write_index(const std::string& group_uri, bool overwrite) {
    tiledb::Context ctx;
    tiledb::VFS vfs(ctx);
    if (vfs.is_dir(group_uri)) {
      if (overwrite == false) {
        return false;
      }
      vfs.remove_dir(group_uri);
    }

    tiledb::Config cfg;
    tiledb::Group::create(ctx, group_uri);
    auto write_group = tiledb::Group(ctx, group_uri, TILEDB_WRITE, cfg);

    for (auto&& [name, value, type] : metadata) {
      write_group.put_metadata(name, type, 1, value);
    }

    auto centroids_uri = group_uri + "/centroids";
    write_matrix(ctx, centroids_, centroids_uri);
    write_group.add_member("centroids", true, "centroids");

    auto pq_vectors_uri = group_uri + "/pq_vectors";
    write_matrix(ctx, pq_vectors_, pq_vectors_uri);
    write_group.add_member("pq_vectors", true, "pq_vectors");

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      std::ostringstream oss;
      oss << std::setw(2) << std::setfill('0') << subspace;
      std::string number = oss.str();

      auto distance_table_uri = group_uri + "/distance_table_" + number;
      write_matrix(ctx, distance_tables_[subspace], distance_table_uri);
      write_group.add_member(
          "distance_table_" + number, true, "distance_table_" + number);
    }
    write_group.close();
    return true;
  }

  /***************************************************************************
   * Methods to aid Testing and Debugging
   **************************************************************************/

  /**
   * @brief Verify that the pq encoding is correct by comparing every feature
   * vector against the corresponding pq encoded value
   * @param feature_vectors
   * @return
   */
  auto verify_pq_encoding(const ColMajorMatrix<feature_type>& feature_vectors) {
    double total_distance = 0.0;
    double total_normalizer = 0.0;

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

        // Reconstruct the encoded vector
        for (size_t j = sub_begin; j < sub_end; ++j) {
          re[j] = centroid[j];
        }
      }
      // std::copy(begin(re), end(re), begin(debug_vectors[i]));

      // Measure the distance between the original vector and the reconstructed
      // vector and accumulate into the total distance as well as the total
      // weight of the feature vector
      auto distance = l2_distance(feature_vectors[i], re);
      total_distance += distance;
      total_normalizer += l2_distance(feature_vectors[i]);
    }
    // debug_slice(debug_vectors, "verify pq encoding re");

    // Return the total accumulated distance between the encoded and original
    // vectors, divided by the total weight of the original feature vectors
    return total_distance / total_normalizer;
  }

  /**
   * @brief Verify that recorded distances between centroids are correct by
   * comparing the distance between every pair of pq vectors against the
   * distance between every pair of the original feature vectors.
   * @param feature_vectors
   * @return
   */
  auto verify_pq_distances(
      const ColMajorMatrix<feature_type>& feature_vectors) {
    double total_diff = 0.0;
    double total_normalizer = 0.0;

    for (size_t i = 0; i < num_vectors(feature_vectors); ++i) {
      for (size_t j = i + 1; j < num_vectors(feature_vectors); ++j) {
        auto real_distance =
            l2_distance(feature_vectors[i], feature_vectors[j]);
        total_normalizer += real_distance;
        auto pq_distance = 0.0;

        for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
          auto sub_distance = distance_tables_[subspace](
              pq_vectors_(subspace, i), pq_vectors_(subspace, j));
          pq_distance += sub_distance;
        }

        auto diff = std::abs(real_distance - pq_distance);
        total_diff += diff;
      }
    }

    return total_diff / total_normalizer;
  }

  auto verify_asymmetric_pq_distances(
      const ColMajorMatrix<feature_type>& feature_vectors) {
    double total_diff = 0.0;
    double total_normalizer = 0.0;

    score_type diff_max = 0.0;
    score_type vec_max = 0.0;
    for (size_t i = 0; i < num_vectors(feature_vectors); ++i) {
      for (size_t j = i + 1; j < num_vectors(feature_vectors); ++j) {
        auto real_distance =
            l2_distance(feature_vectors[i], feature_vectors[j]);
        total_normalizer += real_distance;

        auto pq_distance =
            this->sub_distance_asymmetric(feature_vectors[i], pq_vectors_[j]);

        auto diff = std::abs(real_distance - pq_distance);
        diff_max = std::max(diff_max, diff);
        total_diff += diff;
      }
      vec_max = std::max(vec_max, l2_distance(feature_vectors[i]));
    }

    return std::make_tuple(diff_max / vec_max, total_diff / total_normalizer);
  }

  template <class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        typename ColMajorMatrix<feature_type>::span_type,
        typename ColMajorMatrix<feature_type>::span_type>
  auto verify_symmetric_pq_distances(
      const ColMajorMatrix<feature_type>& feature_vectors) {
    double total_diff = 0.0;
    double total_normalizer = 0.0;
    auto local_sub_distance = SubDistance{0, dimension_};

    score_type diff_max = 0.0;
    score_type vec_max = 0.0;
    for (size_t i = 0; i < num_vectors(feature_vectors); ++i) {
      for (size_t j = i + 1; j < num_vectors(feature_vectors); ++j) {
        auto real_distance =
            local_sub_distance(feature_vectors[i], feature_vectors[j]);
        total_normalizer += real_distance;

        auto pq_distance =
            this->sub_distance_symmetric(pq_vectors_[i], pq_vectors_[j]);

        auto diff = std::abs(real_distance - pq_distance);
        diff_max = std::max(diff_max, diff);
        total_diff += diff;
      }
      auto zeros = std::vector<feature_type>(dimension_);
      vec_max =
          std::max(vec_max, local_sub_distance(feature_vectors[i], zeros));
    }

    return std::make_tuple(diff_max / vec_max, total_diff / total_normalizer);
  }

  /**
   * @brief Compare the metadata information between two flatpq_index
   * @param rhs
   * @return
   */
  bool compare_metadata(const flatpq_index& rhs) {
    if (dimension_ != rhs.dimension_) {
      std::cout << "dimension_ " << dimension_ << " != " << rhs.dimension_
                << std::endl;
      return false;
    }
    if (num_subspaces_ != rhs.num_subspaces_) {
      std::cout << "num_subspaces_ " << num_subspaces_
                << " != " << rhs.num_subspaces_ << std::endl;
      return false;
    }
    if (sub_dimension_ != rhs.sub_dimension_) {
      std::cout << "sub_dimension_ " << sub_dimension_
                << " != " << rhs.sub_dimension_ << std::endl;
      return false;
    }
    if (bits_per_subspace_ != rhs.bits_per_subspace_) {
      std::cout << "bits_per_subspace_ " << bits_per_subspace_
                << " != " << rhs.bits_per_subspace_ << std::endl;
      return false;
    }
    if (num_clusters_ != rhs.num_clusters_) {
      std::cout << "num_clusters_ " << num_clusters_
                << " != " << rhs.num_clusters_ << std::endl;
      return false;
    }
    if (tol_ != rhs.tol_) {
      std::cout << "tol_ " << tol_ << " != " << rhs.tol_ << std::endl;
      return false;
    }
    if (max_iter_ != rhs.max_iter_) {
      std::cout << "max_iter_ " << max_iter_ << " != " << rhs.max_iter_
                << std::endl;
      return false;
    }
    if (num_threads_ != rhs.num_threads_) {
      std::cout << "num_threads_ " << num_threads_ << " != " << rhs.num_threads_
                << std::endl;
      return false;
    }
    return true;
  }

  /**
   * @brief Compare the pq vectors information between two flatpq_index
   * @param rhs
   * @return
   */
  auto compare_pq_vectors(const flatpq_index& rhs) {
    // @todo use std::equal
    if (pq_vectors_.size() != rhs.pq_vectors_.size() ||
        num_vectors(pq_vectors_) != num_vectors(rhs.pq_vectors_)) {
      std::cout << "pq_vectors_.size() " << pq_vectors_.size()
                << " != " << rhs.pq_vectors_.size() << std::endl;
      return false;
    }
    for (size_t i = 0; i < num_vectors(pq_vectors_); ++i) {
      if (!std::equal(
              begin(pq_vectors_[i]),
              end(pq_vectors_[i]),
              begin(rhs.pq_vectors_[i]))) {
        std::cout << "pq_vectors_[" << i << "] != rhs.pq_vectors_[" << i << "]"
                  << std::endl;
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Compare the centroids information between two flatpq_index
   * @param rhs
   * @return
   */
  auto compare_centroids(const flatpq_index& rhs) {
    // @todo use std::equal
    if (centroids_.size() != rhs.centroids_.size() ||
        num_vectors(centroids_) != num_vectors(rhs.centroids_)) {
      std::cout << "centroids_.size() " << centroids_.size()
                << " != " << rhs.centroids_.size() << std::endl;
      return false;
    }
    for (size_t i = 0; i < num_vectors(centroids_); ++i) {
      if (!std::equal(
              begin(centroids_[i]),
              end(centroids_[i]),
              begin(rhs.centroids_[i]))) {
        std::cout << "centroids_[" << i << "] != rhs.centroids_[" << i << "]"
                  << std::endl;
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Compare the distance table information between two flatpq_index
   * @param rhs
   * @return
   */
  auto compare_distance_tables(const flatpq_index& rhs) {
    if (distance_tables_.size() != rhs.distance_tables_.size()) {
      std::cout << "distance_tables_.size() " << distance_tables_.size()
                << " != " << rhs.distance_tables_.size() << std::endl;
      return false;
    }
    for (size_t i = 0; i < distance_tables_.size(); ++i) {
      if (distance_tables_[i] != rhs.distance_tables_[i]) {
        std::cout << "distance_tables_[" << i << "] != rhs.distance_tables_["
                  << i << "]" << std::endl;
        return false;
      }
    }
    return true;
  }
};

#endif  // TILEDB_FLATPQ_INDEX_
