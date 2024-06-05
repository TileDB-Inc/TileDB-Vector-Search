/**
 * @file kmeans.h
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

#ifndef TILEDB_KMEANS_H
#define TILEDB_KMEANS_H

#include <random>

#include "detail/flat/qv.h"
#include "utils/logging.h"

enum class kmeans_init { none, kmeanspp, random };

/****************************************************************************
 * kmeans algorithm for clustering
 * @todo Move out of this class (and implement as free functions)
 ****************************************************************************/
/**
 * @brief Use the kmeans++ algorithm to choose initial centroids.
 * The current implementation follows the algorithm described in
 * the literature (Arthur and Vassilvitskii, 2007):
 *
 *  1a. Choose an initial centroid uniformly at random from the training set
 *  1b. Choose the next centroid from the training set
 *      2b. For each data point x not chosen yet, compute D(x), the distance
 *          between x and the nearest centroid that has already been chosen.
 *      3b. Choose one new data point at random as a new centroid, using a
 *          weighted probability distribution where a point x is chosen
 *          with probability proportional to D(x)2.
 *  3. Repeat Steps 2b and 3b until k centers have been chosen.
 *
 *  The initial centroids are stored in the member variable `centroids_`.
 *  It is expected that the centroids will be further refined by the
 *  kmeans algorithm.
 *
 * @param training_set Array of vectors to cluster.
 *
 * @todo Implement greedy kmeans++: choose several new centers during each
 * iteration, and then greedily chose the one that most decreases φ
 * @todo Finish implementation using triangle inequality.
 */

namespace {
static std::mt19937 gen_;
}
template <
    feature_vector_array V,
    feature_vector_array C,
    class Distance = sum_of_squares_distance>
void kmeans_pp(
    const V& training_set,
    C& centroids_,
    size_t num_partitions_,
    size_t num_threads_,
    Distance distancex = Distance{}) {
  scoped_timer _{__FUNCTION__};
  if (::num_vectors(training_set) == 0) {
    return;
  }
  using score_type = typename C::value_type;

  std::uniform_int_distribution<> dis(0, training_set.num_cols() - 1);
  auto choice = dis(gen_);

  std::copy(
      begin(training_set[choice]),
      end(training_set[choice]),
      begin(centroids_[0]));

  // Initialize distances, leaving some room to grow
  std::vector<score_type> distances(
      training_set.num_cols(), std::numeric_limits<score_type>::max() / 8192);

#ifdef _TRIANGLE_INEQUALITY
  std::vector<centroid_feature_type> centroid_centroid(num_partitions_, 0.0);
  std::vector<index_type> nearest_centroid(training_set.num_cols(), 0);
#endif

  // Calculate the remaining centroids using K-means++ algorithm
  for (size_t i = 1; i < num_partitions_; ++i) {
    stdx::execution::indexed_parallel_policy par{num_threads_};
    stdx::range_for_each(
        std::move(par),
        training_set,
        [&distancex, &centroids_, &distances, i](
            auto&& vec, size_t n, size_t j) {

    // Note: centroid i-1 is the newest centroid

#ifdef _TRIANGLE_INEQUALITY
          // using triangle inequality, only need to calculate distance to the
          // newest centroid if distance between vec and its current nearest
          // centroid is greater than half the distance between the newest
          // centroid and vectors nearest centroid (1/4 distance squared)

          float min_distance = distances[j];
          if (centroid_centroid[nearest_centroid[j]] < 4 * min_distance) {
            float new_distance = distancex(vec, centroids_[i - 1]);
            if (new_distance < min_distance) {
              min_distance = new_distance;
              nearest_centroid[j] = i - 1;
              distances[j] = min_distance;
            }
          }
#else
          auto new_distance = distancex(vec, centroids_[i - 1]);
          auto min_distance = std::min(distances[j], new_distance);
          distances[j] = min_distance;
#endif
        });

    // Select the next centroid based on the probability proportional to
    // distance squared -- note we did not normalize the vectors ourselves
    // since `discrete_distribution` implicitly does that for us
    std::discrete_distribution<size_t> probabilityDistribution(
        distances.begin(), distances.end());
    size_t nextIndex = probabilityDistribution(gen_);
    std::copy(
        begin(training_set[nextIndex]),
        end(training_set[nextIndex]),
        begin(centroids_[i]));
    distances[nextIndex] = 0.0;

#ifdef _TRIANGLE_INEQUALITY
    // Update centroid-centroid distances -- only need distances from each
    // existing to the new one
    centroid_centroid[i] = distancex(centroids_[i], centroids_[i - 1]);
    for (size_t j = 0; j < i; ++j) {
      centroid_centroid[j] = distancex(centroids_[i], centroids_[j]);
    }
#endif
  }
}

/**
 * @brief Initialize centroids by choosing them at random from training set.
 * @param training_set Array of vectors to cluster.
 */
template <feature_vector_array V, feature_vector_array C>
void kmeans_random_init(
    const V& training_set, C& centroids_, size_t num_partitions_) {
  scoped_timer _{__FUNCTION__};
  if (::num_vectors(training_set) == 0) {
    return;
  }

  std::vector<size_t> indices(num_partitions_);

  std::vector<bool> visited(training_set.num_cols(), false);
  std::uniform_int_distribution<> dis(0, training_set.num_cols() - 1);
  for (size_t i = 0; i < num_partitions_; ++i) {
    size_t index;
    do {
      index = dis(gen_);
    } while (visited[index]);
    indices[i] = index;
    visited[index] = true;
  }

  // std::iota(begin(indices), end(indices), 0);
  // std::shuffle(begin(indices), end(indices), gen_);
  for (size_t i = 0; i < num_partitions_; ++i) {
    std::copy(
        begin(training_set[indices[i]]),
        end(training_set[indices[i]]),
        begin(centroids_[i]));
  }
}

/**
 * @brief Use kmeans algorithm to cluster vectors into centroids.  Beginning
 * with an initial set of centroids, the algorithm iteratively partitions
 * the training_set into clusters, and then recomputes new centroids based
 * on the clusters.  The algorithm terminates when the change in centroids
 * is less than a threshold tolerance, or when a maximum number of
 * iterations is reached.
 *
 * @param training_set Array of vectors to cluster.
 * @todo make reassignment optional
 */
template <
    feature_vector_array V,
    feature_vector_array C,
    class Distance = sum_of_squares_distance>
void train_no_init(
    const V& training_set,
    C& centroids_,
    size_t dimension_,
    size_t num_partitions_,
    size_t max_iter_,
    float tol_,
    size_t num_threads_,
    float reassign_ratio_ = 0.05,
    Distance distancex = Distance{}) {
  scoped_timer _{__FUNCTION__};
  if (::num_vectors(training_set) == 0) {
    return;
  }
  using feature_type = typename V::value_type;
  using centroid_feature_type = typename C::value_type;
  using index_type = size_t;

  std::vector<size_t> degrees(num_partitions_, 0);
  auto new_centroids =
      ColMajorMatrix<centroid_feature_type>(dimension_, num_partitions_);

  for (size_t iter = 0; iter < max_iter_; ++iter) {
    auto [scores, parts] = detail::flat::qv_partition_with_scores(
        centroids_, training_set, num_threads_, distancex);

    std::fill(
        new_centroids.data(),
        new_centroids.data() +
            new_centroids.num_rows() * new_centroids.num_cols(),
        0.0);
    std::fill(begin(degrees), end(degrees), 0);

    // How many centroids should we try to fix up
    size_t heap_size = std::ceil(reassign_ratio_ * num_partitions_) + 5;
    auto high_scores = fixed_min_pair_heap<
        feature_type,
        index_type,
        std::greater<feature_type>>(heap_size, std::greater<feature_type>());
    auto low_degrees = fixed_min_pair_heap<index_type, index_type>(heap_size);

    // @todo parallelize -- by partition
    for (size_t i = 0; i < ::num_vectors(training_set); ++i) {
      auto part = parts[i];
      auto centroid = new_centroids[part];
      auto vector = training_set[i];
      for (size_t j = 0; j < dimension_; ++j) {
        centroid[j] += vector[j];
      }
      ++degrees[part];
      high_scores.insert(scores[i], i);
    }

    size_t max_degree = 0;
    for (size_t i = 0; i < num_partitions_; ++i) {
      auto degree = degrees[i];
      max_degree = std::max<size_t>(max_degree, degree);
      low_degrees.insert(degree, i);
    }
    size_t lower_degree_bound = std::ceil(max_degree * reassign_ratio_);

    // Don't reassign if we are on last iteration
    if (iter != max_iter_ - 1) {
// Experiment with random reassignment
#if 0
        // Pick a random vector to be a new centroid
        std::uniform_int_distribution<> dis(0, training_set.num_cols() - 1);
        for (auto&& [degree, zero_part] : low_degrees) {
          if (degree < lower_degree_bound) {
            auto index = dis(gen_);
            auto rand_vector = training_set[index];
            auto low_centroid = new_centroids[zero_part];
            std::copy(begin(rand_vector), end(rand_vector), begin(low_centroid));
            for (size_t i = 0; i < dimension_; ++i) {
              new_centroids[parts[index]][i] -= rand_vector[i];
            }
          }
        }
#endif
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
        std::copy(begin(high_vector), end(high_vector), begin(low_centroid));
        for (size_t i = 0; i < dimension_; ++i) {
          new_centroids[parts[high_vector_id]][i] -= high_vector[i];
        }
        ++degrees[zero_part];
        --degrees[parts[high_vector_id]];
      }
    }
    /**
     * Check for convergence
     */
    // @todo parallelize?
    float max_diff = 0.0;
    float total_weight = 0.0;
    for (size_t j = 0; j < num_partitions_; ++j) {
      if (degrees[j] != 0) {
        auto centroid = new_centroids[j];
        for (size_t k = 0; k < dimension_; ++k) {
          centroid[k] /= degrees[j];
          total_weight += centroid[k] * centroid[k];
        }
      }
      auto diff = distancex(centroids_[j], new_centroids[j]);
      max_diff = std::max<float>(max_diff, diff);
    }
    centroids_.swap(new_centroids);
    if (max_diff < tol_ * total_weight) {
      break;
    }

// Temporary printf debugging
#if 0
        auto mm = std::minmax_element(begin(degrees), end(degrees));
        float sum = std::accumulate(begin(degrees), end(degrees), 0);
        float average = sum / (float)size(degrees);

        auto min = *mm.first;
        auto max = *mm.second;
        auto diff = max - min;
        std::cout << "avg: " << average << " sum: " << sum << " min: " << min
                  << " max: " << max << " diff: " << diff << std::endl;
#endif
  }

// Temporary printf debugging to file (for post-processing)
#ifdef _SAVE_PARTITIONS
  {
    char tempFileName[L_tmpnam];
    tmpnam(tempFileName);

    std::ofstream file(tempFileName);
    if (!file) {
      std::cout << "Error opening the file." << std::endl;
      return;
    }

    for (const auto& element : degrees) {
      file << element << ',';
    }
    file << std::endl;

    for (auto s = 0; s < training_set.num_cols(); ++s) {
      for (auto t = 0; t < training_set.num_rows(); ++t) {
        file << std::to_string(training_set(t, s)) << ',';
      }
      file << std::endl;
    }
    file << std::endl;

    for (auto s = 0; s < centroids_.num_cols(); ++s) {
      for (auto t = 0; t < centroids_.num_rows(); ++t) {
        file << std::to_string(centroids_(t, s)) << ',';
      }
      file << std::endl;
    }

    file.close();

    std::cout << "Data written to file: " << tempFileName << std::endl;
  }
#endif
}

template <feature_vector_array V, feature_vector_array C>
void sub_kmeans_random_init(
    const V& training_set,
    C& centroids,
    size_t sub_begin,
    size_t sub_end,
    size_t seed = 0) {
  scoped_timer _{__FUNCTION__};

  size_t num_clusters =
      std::min(num_vectors(training_set), num_vectors(centroids));
  if (num_clusters == 0) {
    return;
  }

  std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
  std::uniform_int_distribution<> dis(0, num_vectors(training_set) - 1);

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
 * @brief Run kmeans on the indicated view of the training set.  The centroids
 * are updated in place, but only in the range [sub_begin, sub_end).
 * @tparam V Type of training set vectors
 * @tparam C Type of centroids vectors
 * @param training_set Training set vectors
 * @param centroids Initial locations of centroids.  Will be updated in
 * locations [sub_begin, sub_end).
 * @param sub_begin Beginning of the subspace to be computed
 * @param sub_end End of the subspace to be computed
 * @param num_clusters Number of clusters to find
 * @return
 *
 * @todo update with concepts
 * @todo fix up zero sized partitions
 * @todo this would be more cache friendly to do sub_begin and sub_end
 * in an inner loop
 * @todo We can probably just reuse plain kmeans at some point.
 */
template <
    feature_vector_array V,
    feature_vector_array C,
    class SubDistance = cached_sub_sum_of_squares_distance>
auto sub_kmeans(
    const V& training_set,
    C& centroids,
    size_t sub_begin,
    size_t sub_end,
    size_t num_clusters,
    double tol,
    size_t max_iter,
    size_t num_threads,
    float reassign_ratio = 0.05,
    bool reassign_later = false) {
  size_t sub_dimension_ = sub_end - sub_begin;
  auto local_sub_distance = SubDistance{sub_begin, sub_end};

  std::vector<size_t> degrees(num_clusters, 0);

  // Copy centroids to new centroids -- note only one subspace will be changing
  // @todo Keep new_centroids outside function so we don't need to copy all
  C new_centroids(dimensions(centroids), num_vectors(centroids));
  for (size_t i = 0; i < num_vectors(new_centroids); ++i) {
    for (size_t j = 0; j < dimensions(new_centroids); ++j) {
      new_centroids(j, i) = centroids(j, i);
    }
  }

  size_t iter = 0;
  double max_diff = 0.0;
  double total_weight = 0.0;
  for (iter = 0; iter < max_iter; ++iter) {
    // The reassignment code should be in runtime if test, but there are some
    // scoping issues that need to be resolved with some of the variables.
    // Some variables (vectors) we don't want to exist at all if we aren't
    // reassigning, e.g., scores

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

#endif  // TILEDB_KMEANS_H
