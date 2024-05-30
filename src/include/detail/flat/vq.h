/**
 * @file   flat/vq.h
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
 *
 */

#ifndef TILEDB_FLAT_VQ_H
#define TILEDB_FLAT_VQ_H

#include <future>
#include <vector>

#include "algorithm.h"
#include "linalg.h"
#include "scoring.h"
#include "utils/timer.h"

namespace detail::flat {

/**
 * This algorithm accumulates top_k as it goes, but in a "transposed" fashion to
 * qv_query.  Namely, it loops over the database vectors on the outer loop,
 * where each thread keeps its own set of heaps for each query vector.  After
 * The database vector loop, the heaps are merged and then copied to `top_k`.
 *
 * @todo Support out of core operation
 * @todo Unify out of core and not out of core versions.
 */

// @todo Support out of core
template <
    class T,
    class DB,
    class Q,
    class ID,
    class Distance = sum_of_squares_distance>
auto vq_query_heap(
    T,
    DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  // @todo Need to get the total number of queries, not just the first block
  // @todo Use Matrix here rather than vector of vectors

  // using feature_type = typename
  // std::remove_reference_t<decltype(db)>::value_type;
  using id_type = typename std::remove_reference_t<decltype(ids)>::value_type;
  using score_type = float;

  std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>> scores(
      nthreads,
      std::vector<fixed_min_pair_heap<score_type, id_type>>(
          num_vectors(q), fixed_min_pair_heap<score_type, id_type>(k_nn)));

  unsigned size_q = num_vectors(q);
  auto par = stdx::execution::indexed_parallel_policy{nthreads};

  log_timer _i{tdb_func__ + " in RAM"};

  // @todo Can we do blocking in the parallel for_each somehow?

  // @note load(db) will always return false for a matrix
  load(db);
  do {
    _i.start();
    auto size_q = num_vectors(q);
    stdx::range_for_each(
        std::move(par),
        db,
        [&, size_q](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
          size_t index = i + col_offset(db);
          for (size_t j = 0; j < size_q; ++j) {
            auto score = distance(q[j], db_vec);
            if constexpr (std::is_same_v<T, with_ids>) {
              scores[n][j].insert(score, ids[index]);
            } else if constexpr (std::is_same_v<T, without_ids>) {
              scores[n][j].insert(score, index);
            } else {
              static_assert(
                  always_false<T>, "T must be with_ids or without_ids");
            }
          }
        });
    _i.stop();
  } while (load(db));

  consolidate_scores(scores);
  return get_top_k_with_scores(scores, k_nn);
}

template <class DB, class Q, class Distance = sum_of_squares_distance>
auto vq_query_heap(
    DB& db, Q& q, int k_nn, unsigned nthreads, Distance distance = Distance{}) {
  return vq_query_heap(
      without_ids{}, db, q, std::vector<uint64_t>{}, k_nn, nthreads, distance);
}

template <
    feature_vector_array DB,
    query_vector_array Q,
    class Index,
    class Distance = sum_of_squares_distance>
auto vq_query_heap(
    DB& db,
    Q& q,
    const std::vector<Index>& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return vq_query_heap(with_ids{}, db, q, ids, k_nn, nthreads, distance);
}

/**
 * @brief Tiled version of the above.  WIP.
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param k
 * @param nthreads
 * @return
 */
template <class T, class DB, class Q, class ID, class Distance>
auto vq_query_heap_tiled(
    T,
    DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance);

template <class DB, class Q, class Distance = sum_of_squares_distance>
auto vq_query_heap_tiled(
    DB& db,
    const Q& q,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return vq_query_heap_tiled(
      without_ids{}, db, q, std::vector<uint64_t>{}, k_nn, nthreads, distance);
}

template <class DB, class Q, class ID, class Distance = sum_of_squares_distance>
auto vq_query_heap_tiled(
    DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return vq_query_heap_tiled(with_ids{}, db, q, ids, k_nn, nthreads, distance);
}

template <class T, class DB, class Q, class ID, class Distance>
auto vq_query_heap_tiled(
    T,
    DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance) {
  // @todo Need to get the total number of queries, not just the first block
  // @todo Use Matrix here rather than vector of vectors

  // using feature_type = typename
  // std::remove_reference_t<decltype(db)>::value_type;
  using id_type = typename std::remove_reference_t<decltype(ids)>::value_type;
  using score_type = float;

  std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>> scores(
      nthreads,
      std::vector<fixed_min_pair_heap<score_type, id_type>>(
          num_vectors(q), fixed_min_pair_heap<score_type, id_type>(k_nn)));

  unsigned size_q = num_vectors(q);
  auto par = stdx::execution::indexed_parallel_policy{nthreads};

  log_timer _i{tdb_func__ + " in RAM"};

  // @todo Can we do blocking in the parallel for_each somehow?
  do {
    _i.start();
    stdx::range_for_each(
        std::move(par),
        db,
        [&, size_q](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
          std::remove_cvref_t<decltype(i)> index = 0;
          index = i + col_offset(db);
          for (size_t j = 0; j < size_q; ++j) {
            auto score = distance(q[j], db_vec);
            if constexpr (std::is_same_v<T, with_ids>) {
              scores[n][j].insert(score, ids[index]);
            } else if constexpr (std::is_same_v<T, without_ids>) {
              scores[n][j].insert(score, index);
            } else {
              static_assert(
                  always_false<T>, "T must be with_ids or without_ids");
            }
          }
        });
    _i.stop();
  } while (load(db));

  consolidate_scores(scores);
  return get_top_k_with_scores(scores, k_nn);
}

// ====================================================================================================

template <
    class T,
    class DB,
    class Q,
    class ID,
    class Distance = sum_of_squares_distance>
auto vq_query_heap_2(
    T,
    DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{});

template <class DB, class Q, class Distance = sum_of_squares_distance>
auto vq_query_heap_2(
    DB& db,
    const Q& q,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return vq_query_heap_2(
      without_ids{}, db, q, std::vector<uint64_t>{}, k_nn, nthreads, distance);
}

template <
    feature_vector_array DB,
    feature_vector_array Q,
    class ID,
    class Distance = sum_of_squares_distance>
auto vq_query_heap_2(
    DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return vq_query_heap_2(with_ids{}, db, q, ids, k_nn, nthreads, distance);
}

template <class T, class DB, class Q, class ID, class Distance>
auto vq_query_heap_2(
    T,
    DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance) {
  // @todo Need to get the total number of queries, not just the first block
  // @todo Use Matrix here rather than vector of vectors

  // using feature_type = typename
  // std::remove_reference_t<decltype(db)>::value_type;
  using id_type = typename std::remove_reference_t<decltype(ids)>::value_type;
  using score_type = float;

  std::vector<std::vector<fixed_min_pair_heap<score_type, id_type>>> scores(
      nthreads,
      std::vector<fixed_min_pair_heap<score_type, id_type>>(
          num_vectors(q), fixed_min_pair_heap<score_type, id_type>(k_nn)));

  unsigned size_q = num_vectors(q);
  auto par = stdx::execution::indexed_parallel_policy{nthreads};

  log_timer _i{tdb_func__ + " in RAM"};

  // @todo Can we do blocking in the parallel for_each somehow?
  do {
    _i.start();
    stdx::range_for_each(
        std::move(par),
        db,
        [&, size_q](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
          std::remove_cvref_t<decltype(i)> index = 0;
          index = i + col_offset(db);

          for (size_t j = 0; j < size_q; ++j) {
            auto score = distance(q[j], db_vec);
            if constexpr (std::is_same_v<T, with_ids>) {
              scores[n][j].insert(score, ids[index]);
            } else if constexpr (std::is_same_v<T, without_ids>) {
              scores[n][j].insert(score, index);
            } else {
              static_assert(
                  always_false<T>, "T must be with_ids or without_ids");
            }
          }
        });
    _i.stop();
  } while (load(db));

  consolidate_scores(scores);
  return get_top_k_with_scores(scores, k_nn);
}

/**
 * @brief Find the single nearest neighbor for each query vector.  This is
 * essentially vq_query_heap, specialized for k = 1.  Note that if we query
 * the centroids using the set of feature vectors, we will get back the id of
 * closest centroid to each feature vector.  I.e., we will get a partition
 * label for each feature vector.
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param nthreads
 * @return
 */
template <class DB, class Q, class Distance = sum_of_squares_distance>
auto vq_partition(
    const DB& db,
    const Q& q,
    unsigned nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  auto num_queries = num_vectors(q);
  auto top_k = Vector<size_t>(num_queries);

  auto min_scores = std::vector<std::vector<size_t>>(
      nthreads,
      std::vector<size_t>(num_queries, std::numeric_limits<size_t>::max()));
  auto min_ids = std::vector<std::vector<size_t>>(
      nthreads,
      std::vector<size_t>(num_queries, std::numeric_limits<size_t>::max()));
  auto par = stdx::execution::indexed_parallel_policy{(size_t)nthreads};
  stdx::range_for_each(
      std::move(par), db, [&](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
        for (size_t j = 0; j < num_queries; ++j) {
          auto score = distance(q[j], db_vec);
          if (score < min_scores[n][j]) {
            min_scores[n][j] = score;
            min_ids[n][j] = i;
          }
        }
      });

  for (size_t j = 0; j < num_queries; ++j) {
    for (size_t n = 1; n < nthreads; ++n) {
      if (min_scores[0][j] > min_scores[n][j]) {
        min_scores[0][j] = min_scores[n][j];
        min_ids[0][j] = min_ids[n][j];
      }
    }
    top_k[j] = min_ids[0][j];
  }

  return top_k;
}

}  // namespace detail::flat

#endif  // TILEDB_FLAT_VQ_H
