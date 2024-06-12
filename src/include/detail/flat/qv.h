/**
 * @file   flat/qv.h
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
 * Flat L2 query implementations using the qv ordering (loop over all query
 * vectors on outer loop and over all database vectors on inner loop).
 *
 * We currently assume that the database is loaded into memory.  This is a
 * reasonable assumption for small databases (and/or large machines), but for
 * large databases we may want to implement a blocked / out-of-core
 *
 * @todo Implement a blocked / out-of-core version?
 * @todo Are there other optimizations to apply?
 */

#ifndef TILEDB_FLAT_QV_H
#define TILEDB_FLAT_QV_H

#include <future>
#include <numeric>
#include <vector>

#include "algorithm.h"
#include "concepts.h"
#include "linalg.h"
#include "scoring.h"
#include "utils/fixed_min_heap.h"
#include "utils/timer.h"

namespace detail::flat {

/**
 * This algorithm computes the scores by looping over all query vector and
 * all database vectors, but rather than forming the entire score matrix,
 * it only computes the scores for the query given in the outer loop.  It
 * then computes the top k for that query and stores the results in the
 * top_k matrix.  The top k values are computed either using the nth_element
 * algorithm or a heap based algorithm, depending on the value of nth.
 *
 * @tparam DB The type of the database.
 * @tparam Q The type of the query.
 * @param db The database.
 * @param q The query.
 * @param k The number of top k results to return.
 * @param nthreads The number of threads to use in parallel execution.
 * @return A matrix of size k x #queries containing the top k results for each
 * query.
 */
template <
    feature_vector_array DB,
    query_vector_array Q,
    class Distance = sum_of_squares_distance>
[[deprecated]] auto qv_query_heap_0(
    const DB& db,
    const Q& q,
    int k_nn,
    unsigned int nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  using id_type = uint64_t;
  using score_type = float;

  // @todo This should work with num_top_k as min, rather than k_nn if
  // k_nn < num_vectors(db) but somehow it doesn't
  // A matrix with many uninitialized values is created....
  auto num_top_k = std::min<decltype(k_nn)>(k_nn, num_vectors(db));
  ColMajorMatrix<id_type> top_k(k_nn, /*num_top_k,*/ num_vectors(q));

  auto par = stdx::execution::indexed_parallel_policy{nthreads};
  stdx::range_for_each(
      std::move(par), q, [&](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
        size_t size_q = num_vectors(q);
        size_t size_db = num_vectors(db);

        // @todo can we do this more efficiently?
        Vector<score_type> scores(size_db);

        for (size_t i = 0; i < size_db; ++i) {
          scores[i] = distance(q_vec, db[i]);
        }
        get_top_k_from_scores(scores, top_k[j], k_nn);
      });

  // This shows uninitialized matrix when k_nn = 10 and num_vectors(db) = 1
  // std::cout << "======\n";
  // debug_matrix(top_k);
  // std::cout << "======\n";
  return top_k;
}

/**
 * This algorithm computes the scores by looping over all query vector and
 * all database vectors, but rather than forming the score matrix, or even a
 * score vector as in the qv_query_nth algorithm, it computes the top k values
 * on the fly, using a fixed_min_heap.
 *
 * It has two overloads for the case with and without ids.
 *
 * @tparam DB The type of the database.
 * @tparam Q The type of the query.
 * @param db The database.
 * @param q The query.
 * @param k The number of top k results to return.
 * @param nthreads The number of threads to use in parallel execution.
 * @return A matrix of size k x #queries containing the top k results for each
 * query.
 */
template <
    class T,
    class DB,
    class Q,
    class ID,
    class Distance = sum_of_squares_distance>
auto qv_query_heap(
    T,
    const DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{});

template <class DB, class Q, class Distance = sum_of_squares_distance>
auto qv_query_heap(
    const DB& db,
    const Q& q,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return qv_query_heap(
      without_ids{}, db, q, std::vector<uint64_t>{}, k_nn, nthreads, distance);
}

template <class DB, class Q, class ID, class Distance = sum_of_squares_distance>
auto qv_query_heap(
    const DB& db,
    const Q& q,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return qv_query_heap(with_ids{}, db, q, ids, k_nn, nthreads, distance);
}

// @todo Add to out of core
template <
    class T,
    feature_vector_array DB,
    feature_vector_array Q,
    class ID,
    class Distance = sum_of_squares_distance>
auto qv_query_heap(
    T,
    const DB& db,
    const Q& query,
    const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  // print_types(distance);
  // load(db);

  // @todo Definitive spec on whether or not feature_vector_arrays are loaded
  // when the function is called.  IVF assumes calls to "infinite" algorithms
  // are loaded and calls to "finite" algorithms are not loaded.
  // load(db);

  // using feature_type = typename
  // std::remove_reference_t<decltype(db)>::value_type;
  using id_type = typename std::remove_reference_t<decltype(ids)>::value_type;
  using score_type = float;

  auto top_k = ColMajorMatrix<id_type>(k_nn, query.num_cols());
  auto top_k_scores = ColMajorMatrix<score_type>(k_nn, query.num_cols());

  // Have to do explicit asynchronous threading here, as the current parallel
  // algorithms have iterator-based interaces, and the `Matrix` class does not
  // yet have iterators.
  // @todo Implement iterator interface to `Matrix` class

  auto size_db = num_vectors(db);
  auto par = stdx::execution::indexed_parallel_policy{nthreads};
  stdx::range_for_each(
      std::move(par),
      query,
      [&, size_db](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
        fixed_min_pair_heap<score_type, id_type> min_scores(k_nn);

        for (size_t i = 0; i < size_db; ++i) {
          auto score = distance(q_vec, db[i]);
          if constexpr (std::is_same_v<T, with_ids>) {
            min_scores.insert(score, ids[i]);
          } else if constexpr (std::is_same_v<T, without_ids>) {
            min_scores.insert(score, i);
          } else {
            static_assert(always_false<T>, "T must be with_ids or without_ids");
          }
        }

        get_top_k_with_scores_from_heap(min_scores, top_k[j], top_k_scores[j]);
      });

  return std::make_tuple(std::move(top_k_scores), std::move(top_k));
}

template <
    feature_vector_array DB,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto qv_query_heap(
    DB& db,
    const Q& q,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return qv_query_heap(
      without_ids{},
      db,
      q,
      std::vector<uint64_t>{},
      k_nn,
      nthreads,
      distance);  /// ????
}

template <
    feature_vector_array DB,
    feature_vector_array Q,
    class Index,
    class Distance = sum_of_squares_distance>
auto qv_query_heap(
    DB& db,
    const Q& q,
    const std::vector<Index>& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return qv_query_heap(with_ids{}, db, q, ids, k_nn, nthreads, distance);
}

/**
 * This algorithm is similar to `qv_query_heap`, but it tiles the query loop
 * and the database loop (2X2). This is done to improve cache locality.  We
 * assume the feature vector array has been loaded into memory prior to the
 * call.
 * @tparam DB The type of the database.
 * @tparam Q The type of the query.
 * @param db The database.
 * @param query The query.
 * @param k The number of top k results to return.
 * @param nthreads The number of threads to use in parallel execution.
 * @return A matrix of size k x #queries containing the top k results for each
 * query.
 */
template <
    class T,
    class DB,
    class Q,
    class ID,
    class Distance = sum_of_squares_distance>
auto qv_query_heap_tiled(
    T,
    DB& db,
    const Q& query,
    [[maybe_unused]] const ID& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  load(db);

  // debug_matrix(db);
  // debug_matrix(query);

  scoped_timer _{tdb_func__};

  // using feature_type = typename
  // std::remove_reference_t<decltype(db)>::value_type;
  using id_type = typename std::remove_reference_t<decltype(ids)>::value_type;
  using score_type = float;

  // Have to do explicit asynchronous threading here, as the current parallel
  // algorithms have iterator-based interaces, and the `Matrix` class does not
  // yet have iterators.
  // @todo Implement iterator interface to `Matrix` class
  size_t size_db = db.num_cols();
  size_t container_size = num_vectors(query);
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
      num_vectors(query), fixed_min_pair_heap<score_type, id_type>(k_nn));

  // @todo: Use range::for_each
  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async,
          [start, stop, &query, &db, &min_scores, &ids, &distance]() {
            (void)ids;  // Suppress warnings
            auto len = 2 * ((stop - start) / 2);
            auto end = start + len;

            // auto min_scores0 = fixed_min_pair_heap<score_type, id_type> (k);
            // auto min_scores1 = fixed_min_pair_heap<score_type, id_type> (k);

            for (auto j = start; j != end; j += 2) {
              auto j0 = j + 0;
              auto j1 = j + 1;
              auto q_vec_0 = query[j0];
              auto q_vec_1 = query[j1];

              auto kstop =
                  std::min<size_t>(num_vectors(db), 2 * (num_vectors(db) / 2));

              for (size_t kp = 0; kp < kstop; kp += 2) {
                auto score_00 = distance(q_vec_0, db[kp + 0]);
                auto score_01 = distance(q_vec_0, db[kp + 1]);
                auto score_10 = distance(q_vec_1, db[kp + 0]);
                auto score_11 = distance(q_vec_1, db[kp + 1]);

                if constexpr (std::is_same_v<T, with_ids>) {
                  min_scores[j0].insert(score_00, ids[kp + 0]);
                  min_scores[j0].insert(score_01, ids[kp + 1]);
                  min_scores[j1].insert(score_10, ids[kp + 0]);
                  min_scores[j1].insert(score_11, ids[kp + 1]);
                } else if constexpr (std::is_same_v<T, without_ids>) {
                  min_scores[j0].insert(score_00, kp + 0);
                  min_scores[j0].insert(score_01, kp + 1);
                  min_scores[j1].insert(score_10, kp + 0);
                  min_scores[j1].insert(score_11, kp + 1);
                } else {
                  static_assert(
                      always_false<T>, "T must be with_ids or without_ids");
                }
              }
              /*
               * Cleanup the last iteration(s) of k
               */
              for (size_t kp = kstop; kp < num_vectors(db); ++kp) {
                auto score_00 = distance(q_vec_0, db[kp + 0]);
                auto score_10 = distance(q_vec_1, db[kp + 0]);

                if constexpr (std::is_same_v<T, with_ids>) {
                  min_scores[j0].insert(score_00, ids[kp + 0]);
                  min_scores[j1].insert(score_10, ids[kp + 0]);
                } else if constexpr (std::is_same_v<T, without_ids>) {
                  min_scores[j0].insert(score_00, kp + 0);
                  min_scores[j1].insert(score_10, kp + 0);
                } else {
                  static_assert(
                      always_false<T>, "T must be with_ids or without_ids");
                }
              }
            }

            /*
             * Cleanup the last iteration(s) of j
             */
            for (auto j = end; j < stop; ++j) {
              auto j0 = j + 0;
              auto q_vec_0 = query[j0];

              auto kstop =
                  std::min<size_t>(num_vectors(db), 2 * (num_vectors(db) / 2));
              for (size_t kp = 0; kp < kstop; kp += 2) {
                auto score_00 = distance(q_vec_0, db[kp + 0]);
                auto score_01 = distance(q_vec_0, db[kp + 1]);

                if constexpr (std::is_same_v<T, with_ids>) {
                  min_scores[j0].insert(score_00, ids[kp + 0]);
                  min_scores[j0].insert(score_01, ids[kp + 1]);
                } else if constexpr (std::is_same_v<T, without_ids>) {
                  min_scores[j0].insert(score_00, kp + 0);
                  min_scores[j0].insert(score_01, kp + 1);
                } else {
                  static_assert(
                      always_false<T>, "T must be with_ids or without_ids");
                }
              }
              for (size_t kp = kstop; kp < num_vectors(db); ++kp) {
                auto score_00 = distance(q_vec_0, db[kp + 0]);
                if constexpr (std::is_same_v<T, with_ids>) {
                  min_scores[j0].insert(score_00, ids[kp + 0]);
                } else if constexpr (std::is_same_v<T, without_ids>) {
                  min_scores[j0].insert(score_00, kp + 0);
                } else {
                  static_assert(
                      always_false<T>, "T must be with_ids or without_ids");
                }
              }
            }
          }));
    }
  }

  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].get();
  }

  return get_top_k_with_scores(min_scores, k_nn);
}

template <
    feature_vector_array DB,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto qv_query_heap_tiled(
    DB& db,
    const Q& q,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return qv_query_heap_tiled(
      without_ids{}, db, q, std::vector<uint64_t>{}, k_nn, nthreads, distance);
}

template <
    feature_vector_array DB,
    feature_vector_array Q,
    class Index,
    class Distance = sum_of_squares_distance>
auto qv_query_heap_tiled(
    DB& db,
    const Q& q,
    const std::vector<Index>& ids,
    int k_nn,
    unsigned nthreads,
    Distance distance = Distance{}) {
  return qv_query_heap_tiled(with_ids{}, db, q, ids, k_nn, nthreads, distance);
}

/**
 * @brief Find the single nearest neighbor of each query vector in the database.
 * This is essentially qv_query_heap, specialized for the case of k = 1.  Note
 * that if we call this to query a set of centroids using the feature vector
 * array as the query vectors, it will return the id of closest centroid to
 * each feature vector.  I.e., it will label each feature vector with a
 * partition number.  This will be used later to reorder the feature vectors
 * so that all vectors in the same partition are contiguous.
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param nthreads
 * @return
 */
template <
    feature_vector_array DB,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto qv_partition(
    const DB& db,
    const Q& q,
    unsigned nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  // Just need a single vector -- creating an index, not ids, so hardcoded
  // size_t is okay to use here
  using id_type = size_t;
  using score_type = float;
  auto size_db = num_vectors(db);

  std::vector<id_type> top_k(num_vectors(q));

  auto par = stdx::execution::indexed_parallel_policy{(size_t)nthreads};
  // For each query vector, find the closest vector in the database (i.e. in
  // centroids).
  stdx::range_for_each(
      std::move(par), q, [&, size_db](auto&& qvec, auto&& n = 0, auto&& j = 0) {
        score_type min_score = std::numeric_limits<score_type>::max();
        size_t idx = 0;

        for (size_t i = 0; i < size_db; ++i) {
          auto score = distance(qvec, db[i]);
          if (score < min_score) {
            min_score = score;
            idx = i;
          }
        }
        top_k[j] = idx;
      });

  return top_k;
}

/**
 * @brief Find the single nearest neighbor of each query vector in the database.
 * This is essentially qv_query_heap, specialized for k = 1.
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param nthreads
 * @return
 */
template <
    feature_vector_array DB,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto qv_partition_with_scores(
    const DB& db,
    const Q& q,
    unsigned nthreads,
    Distance distance = Distance{}) {
  scoped_timer _{tdb_func__};

  auto size_db = ::num_vectors(db);

  // Just need a single vector
  std::vector<size_t> top_k(q.num_cols());
  std::vector<size_t> top_k_scores(q.num_cols());

  auto par = stdx::execution::indexed_parallel_policy{(size_t)nthreads};
  stdx::range_for_each(
      std::move(par), q, [&, size_db](auto&& qvec, auto&& n = 0, auto&& j = 0) {
        float min_score = std::numeric_limits<float>::max();
        size_t idx = 0;

        for (size_t i = 0; i < size_db; ++i) {
          auto score = distance(qvec, db[i]);
          if (score < min_score) {
            min_score = score;
            idx = i;
          }
        }
        top_k[j] = idx;
        top_k_scores[j] = min_score;
      });

  return std::make_tuple(top_k_scores, top_k);
}

}  // namespace detail::flat

#endif  // TILEDB_FLAT_QV_H
