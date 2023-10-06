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
template <class DB, class Q>
[[deprecated]] auto qv_query_heap_0(
    DB& db, const Q& q, int k_nn, unsigned int nthreads) {
  scoped_timer _{tdb_func__};

  ColMajorMatrix<size_t> top_k(k_nn, size(q));

  auto par = stdx::execution::indexed_parallel_policy{nthreads};
  stdx::range_for_each(
      std::move(par), q, [&](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
        size_t size_q = size(q);
        size_t size_db = size(db);

        // @todo can we do this more efficiently?
        Vector<float> scores(size_db);

        for (size_t i = 0; i < size_db; ++i) {
          scores[i] = L2(q_vec, db[i]);
        }
        get_top_k_from_scores(scores, top_k[j], k_nn);
      });

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
template <class T, class DB, class Q, class Index>
auto qv_query_heap(
    T,
    DB& db,
    Q& q,
    const std::vector<Index>& ids,
    int k_nn,
    unsigned nthreads);

template <class DB, class Q>
auto qv_query_heap(DB& db, Q& q, int k_nn, unsigned nthreads) {
  return qv_query_heap(
      without_ids{}, db, q, std::vector<size_t>{}, k_nn, nthreads);
}

template <class DB, class Q, class Index>
auto qv_query_heap(
    DB& db, Q& q, const std::vector<Index>& ids, int k_nn, unsigned nthreads) {
  return qv_query_heap(with_ids{}, db, q, ids, k_nn, nthreads);
}

// @todo Add to out of core
template <class T, class DB, class Q, class Index>
auto qv_query_heap(
    T,
    DB& db,
    Q& query,
    const std::vector<Index>& ids,
    int k_nn,
    unsigned nthreads) {
  scoped_timer _{tdb_func__};

  auto top_k = ColMajorMatrix<size_t>(k_nn, query.num_cols());
  auto top_k_scores = ColMajorMatrix<float>(k_nn, query.num_cols());

  // Have to do explicit asynchronous threading here, as the current parallel
  // algorithms have iterator-based interaces, and the `Matrix` class does not
  // yet have iterators.
  // @todo Implement iterator interface to `Matrix` class

  auto size_db = db.num_cols();
  auto par = stdx::execution::indexed_parallel_policy{nthreads};
  stdx::range_for_each(
      std::move(par),
      query,
      [&, size_db](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
        fixed_min_pair_heap<float, size_t> min_scores(k_nn);

        for (size_t i = 0; i < size_db; ++i) {
          auto score = L2(q_vec, db[i]);
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

/**
 * This algorithm is similar to `qv_query_heap`, but it tiles the query loop
 * and the database loop (2X2). This is done to improve cache locality.
 * @tparam DB The type of the database.
 * @tparam Q The type of the query.
 * @param db The database.
 * @param query The query.
 * @param k The number of top k results to return.
 * @param nthreads The number of threads to use in parallel execution.
 * @return A matrix of size k x #queries containing the top k results for each
 * query.
 */
template <class T, class DB, class Q, class Index>
auto qv_query_heap_tiled(
    T,
    DB& db,
    Q& q,
    const std::vector<Index>& ids,
    int k_nn,
    unsigned nthreads);

template <class DB, class Q>
auto qv_query_heap_tiled(DB& db, Q& q, int k_nn, unsigned nthreads) {
  return qv_query_heap_tiled(
      without_ids{}, db, q, std::vector<size_t>{}, k_nn, nthreads);
}

template <class DB, class Q, class Index>
auto qv_query_heap_tiled(
    DB& db, Q& q, const std::vector<Index>& ids, int k_nn, unsigned nthreads) {
  return qv_query_heap_tiled(with_ids{}, db, q, ids, k_nn, nthreads);
}

template <class T, class DB, class Q, class Index>
auto qv_query_heap_tiled(
    T,
    DB& db,
    Q& query,
    [[maybe_unused]] const std::vector<Index>& ids,
    int k_nn,
    unsigned nthreads) {
  if constexpr (is_loadable_v<decltype(db)>) {
    db.load();
  }

  scoped_timer _{tdb_func__};

  // Have to do explicit asynchronous threading here, as the current parallel
  // algorithms have iterator-based interaces, and the `Matrix` class does not
  // yet have iterators.
  // @todo Implement iterator interface to `Matrix` class
  size_t size_db = db.num_cols();
  size_t container_size = size(query);
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  auto min_scores = std::vector<fixed_min_pair_heap<float, size_t>>(
      size(query), fixed_min_pair_heap<float, size_t>(k_nn));

  // @todo: Use range::for_each
  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async, [start, stop, &query, &db, &min_scores, &ids]() {
            (void)ids;  // Suppress warnings
            auto len = 2 * ((stop - start) / 2);
            auto end = start + len;

            // auto min_scores0 = fixed_min_pair_heap<float, size_t> (k);
            // auto min_scores1 = fixed_min_pair_heap<float, size_t> (k);

            for (auto j = start; j != end; j += 2) {
              auto j0 = j + 0;
              auto j1 = j + 1;
              auto q_vec_0 = query[j0];
              auto q_vec_1 = query[j1];

              auto kstop = std::min<size_t>(size(db), 2 * (size(db) / 2));

              for (size_t kp = 0; kp < kstop; kp += 2) {
                auto score_00 = L2(q_vec_0, db[kp + 0]);
                auto score_01 = L2(q_vec_0, db[kp + 1]);
                auto score_10 = L2(q_vec_1, db[kp + 0]);
                auto score_11 = L2(q_vec_1, db[kp + 1]);

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
              for (size_t kp = kstop; kp < size(db); ++kp) {
                auto score_00 = L2(q_vec_0, db[kp + 0]);
                auto score_10 = L2(q_vec_1, db[kp + 0]);

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

              auto kstop = std::min<size_t>(size(db), 2 * (size(db) / 2));
              for (size_t kp = 0; kp < kstop; kp += 2) {
                auto score_00 = L2(q_vec_0, db[kp + 0]);
                auto score_01 = L2(q_vec_0, db[kp + 1]);

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
              for (size_t kp = kstop; kp < size(db); ++kp) {
                auto score_00 = L2(q_vec_0, db[kp + 0]);
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

  auto top_k = get_top_k_with_scores(min_scores, k_nn);

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
template <class DB, class Q>
auto qv_partition(const DB& db, const Q& q, unsigned nthreads) {
  scoped_timer _{tdb_func__};

  auto size_db = size(db);

  // Just need a single vector
  std::vector<size_t> top_k(q.num_cols());

  auto par = stdx::execution::indexed_parallel_policy{(size_t)nthreads};
  stdx::range_for_each(
      std::move(par), q, [&, size_db](auto&& qvec, auto&& n = 0, auto&& j = 0) {
        float min_score = std::numeric_limits<float>::max();
        size_t idx = 0;

        for (size_t i = 0; i < size_db; ++i) {
          auto score = L2(qvec, db[i]);
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
template <class DB, class Q>
auto qv_partition_with_scores(const DB& db, const Q& q, unsigned nthreads) {
  scoped_timer _{tdb_func__};

  auto size_db = size(db);

  // Just need a single vector
  std::vector<size_t> top_k(q.num_cols());
  std::vector<size_t> top_k_scores(q.num_cols());

  auto par = stdx::execution::indexed_parallel_policy{(size_t)nthreads};
  stdx::range_for_each(
      std::move(par), q, [&, size_db](auto&& qvec, auto&& n = 0, auto&& j = 0) {
        float min_score = std::numeric_limits<float>::max();
        size_t idx = 0;

        for (size_t i = 0; i < size_db; ++i) {
          auto score = L2(qvec, db[i]);
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
