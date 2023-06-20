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
 *
 */

#ifndef TILEDB_FLAT_QV_H
#define TILEDB_FLAT_QV_H

#include <future>
#include <numeric>
#include <vector>

#include "algorithm.h"
#include "concepts.h"
#include "linalg.h"
#include "utils/timer.h"

namespace detail::flat {

/**
 * Query using the qv ordering (loop over query vectors on outer loop and over
 * database vectors on inner loop).
 *
 * This algorithm does not form the scores matrix but rather computes the
 * relevant portion of the top_k query by query, only working on a single
 * scores vector (rather than matrix).
 *
 * @note The qv_query algorithm in ivf_query.h is essentially this, but has
 * get_top_k hard-coded to use a heap based algorithm.  This version can
 * use either a heap or the nth_element algorithm, depending on value of nth.
 *
 * @todo Implement a blocked version
 * @todo Are there other optimizations to apply?
 */

template <class DB, class Q>
auto qv_query_nth(
    const DB& db, const Q& q, int k, bool nth, unsigned int nthreads) {
  scoped_timer _{tdb_func__};

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  auto par = stdx::execution::indexed_parallel_policy{nthreads};
  stdx::range_for_each(
      std::move(par), q, [&, nth](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
        size_t size_q = size(q);
        size_t size_db = size(db);

        // @todo can we do this more efficiently?
        std::vector<float> scores(size_db);

        for (int i = 0; i < size_db; ++i) {
          scores[i] = L2(q_vec, db[i]);
        }
        if (nth) {
          std::vector<int> index(size_db);
          std::iota(begin(index), end(index), 0);
          get_top_k_nth(scores, top_k[j], index, k);
        } else {
          get_top_k(scores, top_k[j], k);
        }
      });

  return top_k;
}

/**
 * @todo Block the query to avoid memory blowup
 *
 * @note qv_query_by_vector in flat_query.h is similar to this, but
 * uses foreach instead of manually spawning threads.
 */
template <vector_database DB, class Q>
auto qv_query_heap(const DB& db, const Q& q, size_t k, unsigned nthreads) {
  scoped_timer _{tdb_func__};

  using element = std::pair<float, int>;

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  // Have to do explicit asynchronous threading here, as the current parallel
  // algorithms have iterator-based interaces, and the `Matrix` class does not
  // yet have iterators.
  // @todo Implement iterator interface to `Matrix` class
  size_t size_db = db.num_cols();
  size_t size_q = q.num_cols();
  size_t container_size = size_q;
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // @todo: Use range::for_each
  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async, [k, start, stop, size_db, &q, &db, &top_k]() {
            for (size_t j = start; j < stop; ++j) {
              fixed_min_heap<element> min_scores(k);
              size_t idx = 0;

              for (int i = 0; i < size_db; ++i) {
                auto score = L2(q[j], db[i]);
                min_scores.insert(element{score, i});
              }

              // @todo use get_top_k_from_heap
              std::sort_heap(min_scores.begin(), min_scores.end());
              std::transform(
                  min_scores.begin(),
                  min_scores.end(),
                  top_k[j].begin(),
                  ([](auto&& e) { return e.second; }));
            }
          }));
    }
  }

  for (int n = 0; n < size(futs); ++n) {
    futs[n].get();
  }

  return top_k;
}

template <class DB, class Q>
auto qv_partition(const DB& db, const Q& q, unsigned nthreads) {
  scoped_timer _{tdb_func__};

  // Just need a single vector
  std::vector<unsigned> top_k(q.num_cols());

  // Again, doing the parallelization by hand here....
  size_t size_db = db.num_cols();
  size_t size_q = q.num_cols();
  size_t container_size = size_q;
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async, [start, stop, size_db, &q, &db, &top_k]() {
            for (size_t j = start; j < stop; ++j) {
              float min_score = std::numeric_limits<float>::max();
              size_t idx = 0;

              for (int i = 0; i < size_db; ++i) {
                auto score = L2(q[j], db[i]);
                if (score < min_score) {
                  min_score = score;
                  idx = i;
                }
              }

              top_k[j] = idx;
            }
          }));
    }
  }
  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].wait();
  }
  return top_k;
}

}  // namespace detail::flat

#endif  // TILEDB_FLAT_QV_H