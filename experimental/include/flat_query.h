/**
 * @file   flat_query.h
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
 * This file contains the query functions for the TileDB vector similarity
 * demo program.
 *
 * The functions have the same API -- they take a database, a query, a ground
 * truth, and a top-k result set. The functions differ in how they iterate over
 * the database and query vectors. They are parallelized over their outer loops,
 * using `std::async`. They time different parts of the query and print the
 * results to `std::cout`. Each query verifies its results against the ground
 * truth and reports any errors. Note that the top k might not be unique (i.e.
 * there might be more than one vector with the same distance) so that the
 * computed top k might not match the ground truth top k for some entries.  It
 * should be obvious on inspection of the error output whether or not reported
 * errors are due to real differences or just to non-uniqueness of the top k.
 *
 * I have started to parallelize the functions using `stdx::for_each`.
 *
 * Note that although the functions are templated on the database and query
 * type, they expect a "vector of spans" interface.  This works well with
 * the current `std::for_each` and is a reasonable way to think about the
 * sets of vectors.  However, having `mdspan` is more lightweight, but does
 * not support the `std::for_each` interface because it does not have
 * iterators.  I have not yet decided which is the best representation.
 *
 * These algorithms have not been blocked yet.
 */

#ifndef TDB_FLAT_QUERY_H
#define TDB_FLAT_QUERY_H

#include "defs.h"
#include "fixed_min_queues.h"
#include "linalg.h"
#include "utils/timer.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>

#include <future>

#include <vector>

#include "algorithm.h"
#include "scoring.h"

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
auto qv_query_nth(const DB& db, const Q& q, int k, bool nth, unsigned int nthreads) {
  life_timer _ { "Total time (vq hard way)" };

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  auto par = stdx::execution::indexed_parallel_policy { nthreads };
  stdx::range_for_each(std::move(par), q, [&, nth](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
    size_t size_q  = size(q);
    size_t size_db = size(db);

    // @todo can we do this more efficiently?
    std::vector<int>   index(size_db);
    std::vector<float> scores(size_db);

    for (int i = 0; i < size_db; ++i) {
      scores[i] = L2(q_vec, db[i]);
    }
    std::iota(begin(index), end(index), 0);
    get_top_k(scores, top_k[j], index, k, nth);
  });

  return top_k;
}

/**
 * This algorithm requires fully forming the scores matrix, which is then
 * inspected for top_k.  The method for getting top_k is selected by the
 * nth argument (true = nth_element, false = heap).
 *
 * @todo Implement a blocked version
 */
template <class DB, class Q>
auto query_vq_nth(const DB& db, const Q& q, int k, bool nth, int nthreads) {
  life_timer _outer { "Total time (vq loop nesting, nth = " + std::to_string(nth) };

  ColMajorMatrix<float>  scores(db.num_cols(), q.num_cols());
  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  auto                           db_block_size = (size(db) + nthreads - 1) / nthreads;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // Parallelize over the database vectors (outer loop)
  for (int n = 0; n < nthreads; ++n) {
    int    db_start = n * db_block_size;
    int    db_stop  = std::min<int>((n + 1) * db_block_size, size(db));
    size_t size_q   = size(q);

    futs.emplace_back(std::async(std::launch::async, [&db, &q, db_start, db_stop, size_q, &scores, &top_k]() {
      // For each database vector
      for (int i = db_start; i < db_stop; ++i) {
        // Compare with each query
        for (int j = 0; j < size_q; ++j) {
          scores[j][i] = L2(q[j], db[i]);
        }
      }
    }));
  }
  for (int n = 0; n < nthreads; ++n) {
    futs[n].get();
  }

  get_top_k(scores, top_k, k, size(q), size(db), nth, nthreads);

  return top_k;
}

/**
 * This algorithm accumulates top_k as it goes, but in a transpose fashion to
 * qv_query.  Namely, it loops over the database vectors on the outer loop,
 * where each thread keeps its own set of heaps for each query vector.  After
 * The database vector loop, the heaps are merged and then copied to top_k.
 *
 * @todo Implement a blocked version
 */
template <class DB, class Q>
void vq_query_heap(const DB& db, const Q& q, int k, bool nth, unsigned nthreads) {
  life_timer _outer { "Total time (vq loop nesting, set way)" };

  using element = std::pair<float, int>;
  std::vector<std::vector<fixed_min_heap<element>>> scores(nthreads, std::vector<fixed_min_heap<element>>(size(q), fixed_min_heap<element>(k)));

  auto par = stdx::execution::indexed_parallel_policy { nthreads };
  stdx::range_for_each(std::move(par), db, [&](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
    unsigned size_q = size(q);
    for (int j = 0; j < size_q; ++j) {
      auto score = L2(q[j], db_vec);
      scores[n][j].insert(element { score, i });
    }
  });

  for (int j = 0; j < size(q); ++j) {
    for (int n = 1; n < nthreads; ++n) {
      for (auto&& e : scores[n][j]) {
        scores[0][j].insert(e);
      }
    }
  }

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  // This might not be a win.
  int                            q_block_size = (size(q) + std::min<int>(nthreads, size(q)) - 1) / std::min<int>(nthreads, size(q));
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // Parallelize over the query vectors (inner loop)
  // Should pick a threshold below which we don't bother with parallelism
  for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
    int q_start = n * q_block_size;
    int q_stop  = std::min<int>((n + 1) * q_block_size, size(q));

    futs.emplace_back(std::async(std::launch::async, [&scores, q_start, q_stop, &top_k, k]() {
      // For each query
      for (int j = q_start; j < q_stop; ++j) {
        std::transform(scores[0][j].begin(), scores[0][j].end(), top_k[j].begin(), ([](auto&& e) { return e.second; }));
        std::sort(begin(top_k[j]), end(top_k[j]));
       }
    }));
  }

  for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
    futs[n].get();
  }
}


#endif    // TDB_FLAT_QUERY_H
