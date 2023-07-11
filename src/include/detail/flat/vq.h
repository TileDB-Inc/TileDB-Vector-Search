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
#include "defs.h"
#include "linalg.h"
#include "utils/timer.h"

namespace detail::flat {

/**
 * This algorithm requires fully forming the scores matrix, which is then
 * inspected for top_k.  The method for getting top_k is selected by the
 * nth argument (true = nth_element, false = heap).
 *
 * @todo Implement a blocked version that does not require fully forming the
 * scores matrix (and which could also be used for out-of core).
 */
template <class DB, class Q>
auto vq_query_nth(DB& db, const Q& q, int k, bool nth, int nthreads) {
  if constexpr (is_loadable_v<decltype(db)>) {
    db.load();
  }
  scoped_timer _{
      tdb_func__ + (nth ? std::string{" nth"} : std::string{" heap"})};

  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());

  auto db_block_size = (size(db) + nthreads - 1) / nthreads;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // Parallelize over the database vectors (outer loop)
  for (int n = 0; n < nthreads; ++n) {
    int db_start = n * db_block_size;
    int db_stop = std::min<int>((n + 1) * db_block_size, size(db));
    size_t size_q = size(q);

    futs.emplace_back(std::async(
        std::launch::async, [&db, &q, db_start, db_stop, size_q, &scores]() {
          // For each database vector
          for (int i = db_start; i < db_stop; ++i) {
            // Compare with each query
            for (size_t j = 0; j < size_q; ++j) {
              // scores[j][i] = L2(q[j], db[i]);
              scores(i, j) = L2(q[j], db[i]);
            }
          }
        }));
  }
  for (int n = 0; n < nthreads; ++n) {
    futs[n].get();
  }

  auto top_k = get_top_k(scores, k, nth, nthreads);

  return top_k;
}

/**
 * This algorithm accumulates top_k as it goes, but in a "transposed" fashion to
 * qv_query.  Namely, it loops over the database vectors on the outer loop,
 * where each thread keeps its own set of heaps for each query vector.  After
 * The database vector loop, the heaps are merged and then copied to `top_k`.
 *
 * @todo Unify out of core and not out of core versions.
 */
template <class DB, class Q>
auto vq_query_heap(DB& db, Q& q, int k, unsigned nthreads) {
  using element = std::pair<float, int>;

  // @todo Need to get the total number of queries, not just the first block
  // @todo Use Matrix here rather than vector of vectors
  std::vector<std::vector<fixed_min_heap<element>>> scores(
      nthreads,
      std::vector<fixed_min_heap<element>>(
          size(q), fixed_min_heap<element>(k)));

  unsigned size_q = size(q);
  auto par = stdx::execution::indexed_parallel_policy{nthreads};

  log_timer _i{tdb_func__ + " in RAM"};

  // @todo Can we do blocking in the parallel for_each somehow?
  do {
    _i.start();
    stdx::range_for_each(
        std::move(par),
        db,
        [&, size_q](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
          for (size_t j = 0; j < size_q; ++j) {
            auto score = L2(q[j], db_vec);
            scores[n][j].insert(element{score, i + db.col_offset()});
          }
        });
    _i.stop();
  } while (db.load());

  _i.start();
  for (size_t j = 0; j < size(q); ++j) {
    for (unsigned n = 1; n < nthreads; ++n) {
      for (auto&& e : scores[n][j]) {
        scores[0][j].insert(e);
      }
    }
  }

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  // This might not be a win.
  int q_block_size = (size(q) + std::min<int>(nthreads, size(q)) - 1) /
                     std::min<int>(nthreads, size(q));
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // Parallelize over the query vectors (inner loop)
  // Should pick a threshold below which we don't bother with parallelism
  for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
    int q_start = n * q_block_size;
    int q_stop = std::min<int>((n + 1) * q_block_size, size(q));

    futs.emplace_back(
        std::async(std::launch::async, [&scores, q_start, q_stop, &top_k]() {
          // For each query

          // @todo get_top_k_from_heap
          for (int j = q_start; j < q_stop; ++j) {
            sort_heap(scores[0][j].begin(), scores[0][j].end());
            std::transform(
                scores[0][j].begin(),
                scores[0][j].end(),
                top_k[j].begin(),
                ([](auto&& e) { return e.second; }));
          }
        }));
  }

  for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
    futs[n].get();
  }
  _i.stop();

  return top_k;
}

template <class DB, class Q>
auto vq_query_heap_tiled(DB& db, Q& q, int k, unsigned nthreads) {
  using element = std::pair<float, int>;

  // @todo Need to get the total number of queries, not just the first block
  // @todo Use Matrix here rather than vector of vectors
  std::vector<std::vector<fixed_min_heap<element>>> scores(
      nthreads,
      std::vector<fixed_min_heap<element>>(
          size(q), fixed_min_heap<element>(k)));

  unsigned size_q = size(q);
  auto par = stdx::execution::indexed_parallel_policy{nthreads};

  log_timer _i{tdb_func__ + " in RAM"};

  // @todo Can we do blocking in the parallel for_each somehow?
  while (db.load()) {
    _i.start();
    stdx::range_for_each(
        std::move(par),
        db,
        [&, size_q](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
          for (size_t j = 0; j < size_q; ++j) {
            auto score = L2(q[j], db_vec);
            scores[n][j].insert(element{score, i + db.col_offset()});
          }
        });
    _i.stop();
  }

  _i.start();
  for (size_t j = 0; j < size(q); ++j) {
    for (unsigned n = 1; n < nthreads; ++n) {
      for (auto&& e : scores[n][j]) {
        scores[0][j].insert(e);
      }
    }
  }

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  // This might not be a win.
  int q_block_size = (size(q) + std::min<int>(nthreads, size(q)) - 1) /
                     std::min<int>(nthreads, size(q));
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // Parallelize over the query vectors (inner loop)
  // Should pick a threshold below which we don't bother with parallelism
  for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
    int q_start = n * q_block_size;
    int q_stop = std::min<int>((n + 1) * q_block_size, size(q));

    futs.emplace_back(
        std::async(std::launch::async, [&scores, q_start, q_stop, &top_k]() {
          // For each query

          // @todo get_top_k_from_heap
          for (int j = q_start; j < q_stop; ++j) {
            sort_heap(scores[0][j].begin(), scores[0][j].end());
            std::transform(
                scores[0][j].begin(),
                scores[0][j].end(),
                top_k[j].begin(),
                ([](auto&& e) { return e.second; }));
          }
        }));
  }

  for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
    futs[n].get();
  }
  _i.stop();

  return top_k;
}

#if 0
template <class DB, class Q>
auto vq_partition(const DB& db, const Q& q, int k, bool nth, int nthreads) {
 scoped_timer _{"Total time " + tdb_func__};


  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());

  auto db_block_size = (size(db) + nthreads - 1) / nthreads;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  // Parallelize over the database vectors (outer loop)
  for (int n = 0; n < nthreads; ++n) {
    int db_start = n * db_block_size;
    int db_stop = std::min<int>((n + 1) * db_block_size, size(db));
    size_t size_q = size(q);

    futs.emplace_back(std::async(
        std::launch::async,
        [&db, &q, db_start, db_stop, size_q, &scores]() {
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

  auto num_queries = scores.num_cols();

  auto top_k = ColMajorMatrix<size_t>(k, num_queries);

  std::vector<int> index(scores.num_rows());

  for (int j = 0; j < num_queries; ++j) {
    std::iota(begin(index), end(index), 0);

    using element = std::pair<float, unsigned>;
    fixed_min_heap<element> s(k);

    for (size_t i = 0; i < index.size(); ++i) {
      s.insert({scores[index[i]], index[i]});
    }

    // @todo get_top_k_from_heap
    std::sort_heap(begin(s), end(s));
    std::transform(
        s.begin(), s.end(), top_k[j].begin(), ([](auto&& e) { return e.second; }));
  }

  return top_k;
}

#endif
}  // namespace detail::flat

#endif  // TILEDB_FLAT_VQ_H
