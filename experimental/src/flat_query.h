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
 */

#ifndef TDB_FLAT_QUERY_H
#define TDB_FLAT_QUERY_H

#include "defs.h"
#include "linalg.h"
#include "timer.h"
#include "fixed_min_queues.h"

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
 * Dispatch to the query that uses the qv ordering (loop over query vectors on
 * outer loop and over database vectors on inner loop).
 * @tparam DB
 * @tparam Q
 * @tparam G
 * @tparam TK
 * @param db The database vectors
 * @param q The query vectors
 * @param g The ground truth indices
 * @param top_k The top-k results
 * @param k How many nearest neighbors to find
 * @param hardway Whether to use the hard way (compute all distances) or the
 * easy way (compute a running set of only the top-k distances)
 * @param nthreads Number of threads to use
 */
template <class DB, class Q, class G, class TK>
void query_qv(const DB& db, const Q& q, const G& g, TK& top_k, int k, bool hardway, int nthreads) {
  if (hardway) {
    query_qv_hw(db, q, g, top_k, k, nthreads);
  } else {
    query_qv_ew(db, q, g, top_k, k, nthreads);
  }
}

/**
 * Dispatch to the query that uses the vq ordering (loop over database vectors
 * on outer loop and over query vectors on inner loop).
 * @tparam DB
 * @tparam Q
 * @tparam G
 * @tparam TK
 * @param db The database vectors
 * @param q The query vectors
 * @param g The ground truth indices
 * @param top_k The top-k results
 * @param k How many nearest neighbors to find
 * @param hardway Whether to use the hard way (compute all distances) or the
 * easy way (compute a running set of only the top-k distances)
 * @param nthreads Number of threads to use
 */
template <class DB, class Q, class G, class TK>
void query_vq(const DB& db, const Q& q, const G& g, TK& top_k, int k, bool hardway, int nthreads) {
  if (hardway) {
    query_vq_hw(db, q, g, top_k, k, nthreads);
  } else {
    query_vq_ew(db, q, g, top_k, k, nthreads);
  }
}

/**
 * Query using the qv ordering (loop over query vectors on outer loop and over
 * database vectors on inner loop). Does this the hard way -- computes all
 * distances and then sorts to find the top k.
 */
template <class DB, class Q, class G, class TK>
void query_qv_hw(const DB& db, const Q& q, const G& g, TK& top_k, int k, unsigned int nthreads) {
  life_timer _ { "Total time (vq hard way)" };

  std::vector<int> i_index(size(db));
  std::iota(begin(i_index), end(i_index), 0);

  auto par = stdx::execution::indexed_parallel_policy { nthreads };
  stdx::range_for_each(std::move(par), q, [&](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
    size_t             size_q  = size(q);
    size_t             size_db = size(db);
    std::vector<int>   index(size_db);
    std::vector<float> scores(size_db);
    for (int i = 0; i < size_db; ++i) {
      scores[i] = L2(q_vec, db[i]);
    }

    if (g.num_rows() > 0) {
      // std::copy(begin(i_index), end(i_index), begin(index));
      std::iota(begin(index), end(index), 0);
      get_top_k(scores, top_k[j], index, k);
      verify_top_k(scores, top_k[j], g[j], k, j);
    }
  });
}

/**
 * Query using the qv ordering (loop over query vectors on outer loop and over
 * database vectors on inner loop). Does this the easy way -- keeps a running
 * total of the top k distances.
 */
template <class DB, class Q, class G, class TK>
void query_qv_ew(const DB& db, const Q& q, const G& g, TK& top_k, int k, unsigned nthreads) {
  life_timer _ { "Total time (qv set way)" };

  using element = std::pair<float, int>;

  std::vector<int> i_index(size(db));
  std::iota(begin(i_index), end(i_index), 0);

  auto par = stdx::execution::indexed_parallel_policy { nthreads };
  stdx::range_for_each(std::move(par), q, [&](auto&& q_vec, auto&& n = 0, auto&& j = 0) {
    size_t size_db = size(db);

    // Create a set of the top k scores
    fixed_min_heap<element> scores(k);

    // Compare with each database vector
    for (int i = 0; i < size_db; ++i) {
      auto score = L2(q[j], db[i]);
      scores.insert(element { score, i });
    }

    // Copy indexes into top_k
    std::transform(scores.begin(), scores.end(), top_k[j].begin(), ([](auto&& e) { return e.second; }));
    if (g.num_rows() > 0) {
      // Try to break ties by sorting the top k
      std::sort(begin(top_k[j]), end(top_k[j]));
      std::sort(begin(g[j]), begin(g[j]) + k);
      verify_top_k(top_k[j], g[j], k, j);
    }
  });
}

/**
 * Query using the vq ordering (loop over database vectors on outer loop and
 * over query vectors on inner loop). Does this the hard way -- computes all
 * distances and then sorts (actually, nth_element) to find the top k.
 */
template <class DB, class Q, class G, class TK>
void query_vq_hw(const DB& db, const Q& q, const G& g, TK& top_k, int k, int nthreads) {
  life_timer _outer { "Total time (vq loop nesting, hard way)" };

  ms_timer init_time("Allocating score array");
  init_time.start();

#if __APPLE__
  // std::vector<std::vector<float>> scores(size(q),
  // std::vector<float>(size(db), 0.0f));
  auto             buf = std::make_unique<float[]>(size(q) * size(db));
  std::span<float> _score_data { buf.get(), size(q) * size(db) };
#else
  auto                          buf = std::make_unique_for_overwrite<float[]>(size(q) * size(db));
  std::span<float>              _score_data { buf.get(), size(q) * size(db) };
#endif

  std::vector<std::span<float>> scores(size(q));

  // Each score[j] is a column of the score matrix
  int size_q = size(q);
  for (int j = 0; j < size_q; ++j) {
    scores[j] = std::span<float>(_score_data.data() + j * size(db), size(db));
  }

  init_time.stop();
  std::cout << init_time << std::endl;

#ifdef __APPLE__
  std::cout << "Apple clang does not yet support make_unique_for_overwrite" << std::endl;
  std::cout << "so this is about 3X slower than it should be" << std::endl;
#endif

  {
    life_timer _ { "L2 distance" };

    auto                           db_block_size = (size(db) + nthreads - 1) / nthreads;
    std::vector<std::future<void>> futs;
    futs.reserve(nthreads);

    // Parallelize over the database vectors (outer loop)
    for (int n = 0; n < nthreads; ++n) {
      int db_start = n * db_block_size;
      int db_stop  = std::min<int>((n + 1) * db_block_size, size(db));

      futs.emplace_back(std::async(std::launch::async, [&db, &q, db_start, db_stop, size_q, &scores]() {
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
  }

  get_top_k(scores, top_k, k, size(q), size(db), nthreads);

  if (g.num_rows() > 0) {
    life_timer _ { "Checking results" };

    // #pragma omp parallel for
    for (int j = 0; j < size_q; ++j) {
      verify_top_k(scores[j], top_k[j], g[j], k, j);
    }
  }
}

/**
 * Query using the vq ordering (loop over database vectors on outer loop and
 * over query vectors on inner loop). Does this the easy way -- keeps a running
 * tally of top k scores.
 */
template <class DB, class Q, class G, class TK>
void query_vq_ew(const DB& db, const Q& q, const G& g, TK& top_k, int k, unsigned nthreads) {
  life_timer _outer { "Total time (vq loop nesting, set way)" };

  using element = std::pair<float, int>;
  std::vector<std::vector<fixed_min_heap<element>>> scores(nthreads, std::vector<fixed_min_heap<element>>(size(q), fixed_min_heap<element>(k)));

  {
    life_timer _ { "L2 distance" };

    // Spawning threads manually is probably more like the distributed
    // memory case
    auto par = stdx::execution::indexed_parallel_policy { nthreads };
    stdx::range_for_each(std::move(par), db, [&](auto&& db_vec, auto&& n = 0, auto&& i = 0) {
      unsigned size_q = size(q);
      for (int j = 0; j < size_q; ++j) {
        auto score = L2(q[j], db_vec);
        scores[n][j].insert(element { score, i });
      }
    });
  }

  // Merge the scores from each thread
  {
    life_timer _ { "Merge" };
    for (int j = 0; j < size(q); ++j) {
      for (int n = 1; n < nthreads; ++n) {
        for (auto&& e : scores[n][j]) {
          scores[0][j].insert(e);
        }
      }
    }
  }

  {
    life_timer _ { "Get top k and check results" };

    int                            q_block_size = (size(q) + std::min<int>(nthreads, size(q)) - 1) / std::min<int>(nthreads, size(q));
    std::vector<std::future<void>> futs;
    futs.reserve(nthreads);

    // Parallelize over the query vectors (inner loop)
    // Should pick a threshold below which we don't bother with parallelism
    for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
      int q_start = n * q_block_size;
      int q_stop  = std::min<int>((n + 1) * q_block_size, size(q));

      futs.emplace_back(std::async(std::launch::async, [&scores, &g, q_start, q_stop, &top_k, k]() {
        // For each query
        for (int j = q_start; j < q_stop; ++j) {
          std::transform(scores[0][j].begin(), scores[0][j].end(), top_k[j].begin(), ([](auto&& e) { return e.second; }));
          std::sort(begin(top_k[j]), end(top_k[j]));
          if (g.num_rows() > 0) {
            std::sort(begin(g[j]), begin(g[j]) + k);
            verify_top_k(top_k[j], g[j], k, j);
          }
        }
      }));
    }
    for (int n = 0; n < std::min<int>(nthreads, size(q)); ++n) {
      futs[n].get();
    }
  }
}

/**
 * Query using dense linear algebra.  This uses the vector generalization of
 * the identity (a - b) * (a - b) = a * a + b * b - 2 * a * b .
 * We use outer products to compute the a * a and b * b terms, and then use
 * a gemm to compute the a * b term.
 *
 * This is extremely fast for large numbers of query vectors, but is not as fast
 * as vq_ew for small numbers of query vectors.
 */
template <class DB, class Q, class TK>
auto gemm_compute_scores(const DB& db, const Q& q, TK& top_k, int k, size_t nthreads) {
  life_timer _outer { "Total time gemm" };

  auto scores = gemm_scores(db, q, nthreads);
  get_top_k(scores, top_k, k, size(q), size(db), nthreads);

  return scores;
}

template <class DB, class Q, class G, class TK>
void query_gemm(const DB& db, const Q& q, const G& g, TK& top_k, int k, [[maybe_unused]] bool hw, size_t nthreads) {
  auto scores = gemm_compute_scores(db, q, top_k, k, nthreads);
  if (g.num_rows() > 0) {
    life_timer _ { "Checking results" };

    size_t size_q = size(q);
    for (size_t j = 0; j < size_q; ++j) {
      verify_top_k(scores[j], top_k[j], g[j], k, j);
    }
  }
}

template <class DB, class Q, class TK>
auto blocked_gemm_compute_scores(DB& db, Q& q, TK& top_k, int k, size_t nthreads) {
  life_timer _outer { "Total time gemm" };

  using element = std::pair<float, unsigned>;

  auto block_db = true;

  // @todo Untangle this some day so that it is Column Major
  ColMajorMatrix<float> scores(q.num_cols(), db.num_cols());
  // ColMajorMatrix<float> scores(q.num_cols(), db.num_cols());
  auto _score_data = raveled(scores);

  std::vector<fixed_min_heap<element>> min_scores(size(q), fixed_min_heap<element>(k));

  for (;;) {

    gemm_scores(db, q, scores, nthreads);

    { life_timer _ { "inserting" };

      for (int i = 0; i < scores.num_cols(); ++i) {
        for (int j = 0; j < scores.num_rows(); ++j) {
          min_scores[j].insert({ scores(j, i), i + db.offset() });
        }
      }
    }

    bool done = false;
    if (block_db) {
      done = !db.advance();
    } else {
      done = !q.advance();
    }
    if (done) {
      break;
    }
  }
  {
    life_timer _ { "Sorting" };
    for (int j = 0; j < scores.num_rows(); ++j) {
      //std::sort(min_scores[j].begin(), min_scores[j].end());
      std::transform(min_scores[j].begin(), min_scores[j].end(), top_k[j].begin(), ([](auto&& e) { return e.second; }));
    }
  }

  return scores;
}


template <class DB, class Q, class G, class TK>
void blocked_query_gemm(DB& db, Q& q, const G& g, TK& top_k, int k, [[maybe_unused]] bool hw, size_t nthreads) {
  auto scores = blocked_gemm_compute_scores(db, q, top_k, k, nthreads);
  if (g.num_rows() > 0) {
    life_timer _ { "Checking results" };

    size_t size_q = size(q);
    for (size_t j = 0; j < size_q; ++j) {
      verify_top_k(scores[j], top_k[j], g[j], k, j);
    }
  }
}
#endif    // TDB_FLAT_QUERY_H
