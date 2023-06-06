/**
 * @file   flat/gemm.h
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

#ifndef TILEDB_FLAT_GEMM_H
#define TILEDB_FLAT_GEMM_H

#include "algorithm.h"
#include "choose_blas.h"
#include "linalg.h"
#include "utils/timer.h"
#include "scoring.h"

namespace detail::flat {

template <class DB, class Q>
auto gemm_query(const DB& db, const Q& q, int k, bool nth, size_t nthreads) {
  life_timer _{"Total time " + tdb_func__};
  auto scores = gemm_scores(db, q, nthreads);
  auto top_k = get_top_k(scores, k, nth, nthreads);
  return top_k;
}

using namespace std::chrono_literals;

template <class DB, class Q>
auto blocked_gemm_query(DB& db, Q& q, int k, bool nth, size_t nthreads) {
  life_timer _{"Total time " + tdb_func__};

  using element = std::pair<float, unsigned>;

  // @todo constexpr block_db and block_q
  auto block_db = db.is_blocked();
  auto block_q = q.is_blocked();
  auto async_db = block_db && db.is_async();
  auto async_q = block_q && q.is_async();
  if (block_db && block_q) {
    throw std::runtime_error("Can't block both db and q");
  }

  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());

  std::vector<fixed_min_heap<element>> min_scores(
      size(q), fixed_min_heap<element>(k));

  for (;;) {
    if (async_db) {
      db.advance_async();
    }
    if (async_q) {
      q.advance_async();
    }
    gemm_scores(db, q, scores, nthreads);

    auto par = stdx::execution::indexed_parallel_policy{nthreads};
    stdx::range_for_each(
        std::move(par), scores, [&](auto&& q_vec, auto&& n = 0, auto&& i = 0) {
          if (block_db) {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i].insert({scores(j, i), j + db.offset()});
            }
          } else if (block_q) {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i + q.offset()].insert({scores(j, i), j});
            }
          } else {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i].insert({scores(j, i), j});
            }
          }
        });

    bool done = true;
    if (block_db) {
      done = async_db ? !db.advance_wait() : !db.advance();
    } else if (block_q) {
      done = async_q ? !q.advance_wait() : !q.advance();
    }
    if (done) {
      break;
    }
  }

  ColMajorMatrix<size_t> top_k(k, q.num_cols());
  for (int j = 0; j < min_scores.size(); ++j) {
    // @todo sort_heap
    std::sort_heap(min_scores[j].begin(), min_scores[j].end());
    std::transform(
        min_scores[j].begin(),
        min_scores[j].end(),
        top_k[j].begin(),
        ([](auto&& e) { return e.second; }));
  }

  return top_k;
}

template <class DB, class Q>
auto gemm_partition(const DB& db, const Q& q, unsigned nthreads) {
  life_timer _{"Total time " + tdb_func__};

  auto scores = gemm_scores(db, q, nthreads);

  auto top_k = std::vector<int>(q.num_cols());
  {
    for (int i = 0; i < scores.num_cols(); ++i) {
      auto min_score = std::numeric_limits<float>::max();
      auto idx = 0;

      for (int j = 0; j < scores.num_rows(); ++j) {
        auto score = scores(j, i);
        if (score < min_score) {
          min_score = score;
          idx = j;
        }
      }
      top_k[i] = idx;
    }
  }

  return top_k;
}

template <class DB, class Q>
auto blocked_gemm_partition(DB& db, Q& q, unsigned nthreads) {
  life_timer _{"Total time " + tdb_func__};

  const auto block_db = db.is_blocked();
  const auto block_q = q.is_blocked();
  if (block_db && block_q) {
    throw std::runtime_error("Can't block both db and q");
  }

  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());
  auto _score_data = raveled(scores);
  auto top_k = std::vector<int>(q.num_cols());
  auto min_scores =
      std::vector<float>(q.num_cols(), std::numeric_limits<float>::max());

  for (;;) {
    gemm_scores(db, q, scores, nthreads);

    for (int i = 0; i < scores.num_cols(); ++i) {
      auto min_score = min_scores[i];
      auto idx = db.offset();

      for (int j = 0; j < scores.num_rows(); ++j) {
        auto score = scores(j, i);
        if (score < min_score) {
          min_score = score;
          idx = j + db.offset();
        }
      }
      top_k[i] = idx;
    }
    bool done = true;
    if (block_db) {
      done = !db.advance();
    } else {
      done = !q.advance();
    }
    if (done) {
      break;
    }
  }
  return top_k;
}
}  // namespace detail::flat

#endif // TILEDB_FLAT_GEMM_H