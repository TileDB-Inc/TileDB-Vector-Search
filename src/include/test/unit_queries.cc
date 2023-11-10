/**
 * @file   unit_queries.cc
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
 * Test correctness of queries.
 *
 */

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/timer.h"

#include "linalg.h"

#include "detail/flat/qv.h"
#include "detail/flat/vq.h"
using namespace detail::flat;

TEST_CASE("test queries", "[queries]") {
  size_t dimension = 128;
  size_t num_vectors = 2000;
  size_t num_queries = GENERATE(1, 10, 100);
  size_t k = GENERATE(1, 10, 100);
  size_t nth = GENERATE(true, false);
  size_t nthreads = GENERATE(1, 8);

  std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(2514908090);
  std::uniform_int_distribution<int> dist(-128, 128);

  auto db_mat = ColMajorMatrix<float>(dimension, num_vectors);
  for (auto& x : raveled(db_mat)) {
    x = dist(gen);
  }

  //  auto b_db_mat =
  //      BlockedMatrix<float, stdx::layout_left>(dimension, num_vectors);
  //  for (auto& x : raveled(b_db_mat)) {
  //    x = dist(gen);
  //  }

  auto q_mat = ColMajorMatrix<float>(dimension, num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    auto n = 17 * (i + 3);
    std::copy(begin(db_mat[n]), end(db_mat[n]), begin(q_mat[i]));
  }

  SECTION("qv_query") {
    auto&& [top_k_scores, top_k] = qv_query_heap(db_mat, q_mat, k, nthreads);
    CHECK(top_k.num_rows() == k);
    CHECK(top_k.num_cols() == num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
      auto n = 17 * (i + 3);
      CHECK(top_k(0, i) == n);
    }
  }

#ifdef TDB_MATRIX_LOAD
  SECTION("vq_query_heap") {
    auto top_k = vq_query_heap(db_mat, q_mat, k, nthreads);
    CHECK(top_k.num_rows() == k);
    CHECK(top_k.num_cols() == num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
      auto n = 17 * (i + 3);
      CHECK(top_k(0, i) == n);
    }
  }

  for (bool nth : {true, false}) {
    SECTION("qv, nth = " + std::to_string(nth)) {
      auto top_k = qv_query_nth(db_mat, q_mat, k, nth, nthreads);
      CHECK(top_k.num_rows() == k);
      CHECK(top_k.num_cols() == num_queries);
      for (size_t i = 0; i < num_queries; ++i) {
        auto n = 17 * (i + 3);
        CHECK(top_k(0, i) == n);
      }
    }

    SECTION("vq, nth = " + std::to_string(nth)) {
      auto top_k = vq_query_nth(db_mat, q_mat, k, nth, nthreads);
      CHECK(top_k.num_rows() == k);
      CHECK(top_k.num_cols() == num_queries);
      for (size_t i = 0; i < num_queries; ++i) {
        auto n = 17 * (i + 3);
        CHECK(top_k(0, i) == n);
      }
    }

#ifdef TILEDB_VS_ENABLE_BLAS
    SECTION("gemm, nth = " + std::to_string(nth)) {
      auto top_k = gemm_query(db_mat, q_mat, k, nth, nthreads);
      CHECK(top_k.num_rows() == k);
      CHECK(top_k.num_cols() == num_queries);
      for (size_t i = 0; i < num_queries; ++i) {
        auto n = 17 * (i + 3);
        CHECK(top_k(0, i) == n);
      }
    }

    SECTION("blocked gemm, nth = " + std::to_string(nth)) {
      auto top_k = blocked_gemm_query(b_db_mat, q_mat, k, nth, nthreads);
      CHECK(top_k.num_rows() == k);
      CHECK(top_k.num_cols() == num_queries);
      CHECK(top_k(0, 0) == 333);  // FIXME: this is broken (maybe)
    }
#endif  // TILEDB_VS_ENABLE_BLAS
  }
#endif
}
