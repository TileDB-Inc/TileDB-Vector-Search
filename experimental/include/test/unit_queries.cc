

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/timer.h"

#include "flat_query.h"
#include "ivf_query.h"
#include "linalg.h"

TEST_CASE("test queries", "[queries]") {
  size_t dimension = 128;
  size_t num_vectors = 2000;
  size_t num_queries = GENERATE(1, 10, 100);
  size_t k = GENERATE(1, 10, 100);
  size_t nth = GENERATE(true, false);
  size_t nthreads = GENERATE(1, 8);

  std::random_device rd;
  //std::mt19937 gen(rd());
  std::mt19937 gen(2514908090);
  std::uniform_int_distribution<int> dist(-128, 128);

  auto db_mat = ColMajorMatrix<float>(dimension, num_vectors);
  for (auto& x : raveled(db_mat)) {
    x = dist(gen);
  }

  auto b_db_mat =
      BlockedMatrix<float, stdx::layout_left>(dimension, num_vectors);
  for (auto& x : raveled(b_db_mat)) {
    x = dist(gen);
  }

  auto q_mat = ColMajorMatrix<float>(dimension, num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    auto n = 17 * (i + 3);
    std::copy(begin(db_mat[n]), end(db_mat[n]), begin(q_mat[i]));
  }

  SECTION("qv_query") {
    auto top_k = qv_query(db_mat, q_mat, k, nthreads);
    CHECK(top_k.num_rows() == k);
    CHECK(top_k.num_cols() == num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
      auto n = 17 * (i + 3);
      CHECK(top_k(0, i) == n);
    }
  }

#if 1
  SECTION("vq_query_heap") {  // FIXME: this is broken
    auto top_k = vq_query_heap(db_mat, q_mat, k, nthreads);
    CHECK(top_k.num_rows() == k);
    CHECK(top_k.num_cols() == num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
      auto n = 17 * (i + 3);
      CHECK(top_k(0, i) == n);
    }
  }
#endif
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

    SECTION("gemm, nth = " + std::to_string(nth)) {
      auto top_k = gemm_query(db_mat, q_mat, k, nth, nthreads);
      CHECK(top_k.num_rows() == k);
      CHECK(top_k.num_cols() == num_queries);
      for (size_t i = 0; i < num_queries; ++i) {
        auto n = 17 * (i + 3);
        CHECK(top_k(0, i) == n);
      }
    }
#if 0
    SECTION("blocked gemm, nth = " + std::to_string(nth)) {
      auto top_k = blocked_gemm_query(b_db_mat, q_mat, k, nth, nthreads);
      CHECK(top_k.num_rows() == k);
      CHECK(top_k.num_cols() == num_queries);
      CHECK(top_k(0, 0) == 333);   // FIXME: this is broken (maybe)
    }
#endif
  }
}
