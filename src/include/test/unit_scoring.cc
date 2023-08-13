/**
 * @file   unit_scoring.cc
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
 */

#include <catch2/catch_all.hpp>
#include <set>
#include <span>
#include <vector>
#include "scoring.h"
#include <span>
#include "detail/linalg/matrix.h"

#ifdef TILEDB_VS_ENABLE_BLAS

TEST_CASE("defs: vector test", "[defs]") {
  std::vector<std::vector<float>> a{
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
  std::vector<float> b{0, 0, 0, 0};

  std::vector<float> c{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<std::span<float>> d;
  for (size_t i = 0; i < 4; ++i) {
    d.push_back(std::span<float>(c.data() + i * 3, 3));
  }

  SECTION("column sum") {
    col_sum(a, b, [](auto x) { return x; });
    CHECK(b[0] == (1 + 2 + 3));
    CHECK(b[1] == (4 + 5 + 6));
    CHECK(b[2] == (7 + 8 + 9));
    CHECK(b[3] == (10 + 11 + 12));
  }

  SECTION("column sum of squares") {
    col_sum(a, b, [](auto x) { return x * x; });
    CHECK(b[0] == (1 + 4 + 9));
    CHECK(b[1] == (16 + 25 + 36));
    CHECK(b[2] == (49 + 64 + 81));
    CHECK(b[3] == (100 + 121 + 144));
  }

  SECTION("column sum of squares with span") {
    col_sum(d, b, [](auto x) { return x * x; });
    CHECK(b[0] == (1 + 4 + 9));
    CHECK(b[1] == (16 + 25 + 36));
    CHECK(b[2] == (49 + 64 + 81));
    CHECK(b[3] == (100 + 121 + 144));
  }
}
#endif

// L2
// cosine
// dot
// jaccard WIP

// get_top_k (heap) from scores array
TEST_CASE("get_top_k (heap) from scores array", "[get_top_k]") {
  std::vector<float> scores = { 8, 6, 7, 5, 3, 0, 9, 1, 2, 4  };

  std::vector<unsigned> top_k (3);
  get_top_k(scores, top_k, 3);
  CHECK(top_k.size() == 3);
  CHECK(top_k[0] == 5); // 0
  CHECK(top_k[1] == 7); // 1
  CHECK(top_k[2] == 8); // 2
}

// get_top_k (heap) from scores matrix, parallel
TEST_CASE("get_top_k (heap) from scores matrix, parallel", "[get_top_k]") {
  ColMajorMatrix<float> scores {
    //  0  1  2  3  4  5  6  7  8
      { 8, 6, 7, 5, 3, 0, 9, 1, 2 },
      { 3, 1, 4, 1, 5, 9, 2, 6, 7 },
      { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
      { 9, 8, 7, 6, 5, 4, 3, 2, 1 },
      { 9, 8, 7, 2, 5, 4, 3, 2, 9 },
      { 9, 8, 3, 6, 5, 4, 3, 9, 1 },
      { 7, 5, 3, 0, 9, 1, 2, 8, 1 },
  };

  CHECK(scores.num_rows() == 9);
  CHECK(scores.num_cols() == 7);

  // top_k
  // 0 1 2
  // 1 1 2
  // 1 2 3
  // 1 2 3
  // 2 2 3
  // 1 1 3
  // 0 1 1
  std::vector<unsigned> gt_scores {
      0, 1, 2,
      1, 1, 2,
      1, 2, 3,
      1, 2, 3,
      2, 2, 3,
      1, 1, 3,
      0, 1, 1,
  };
  std::vector<unsigned> gt_neighbors {
      5, 7, 8,
      1, 3, 6,
      0, 1, 2,
      8, 7, 6,
      7, 3, 6,
      8, 6, 2,
      3, 8, 5,
  };

  SECTION("single thread") {
    auto top_k = get_top_k(scores, 3);
    CHECK(top_k.num_rows() == 3);
    CHECK(top_k.num_cols() == 7);
    std::vector<unsigned> foo(top_k.data(), top_k.data() + top_k.size());
    CHECK(std::equal(begin(gt_neighbors), end(gt_neighbors), top_k.data()));
  }
  SECTION("multiple threads") {
    auto top_k = get_top_k(scores, 3, 5);
    CHECK(top_k.num_rows() == 3);
    CHECK(top_k.num_cols() == 7);
    std::vector<unsigned> foo(top_k.data(), top_k.data() + top_k.size());
    CHECK(std::equal(begin(gt_neighbors), end(gt_neighbors), top_k.data()));
  }
}


// consolidate scores
TEST_CASE("scoring consolidate scores", "[scoring]") {
  std::vector<fixed_min_pair_heap<float, unsigned>> scores00 {
      fixed_min_pair_heap<float, unsigned>(3, {{0.1, 0}, {0.2, 1}, {0.3, 2}, {0.4, 3}, {0.5, 4},}),
      fixed_min_pair_heap<float, unsigned>(3, {{0.9, 0}, {0.8, 1}, {0.7, 2}, {0.6, 3}, {0.5, 4},}),
      fixed_min_pair_heap<float, unsigned>(3, {{0.6, 5}, {0.7, 6}, {0.8, 7}, {0.9, 8}, {1.0, 9},}),
      fixed_min_pair_heap<float, unsigned>(3, {{0.4, 5}, {0.3, 6}, {0.2, 7}, {0.1, 8}, {0.0, 9},}),
  };
  std::vector<fixed_min_pair_heap<float, unsigned>> scores01 {
      fixed_min_pair_heap<float, unsigned>(3, {{0.1, 4}, {0.2, 5}, {0.3, 6}, {0.4, 7}, {0.5, 8},}),
      fixed_min_pair_heap<float, unsigned>(3, {{0.6, 4}, {0.7, 3}, {0.2, 5}, {0.9, 7}, {1.0, 8},}),
      fixed_min_pair_heap<float, unsigned>(3, {{0.9, 0}, {0.8, 1}, {0.7, 2}, {0.6, 3}, {0.5, 4},}),
      fixed_min_pair_heap<float, unsigned>(3, {{0.4, 9}, {0.3, 8}, {0.2, 7}, {0.1, 5}, {0.0, 6},}),
  };
  auto scores = std::vector<std::vector<fixed_min_pair_heap<float, unsigned>>> {scores00, scores01};
  CHECK(scores.size() == 2);
  consolidate_scores(scores);
  CHECK(scores.size() == 2);
  auto s = scores[0];
  CHECK(s.size() == 4);
  for ( auto&& j : s) {
    std::sort_heap(j.begin(), j.end());
  }
  CHECK(std::equal(begin(s[0]), end(s[0]), std::vector<std::tuple<float, unsigned>>({{0.1, 0}, {0.1, 4}, {0.2, 5}}).begin()));
  CHECK(std::equal(begin(s[1]), end(s[1]), std::vector<std::tuple<float, unsigned>>({{0.2, 5}, {0.5, 4}, {0.6, 4}}).begin()));
  CHECK(std::equal(begin(s[2]), end(s[2]), std::vector<std::tuple<float, unsigned>>({{0.5, 4}, {0.6, 3}, {0.6, 5}}).begin()));
  CHECK(std::equal(begin(s[3]), end(s[3]), std::vector<std::tuple<float, unsigned>>({{0.0, 6}, {0.0, 9}, {0.1, 5}}).begin()));

}

TEST_CASE("scoring get_top_k_from_heap one min_heap", "[scoring]") {
  fixed_min_pair_heap<float, unsigned> a(
      5,
      {
          {10, 0},
          {9, 1},
          {8, 2},
          {7, 3},
          {6, 4},
          {5, 5},
          {4, 6},
          {3, 7},
          {2, 8},
          {1, 9},
      });
  std::vector<unsigned> gt_neighbors {
      9, 8, 7, 6, 5,
  };

  SECTION("std::vector") {
    std::vector<unsigned> top_k(5);
    get_top_k_from_heap(a, top_k);
    CHECK(top_k.size() == 5);
    CHECK(std::equal(begin(gt_neighbors), end(gt_neighbors), top_k.begin()));
  }
  SECTION("std::span") {
    std::vector<unsigned> top_k(5);
    get_top_k_from_heap(a, std::span(top_k.data(), 5));
    CHECK(top_k.size() == 5);
    CHECK(std::equal(begin(gt_neighbors), end(gt_neighbors), top_k.begin()));
  }
}

TEST_CASE("scoring get_top_k_from_heap vector of min_heap", "[scoring]") {
  fixed_min_pair_heap<float, unsigned> a(
      5,
      {
          {10, 0},
          {9, 1},
          {8, 2},
          {7, 3},
          {6, 4},
          {5, 5},
          {4, 6},
          {3, 7},
          {2, 8},
          {1, 9},
      });
  fixed_min_pair_heap<float, unsigned> b(
      5,
      {
          {2, 0},
          {4, 1},
          {6, 2},
          {8, 3},
          {1, 4},
          {3, 5},
          {5, 6},
          {7, 7},
          {9, 8},
          {0, 9},
      });
  ColMajorMatrix<unsigned> gt_neighbors_mat {
        {9, 8, 7, 6, 5},
        {9, 4, 0, 5, 1},
  };
  std::vector<unsigned> gt_neighbors_vec {
      9, 8, 7, 6, 5, 9, 4, 0, 5, 1,
  };
  ColMajorMatrix<unsigned> gt_scores_mat {
      {1, 2, 3, 4, 5},
      {0, 1, 2, 3, 4},
  };
  std::vector<unsigned> gt_scores_vec {
      1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
  };
  std::vector <fixed_min_pair_heap<float, unsigned>> scores {a, b};
  SECTION("std::vector") {
    auto top_k = get_top_k(scores, 5);
    CHECK(top_k.num_rows() == 5);
    CHECK(top_k.num_cols() == 2);
    CHECK(std::equal(begin(gt_neighbors_vec), end(gt_neighbors_vec), top_k.data()));
  }
  SECTION("std::span") {
    auto top_k = get_top_k(scores, 5);
    CHECK(top_k.num_rows() == 5);
    CHECK(top_k.num_cols() == 2);
    CHECK(std::equal(begin(gt_neighbors_vec), end(gt_neighbors_vec), top_k.data()));
  }
  SECTION("scores, std::vector") {
    auto&& [top_scores, top_k] = get_top_k_with_scores(scores, 5);
    CHECK(top_k.num_rows() == 5);
    CHECK(top_k.num_cols() == 2);
    CHECK(std::equal(
        begin(gt_neighbors_vec), end(gt_neighbors_vec), top_k.data()));
    CHECK(std::equal(
        begin(gt_scores_vec), end(gt_scores_vec), top_scores.data()));
  }

  SECTION("scores, std::span") {
    auto&& [top_scores, top_k] = get_top_k_with_scores(scores, 5);
    CHECK(top_k.num_rows() == 5);
    CHECK(top_k.num_cols() == 2);
    CHECK(std::equal(begin(gt_neighbors_vec), end(gt_neighbors_vec), top_k.data()));
    CHECK(std::equal(
        begin(gt_scores_vec), end(gt_scores_vec), top_scores.data()));
  }
}


// get_top_k_from_heap (vector of vectors of min_heaps)
// get_top_k_with_scores_from_heap (one min_heap)
// get_top_k_with_scores_from_heap (vector of min_heaps)
// get_top_k_with_scores_from_heap (vector of vectors of min_heaps)

// verify_top_k_index
// verify_top_k_scores

#ifdef TILEDB_VS_ENABLE_BLAS
// mat_col_sum
// gemm_scores
#endif