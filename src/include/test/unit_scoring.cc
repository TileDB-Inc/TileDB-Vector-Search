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
#include "detail/linalg/matrix.h"
#include "scoring.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#if defined(TILEDB_VS_ENABLE_BLAS) && 0

TEST_CASE("vector test", "[scoring]") {
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
using scoring_typelist = std::tuple<
    std::tuple<float, int, int>,
    std::tuple<float, unsigned, unsigned>,
    std::tuple<float, size_t, size_t>,
    std::tuple<float, unsigned, size_t>,
    std::tuple<float, int, size_t>,
    std::tuple<float, size_t, int>>;
// get_top_k (heap) from scores array
TEMPLATE_LIST_TEST_CASE(
    "get_top_k_from_scores vector", "[scoring][get_top_k]", scoring_typelist) {
  using score_type = std::tuple_element_t<0, TestType>;
  using index_type = std::tuple_element_t<1, TestType>;
  // using groundtruth_type = std::tuple_element_t<2, TestType>;
  std::vector<score_type> scores = {8, 6, 7, 5, 3, 0, 9, 1, 2, 4, 3.14159};

  std::vector<index_type> top_k(5);
  get_top_k_from_scores(scores, top_k, 5);
  CHECK(top_k.size() == 5);
  CHECK(top_k[0] == 5);   // 0
  CHECK(top_k[1] == 7);   // 1
  CHECK(top_k[2] == 8);   // 2
  CHECK(top_k[3] == 4);   // 3
  CHECK(top_k[4] == 10);  // 3.14159
}

// get_top_k (heap) from scores matrix, parallel
TEMPLATE_LIST_TEST_CASE(
    "get_top_k_from_scores matrix", "[scoring][get_top_k]", scoring_typelist) {
  using score_type = std::tuple_element_t<0, TestType>;
  using index_type = std::tuple_element_t<1, TestType>;
  using groundtruth_type = std::tuple_element_t<2, TestType>;

  ColMajorMatrix<score_type> scores{
      //  0  1  2  3  4  5  6  7  8
      {8, 6, 7, 5, 3, 0, 9, 1, 2},
      {3, 1, 4, 1, 5, 9, 2, 6, 7},
      {1, 2, 3, 4, 5, 6, 7, 8, 9},
      {9, 8, 7, 6, 5, 4, 3, 2, 1},
      {9, 8, 7, 2, 5, 4, 3, 2, 9},
      {9, 8, 3, 6, 5, 4, 3, 9, 1},
      {7, 5, 3, 0, 9, 1, 2, 8, 1},
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
  std::vector<score_type> gt_scores{
      0, 1, 2, 1, 1, 2, 1, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 3, 0, 1, 1,
  };
  std::vector<groundtruth_type> gt_neighbors{
      5, 7, 8, 1, 3, 6, 0, 1, 2, 8, 7, 6, 7, 3, 6, 8, 6, 2, 3, 8, 5,
  };

  SECTION("single thread") {
    auto top_k = get_top_k_from_scores<index_type>(scores, 3);
    CHECK(top_k.num_rows() == 3);
    CHECK(top_k.num_cols() == 7);
    std::vector<index_type> foo(top_k.data(), top_k.data() + top_k.size());
    CHECK(std::equal(begin(gt_neighbors), end(gt_neighbors), top_k.data()));
  }
// no multithreaded version WIP
#if 0
  SECTION("multiple threads") {
    auto top_k = get_top_k_from_scores(scores, 3, 5);
    CHECK(top_k.num_rows() == 3);
    CHECK(top_k.num_cols() == 7);
    std::vector<unsigned> foo(top_k.data(), top_k.data() + top_k.size());
    CHECK(std::equal(begin(gt_neighbors), end(gt_neighbors), top_k.data()));
  }
#endif
}

// consolidate scores
TEMPLATE_LIST_TEST_CASE("consolidate scores", "[scoring]", scoring_typelist) {
  using score_type = std::tuple_element_t<0, TestType>;
  using index_type = std::tuple_element_t<1, TestType>;
  // using groundtruth_type = std::tuple_element_t<2, TestType>;

  std::vector<fixed_min_pair_heap<score_type, index_type>> scores00{
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.1, 0},
              {0.2, 1},
              {0.3, 2},
              {0.4, 3},
              {0.5, 4},
          }),
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.9, 0},
              {0.8, 1},
              {0.7, 2},
              {0.6, 3},
              {0.5, 4},
          }),
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.6, 5},
              {0.7, 6},
              {0.8, 7},
              {0.9, 8},
              {1.0, 9},
          }),
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.4, 5},
              {0.3, 6},
              {0.2, 7},
              {0.1, 8},
              {0.0, 9},
          }),
  };
  std::vector<fixed_min_pair_heap<score_type, index_type>> scores01{
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.1, 4},
              {0.2, 5},
              {0.3, 6},
              {0.4, 7},
              {0.5, 8},
          }),
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.6, 4},
              {0.7, 3},
              {0.2, 5},
              {0.9, 7},
              {1.0, 8},
          }),
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.9, 0},
              {0.8, 1},
              {0.7, 2},
              {0.6, 3},
              {0.5, 4},
          }),
      fixed_min_pair_heap<score_type, index_type>(
          3,
          {
              {0.4, 9},
              {0.3, 8},
              {0.2, 7},
              {0.1, 5},
              {0.0, 6},
          }),
  };
  auto scores =
      std::vector<std::vector<fixed_min_pair_heap<score_type, index_type>>>{
          scores00, scores01};
  CHECK(scores.size() == 2);
  consolidate_scores(scores);
  CHECK(scores.size() == 2);
  auto s = scores[0];
  CHECK(size(s) == 4);
  for (auto& j : s) {
    std::sort_heap(begin(j), end(j));
  }

  CHECK(size(s[0]) == 3);
  CHECK(size(s[1]) == 3);
  CHECK(size(s[2]) == 3);
  CHECK(size(s[3]) == 3);
  CHECK(std::equal(
      begin(s[0]),
      end(s[0]),
      std::vector<std::tuple<score_type, index_type>>(
          {{0.1, 0}, {0.1, 4}, {0.2, 1}})
          .begin()));
  CHECK(std::equal(
      begin(s[1]),
      end(s[1]),
      std::vector<std::tuple<score_type, index_type>>(
          {{0.2, 5}, {0.5, 4}, {0.6, 4}})
          .begin()));
  CHECK(std::equal(
      begin(s[2]),
      end(s[2]),
      std::vector<std::tuple<score_type, index_type>>(
          {{0.5, 4}, {0.6, 3}, {0.6, 5}})
          .begin()));
  CHECK(std::equal(
      begin(s[3]),
      end(s[3]),
      std::vector<std::tuple<score_type, index_type>>(
          {{0.0, 6}, {0.0, 9}, {0.1, 5}})
          .begin()));
}

TEMPLATE_LIST_TEST_CASE(
    "get_top_k_from_heap one min_heap", "[scoring]", scoring_typelist) {
  using score_type = std::tuple_element_t<0, TestType>;
  using index_type = std::tuple_element_t<1, TestType>;
  using groundtruth_type = std::tuple_element_t<2, TestType>;

  groundtruth_type k_nn = GENERATE(1, 3, 5);
  groundtruth_type asize = GENERATE(1, 3, 5);

  fixed_min_pair_heap<score_type, index_type> a(
      asize,
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
  std::vector<groundtruth_type> gt_neighbors{9, 8, 7, 6, 5, 4, 3, 2, 1};

  SECTION("std::vector") {
    std::vector<index_type> top_k(k_nn);
    get_top_k_from_heap(a, top_k);
    REQUIRE((size_t)top_k.size() == (size_t)k_nn);
    auto l_nn = std::min<size_t>(k_nn, a.size());
    CHECK(std::equal(
        begin(gt_neighbors), begin(gt_neighbors) + l_nn, top_k.begin()));
  }

  SECTION("std::span") {
    std::vector<index_type> top_k(k_nn);
    get_top_k_from_heap(a, std::span(top_k.data(), k_nn));
    REQUIRE((size_t)top_k.size() == (size_t)k_nn);
    auto l_nn = std::min<size_t>(k_nn, a.size());
    CHECK(std::equal(
        begin(gt_neighbors), begin(gt_neighbors) + l_nn, top_k.begin()));
  }

  SECTION("std::vector, pad") {
    groundtruth_type pad = 2;
    std::vector<index_type> top_k(k_nn + pad);
    get_top_k_from_heap(a, top_k);
    REQUIRE((size_t)top_k.size() == (size_t)(k_nn + pad));
    auto l_nn = std::min<size_t>(k_nn + pad, a.size());
    CHECK(std::equal(
        begin(gt_neighbors), begin(gt_neighbors) + l_nn, top_k.begin()));
    CHECK(end(top_k) == begin(top_k) + k_nn + pad);
    CHECK((size_t)(end(top_k) - begin(top_k)) == (size_t)(k_nn + pad));
    CHECK(std::equal(
        begin(top_k) + l_nn,
        begin(top_k) + k_nn + pad,
        std::vector<index_type>(
            k_nn + pad - l_nn, std::numeric_limits<index_type>::max())
            .begin()));
  }

  SECTION("std::span, pad") {
    groundtruth_type pad = 2;
    std::vector<index_type> top_k(k_nn + pad);
    get_top_k_from_heap(a, std::span(top_k.data(), k_nn + pad));
    REQUIRE((size_t)top_k.size() == (size_t)(k_nn + pad));
    auto l_nn = std::min<size_t>(k_nn + pad, a.size());
    CHECK(std::equal(
        begin(gt_neighbors), begin(gt_neighbors) + l_nn, top_k.begin()));
    CHECK(end(top_k) == begin(top_k) + k_nn + pad);
    CHECK((size_t)(end(top_k) - begin(top_k)) == (size_t)(k_nn + pad));
    CHECK(std::equal(
        begin(top_k) + l_nn,
        begin(top_k) + k_nn + pad,
        std::vector<index_type>(
            k_nn + pad - l_nn, std::numeric_limits<index_type>::max())
            .begin()));
  }
}

TEMPLATE_LIST_TEST_CASE("get_top_k", "[scoring]", scoring_typelist) {
  using score_type = std::tuple_element_t<0, TestType>;
  using index_type = std::tuple_element_t<1, TestType>;
  using groundtruth_type = std::tuple_element_t<2, TestType>;

  groundtruth_type k_nn = GENERATE(1, 3, 5);
  groundtruth_type asize = GENERATE(1, 3, 5);
  groundtruth_type bsize = GENERATE(0, 1);
  size_t num_vectors{2};

  std::vector<fixed_min_pair_heap<score_type, index_type>> scores00{
      fixed_min_pair_heap<score_type, index_type>(
          asize,
          {
              {0.1, 0},
              {0.2, 1},
              {0.3, 2},
              {0.4, 3},
              {0.5, 4},
              {0.6, 5},
              {0.7, 6},
              {0.8, 7},
              {0.9, 8},
              {1.0, 9},
          }),
      fixed_min_pair_heap<score_type, index_type>(
          asize + bsize,
          {
              {0.9, 0},
              {0.8, 1},
              {0.7, 2},
              {0.6, 3},
              {0.5, 4},
              {0.4, 5},
              {0.3, 6},
              {0.2, 7},
              {0.1, 8},
              {0.0, 9},
          })};

  // Matrix not used
  ColMajorMatrix<groundtruth_type> gt_neighbors_mat{
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {9, 8, 7, 6, 5, 4, 3, 2, 1, 0},
  };
  ColMajorMatrix<score_type> gt_scores_mat{
      {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
  };

  SECTION("std::vector get_top_k") {
    CHECK((size_t)size(scores00[0]) == (size_t)(asize));
    CHECK((size_t)size(scores00[1]) == size_t(asize + bsize));

    auto top_k = get_top_k(scores00, k_nn);

    for (size_t i = 0; i < num_vectors; ++i) {
      auto l_nn = std::min<size_t>(top_k[0].size(), scores00[i].size());
      CHECK(std::equal(
          begin(gt_neighbors_mat[i]),
          begin(gt_neighbors_mat[i]) + l_nn,
          begin(top_k[i])));
      CHECK(std::equal(
          begin(top_k[i]) + l_nn,
          begin(top_k[i]) + k_nn,
          std::vector<index_type>(
              k_nn - l_nn, std::numeric_limits<index_type>::max())
              .begin()));
    }
  }
  SECTION("std::vector get_top_k_with_scores") {
    CHECK((size_t)size(scores00[0]) == (size_t)asize);
    CHECK((size_t)size(scores00[1]) == (size_t)(asize + bsize));

    auto&& [top_k_scores, top_k] = get_top_k_with_scores(scores00, k_nn);

    for (size_t i = 0; i < num_vectors; ++i) {
      auto l_nn = std::min<size_t>(top_k[0].size(), scores00[i].size());
      CHECK(std::equal(
          begin(gt_neighbors_mat[i]),
          begin(gt_neighbors_mat[i]) + l_nn,
          begin(top_k[i])));
      CHECK(std::equal(
          begin(top_k[i]) + l_nn,
          begin(top_k[i]) + k_nn,
          std::vector<index_type>(
              k_nn - l_nn, std::numeric_limits<index_type>::max())
              .begin()));
      CHECK(std::equal(
          begin(gt_scores_mat[i]),
          begin(gt_scores_mat[i]) + l_nn,
          begin(top_k_scores[i])));
      CHECK(std::equal(
          begin(top_k_scores[i]) + l_nn,
          begin(top_k_scores[i]) + k_nn,
          std::vector<score_type>(
              k_nn - l_nn, std::numeric_limits<score_type>::max())
              .begin()));
    }
  }

#if 0
  SECTION("std::span get_top_k") {
    std::vector<index_type> top_k(k_nn);
    get_top_k_from_heap(a, std::span(top_k.data(), k_nn));
    REQUIRE(top_k.size() == k_nn);
    auto l_nn = std::min<size_t>(k_nn, a.size());
    CHECK(std::equal(
        begin(gt_neighbors), begin(gt_neighbors) + l_nn, top_k.begin()));
  }

  SECTION("std::vector get_top_k, pad") {
    groundtruth_type pad = 2;
    std::vector<index_type> top_k(k_nn + pad);
    get_top_k_from_heap(a, top_k);
    REQUIRE(top_k.size() == k_nn + pad);
    auto l_nn = std::min<size_t>(k_nn + pad, a.size());
    CHECK(std::equal(
        begin(gt_neighbors), begin(gt_neighbors) + l_nn, top_k.begin()));
    CHECK(end(top_k) == begin(top_k) + k_nn + pad);
    CHECK(end(top_k) - begin(top_k) == k_nn + pad);
    CHECK(std::equal(
        begin(top_k) + l_nn,
        begin(top_k) + k_nn + pad,
        std::vector<index_type>(
            k_nn + pad - l_nn, std::numeric_limits<index_type>::max())
            .begin()));
  }

  SECTION("std::span get_top_k, pad") {
    groundtruth_type pad = 2;
    std::vector<index_type> top_k(k_nn + pad);
    get_top_k_from_heap(a, std::span(top_k.data(), k_nn + pad));
    REQUIRE(top_k.size() == k_nn + pad);
    auto l_nn = std::min<size_t>(k_nn + pad, a.size());
    CHECK(std::equal(
        begin(gt_neighbors), begin(gt_neighbors) + l_nn, top_k.begin()));
    CHECK(end(top_k) == begin(top_k) + k_nn + pad);
    CHECK(end(top_k) - begin(top_k) == k_nn + pad);
    CHECK(std::equal(
        begin(top_k) + l_nn,
        begin(top_k) + k_nn + pad,
        std::vector<index_type>(
            k_nn + pad - l_nn, std::numeric_limits<index_type>::max())
            .begin()));
  }
#endif
}

TEMPLATE_LIST_TEST_CASE(
    "get_top_k_from_heap vector of min_heap", "[scoring]", scoring_typelist) {
  using score_type = std::tuple_element_t<0, TestType>;
  using index_type = std::tuple_element_t<1, TestType>;
  using groundtruth_type = std::tuple_element_t<2, TestType>;
  fixed_min_pair_heap<score_type, index_type> a(
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
  fixed_min_pair_heap<score_type, index_type> b(
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
  std::vector<groundtruth_type> gt_neighbors_vec{
      9,
      8,
      7,
      6,
      5,
      9,
      4,
      0,
      5,
      1,
  };
  std::vector<score_type> gt_scores_vec{
      1,
      2,
      3,
      4,
      5,
      0,
      1,
      2,
      3,
      4,
  };

  // Matrix not used
  ColMajorMatrix<groundtruth_type> gt_neighbors_mat{
      {9, 8, 7, 6, 5},
      {9, 4, 0, 5, 1},
  };
  ColMajorMatrix<score_type> gt_scores_mat{
      {1, 2, 3, 4, 5},
      {0, 1, 2, 3, 4},
  };
  std::vector<fixed_min_pair_heap<score_type, index_type>> scores{a, b};
  SECTION("std::vector") {
    auto top_k = get_top_k(scores, 5);
    CHECK(top_k.num_rows() == 5);
    CHECK(top_k.num_cols() == 2);
    CHECK(std::equal(
        begin(gt_neighbors_vec), end(gt_neighbors_vec), top_k.data()));
  }
  SECTION("std::span") {
    auto top_k = get_top_k(scores, 5);
    CHECK(top_k.num_rows() == 5);
    CHECK(top_k.num_cols() == 2);
    CHECK(std::equal(
        begin(gt_neighbors_vec), end(gt_neighbors_vec), top_k.data()));
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
    CHECK(std::equal(
        begin(gt_neighbors_vec), end(gt_neighbors_vec), top_k.data()));
    CHECK(std::equal(
        begin(gt_scores_vec), end(gt_scores_vec), top_scores.data()));
  }
}

// test pad with sentinel
TEMPLATE_LIST_TEST_CASE("pad_with_sentinels", "[scoring]", scoring_typelist) {
  using score_type = std::tuple_element_t<0, TestType>;
  using index_type = std::tuple_element_t<1, TestType>;
  using groundtruth_type = std::tuple_element_t<2, TestType>;

  auto v = std::vector<index_type>{8, 6, 7, 5, 3, 0, 9, 1, 2, 4, 3};
  auto w = v;
  auto x = std::vector<score_type>{
      3.1, 4.1, 5.9, 2.6, 5.3, 5.8, 9.7, 9.3, 2.3, 8.4, 6.2};
  auto y = x;
  groundtruth_type start = GENERATE(1, 3, 9, 11);

  SECTION("top_k") {
    pad_with_sentinels(start, v);
    CHECK(std::equal(begin(v), begin(v) + start, begin(w)));
    CHECK(std::equal(
        begin(v) + start,
        end(v),
        std::vector<index_type>(
            size(v) - start, std::numeric_limits<index_type>::max())
            .begin()));
  }
  SECTION("top_k_with_scores") {
    pad_with_sentinels(start, v, x);
    CHECK(std::equal(begin(v), begin(v) + start, begin(w)));
    CHECK(std::equal(
        begin(v) + start,
        end(v),
        std::vector<index_type>(
            size(v) - start, std::numeric_limits<index_type>::max())
            .begin()));

    CHECK(std::equal(begin(x), begin(x) + start, begin(y)));
    CHECK(std::equal(
        begin(x) + start,
        end(x),
        std::vector<score_type>(
            size(x) - start, std::numeric_limits<score_type>::max())
            .begin()));
  }
}

// test get_top_k_from_heap and get_top_k_from_heap_with_scores with padding
// test get_top_k and get_top_k_with_scores with padding

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

#ifdef __AVX2__

template <class V, class W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float sum_of_squares_avx2(const V& a, const W& b) {
  // @todo Align on 256 bit boundaries
  const size_t start = 0;
  const size_t size_a = size(a);
  const size_t stop = size_a - (size_a % 8);

  const float* a_ptr = a.data();
  const float* b_ptr = b.data();

  __m256 sum_vec = _mm256_setzero_ps();

  for (int i = start; i < stop; i += 8) {
    // @todo Align on 256 bit boundaries
    __m256 vec_a = _mm256_loadu_ps(a_ptr + i + 0);
    __m256 vec_b = _mm256_loadu_ps(b_ptr + i + 0);

    __m256 diff = _mm256_sub_ps(vec_a, vec_b);
    sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
  }

  __m128 lo = _mm256_castps256_ps128(sum_vec);
  __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
  __m128 combined = _mm_add_ps(lo, hi);
  combined = _mm_hadd_ps(combined, combined);
  combined = _mm_hadd_ps(combined, combined);

  float sum = _mm_cvtss_f32(combined);

  for (size_t i = stop; i < size_a; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

TEST_CASE("avx2", "[scoring]") {
  ColMajorMatrix<float> rand_a{
      {0, 1, 2, 3, 4, 5, 6, 7, 3, 1, 4},
      {8, 9, 10, 11, 12, 13, 14, 15, 1, 5, 9},
      {16, 17, 18, 19, 20, 21, 22, 23, 2, 6, 5},
      {24, 25, 26, 27, 28, 29, 30, 31, 3, 5, 8},
      {
          32,
          33,
          34,
          35,
          36,
          37,
          38,
          39,
          9,
          7,
          9,
      },
      {40, 41, 42, 43, 44, 45, 46, 47, 3, 2, 3},
      {48, 49, 50, 51, 52, 53, 54, 55, 8, 4, 6},
      {56, 57, 58, 59, 60, 61, 62, 63, 2, 6, 4},
      {46, 65, 66, 67, 68, 69, 70, 71, 3, 3, 8},
  };

  ColMajorMatrix<float> rand_b{
      {136, 135, 134, 33, 132, 131, 130, 129, 3, 8, 3},
      {128, 127, 126, 125, 124, 123, 122, 121, 3, 4, 6},
      {120, 119, 118, 117, 116, 115, 114, 113, 2, 6, 4},
      {112, 111, 110, 109, 108, 107, 106, 105, 8, 3, 2},
      {104, 103, 102, 101, 100, 99, 98, 97, 3, 9, 7},
      {96, 95, 94, 93, 92, 91, 90, 89, 9, 8, 5},
      {88, 87, 86, 85, 84, 83, 82, 81, 3, 5, 6},
      {80, 79, 78, 77, 76, 75, 74, 73, 2, 9, 5},
      {72, 71, 70, 69, 68, 67, 66, 65, 1, 4, 1},
  };

  ColMajorMatrix<float> rand_0{
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  float sum_0 = 0;
  for (size_t i = 0; i < num_vectors(rand_a); ++i) {
    for (size_t j = 0; j < dimensions(rand_b); ++j) {
      float diff = rand_a[i][j] - rand_b[i][j];
      sum_0 += diff * diff;
    }
  }
  CHECK(sum_0 == 413544);

  float sum_avx2 = 0.0;

  for (size_t i = 0; i < num_vectors(rand_a); ++i) {
    sum_avx2 += sum_of_squares_avx2(rand_a[i], rand_b[i]);
  }
  CHECK(sum_avx2 == 413544);
}

#endif
