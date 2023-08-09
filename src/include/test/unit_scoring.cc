/**
 * @file   unit_fixed_min_heap.cc
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



// consolidate scores
// get_top_k_from_heap (one min_heap)
// get_top_k_from_heap (vector of min_heaps)
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