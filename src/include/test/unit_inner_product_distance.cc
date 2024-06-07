/**
 * @file   unit_inner_product_distance.cc
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
 * Test the inner product distance functions
 */
#include <catch2/catch_all.hpp>
#include "detail/scoring/inner_product.h"

TEST_CASE("simple vectors", "[inner_product_distance]") {
  auto u = std::vector<uint8_t>{1, 2, 3, 4};
  auto v = std::vector<uint8_t>{5, 6, 7, 8};
  auto x = std::vector<float>{1, 2, 3, 4};
  auto y = std::vector<float>{5, 6, 7, 8};

  auto z = std::vector<float>{1, 0, 1, -1};

  auto a = naive_inner_product(x, y);
  CHECK(a == 70);

  auto b = unroll4_inner_product(x, y);
  CHECK(b == 70);

  auto c = naive_inner_product(x, v);
  CHECK(c == 70);

  auto d = unroll4_inner_product(u, y);
  CHECK(d == 70);

  auto e = naive_inner_product(u, v);
  CHECK(e == 70);

  auto f = unroll4_inner_product(u, v);
  CHECK(f == 70);

  auto g = naive_inner_product(u, z);
  CHECK(g == 0);

  auto h = unroll4_inner_product(u, z);
  CHECK(h == 0);

  auto i = naive_inner_product(x, z);
  CHECK(i == 0);

  auto j = unroll4_inner_product(x, z);
  CHECK(j == 0);

#ifdef __AVX2__
  auto k = avx2_inner_product(x, y);
  CHECK(k == 70);

  auto l = avx2_inner_product(x, z);
  CHECK(l == 0);

  auto m = avx2_inner_product(u, y);
  CHECK(m == 70);

  auto n = avx2_inner_product(u, z);
  CHECK(n == 0);
#endif
}

TEST_CASE(
    "inner_distance: naive_inner_product with longer vectors",
    "[inner_product_distance]") {
  // size_t n = GENERATE(1, 3, 127, 1021, 1024);

  // @todo generate a range of sizes (will require range of expected answers)
  size_t n = GENERATE(127);

  auto u = std::vector<uint8_t>(n);
  auto v = std::vector<uint8_t>(n);
  auto x = std::vector<float>(n);
  auto y = std::vector<float>(n);

  auto z = std::vector<float>(n);

  std::iota(begin(u), end(u), 0);
  std::iota(begin(v), end(v), 13);
  std::iota(begin(x), end(x), -3.14159);
  std::iota(begin(y), end(y), 17.8675309);

  std::iota(begin(z), end(z), 13);

  auto a = naive_inner_product(x, y);
  CHECK(std::abs(a - 785444.4375) < 0.01);
  auto ax = naive_inner_product(y, x);
  CHECK(ax == a);
  CHECK(std::abs(ax - 785444.4375) < 0.01);

  auto b = unroll4_inner_product(x, y);
  CHECK(std::abs(b - 785444.4375) < 0.01);
  CHECK(a == b);
  auto bx = unroll4_inner_product(y, x);
  CHECK(bx == b);
  CHECK(std::abs(bx - 785444.4375) < 0.01);

  auto a2 = naive_inner_product(u, v);
  CHECK(a2 == 778764);
  auto b2 = unroll4_inner_product(u, v);
  CHECK(b2 == 778764);
  CHECK(a2 == b2);

  auto a2x = naive_inner_product(v, u);
  CHECK(a2x == 778764);
  CHECK(a2 == a2x);
  auto b2x = unroll4_inner_product(v, u);
  CHECK(b2x == 778764);
  CHECK(a2x == b2x);

  auto a3 = naive_inner_product(u, x);
  CHECK(std::abs(a3 - 649615.1875) < 0.25);
  auto b3 = unroll4_inner_product(u, x);
  CHECK(std::abs(a3 - 649615.1875) < 0.25);
  CHECK(std::abs(a3 - b3) < 0.25);

  auto a3x = naive_inner_product(x, u);
  CHECK(std::abs(a3x - 649615.1875) < 0.25);
  CHECK(a3 == a3x);
  auto b3x = unroll4_inner_product(x, u);
  CHECK(std::abs(a3x - 649615.1875) < 0.25);
  CHECK(std::abs(a3x - b3x) < 0.25);

#ifdef __AVX2__
  {
    auto a = avxs_inner_product(x, y);
    CHECK(std::abs(a - 785444.4375) < 0.01);

    auto ax = avx2_inner_product(y, x);
    CHECK(ax == a);
    CHECK(std::abs(ax - 785444.4375) < 0.01);

    auto a2 = avx2_inner_product(u, v);
    CHECK(a2 == 778764);

    auto a2x = avx2_inner_product(v, u);
    CHECK(a2x == 778764);
    CHECK(a2 == a2x);

    auto a3 = avx2_inner_product(u, x);
    CHECK(std::abs(a3 - 649615.1875) < 0.25);

    auto a3x = avx2_inner_product(x, u);
    CHECK(std::abs(a3x - 649615.1875) < 0.25);
    CHECK(a3 == a3x);
  }
#endif

// @todo: inner_product does not yet have a partitioned version implemented yet.
// Leaving code here for future reference
#if 0
 // auto c = naive_inner_product(x, y, 0, size(x));
 // CHECK(c == 16 * n);

 auto d = unroll4_sum_of_squares(x, y, 0, size(x));
 CHECK(d == 16 * n);

 auto e = naive_sum_of_squares(x, y, 0, size(x) / 2);
 auto f = naive_sum_of_squares(x, y, size(x) / 2, size(x));
 CHECK(e == 16 * (size(x) / 2));
 CHECK(f == 16 * (size(x) - size(x) / 2));
 CHECK(e + f == 16 * n);

 auto g = unroll4_sum_of_squares(x, y, 0, size(x) / 2);
 auto h = unroll4_sum_of_squares(x, y, size(x) / 2, size(x));
 CHECK(g == 16 * (size(x) / 2));
 CHECK(h == 16 * (size(x) - size(x) / 2));

 CHECK(e == g);
 CHECK(f == h);

 CHECK(g + h == 16 * n);
#endif
}
