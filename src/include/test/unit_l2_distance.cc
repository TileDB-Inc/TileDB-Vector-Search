/**
 * @file   unit_l2_distance.cc
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
 * Test the l2 distance functions
 */

#include <catch2/catch_all.hpp>
#include "detail/linalg/matrix.h"
#include "detail/scoring/l2_distance.h"

TEMPLATE_TEST_CASE("naive_sum_of_squares", "[l2_distance]", int8_t, uint8_t) {
  // size_t n = GENERATE(1, 3, 127, 1021, 1024);

  size_t n = GENERATE(127);

  auto u = std::vector<TestType>(n);
  auto v = std::vector<TestType>(n);
  auto x = std::vector<float>(n);
  auto y = std::vector<float>(n);

  std::iota(begin(u), end(u), 0);
  std::iota(begin(v), end(v), -9);
  std::iota(begin(x), end(x), 13);
  std::iota(begin(y), end(y), 17);

  auto a = naive_sum_of_squares(x, y);
  CHECK(a == 16 * n);

  auto b = unroll4_sum_of_squares(x, y);
  CHECK(b == 16 * n);

  auto c = naive_sum_of_squares(x, y, 0, size(x));
  CHECK(c == 16 * n);

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
}
