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
#include "detail/scoring/l2_distance.h"


TEST_CASE("l2_distance: null test", "[l2_distance]") {

}


TEST_CASE("l2_distance: naive_sum_of_squares", "[l2_distance]") {

  auto u = std::vector<uint8_t>(1021);
  auto v = std::vector<uint8_t>(1021);
  auto x = std::vector<float>(1021);
  auto y = std::vector<float>(1021);

  std::iota(begin(u), end(u), 0);
  std::iota(begin(v), end(v), -9);
  std::iota(begin(x), end(x), 13);  
  std::iota(begin(y), end(y), 17);  

  auto a = naive_sum_of_squares(u, v);
  CHECK(a == 81 * 1021);
}
