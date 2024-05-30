/**
 * @file   unit_vector.cc
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
 * Tests for Vector class.
 *
 */

#include <catch2/catch_all.hpp>
#include <set>
#include <vector>
#include "detail/linalg/vector.h"

TEST_CASE("test constructor", "[vector]") {
  Vector<int> v(10);
  REQUIRE(v.num_rows() == 10);
  REQUIRE(v.size() == 10);
  REQUIRE(v.data() != nullptr);
}

TEST_CASE("test move constructor", "[vector]") {
  Vector<int> v(10);
  auto p = v.data();
  Vector<int> w(std::move(v));
  REQUIRE(w.num_rows() == 10);
  REQUIRE(w.size() == 10);
  REQUIRE(w.data() == p);
}

TEST_CASE("test operator()", "[vector]") {
  Vector<int> v(10);
  for (int i = 0; i < 10; ++i) {
    v(i) = i;
  }
  for (int i = 0; i < 10; ++i) {
    REQUIRE(v(i) == i);
  }
}

TEST_CASE("test initializer_list", "[vector]") {
  std::vector<int> u{8, 6, 7, 5, 3, 0, 9};
  Vector<int> v{8, 6, 7, 5, 3, 0, 9};

  REQUIRE(v.num_rows() == 7);
  REQUIRE(v.size() == 7);
  REQUIRE(std::equal(v.data(), v.data() + 7, u.begin()));
  REQUIRE(std::equal(begin(v), end(v), u.begin()));
}

TEST_CASE("test move constructor too", "[vector]") {
  std::vector<int> u{8, 6, 7, 5, 3, 0, 9};
  Vector<int> v{8, 6, 7, 5, 3, 0, 9};
  Vector<int> w{std::move(v)};

  REQUIRE(v.data() == nullptr);
  REQUIRE(w.size() == 7);
  REQUIRE(std::equal(begin(w), end(w), u.begin()));
}
