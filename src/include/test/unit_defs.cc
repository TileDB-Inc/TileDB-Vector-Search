/**
 * @file   unit_defs.cc
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
#include <vector>
#include "../defs.h"

TEST_CASE("defs: test test", "[defs]") {
  REQUIRE(true);
}

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

TEST_CASE("defs: std::set", "[defs]") {
  std::set<int> a;

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert(i);
    }
  }
  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert(i);
    }
  }
  CHECK(a.size() == 10);
  CHECK(a.count(0) == 1);
  CHECK(*begin(a) == 0);
  CHECK(*rbegin(a) == 9);
}

TEST_CASE("defs: std::set with pairs", "[defs]") {
  using element = std::pair<float, int>;
  std::set<element> a;

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert({10 - i, i});
    }
    CHECK(begin(a)->first == 1);
    CHECK(begin(a)->second == 9);
    CHECK(rbegin(a)->first == 10.0);
    CHECK(rbegin(a)->second == 0);
  }
  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert({10 + i, i});
    }
    CHECK(begin(a)->first == 10.0);
    CHECK(begin(a)->second == 0);
    CHECK(rbegin(a)->first == 19.0);
    CHECK(rbegin(a)->second == 9);
  }
  CHECK(a.size() == 10);
  // CHECK(*begin(a) == element{10, 0});
  // CHECK(*rbegin(a) == element{9, 1});
}

TEST_CASE("defs: fixed_min_heap", "[defs]") {
  fixed_min_heap<int> a(5);

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert(i);
    }
  }

  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert(i);
    }
  }
  CHECK(a.size() == 5);
  // CHECK(a.count(0) == 1);
  // CHECK(*begin(a) == 0);
  // CHECK(*rbegin(a) == 4);
}

TEST_CASE("defs: fixed_min_heap with pairs", "[defs]") {
  using element = std::pair<float, int>;
  fixed_min_heap<element> a(5);

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert({10 - i, i});
    }
    std::sort(begin(a), end(a));
    CHECK(begin(a)->first == 1);
    CHECK(begin(a)->second == 9);
    CHECK(rbegin(a)->first == 5.0);
    CHECK(rbegin(a)->second == 5);
  }
  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert({10 + i, i});
    }
    std::sort(begin(a), end(a));
    CHECK(begin(a)->first == 10.0);
    CHECK(begin(a)->second == 0);
    CHECK(rbegin(a)->first == 14.0);
    CHECK(rbegin(a)->second == 4);

    for (size_t i = 0; i < size(a); ++i) {
      CHECK(a[i].first == 10.0 + i);
      CHECK((size_t)a[i].second == i);
    }
  }
  CHECK(a.size() == 5);
}

TEST_CASE("defs: fixed_min_heap with a large vector", "[defs]") {
  using element = std::pair<float, int>;
  fixed_min_heap<element> a(7);

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  for (auto&& i : v) {
    a.insert(i);
  }
  CHECK(a.size() == 7);

  std::vector<element> a2(begin(a), end(a));
  std::sort(begin(a2), end(a2));

  std::vector<element> u(v.begin(), v.begin() + 7);

  std::nth_element(v.begin(), v.begin() + 7, v.end());
  std::vector<element> w(v.begin(), v.begin() + 7);

  CHECK(u != w);

  std::vector<element> v3(v.begin(), v.begin() + 7);
  std::sort(begin(v3), end(v3));
  CHECK(a2 == v3);
}
