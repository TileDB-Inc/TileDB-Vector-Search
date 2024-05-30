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

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <iostream>
#include <set>
#include <span>
#include <vector>
#include "detail/linalg/vector.h"
#include "utils/fixed_min_heap.h"

bool debug = false;

TEST_CASE("std::heap", "[fixed_min_heap]") {
  std::vector<int> v{3, 1, 4, 1, 5, 9};

  if (debug) {
    debug_vector(v, "initial vector");
  }

  std::make_heap(v.begin(), v.end());
  CHECK(std::is_heap(v.begin(), v.end(), std::less<>()));

  if (debug) {
    debug_vector(v, "initial max heap");
  }

  std::pop_heap(v.begin(), v.end());

  if (debug) {
    debug_vector(v, "max heap after pop heap");
  }

  v.pop_back();
  CHECK(std::is_heap(v.begin(), v.end(), std::less<>()));

  if (debug) {
    debug_vector(v, "max heap after pop back");
  }

  std::make_heap(v.begin(), v.end(), std::greater<>());
  CHECK(std::is_heap(v.begin(), v.end(), std::greater<>()));

  if (debug) {
    debug_vector(v, "min heap after make heap");
  }

  std::pop_heap(v.begin(), v.end(), std::greater<>());

  if (debug) {
    debug_vector(v, "min heap after pop heap");
  }

  v.pop_back();

  CHECK(std::is_heap(v.begin(), v.end(), std::greater<>()));

  if (debug) {
    debug_vector(v, "min heap after pop back");
  }
}

TEST_CASE("std::set", "[fixed_min_heap]") {
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

TEST_CASE("std::set with pairs", "[fixed_min_heap]") {
  using element = std::tuple<float, int>;
  std::set<element> a;

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert({10 - i, i});
    }
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 10.0);
    CHECK(std::get<1>(*(rbegin(a))) == 0);
  }
  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert({10 + i, i});
    }
    CHECK(std::get<0>(*(begin(a))) == 10.0);
    CHECK(std::get<1>(*(begin(a))) == 0);
    CHECK(std::get<0>(*(rbegin(a))) == 19.0);
    CHECK(std::get<1>(*(rbegin(a))) == 9);
  }
  CHECK(a.size() == 10);
  // CHECK(*begin(a) == element{10, 0});
  // CHECK(*rbegin(a) == element{9, 1});
}

TEST_CASE("initializer constructor", "[fixed_min_heap]") {
  fixed_min_pair_heap<float, int> a(
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
  CHECK(std::is_heap(begin(a), end(a)));
  CHECK(std::is_heap(begin(a), end(a), std::less<>()));
  CHECK(std::is_heap(begin(a), end(a), [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  }));

  SECTION("pop") {
    CHECK(a.size() == 5);
    // Note that the "fixed_min_heap" is a max heap, so the first element
    // is actually the largest.
    CHECK(std::get<0>(*(begin(a))) == 5);
    CHECK(std::get<0>(a.front()) == 5);
    a.pop();
    CHECK(std::get<0>(*(begin(a))) == 4);
    CHECK(std::get<0>(a.front()) == 4);
    a.pop();
    CHECK(std::get<0>(*(begin(a))) == 3);
    CHECK(std::get<0>(a.front()) == 3);
    a.pop();
    CHECK(std::get<0>(*(begin(a))) == 2);
    CHECK(std::get<0>(a.front()) == 2);
    a.pop();
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<0>(a.front()) == 1);
    a.pop();
    CHECK(a.empty());
  }

  SECTION("pop and insert") {
    CHECK(a.size() == 5);
    // Note that the "fixed_min_heap" is a max heap, so the first element
    // is actually the largest.
    CHECK(std::get<0>(*(begin(a))) == 5);
    CHECK(std::get<0>(a.front()) == 5);
    a.pop();
    CHECK(std::get<0>(*(begin(a))) == 4);
    CHECK(std::get<0>(a.front()) == 4);
    a.insert(10, 10);
    CHECK(std::get<0>(*(begin(a))) == 10);
    CHECK(std::get<0>(a.front()) == 10);
    a.insert(3.3, 33);
    CHECK(std::get<0>(*(begin(a))) == 4);
    CHECK(std::get<0>(a.front()) == 4);
    a.pop();
    CHECK(std::get<0>(*(begin(a))) == 3.3F);
    CHECK(std::get<0>(a.front()) == 3.3F);
    CHECK(std::get<1>(a.front()) == 33);
  }

  SECTION("sort") {
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }
  SECTION("sort_heap") {
    std::sort_heap(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }
  SECTION("sort, 0") {
    std::sort(begin(a), end(a), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }
  SECTION("sort_heap, 0") {
    std::sort_heap(begin(a), end(a), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }
}

TEST_CASE("fixed_min_pair_heap", "[fixed_min_heap]") {
  fixed_min_pair_heap<float, int> a(5);

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert(10 - i, i);
    }
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }
  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert(10 + i, i);
    }
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 10.0);
    CHECK(std::get<1>(*(begin(a))) == 0);
    CHECK(std::get<0>(*(rbegin(a))) == 14.0);
    CHECK(std::get<1>(*(rbegin(a))) == 4);

    for (size_t i = 0; i < size(a); ++i) {
      CHECK(std::get<0>(a[i]) == 10.0 + i);
      CHECK((size_t)std::get<1>(a[i]) == i);
    }
  }
  CHECK(a.size() == 5);
}

TEST_CASE("fixed_min_heap with a 5500 vector", "[fixed_min_heap]") {
  using element = std::tuple<float, int>;
  fixed_min_pair_heap<float, int> a(7);
  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  for (auto&& [e, f] : v) {
    a.insert(e, f);
  }
  CHECK(a.size() == 7);
  std::vector<element> a2(begin(a), end(a));
  std::sort(begin(a2), end(a2));
  CHECK(a2.size() == 7);
  std::vector<element> u(v.begin(), v.begin() + 7);
  CHECK(u.size() == 7);
  std::nth_element(v.begin(), v.begin() + 7, v.end());
  std::vector<element> w(v.begin(), v.begin() + 7);
  CHECK(w.size() == 7);
  CHECK(u != w);

  std::vector<element> v3(v.begin(), v.begin() + 7);
  std::sort(begin(v3), end(v3));
  CHECK(std::equal(begin(v3), end(v3), begin(a2), [](auto&& f, auto&& s) {
    return std::get<0>(f) == std::get<0>(s);
  }));

  std::sort_heap(begin(a), end(a), first_less<element>{});
  CHECK(a == a2);
}

TEMPLATE_TEST_CASE(
    "first_less", "[fixed_min_heap]", float, double, int, unsigned) {
  first_less<std::tuple<TestType, size_t>> a;
  auto v = std::vector<std::tuple<TestType, size_t>>{
      {0, 0},
      {1, 1},
      {0, 2},
      {3, 3},
      {4, 4},
      {5, 5},
      {6, 6},
      {7, 7},
      {8, 8},
      {9, 9},
  };
  CHECK(a(v[0], v[1]));
  CHECK(!a(v[1], v[0]));
  CHECK(a(v[2], v[1]));
  CHECK(!a(v[1], v[2]));
}

TEST_CASE(
    "threshold_heap: threshold_heap", "[fixed_min_heap][threshold_heap]") {
  threshold_heap<float, int> a(5);

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert(10 - i, i);
    }
    CHECK(a.size() == 4);
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 4.0);
    CHECK(std::get<1>(*(rbegin(a))) == 6);
  }
  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert(10 + i, i);
    }
    CHECK(a.size() == 0);
  }
}

TEST_CASE(
    "threshold_heap: threshold_heap with a large vector", "[threshold_heap]") {
  using element = std::tuple<float, int>;

  float thresh = std::rand();

  threshold_heap<float, int> a(thresh);

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  v[0] = {thresh, 0};
  for (auto&& [e, f] : v) {
    a.insert(e, f);
  }
  CHECK(a.size() != 0);
  for (auto&& [e, f] : a) {
    CHECK(e < thresh);
  }

  a.resize(size(v));
  std::copy(begin(v), end(v), begin(a));
  a.filtered_heapify();
  CHECK(a.size() != 0);
  for (auto&& [e, f] : a) {
    CHECK(e < thresh);
  }
}

TEST_CASE("threshold_heap: new threshold", "[threshold_heap]") {
  using element = std::tuple<float, int>;

  float thresh = std::rand();
  threshold_heap<float, int> a(thresh);

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  v[0] = {thresh, 0};

  a.resize(size(v));
  std::copy(begin(v), end(v), begin(a));
  a.filtered_heapify();
  for (auto&& [e, f] : a) {
    CHECK(e < thresh);
  }

  thresh -= thresh / 2.0;
  a.set_threshold(thresh);
  for (auto&& [e, f] : a) {
    CHECK(e < thresh);
  }
}

TEST_CASE(
    "fixed_max_heap with a large vector and compare function",
    "[fixed_min_heap]") {
  using element = std::tuple<float, int>;

  fixed_min_pair_heap<float, int, std::greater<float>> a(
      7, std::greater<float>{});

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  for (auto&& [e, f] : v) {
    a.insert(e, f);
  }
  CHECK(a.size() == 7);

  std::vector<element> a2(begin(a), end(a));
  std::sort(begin(a2), end(a2), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });

  std::vector<element> u(v.begin(), v.begin() + 7);

  std::nth_element(v.begin(), v.begin() + 7, v.end(), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  std::vector<element> w(v.begin(), v.begin() + 7);

  CHECK(u != w);

  std::vector<element> v3(v.begin(), v.begin() + 7);
  std::sort(begin(v3), end(v3), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  CHECK(a2 == v3);
}

TEST_CASE("fixed_max_heap with a large vector", "[fixed_min_heap]") {
  using element = std::tuple<float, int>;

  fixed_min_pair_heap<float, int, std::greater<float>> a(
      7, std::greater<float>{});

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  for (auto&& [e, f] : v) {
    a.insert(e, f);
  }
  CHECK(a.size() == 7);

  std::vector<element> a2(begin(a), end(a));
  std::sort(begin(a2), end(a2), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });

  std::vector<element> u(v.begin(), v.begin() + 7);

  std::nth_element(v.begin(), v.begin() + 7, v.end(), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  std::vector<element> w(v.begin(), v.begin() + 7);

  CHECK(u != w);

  std::vector<element> v3(v.begin(), v.begin() + 7);
  std::sort(begin(v3), end(v3), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  CHECK(a2 == v3);
}

// This seems to duplicate above
#if 0
TEST_CASE(
    "fixed_max_heap with a large vector", "[fixed_min_heap]") {
  using element = std::tuple<float, int>;

  fixed_min_pair_heap<float, int, std::greater<float>> a(
      7, std::greater<float>{});

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  for (auto&& [e, f] : v) {
    a.insert(e, f);
  }
  CHECK(a.size() == 7);

  std::vector<element> a2(begin(a), end(a));
  std::sort(begin(a2), end(a2), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });

  std::vector<element> u(v.begin(), v.begin() + 7);

  std::nth_element(v.begin(), v.begin() + 7, v.end(), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  std::vector<element> w(v.begin(), v.begin() + 7);

  CHECK(u != w);

  std::vector<element> v3(v.begin(), v.begin() + 7);
  std::sort(begin(v3), end(v3), [](auto&& a, auto&& b) {
    return std::get<0>(a) > std::get<0>(b);
  });
  CHECK(a2 == v3);
}
#endif
