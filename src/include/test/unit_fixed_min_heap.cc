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
#include <vector>
#include "detail/linalg/vector.h"
#include "utils/fixed_min_heap.h"

bool debug = false;

////////////////////////////////////////////////////////////////////////////////////////////////////
// fixed_min_pair_heap
////////////////////////////////////////////////////////////////////////////////////////////////////

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
      a.insert(static_cast<float>(10 - i), i);
    }
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }
  SECTION("insert in descending order") {
    for (auto&& i : {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert(static_cast<float>(10 + i), i);
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// fixed_min_triplet_heap
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("std::heap with triplets", "[fixed_min_heap]") {
  std::vector<std::tuple<int, int, int>> v{
      {3, 1, 0}, {1, 2, 0}, {4, 3, 0}, {1, 4, 0}, {5, 5, 0}, {9, 6, 0}};

  std::make_heap(v.begin(), v.end());
  CHECK(std::is_heap(v.begin(), v.end(), std::less<>()));

  std::pop_heap(v.begin(), v.end());
  v.pop_back();
  CHECK(std::is_heap(v.begin(), v.end(), std::less<>()));

  std::make_heap(v.begin(), v.end(), std::greater<>());
  CHECK(std::is_heap(v.begin(), v.end(), std::greater<>()));

  std::pop_heap(v.begin(), v.end(), std::greater<>());
  v.pop_back();
  CHECK(std::is_heap(v.begin(), v.end(), std::greater<>()));
}

TEST_CASE("std::set with triplets", "[fixed_min_heap]") {
  using element = std::tuple<int, int, int>;
  std::set<element> a;

  SECTION("insert in ascending order") {
    for (int i = 0; i < 10; ++i) {
      a.insert({i, i + 1, i + 2});
    }
    CHECK(std::get<0>(*begin(a)) == 0);
    CHECK(std::get<0>(*rbegin(a)) == 9);
  }

  SECTION("insert in descending order") {
    for (int i = 9; i >= 0; --i) {
      a.insert({i, i + 1, i + 2});
    }
    CHECK(std::get<0>(*begin(a)) == 0);
    CHECK(std::get<0>(*rbegin(a)) == 9);
  }

  CHECK(a.size() == 10);
}

TEST_CASE("std::set with triplets (complex)", "[fixed_min_heap]") {
  using element = std::tuple<float, int, int>;
  std::set<element> a;

  SECTION("insert in ascending order") {
    for (int i = 0; i < 10; ++i) {
      a.insert({10.0f - i, i, i + 2});
    }
    CHECK(std::get<0>(*begin(a)) == 1.0f);
    CHECK(std::get<1>(*begin(a)) == 9);
    CHECK(std::get<0>(*rbegin(a)) == 10.0f);
    CHECK(std::get<1>(*rbegin(a)) == 0);
  }

  SECTION("insert in descending order") {
    for (int i = 9; i >= 0; --i) {
      a.insert({10.0f + i, i, i + 2});
    }
    CHECK(std::get<0>(*begin(a)) == 10.0f);
    CHECK(std::get<1>(*begin(a)) == 0);
    CHECK(std::get<0>(*rbegin(a)) == 19.0f);
    CHECK(std::get<1>(*rbegin(a)) == 9);
  }

  CHECK(a.size() == 10);
}

TEST_CASE("initializer constructor with triplets", "[fixed_min_heap]") {
  fixed_min_triplet_heap<float, int, int> a(
      5,
      {
          {10.0f, 0, 0},
          {9.0f, 1, 0},
          {8.0f, 2, 0},
          {7.0f, 3, 0},
          {6.0f, 4, 0},
          {5.0f, 5, 0},
          {4.0f, 6, 0},
          {3.0f, 7, 0},
          {2.0f, 8, 0},
          {1.0f, 9, 0},
      });
  CHECK(std::is_heap(begin(a), end(a)));
  CHECK(std::is_heap(begin(a), end(a), std::less<>()));
  CHECK(std::is_heap(begin(a), end(a), [](auto&& a, auto&& b) {
    return std::get<0>(a) < std::get<0>(b);
  }));

  SECTION("pop") {
    CHECK(a.size() == 5);
    CHECK(std::get<0>(a.front()) == 5.0f);
    a.pop();
    CHECK(std::get<0>(a.front()) == 4.0f);
    a.pop();
    CHECK(std::get<0>(a.front()) == 3.0f);
    a.pop();
    CHECK(std::get<0>(a.front()) == 2.0f);
    a.pop();
    CHECK(std::get<0>(a.front()) == 1.0f);
    a.pop();
    CHECK(a.empty());
  }

  SECTION("pop and insert") {
    CHECK(a.size() == 5);
    CHECK(std::get<0>(a.front()) == 5.0f);
    a.pop();
    CHECK(std::get<0>(a.front()) == 4.0f);
    a.insert(10.0f, 10, 10);
    CHECK(std::get<0>(a.front()) == 10.0f);
    a.insert(3.3f, 33, 33);
    CHECK(std::get<0>(a.front()) == 4.0f);
    a.pop();
    CHECK(std::get<0>(a.front()) == 3.3f);
    CHECK(std::get<1>(a.front()) == 33);
  }

  SECTION("sort") {
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1.0f);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0f);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }

  SECTION("sort_heap") {
    std::sort_heap(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1.0f);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0f);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }

  SECTION("sort, 0") {
    std::sort(begin(a), end(a), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });
    CHECK(std::get<0>(*(begin(a))) == 1.0f);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0f);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }

  SECTION("sort_heap, 0") {
    std::sort_heap(begin(a), end(a), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });
    CHECK(std::get<0>(*(begin(a))) == 1.0f);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0f);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }
}

TEST_CASE("fixed_min_triplet_heap", "[fixed_min_heap]") {
  fixed_min_triplet_heap<float, int, int> a(5);

  SECTION("insert in ascending order") {
    for (int i = 0; i < 10; ++i) {
      a.insert(10.0f - i, i, i + 2);
    }
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 1.0f);
    CHECK(std::get<1>(*(begin(a))) == 9);
    CHECK(std::get<0>(*(rbegin(a))) == 5.0f);
    CHECK(std::get<1>(*(rbegin(a))) == 5);
  }

  SECTION("insert in descending order") {
    for (int i = 9; i >= 0; --i) {
      a.insert(10.0f + i, i, i + 2);
    }
    std::sort(begin(a), end(a));
    CHECK(std::get<0>(*(begin(a))) == 10.0f);
    CHECK(std::get<1>(*(begin(a))) == 0);
    CHECK(std::get<0>(*(rbegin(a))) == 14.0f);
    CHECK(std::get<1>(*(rbegin(a))) == 4);

    for (size_t i = 0; i < size(a); ++i) {
      CHECK(std::get<0>(a[i]) == 10.0f + i);
      CHECK((size_t)std::get<1>(a[i]) == i);
    }
  }
  CHECK(a.size() == 5);
}

TEST_CASE("fixed_min_triplet_heap with a 5500 vector", "[fixed_min_heap]") {
  using element = std::tuple<float, int, int>;
  fixed_min_triplet_heap<float, int, int> a(7);
  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand(), std::rand()};
    CHECK(i != element{});
  }
  for (auto&& [e, f, g] : v) {
    a.insert(e, f, g);
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
    "first_less with triplets",
    "[fixed_min_heap]",
    float,
    double,
    int,
    unsigned) {
  first_less<std::tuple<TestType, size_t, size_t>> a;
  auto v = std::vector<std::tuple<TestType, size_t, size_t>>{
      {0, 0, 0},
      {1, 1, 1},
      {0, 2, 2},
      {3, 3, 3},
      {4, 4, 4},
      {5, 5, 5},
      {6, 6, 6},
      {7, 7, 7},
      {8, 8, 8},
      {9, 9, 9},
  };
  CHECK(a(v[0], v[1]));
  CHECK(!a(v[1], v[0]));
  CHECK(a(v[2], v[1]));
  CHECK(!a(v[1], v[2]));
}

TEST_CASE(
    "fixed_max_triplet_heap with a large vector and compare function",
    "[fixed_min_heap]") {
  using element = std::tuple<float, int, int>;

  fixed_min_triplet_heap<float, int, int, std::greater<float>> a(
      7, std::greater<float>{});

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand(), std::rand()};
    CHECK(i != element{});
  }
  for (auto&& [e, f, g] : v) {
    a.insert(e, f, g);
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

TEST_CASE("fixed_max_triplet_heap with a large vector", "[fixed_min_heap]") {
  using triplet = std::tuple<float, int, int>;

  fixed_min_triplet_heap<float, int, int, std::greater<float>> triplet_heap(
      7, std::greater<float>{});

  std::vector<triplet> random_triplets(55);
  for (auto&& current_triplet : random_triplets) {
    current_triplet = {std::rand(), std::rand(), std::rand()};
    CHECK(current_triplet != triplet{});
  }
  for (auto&& [value, x, y] : random_triplets) {
    triplet_heap.insert(value, x, y);
  }
  CHECK(triplet_heap.size() == 7);

  std::vector<triplet> sorted_heap_triplets(
      begin(triplet_heap), end(triplet_heap));
  std::sort(
      begin(sorted_heap_triplets),
      end(sorted_heap_triplets),
      [](auto&& lhs, auto&& rhs) {
        return std::get<0>(lhs) > std::get<0>(rhs);
      });

  std::vector<triplet> first_seven_random_triplets(
      random_triplets.begin(), random_triplets.begin() + 7);

  std::nth_element(
      random_triplets.begin(),
      random_triplets.begin() + 7,
      random_triplets.end(),
      [](auto&& lhs, auto&& rhs) {
        return std::get<0>(lhs) > std::get<0>(rhs);
      });
  std::vector<triplet> top_seven_random_triplets(
      random_triplets.begin(), random_triplets.begin() + 7);

  CHECK(first_seven_random_triplets != top_seven_random_triplets);

  std::vector<triplet> sorted_top_seven_random_triplets(
      random_triplets.begin(), random_triplets.begin() + 7);
  std::sort(
      begin(sorted_top_seven_random_triplets),
      end(sorted_top_seven_random_triplets),
      [](auto&& lhs, auto&& rhs) {
        return std::get<0>(lhs) > std::get<0>(rhs);
      });
  CHECK(sorted_heap_triplets == sorted_top_seven_random_triplets);
}

TEST_CASE("basic insertion and size check", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  REQUIRE(heap.insert(10, 1, 100));
  REQUIRE(heap.insert(20, 2, 200));
  REQUIRE(heap.insert(5, 3, 300));

  REQUIRE(heap.size() == 3);
}

TEST_CASE(
    "insertion beyond capacity and eviction check",
    "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  heap.insert(10, 1, 100);
  heap.insert(20, 2, 200);
  heap.insert(5, 3, 300);

  // Inserting an element smaller than the largest element in the heap
  REQUIRE(heap.insert(15, 4, 400));
  REQUIRE(heap.size() == 3);

  // Largest element (20, 2, 200) should have been evicted
  auto [score, id, val] = heap.front();
  REQUIRE(score == 15);
  REQUIRE(id == 4);
  REQUIRE(val == 400);

  // Inserting an element larger than the largest element in the heap
  REQUIRE_FALSE(heap.insert(25, 5, 500));
  REQUIRE(heap.size() == 3);
}

TEST_CASE("insert with unique_id constraint", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  REQUIRE(heap.insert(10, 1, 100));
  REQUIRE(heap.insert(20, 2, 200));
  REQUIRE(heap.insert(5, 3, 300));

  // Attempt to insert an element with an existing id (2)
  REQUIRE_FALSE(heap.insert<unique_id>(200, 2, 400));
  REQUIRE(heap.size() == 3);

  // Ensure the existing element with id 2 has not changed
  auto [score, id, val] = heap.front();
  REQUIRE(score == 20);
  REQUIRE(id == 2);
  REQUIRE(val == 200);
}

TEST_CASE("evict_insert functionality", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  heap.insert(10, 1, 100);
  heap.insert(20, 2, 200);
  heap.insert(5, 3, 300);

  auto [inserted, evicted, old_score, old_id] = heap.evict_insert(15, 4, 400);

  REQUIRE(inserted);
  REQUIRE(evicted);
  REQUIRE(old_score == 20);
  REQUIRE(old_id == 2);

  std::tie(inserted, evicted, old_score, old_id) =
      heap.evict_insert(25, 5, 500);
  REQUIRE_FALSE(inserted);
  REQUIRE_FALSE(evicted);

  std::tie(inserted, evicted, old_score, old_id) =
      heap.evict_insert<unique_id>(5, 4, 450);
  REQUIRE_FALSE(inserted);
  REQUIRE_FALSE(evicted);
}

template <typename T, typename U, typename V>
void check_front(
    const fixed_min_triplet_heap<T, U, V>& heap,
    const T& expected_first,
    const U& expected_second,
    const V& expected_third) {
  auto [first, second, third] = heap.front();
  REQUIRE(first == expected_first);
  REQUIRE(second == expected_second);
  REQUIRE(third == expected_third);
}

TEST_CASE("self_heapify and self_sort", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(5);

  heap.insert(20, 2, 200);
  heap.insert(10, 1, 100);
  heap.insert(30, 3, 300);
  heap.insert(15, 4, 400);
  heap.insert(5, 5, 500);
  REQUIRE(heap.size() == 5);

  heap.pop();
  REQUIRE(heap.size() == 4);
  heap.pop();
  REQUIRE(heap.size() == 3);

  heap.insert(25, 6, 600);
  REQUIRE(heap.size() == 4);

  // Heapify the remaining elements
  heap.self_heapify();
  REQUIRE(heap.size() == 4);

  // After heapify, the largest element should be on top
  check_front(heap, 25, 6, 600);
}

TEST_CASE("insertion of duplicate values", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  REQUIRE(heap.insert(10, 1, 100));
  REQUIRE(heap.insert(20, 2, 200));
  REQUIRE(heap.insert(10, 3, 300));

  REQUIRE(heap.size() == 3);

  // Inserting a duplicate score but different id should succeed
  REQUIRE(heap.insert(10, 4, 400));
  REQUIRE(heap.size() == 3);

  // Heap should still maintain min-heap properties
  check_front(heap, 10, 1, 100);
}

TEST_CASE(
    "inserting elements with min and max values", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  REQUIRE(heap.insert(std::numeric_limits<int>::max(), 1, 100));
  REQUIRE(heap.dump() == "(2147483647, 1, 100) ");
  REQUIRE(heap.size() == 1);

  REQUIRE(heap.insert(std::numeric_limits<int>::min(), 2, 200));
  REQUIRE(heap.dump() == "(2147483647, 1, 100) (-2147483648, 2, 200) ");
  REQUIRE(heap.size() == 2);

  REQUIRE(heap.insert(0, 3, 300));
  REQUIRE(
      heap.dump() == "(2147483647, 1, 100) (-2147483648, 2, 200) (0, 3, 300) ");
  REQUIRE(heap.size() == 3);

  check_front(heap, std::numeric_limits<int>::max(), 1, 100);
  REQUIRE(
      heap.dump() == "(2147483647, 1, 100) (-2147483648, 2, 200) (0, 3, 300) ");

  REQUIRE_FALSE(heap.insert(std::numeric_limits<int>::max(), 4, 400));
  check_front(heap, std::numeric_limits<int>::max(), 1, 100);
  REQUIRE(
      heap.dump() == "(2147483647, 1, 100) (-2147483648, 2, 200) (0, 3, 300) ");

  REQUIRE_FALSE(heap.insert(std::numeric_limits<int>::max(), -10, -10));
  REQUIRE(heap.size() == 3);
  check_front(heap, std::numeric_limits<int>::max(), 1, 100);
  REQUIRE(
      heap.dump() == "(2147483647, 1, 100) (-2147483648, 2, 200) (0, 3, 300) ");

  REQUIRE(heap.insert(std::numeric_limits<int>::min(), 5, 500));
  REQUIRE(heap.size() == 3);
  check_front(heap, 0, 3, 300);
  REQUIRE(
      heap.dump() ==
      "(0, 3, 300) (-2147483648, 2, 200) (-2147483648, 5, 500) ");
}

TEST_CASE(
    "handling of insertions when heap is at capacity with equal elements",
    "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  heap.insert(10, 1, 100);
  heap.insert(10, 2, 200);
  heap.insert(10, 3, 300);

  REQUIRE(heap.size() == 3);

  // Insert an element with the same score but different id and value
  REQUIRE_FALSE(heap.insert(10, 4, 400));
  REQUIRE(heap.size() == 3);

  auto [score, id, val] = heap.front();
  REQUIRE(score == 10);
  REQUIRE(id == 1);
  REQUIRE(val == 100);

  // Insert an element with a smaller score should cause an eviction
  REQUIRE(heap.insert(5, 4, 400));
  REQUIRE(heap.size() == 3);
}

TEST_CASE("evict_insert on empty heap", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  auto [inserted, evicted, old_score, old_id] = heap.evict_insert(10, 1, 100);

  REQUIRE(inserted);
  REQUIRE_FALSE(evicted);
  REQUIRE(old_score == 10);
  REQUIRE(old_id == 1);
  REQUIRE(heap.size() == 1);
}

TEST_CASE("evict_insert with identical elements", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  heap.insert(10, 1, 100);
  heap.insert(10, 2, 200);
  heap.insert(10, 3, 300);

  auto [inserted, evicted, old_score, old_id] = heap.evict_insert(10, 4, 400);

  // No element should be evicted since all elements have the same score
  REQUIRE_FALSE(inserted);
  REQUIRE_FALSE(evicted);
  REQUIRE(heap.size() == 3);

  // Insert an element with a lower score
  std::tie(inserted, evicted, old_score, old_id) = heap.evict_insert(5, 5, 500);
  REQUIRE(inserted);
  REQUIRE(evicted);
  REQUIRE(old_score == 10);
  REQUIRE(heap.size() == 3);
}

TEST_CASE("pop on an empty heap", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(3);

  REQUIRE_NOTHROW(heap.pop());

  heap.insert(10, 1, 100);
  REQUIRE_NOTHROW(heap.pop());
  REQUIRE(heap.empty());

  REQUIRE_NOTHROW(heap.pop());
}

TEST_CASE(
    "large number of insertions to test heap integrity",
    "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<int, int, int> heap(10);

  for (int i = 0; i < 100; ++i) {
    heap.insert(i, i, i * 10);
  }

  REQUIRE(heap.size() == 10);
}

TEST_CASE("handling insertion of complex types", "[fixed_min_triplet_heap]") {
  fixed_min_triplet_heap<std::string, int, double> heap(3);

  REQUIRE(heap.insert("b", 1, 1.1));
  REQUIRE(heap.insert("c", 2, 2.2));
  REQUIRE(heap.insert("d", 3, 3.3));
  REQUIRE(heap.size() == 3);
  REQUIRE(heap.dump() == "(d, 3, 3.3) (b, 1, 1.1) (c, 2, 2.2) ");

  auto [score, id, val] = heap.front();
  check_front(heap, std::string("d"), 3, 3.3);

  REQUIRE_FALSE(heap.insert("e", 4, 4.4));
  REQUIRE(heap.dump() == "(d, 3, 3.3) (b, 1, 1.1) (c, 2, 2.2) ");

  REQUIRE(heap.insert("a", 5, 5.5));
  REQUIRE(heap.size() == 3);
  REQUIRE(heap.dump() == "(c, 2, 2.2) (b, 1, 1.1) (a, 5, 5.5) ");
}
