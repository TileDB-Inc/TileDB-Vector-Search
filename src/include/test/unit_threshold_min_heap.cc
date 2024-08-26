/**
 * @file unit_threshold_min_heap.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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
#include "utils/threshold_min_heap.h"

TEST_CASE(
    "threshold_min_pair_heap: threshold_min_pair_heap",
    "[fixed_min_heap][threshold_min_pair_heap]") {
  threshold_min_pair_heap<float, int> a(5);

  SECTION("insert in ascending order") {
    for (auto&& i : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert(static_cast<float>(10 - i), i);
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
      a.insert(static_cast<float>(10 + i), i);
    }
    CHECK(a.empty());
  }
}

TEST_CASE(
    "threshold_min_pair_heap: threshold_min_pair_heap with a large vector",
    "[threshold_min_pair_heap]") {
  using element = std::tuple<float, int>;

  float thresh = std::rand();

  threshold_min_pair_heap<float, int> a(thresh);

  std::vector<element> v(5500);
  for (auto&& i : v) {
    i = {std::rand(), std::rand()};
    CHECK(i != element{});
  }
  v[0] = {thresh, 0};
  for (auto&& [e, f] : v) {
    a.insert(e, f);
  }
  CHECK(!a.empty());
  for (auto&& [e, f] : a) {
    CHECK(e < thresh);
  }

  a.resize(size(v));
  std::copy(begin(v), end(v), begin(a));
  a.filtered_heapify();
  CHECK(!a.empty());
  for (auto&& [e, f] : a) {
    CHECK(e < thresh);
  }
}

TEST_CASE(
    "threshold_min_pair_heap: new threshold", "[threshold_min_pair_heap]") {
  using element = std::tuple<float, int>;

  float thresh = static_cast<float>(std::rand());
  threshold_min_pair_heap<float, int> a(thresh);

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

  thresh -= thresh / 2.f;
  a.set_threshold(thresh);
  for (auto&& [e, f] : a) {
    CHECK(e < thresh);
  }
}
