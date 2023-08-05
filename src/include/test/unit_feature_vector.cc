/**
 * @file   unit_feature_vector.cc
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
 * Tests for FeatureVector clas.
 *
 */

#include <catch2/catch_all.hpp>
#include <vector>
#include "api.h"
#include "feature_vector.h"

TEST_CASE("FeatureVector: test test", "[FeatureVector]") {
  CHECK(true);
}

TEST_CASE("FeatureVector: feature_vector", "[FeatureVector]") {
  CHECK(feature_vector<FeatureVector<int>>);
  CHECK(feature_vector<FeatureVector<double>>);
  CHECK(feature_vector<FeatureVector<std::vector<char>>>);
  CHECK(!feature_vector<std::vector<int>>);
  CHECK(!feature_vector<int>);
}

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

TEMPLATE_LIST_TEST_CASE(
    "FeatureVector: test FeatureVector constructor",
    "[linalg][FeatureVector][create]",
    TestTypes) {
  auto a = FeatureVector<TestType>(7);
  std::iota(begin(a), end(a), 1);

  SECTION("dims") {
    CHECK(dimension(a) == 7);
  }
  SECTION("values") {
    for (size_t i = 0; i < dimension(a); ++i) {
      CHECK((size_t)a(i) == i + 1);
    }
    for (size_t i = 0; i < dimension(a); ++i) {
      a(i) *= a[i];
    }
    for (size_t i = 0; i < dimension(a); ++i) {
      CHECK((size_t)a(i) == (i + 1) * (i + 1));
    }
  }
  SECTION("values, copy") {
    auto b = std::move(a);
    for (size_t i = 0; i < dimension(a); ++i) {
      CHECK((size_t)b(i) == i + 1);
    }
    for (size_t i = 0; i < dimension(a); ++i) {
      b(i) *= b[i];
    }
    for (size_t i = 0; i < dimension(a); ++i) {
      CHECK((size_t)b(i) == (i + 1) * (i + 1));
    }
  }
}

TEST_CASE("FeatureVector: test constructor", "[FeatureVector]") {
  FeatureVector<int> v(10);
  CHECK(dimension(v) == 10);
  CHECK(data(v) != nullptr);
  CHECK(v.dimension() == 10);
  CHECK(v.data() != nullptr);
}

TEST_CASE("FeatureVector: test move constructor", "[FeatureVector]") {
  FeatureVector<int> v(10);
  auto p = v.data();
  FeatureVector<int> w(std::move(v));
  CHECK(dimension(w) == 10);
  CHECK(data(w) == p);
  CHECK(w.dimension() == 10);
  CHECK(w.data() == p);
}

TEST_CASE("FeatureVector: test operator[]", "[FeatureVector]") {
  FeatureVector<int> v(10);
  for (int i = 0; i < 10; ++i) {
    v[i] = i;
  }
  for (int i = 0; i < 10; ++i) {
    CHECK(v[i] == i);
  }
}

TEST_CASE("FeatureVector: test operator()", "[FeatureVector]") {
  FeatureVector<int> v(10);
  for (int i = 0; i < 10; ++i) {
    v(i) = i;
  }
  for (int i = 0; i < 10; ++i) {
    CHECK(v(i) == i);
  }
}

TEST_CASE("FeatureVector: test initializer_list", "[FeatureVector]") {
  std::vector<int> u{8, 6, 7, 5, 3, 0, 9};
  FeatureVector<int> v{8, 6, 7, 5, 3, 0, 9};

  CHECK(dimension(v) == 7);
  CHECK(std::equal(data(v), data(v) + 7, begin(u)));
  CHECK(std::equal(begin(v), end(v), begin(u)));
}

TEST_CASE("FeatureVector: test move constructor too", "[FeatureVector]") {
  std::vector<int> u{8, 6, 7, 5, 3, 0, 9};
  FeatureVector<int> v{8, 6, 7, 5, 3, 0, 9};
  FeatureVector<int> w{std::move(v)};

  CHECK(data(v) == nullptr);
  CHECK(dimension(v) == 0);
  CHECK(dimension(w) == 7);
  CHECK(std::equal(begin(w), end(w), begin(u)));
}
