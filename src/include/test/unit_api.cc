/**
* @file   unit_concepts.cc
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
* Test application of concepts with TileDB-Vector-Search types
*
 */
#include <catch2/catch_all.hpp>
#include "api.h"
#include "detail/linalg/tdb_vector.h"
#include "detail/linalg/vector.h"

TEST_CASE("api: test test", "[api]") {
  REQUIRE(true);
}

TEST_CASE("api: FeatureVector data", "[api]") {
  auto v = std::vector<int>{1, 2, 3};
  auto w = Vector<int>{1, 2, 3};
  auto dv = v.data();
  auto dw = w.data();

  auto t = std::vector<int>{1, 2, 3};
  auto u = Vector<int>{1, 2, 3};
  auto dt = t.data();
  auto du = u.data();

  auto x = FeatureVector{std::move(t)};
  auto y = FeatureVector{std::move(u)};
  auto dx = x.data();
  auto dy = y.data();

  CHECK(dx == dt);
  CHECK(dy == du);

  SECTION("v, w, x, y") {
    CHECK(dv == data(v));
    CHECK(dw == data(w));
    CHECK(dx == data(x));
    CHECK(dy == data(y));
  }

  SECTION("move(v), move(w), move(x), move(y)") {
    CHECK(dv == data(FeatureVector(std::move(v))));
    CHECK(dw == data(FeatureVector(std::move(w))));
    CHECK(dx == data(FeatureVector(std::move(x))));
    CHECK(dy == data(FeatureVector(std::move(y))));
  }

  SECTION("rvalues") {
    CHECK(data(std::vector<int>{1, 2, 3}) != nullptr);
    CHECK(data(Vector<int>{1, 2, 3}) != nullptr);
    CHECK(data(FeatureVector(std::vector<int>{1, 2, 3})) != nullptr);
    CHECK(data(FeatureVector(Vector<int>{1, 2, 3})) != nullptr);
  }
}

TEST_CASE("api: FeatureVector dimension", "[api]") {
  auto v = std::vector<int>{1, 2, 3};
  auto w = Vector<int> {1, 2, 3};
  auto t = std::vector<int>{1, 2, 3};
  auto u = Vector<int> {1, 2, 3};
  auto x = FeatureVector{std::move(t)};
  auto y = FeatureVector{std::move(u)};

  SECTION("v, w, x, y") {
    CHECK(dimension(v) == 3);
    CHECK(dimension(w) == 3);
    CHECK(dimension(x) == 3);
    CHECK(dimension(y) == 3);
  }

  SECTION("move(v), move(w), move(x), move(y)") {
    CHECK(dimension(FeatureVector(std::move(v))) == 3);
    CHECK(dimension(FeatureVector(std::move(w))) == 3);
    CHECK(dimension(FeatureVector(std::move(x))) == 3);
    CHECK(dimension(FeatureVector(std::move(y))) == 3);
  }

  CHECK(dimension(std::vector<int>{1,2,3}) == 3);
  CHECK(dimension(Vector<int>{1,2,3}) == 3);
  CHECK(dimension(FeatureVector(std::vector<int>{1,2,3})) == 3);
  CHECK(dimension(FeatureVector(Vector<int>{1,2,3})) == 3);
}

