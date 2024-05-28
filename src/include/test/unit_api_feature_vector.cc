/**
 * @file   unit_api_feature_vector.cc
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

#include "api/feature_vector.h"
#include "catch2/catch_all.hpp"
#include "test/utils/randomize.h"
#include "test/utils/test_utils.h"
#include "utils/utils.h"

// ----------------------------------------------------------------------------
// FeatureVector tests
// ----------------------------------------------------------------------------
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
  auto w = Vector<int>{1, 2, 3};
  auto t = std::vector<int>{1, 2, 3};
  auto u = Vector<int>{1, 2, 3};
  auto x = FeatureVector{std::move(t)};
  auto y = FeatureVector{std::move(u)};

  SECTION("v, w, x, y") {
    CHECK(dimensions(v) == 3);
    CHECK(dimensions(w) == 3);
    CHECK(dimensions(x) == 3);
    CHECK(dimensions(y) == 3);
  }

  SECTION("move(v), move(w), move(x), move(y)") {
    CHECK(dimensions(FeatureVector(std::move(v))) == 3);
    CHECK(dimensions(FeatureVector(std::move(w))) == 3);
    CHECK(dimensions(FeatureVector(std::move(x))) == 3);
    CHECK(dimensions(FeatureVector(std::move(y))) == 3);
  }

  CHECK(dimensions(std::vector<int>{1, 2, 3}) == 3);
  CHECK(dimensions(Vector<int>{1, 2, 3}) == 3);
  CHECK(dimensions(FeatureVector(std::vector<int>{1, 2, 3})) == 3);
  CHECK(dimensions(FeatureVector(Vector<int>{1, 2, 3})) == 3);
}

using TestTypes = std::tuple<float, uint8_t, int32_t, uint32_t, uint64_t>;

int api_counter = 0;
TEMPLATE_LIST_TEST_CASE("api: FeatureVector read", "[api]", TestTypes) {
  size_t N = GENERATE(1UL, 2UL, 8191UL, 8192UL, 8193UL);

  std::vector<TestType> v(N);
  randomize(v, {0, 128});
  auto w{v};
  auto ctx = tiledb::Context{};
  auto vname = (std::filesystem::temp_directory_path() /
                ("test_vector_" + std::to_string(api_counter++)))
                   .string();
  if (local_array_exists(vname)) {
    std::filesystem::remove_all(vname);
  }
  write_vector(ctx, v, vname);
  auto x = read_vector<TestType>(ctx, vname);

  CHECK(dimensions(x) == N);
  CHECK(std::equal(begin(v), end(v), begin(x)));

  auto y = FeatureVector{std::move(v)};
  CHECK(dimensions(y) == N);

  auto by = (TestType*)y.data();
  auto ey = by + dimensions(y);
  CHECK(std::equal(by, ey, begin(x)));

  auto z = FeatureVector{ctx, vname};
  CHECK(dimensions(z) == N);

  auto bz = (TestType*)y.data();
  auto ez = by + dimensions(y);
  CHECK(std::equal(bz, ez, begin(x)));
}

TEMPLATE_TEST_CASE(
    "api: FeatureVector feature_type",
    "[api]",
    int,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  auto a = std::vector<TestType>{1, 2, 3};
  auto b = FeatureVector(a);
  CHECK(b.feature_type() == t);

  auto c = FeatureVector{std::vector<TestType>{1, 2, 3}};
  CHECK(c.feature_type() == t);

  auto f = std::vector<TestType>{1, 2, 3};
  auto d = FeatureVector{std::move(f)};
  CHECK(d.feature_type() == t);

  auto e = FeatureVector{std::move(std::vector<TestType>{1, 2, 3})};
  CHECK(e.feature_type() == t);

  auto g = std::move(e);
  CHECK(g.feature_type() == t);

  auto h = FeatureVector{FeatureVector(std::vector<TestType>{1, 2, 3})};
  CHECK(h.feature_type() == t);

  auto i = FeatureVector{FeatureVector(std::vector<TestType>{1, 2, 3})};
  CHECK(i.feature_type() == t);
}
