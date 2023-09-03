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
#include <filesystem>
#include "api.h"
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/tdb_vector.h"
#include "detail/linalg/vector.h"
#include "query_common.h"
#include "test_utils.h"
#include "utils/utils.h"

bool global_debug = false;

TEST_CASE("api: test test", "[api]") {
  REQUIRE(true);
}

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

  CHECK(dimension(std::vector<int>{1, 2, 3}) == 3);
  CHECK(dimension(Vector<int>{1, 2, 3}) == 3);
  CHECK(dimension(FeatureVector(std::vector<int>{1, 2, 3})) == 3);
  CHECK(dimension(FeatureVector(Vector<int>{1, 2, 3})) == 3);
}

using TestTypes = std::tuple<float, uint8_t, int32_t, uint32_t, uint64_t>;

int counter = 0;
TEMPLATE_LIST_TEST_CASE("api: FeatureVector read", "[api]", TestTypes) {
  size_t N = GENERATE(1UL, 2UL, 8191UL, 8192UL, 8193UL);

  std::vector<TestType> v(N);
  randomize(v, {0, 128});
  auto w{v};
  auto ctx = tiledb::Context{};
  auto vname = "/tmp/test_vector_" + std::to_string(counter++);
  if (local_array_exists(vname)) {
    std::filesystem::remove_all(vname);
  }
  write_vector(ctx, v, vname);
  auto x = read_vector<TestType>(ctx, vname);

  CHECK(dimension(x) == N);
  CHECK(std::equal(begin(v), end(v), begin(x)));

  auto y = FeatureVector{std::move(v)};
  CHECK(dimension(y) == N);

  auto by = (TestType*)y.data();
  auto ey = by + dimension(y);
  CHECK(std::equal(by, ey, begin(x)));

  auto z = FeatureVector{ctx, vname};
  CHECK(dimension(z) == N);

  auto bz = (TestType*)y.data();
  auto ez = by + dimension(y);
  CHECK(std::equal(bz, ez, begin(x)));
}

// ----------------------------------------------------------------------------
// FeatureVectorArray tests
// ----------------------------------------------------------------------------

TEST_CASE("api: feature vector array open", "[api]") {
  tiledb::Context ctx;

  auto a = FeatureVectorArray(ctx, db_uri);
  CHECK(a.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == 1'000'000);

  auto b = FeatureVectorArray(ctx, bigann1M_base_uri);
  CHECK(b.datatype() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == 1'000'000);

  auto c = FeatureVectorArray(ctx, fmnist_train_uri);
  CHECK(c.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = FeatureVectorArray(ctx, sift_base_uri);
  CHECK(d.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == 1'000'000);
}

// ----------------------------------------------------------------------------
// Index tests
// ----------------------------------------------------------------------------
TEST_CASE("api: uri index constructors, context","[api][index]"){
  tiledb::Context ctx;

  auto a = Index(ctx, db_uri);
  CHECK(a.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == 1'000'000);

  auto b = Index(ctx, bigann1M_base_uri);
  CHECK(b.datatype() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == 1'000'000);

  auto c = Index(ctx, fmnist_train_uri);
  CHECK(c.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = Index(ctx, sift_base_uri);
  CHECK(d.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == 1'000'000);
}

TEST_CASE("api: uri index constructors, no context","[api][index]"){
  auto a = Index(db_uri);
  CHECK(a.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == 1'000'000);

  auto b = Index(bigann1M_base_uri);
  CHECK(b.datatype() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == 1'000'000);

  auto c = Index(fmnist_train_uri);
  CHECK(c.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = Index(sift_base_uri);
  CHECK(d.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == 1'000'000);
}

TEST_CASE("api: uri query","[api][index]") {
  tiledb::Context ctx;
  size_t k_nn = 10;
  size_t nthreads = 5;
  size_t num_queries = 100;

  auto a = Index(ctx, db_uri);
  CHECK(a.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == 1'000'000);
  auto aq = QueryVectorArray(ctx, query_uri, num_queries);
  load(aq);
  auto [aq_scores, aq_top_k] = a.query(aq, k_nn);
  auto ag = QueryVectorArray(ctx, groundtruth_uri);
  load(ag);

  auto ck = tdbColMajorMatrix<float>(ctx, db_uri);
  load(ck);
  auto qk = tdbColMajorMatrix<float>(ctx, query_uri, num_queries);
  load(qk);
  auto gk = tdbColMajorMatrix<groundtruth_type>(ctx, groundtruth_uri);
  load(gk);


  auto [ck_scores, ck_top_k] = detail::flat::qv_query_heap(ck, qk, k_nn, nthreads);
  CHECK(validate_top_k(ck_top_k, gk));
  CHECK(validate_top_k(ag_top_k, gk));

  auto b = Index(ctx, bigann1M_base_uri);
  CHECK(b.datatype() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == 1'000'000);

  auto c = Index(ctx, fmnist_train_uri);
  CHECK(c.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = Index(ctx, sift_base_uri);
  CHECK(d.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == 1'000'000);
}