/**
 * @file   unit_api_feature_vector_array.cc
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

#include "api/feature_vector_array.h"
#include "array_defs.h"
#include "catch2/catch_all.hpp"
#include "detail/ivf/qv.h"
#include "query_common.h"

TEST_CASE("api_feature_vector_array: test test", "[api_feature_vector_array]") {
  REQUIRE(true);
}

// ----------------------------------------------------------------------------
// FeatureVectorArray tests
// ----------------------------------------------------------------------------

TEST_CASE("api: feature vector array open", "[api]") {
  tiledb::Context ctx;

  auto a = FeatureVectorArray(ctx, sift_inputs_uri);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = FeatureVectorArray(ctx, bigann1M_inputs_uri);
  CHECK(b.feature_type() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == num_bigann1M_vectors);

  auto c = FeatureVectorArray(ctx, fmnist_inputs_uri);
  CHECK(c.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == num_fmnist_vectors);

  auto d = FeatureVectorArray(ctx, sift_inputs_uri);
  CHECK(d.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == num_sift_vectors);
}

template <query_vector_array M>
auto _ack(const M& m) {
}

auto ack() {
  _ack(MatrixView<float>{});
}

TEST_CASE("api: Matrix constructors and destructors", "[api]") {
  auto a = ColMajorMatrix<int>(3, 7);
  auto b = FeatureVectorArray(a);

  auto c = ColMajorMatrix<int>(3, 7);
  auto d = FeatureVectorArray(std::move(c));
}

TEMPLATE_TEST_CASE(
    "api: FeatureVectorArray feature_type",
    "[api]",
    int,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  auto a = ColMajorMatrix<TestType>{3, 17};
  auto b = FeatureVectorArray(a);

  CHECK(b.feature_type() == t);
  CHECK(b.feature_size() == sizeof(TestType));

  auto c = FeatureVectorArray{ColMajorMatrix<TestType>{17, 3}};
  CHECK(c.feature_type() == t);
  CHECK(c.feature_size() == sizeof(TestType));

  auto f = ColMajorMatrix<TestType>{3, 17};
  auto d = FeatureVectorArray{std::move(f)};
  CHECK(d.feature_type() == t);
  CHECK(d.feature_size() == sizeof(TestType));

  auto e = FeatureVectorArray{std::move(ColMajorMatrix<TestType>{3, 9})};
  CHECK(e.feature_type() == t);
  CHECK(e.feature_size() == sizeof(TestType));

  auto g = std::move(e);
  CHECK(g.feature_type() == t);
  CHECK(g.feature_size() == sizeof(TestType));
}

TEST_CASE("api: tdbMatrix constructors and destructors", "[api]") {
  tiledb::Context ctx;
  auto c = ColMajorMatrix<int>(3, 7);

  std::filesystem::remove_all("/tmp/a");
  write_matrix(ctx, c, "/tmp/a");

  auto a = tdbColMajorMatrix<int>(ctx, "/tmp/a");
  a.load();
  auto b = FeatureVectorArray(a);

  auto d = tdbColMajorMatrix<int>(ctx, "/tmp/a");
  d.load();
  auto e = FeatureVectorArray(std::move(d));
}

#if 0  // This fails with 2.16.0
TEST_CASE("api: Arrays going out of scope", "[api]") {
  auto ctx = tiledb::Context{};
  auto foo = tiledb::Array(ctx, "/tmp/a", TILEDB_READ);
  auto bar = std::move(foo);
}
#endif

TEMPLATE_TEST_CASE(
    "api: tdb FeatureVectorArray feature_type",
    "[api]",
    int,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  tiledb::Context ctx;
  auto uri = std::string{"/tmp/a"};

  auto cc = ColMajorMatrix<TestType>(3, 7);

  std::filesystem::remove_all(uri);
  write_matrix(ctx, cc, uri);

  {
    auto a = tdbColMajorMatrix<TestType>{ctx, uri};
    auto b = FeatureVectorArray(a);
    CHECK(b.feature_type() == t);
  }

  {
    auto c = FeatureVectorArray(tdbColMajorMatrix<TestType>{ctx, uri});
    CHECK(c.feature_type() == t);
  }

  {
    auto f = tdbColMajorMatrix<TestType>{ctx, uri};
    auto d = FeatureVectorArray{std::move(f)};
    CHECK(d.feature_type() == t);
  }

  {
    auto e =
        FeatureVectorArray{std::move(tdbColMajorMatrix<TestType>{ctx, uri})};
    CHECK(e.feature_type() == t);

    auto g = std::move(e);
    CHECK(g.feature_type() == t);
  }
}

TEST_CASE("api: query checks", "[api][index]") {
  tiledb::Context ctx;
  size_t k_nn = 10;
  size_t nthreads = 8;
  size_t num_queries = 50;

  SECTION("simple check") {
    auto z = FeatureVectorArray(ctx, sift_inputs_uri);
    auto nn = dimension(z);
    auto nnn = num_vectors(z);
    CHECK(dimension(z) == 128);
    CHECK(num_vectors(z) == num_sift_vectors);
  }

  SECTION("tdbMatrix") {
    auto ck = tdbColMajorMatrix<float>(ctx, sift_inputs_uri);
    ck.load();

    auto qk = tdbColMajorMatrix<float>(ctx, sift_query_uri, num_queries);
    load(qk);

    auto [ck_scores, ck_top_k] =
        detail::flat::qv_query_heap(ck, qk, k_nn, nthreads);

    auto gk =
        tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
    load(gk);

    auto ok = validate_top_k(ck_top_k, gk);
    CHECK(ok);
  }
}
