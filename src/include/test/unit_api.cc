/**
 * @file   unit_api.cc
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
#include "tdb_defs.h"

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

  auto a = FeatureVectorArray(ctx, sift_inputs_uri);
  CHECK(a.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = FeatureVectorArray(ctx, bigann1M_inputs_uri);
  CHECK(b.datatype() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == num_bigann1M_vectors);

  auto c = FeatureVectorArray(ctx, fmnist_inputs_uri);
  CHECK(c.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = FeatureVectorArray(ctx, sift_inputs_uri);
  CHECK(d.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == num_sift_vectors);
}

// ----------------------------------------------------------------------------
// Index tests
// ----------------------------------------------------------------------------
TEST_CASE("api: uri index constructors, context", "[api][index]") {
  tiledb::Context ctx;

  auto a = Index(ctx, sift_inputs_uri);
  CHECK(a.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = Index(ctx, bigann1M_inputs_uri);
  CHECK(b.datatype() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == num_bigann1M_vectors);

  auto c = Index(ctx, fmnist_inputs_uri);
  CHECK(c.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = Index(ctx, sift_inputs_uri);
  CHECK(d.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == num_sift_vectors);
}

TEST_CASE("api: uri index constructors, no context", "[api][index]") {
  auto a = Index(sift_inputs_uri);
  CHECK(a.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = Index(bigann1M_inputs_uri);
  CHECK(b.datatype() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == num_bigann1M_vectors);

  auto c = Index(fmnist_inputs_uri);
  CHECK(c.datatype() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = Index(sift_inputs_uri);
  CHECK(d.datatype() == TILEDB_FLOAT32);
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
    "api: FeatureVector datatype",
    "[api]",
    int,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  auto a = std::vector<TestType>{1, 2, 3};
  auto b = FeatureVector(a);
  CHECK(b.datatype() == t);

  auto c = FeatureVector{std::vector<TestType>{1, 2, 3}};
  CHECK(c.datatype() == t);

  auto f = std::vector<TestType>{1, 2, 3};
  auto d = FeatureVector{std::move(f)};
  CHECK(d.datatype() == t);

  auto e = FeatureVector{std::move(std::vector<TestType>{1, 2, 3})};
  CHECK(e.datatype() == t);

  auto g = std::move(e);
  CHECK(g.datatype() == t);
}

TEMPLATE_TEST_CASE(
    "api: FeatureVectorArray datatype",
    "[api]",
    int,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  auto a = ColMajorMatrix<TestType>{3, 17};
  auto b = FeatureVectorArray(a);

  CHECK(b.datatype() == t);

  auto c = FeatureVectorArray{ColMajorMatrix<TestType>{17, 3}};
  CHECK(c.datatype() == t);

  auto f = ColMajorMatrix<TestType>{3, 17};
  auto d = FeatureVectorArray{std::move(f)};
  CHECK(d.datatype() == t);

  auto e = FeatureVectorArray{std::move(ColMajorMatrix<TestType>{3, 9})};
  CHECK(e.datatype() == t);

  auto g = std::move(e);
  CHECK(g.datatype() == t);
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

TEMPLATE_TEST_CASE(
    "api: tdb FeatureVectorArray datatype",
    "[api]",
    int,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  tiledb::Context ctx;
  auto cc = ColMajorMatrix<TestType>(3, 7);

  auto uri = std::string{"/tmp/a"};
  std::filesystem::remove_all(uri);
  write_matrix(ctx, cc, uri);

  auto a = tdbColMajorMatrix<TestType>{ctx, uri};
  auto b = FeatureVectorArray(a);

  CHECK(b.datatype() == t);

  auto c = FeatureVectorArray{tdbColMajorMatrix<TestType>{ctx, uri}};
  CHECK(c.datatype() == t);

  auto f = tdbColMajorMatrix<TestType>{ctx, uri};
  auto d = FeatureVectorArray{std::move(f)};
  CHECK(d.datatype() == t);

  auto e = FeatureVectorArray{std::move(tdbColMajorMatrix<TestType>{ctx, uri})};
  CHECK(e.datatype() == t);

  auto g = std::move(e);
  CHECK(g.datatype() == t);
}

TEST_CASE("api: types", "[types]") {
  CHECK(tiledb::impl::type_to_tiledb<float>::name == std::string("FLOAT32"));
  CHECK(tiledb::impl::type_to_tiledb<int>::name == std::string("INT32"));
  CHECK(tiledb::impl::type_to_tiledb<unsigned>::name == std::string("UINT32"));
  CHECK(tiledb::impl::type_to_tiledb<int64_t>::name == std::string("INT64"));
  CHECK(tiledb::impl::type_to_tiledb<uint64_t>::name == std::string("UINT64"));

  CHECK(tiledb::impl::type_to_tiledb<long>::name != std::string("INT64"));
  CHECK(
      tiledb::impl::type_to_tiledb<unsigned long>::name !=
      std::string("UINT64"));
  CHECK(tiledb::impl::type_to_tiledb<size_t>::name != std::string("UINT64"));
}

template <_load::_member_load T>
void _yack(T&& t) {
  auto x = load(t);
}

void yack() {
  tiledb::Context ctx;
  //  _yack(tdbColMajorMatrix<float>{ctx, "17"});

  auto f = FeatureVectorArray(tdbColMajorMatrix<float>{ctx, "17"});
  //_yack(f);

  //  auto k = tdbColMajorMatrix<float>{ctx, "17"};
  //  _yack(k);

  //    auto g = FeatureVectorArray(k);
  //    _yack(g);

  //    _yack(std::move(g));
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

  // Groundtruth types aren't the same for sift vs bigann (at least as we have
  // generated them so far)
#if 0
  SECTION("tdbMatrix") {
    auto ck = tdbColMajorMatrix<float>(ctx, sift_inputs_uri);

    auto qk = tdbColMajorMatrix<float>(ctx, sift_query_uri, num_queries);
    load(qk);

    auto [ck_scores, ck_top_k] =
        detail::flat::qv_query_heap(ck, qk, k_nn, nthreads);

    auto gk = tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
    load(gk);

    auto ok = validate_top_k(ck_top_k, gk);
    CHECK(ok);
  }
#endif
}

TEST_CASE("api: queries", "[api][index]") {
  tiledb::Context ctx;
  size_t k_nn = 10;
  size_t nthreads = 8;
  size_t num_queries = 50;
  auto sift_test_tuple = std::make_tuple(
      sift_inputs_uri,
      sift_groundtruth_uri,
      sift_query_uri, TILEDB_FLOAT32, 128, num_sift_vectors);

  auto bigann1M_tuple = std::make_tuple(
      bigann1M_inputs_uri,
      bigann1M_groundtruth_uri,
      bigann1M_query_uri,
      TILEDB_UINT8,
      128,
      num_bigann1M_vectors);

  auto fmnist_tuple = std::make_tuple(
      fmnist_inputs_uri,
      fmnist_groundtruth_uri,
      fmnist_query_uri,
      TILEDB_FLOAT32,
      784,
      60'000);

  std::vector<std::tuple<
      std::string,
      std::string,
      std::string,
      tiledb_datatype_t,
      size_t,
      size_t>>
      tuples{sift_test_tuple, bigann1M_tuple, fmnist_tuple};

  SECTION("FeatureVectorArray - queries") {
    for (auto&& t : tuples) {
      auto [uri, gt_uri, q_uri, dtype, dim, numv] = t;

      std::cout << uri << std::endl;

      auto a = Index(ctx, uri);

      CHECK(a.datatype() == dtype);
      CHECK(dimension(a) == dim);
      CHECK(num_vectors(a) == numv);

      auto aq = QueryVectorArray(ctx, q_uri, num_queries);
      load(aq);

      auto [aq_scores, aq_top_k] = a.query(aq, k_nn);
      CHECK(num_vectors(aq_top_k) == num_queries);
      CHECK(dimension(aq_top_k) == k_nn);
      CHECK(num_vectors(aq_scores) == num_queries);
      CHECK(dimension(aq_scores) == k_nn);

#if 0
auto hk = tdbColMajorMatrix<test_groundtruth_type>(ctx, gt_uri);
load(hk);

auto ok = validate_top_k(aq_top_k, FeatureVectorArray{std::move(hk)});
CHECK(ok);
#else
      auto hk = FeatureVectorArray(ctx, gt_uri);

      auto ok = validate_top_k(aq_top_k, hk);
      CHECK(ok);
#endif
    }
  }
}
