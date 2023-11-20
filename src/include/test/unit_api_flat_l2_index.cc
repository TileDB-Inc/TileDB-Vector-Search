/**
 * @file   unit_api_flat_l2_index.cc
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
#include "api/flat_l2_index.h"
#include "catch2/catch_all.hpp"
#include "query_common.h"

TEST_CASE("api_flat_l2_index: test test", "[api_flat_l2_index]") {
  REQUIRE(true);
}

// ----------------------------------------------------------------------------
// Index tests
// ----------------------------------------------------------------------------

TEST_CASE("api: flat_l2_index", "[api][flat_l2_index]") {
  tiledb::Context ctx;
  auto a = IndexFlatL2(ctx, fmnist_train_uri);
  // auto b = Index(ctx, fmnist_train_uri, IndexKind::FlatL2);
  // auto c = Index(ctx, fmnist_train_uri, IndexKind::IVFFlat);
}

TEST_CASE(
    "api: uri flat_l2_index constructors, context", "[api][flat_l2_index]") {
  tiledb::Context ctx;

  auto a = IndexFlatL2(ctx, db_uri);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == 1'000'000);

  auto b = IndexFlatL2(ctx, bigann1M_base_uri);
  CHECK(b.feature_type() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == 1'000'000);

  auto c = IndexFlatL2(ctx, fmnist_train_uri);
  CHECK(c.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = IndexFlatL2(ctx, sift_base_uri);
  CHECK(d.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == 1'000'000);
}

TEST_CASE(
    "api: uri flat_l2_index constructors, no context", "[api][flat_l2_index]") {
  tiledb::Context ctx;
  auto a = IndexFlatL2(ctx, db_uri);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(a) == 128);
  CHECK(num_vectors(a) == 1'000'000);

  auto b = IndexFlatL2(ctx, bigann1M_base_uri);
  CHECK(b.feature_type() == TILEDB_UINT8);
  CHECK(dimension(b) == 128);
  CHECK(num_vectors(b) == 1'000'000);

  auto c = IndexFlatL2(ctx, fmnist_train_uri);
  CHECK(c.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(c) == 784);
  CHECK(num_vectors(c) == 60'000);

  auto d = IndexFlatL2(ctx, sift_base_uri);
  CHECK(d.feature_type() == TILEDB_FLOAT32);
  CHECK(dimension(d) == 128);
  CHECK(num_vectors(d) == 1'000'000);
}

TEST_CASE("api: queries", "[api][flat_l2_index]") {
  tiledb::Context ctx;
  size_t k_nn = 10;
  size_t nthreads = 8;
  size_t num_queries = 50;
  auto sift_test_tuple = std::make_tuple(
      db_uri, groundtruth_uri, query_uri, TILEDB_FLOAT32, 128, 1'000'000);

  auto bigann1M_tuple = std::make_tuple(
      bigann1M_base_uri,
      bigann1M_groundtruth_uri,
      bigann1M_query_uri,
      TILEDB_UINT8,
      128,
      1'000'000);

  auto fmnist_tuple = std::make_tuple(
      fmnist_train_uri,
      fmnist_groundtruth_uri,
      fmnist_test_uri,
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
      auto a = IndexFlatL2(ctx, uri);

      CHECK(a.feature_type() == dtype);
      CHECK(dimension(a) == dim);
      CHECK(num_vectors(a) == numv);

      auto aq = QueryVectorArray(ctx, q_uri, num_queries);
      load(aq);

      auto [aq_scores, aq_top_k] = a.query(aq, k_nn);
      CHECK(num_vectors(aq_top_k) == num_queries);
      CHECK(dimension(aq_top_k) == k_nn);
      CHECK(num_vectors(aq_scores) == num_queries);
      CHECK(dimension(aq_scores) == k_nn);

      auto hk = tdbColMajorMatrix<groundtruth_type>(ctx, gt_uri);
      load(hk);

      auto ok = validate_top_k(aq_top_k, FeatureVectorArray{std::move(hk)});
      CHECK(ok);
    }
  }
}
