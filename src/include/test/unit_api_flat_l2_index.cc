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
#include "test/utils/query_common.h"

// ----------------------------------------------------------------------------
// Index tests
// ----------------------------------------------------------------------------

TEST_CASE("api: flat_l2_index", "[api][flat_l2_index]") {
  tiledb::Context ctx;
  auto a = IndexFlatL2(ctx, fmnist_inputs_uri);
  // auto b = Index(ctx, fmnist_inputs_uri, IndexKind::FlatL2);
  // auto c = Index(ctx, fmnist_inputs_uri, IndexKind::IVFFlat);
}

TEST_CASE(
    "api: uri flat_l2_index constructors, context", "[api][flat_l2_index]") {
  tiledb::Context ctx;

  auto a = IndexFlatL2(ctx, sift_inputs_uri);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(a) == sift_dimensions);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = IndexFlatL2(ctx, bigann1M_inputs_uri);
  CHECK(b.feature_type() == TILEDB_UINT8);
  CHECK(dimensions(b) == bigann1M_dimension);
  CHECK(num_vectors(b) == num_bigann1M_vectors);

  auto c = IndexFlatL2(ctx, fmnist_inputs_uri);
  CHECK(c.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(c) == fmnist_dimension);
  CHECK(num_vectors(c) == num_fmnist_vectors);

  auto d = IndexFlatL2(ctx, sift_inputs_uri);
  CHECK(d.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(d) == sift_dimensions);
  CHECK(num_vectors(d) == num_sift_vectors);

  auto e = IndexFlatL2(ctx, siftsmall_inputs_uri);
  CHECK(d.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(d) == siftsmall_dimensions);
  CHECK(num_vectors(d) == num_siftsmall_vectors);
}

TEST_CASE(
    "api: uri flat_l2_index constructors, no context", "[api][flat_l2_index]") {
  tiledb::Context ctx;
  auto a = IndexFlatL2(ctx, sift_inputs_uri);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(a) == sift_dimensions);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = IndexFlatL2(ctx, bigann1M_inputs_uri);
  CHECK(b.feature_type() == TILEDB_UINT8);
  CHECK(dimensions(b) == bigann1M_dimension);
  CHECK(num_vectors(b) == num_bigann1M_vectors);

  auto c = IndexFlatL2(ctx, fmnist_inputs_uri);
  CHECK(c.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(c) == fmnist_dimension);
  CHECK(num_vectors(c) == num_fmnist_vectors);

  auto d = IndexFlatL2(ctx, sift_inputs_uri);
  CHECK(d.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(d) == sift_dimensions);
  CHECK(num_vectors(d) == num_sift_vectors);

  auto e = IndexFlatL2(ctx, siftsmall_inputs_uri);
  CHECK(e.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(e) == siftsmall_dimensions);
  CHECK(num_vectors(e) == num_siftsmall_vectors);
}

TEST_CASE("api: queries", "[api][flat_l2_index]") {
  tiledb::Context ctx;
  size_t k_nn = 10;
  size_t nthreads = 8;
  size_t num_queries = 50;
  auto sift_test_tuple = std::make_tuple(
      sift_inputs_uri,
      sift_groundtruth_uri,
      sift_query_uri,
      TILEDB_FLOAT32,
      sift_dimensions,
      num_sift_vectors);

  auto bigann1M_tuple = std::make_tuple(
      bigann1M_inputs_uri,
      bigann1M_groundtruth_uri,
      bigann1M_query_uri,
      TILEDB_UINT8,
      bigann1M_dimension,
      num_bigann1M_vectors);

  auto fmnist_tuple = std::make_tuple(
      fmnist_inputs_uri,
      fmnist_groundtruth_uri,
      fmnist_query_uri,
      TILEDB_FLOAT32,
      fmnist_dimension,
      num_fmnist_vectors);

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
      CHECK(dimensions(a) == dim);
      CHECK(num_vectors(a) == numv);

      auto aq = QueryVectorArray(ctx, q_uri, "", num_queries);
      load(aq);

      auto [aq_scores, aq_top_k] = a.query(aq, k_nn);
      CHECK(num_vectors(aq_top_k) == num_queries);
      CHECK(dimensions(aq_top_k) == k_nn);
      CHECK(num_vectors(aq_scores) == num_queries);
      CHECK(dimensions(aq_scores) == k_nn);

      auto hk = tdbColMajorMatrix<test_groundtruth_type>(ctx, gt_uri);
      load(hk);

      auto ok = validate_top_k(aq_top_k, FeatureVectorArray{std::move(hk)});
      CHECK(ok);
    }
  }
}
