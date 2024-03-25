/**
 * @file   unit_api_vamana_index.cc
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

#include "api/vamana_index.h"
#include "catch2/catch_all.hpp"
#include "test/query_common.h"

TEST_CASE("api_vamana_index: test test", "[api_vamana_index]") {
  REQUIRE(true);
}

TEST_CASE("api_vamana_index: init constructor", "[api_vamana_index]") {
  SECTION("default") {
    auto a = IndexVamana();
    CHECK(a.feature_type() == TILEDB_ANY);
    CHECK(a.feature_type_string() == datatype_to_string(TILEDB_ANY));
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.id_type_string() == datatype_to_string(TILEDB_UINT32));
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
    CHECK(
        a.adjacency_row_index_type_string() ==
        datatype_to_string(TILEDB_UINT32));
    CHECK(dimension(a) == 0);
  }

  SECTION("float uint32 uint32") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint32"},
         {"adjacency_row_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
    CHECK(dimension(a) == 0);
  }

  SECTION("uint8 uint32 uint32") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint32"},
         {"adjacency_row_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
  }

  SECTION("float uint64 uint32") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint64"},
         {"adjacency_row_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
  }

  SECTION("float uint32 uint64") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint32"},
         {"adjacency_row_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint64 uint32") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint64"},
         {"adjacency_row_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
  }

  SECTION("uint8 uint32 uint64") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint32"},
         {"adjacency_row_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
  }

  SECTION("float uint64 uint64") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint64"},
         {"adjacency_row_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint64 uint64") {
    auto a = IndexVamana(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint64"},
         {"adjacency_row_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
  }
}

TEST_CASE("api_vamana_index: infer feature type", "[api_vamana_index]") {
  auto a = IndexVamana(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"adjacency_row_index_type", "uint32"}}));
  auto ctx = tiledb::Context{};
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(a.id_type() == TILEDB_UINT32);
  CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
}

TEST_CASE("api_vamana_index: infer dimension", "[api_vamana_index]") {
  auto a = IndexVamana(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"adjacency_row_index_type", "uint32"}}));
  auto ctx = tiledb::Context{};
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  CHECK(dimension(a) == 0);
  a.train(training_set);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(a.id_type() == TILEDB_UINT32);
  CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
  CHECK(dimension(a) == 128);
}

TEST_CASE(
    "api_vamana_index: api_vamana_index write and read", "[api_vamana_index]") {
  auto ctx = tiledb::Context{};
  std::string api_vamana_index_uri =
      (std::filesystem::temp_directory_path() / "api_vamana_index").string();

  auto a = IndexVamana(std::make_optional<IndexOptions>(
      {{"feature_type", "float32"},
       {"id_type", "uint32"},
       {"adjacency_row_index_type", "uint32"}}));
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set);
  a.add(training_set);
  a.write_index(ctx, api_vamana_index_uri, true);

  auto b = IndexVamana(ctx, api_vamana_index_uri);

  CHECK(dimension(a) == dimension(b));
  CHECK(a.feature_type() == b.feature_type());
  CHECK(a.id_type() == b.id_type());
  CHECK(a.adjacency_row_index_type() == b.adjacency_row_index_type());
}

TEST_CASE("api_vamana_index: build index and query", "[api_vamana_index]") {
  auto ctx = tiledb::Context{};
  size_t k_nn = 10;
  size_t nprobe = GENERATE(8, 32);

  auto a = IndexVamana(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"adjacency_row_index_type", "uint32"}}));
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);
  a.train(training_set);
  a.add(training_set);

  auto&& [s, t] = a.query(query_set, k_nn);

  auto intersections = count_intersections(t, groundtruth_set, k_nn);
  auto nt = num_vectors(t);
  auto recall = ((double)intersections) / ((double)nt * k_nn);
  CHECK(recall == 1.0);
}

TEST_CASE("api_vamana_index: read index and query", "[api_vamana_index]") {
  auto ctx = tiledb::Context{};
  size_t k_nn = 10;
  size_t nprobe = GENERATE(8, 32);

  std::string api_vamana_index_uri =
      (std::filesystem::temp_directory_path() / "api_vamana_index").string();

  auto a = IndexVamana(std::make_optional<IndexOptions>(
      {{"feature_type", "float32"},
       {"id_type", "uint32"},
       {"adjacency_row_index_type", "uint32"}}));

  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set);
  a.add(training_set);
  a.write_index(ctx, api_vamana_index_uri, true);
  auto b = IndexVamana(ctx, api_vamana_index_uri);

  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);

  auto&& [s, t] = a.query(query_set, k_nn);
  auto&& [u, v] = b.query(query_set, k_nn);

  auto intersections_a = count_intersections(t, groundtruth_set, k_nn);
  auto intersections_b = count_intersections(v, groundtruth_set, k_nn);
  CHECK(intersections_a == intersections_b);
  auto nt = num_vectors(t);
  auto nv = num_vectors(v);
  CHECK(nt == nv);
  auto recall = ((double)intersections_a) / ((double)nt * k_nn);
  CHECK(recall == 1.0);
}
