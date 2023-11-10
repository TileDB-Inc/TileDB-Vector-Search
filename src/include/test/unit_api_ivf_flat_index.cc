/**
 * @file   unit_api_ivf_flat_index.cc
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

#include "api/ivf_flat_index.h"
#include "catch2/catch_all.hpp"
#include "test/query_common.h"

TEST_CASE("api_ivf_flat_index: test test", "[api_ivf_flat_index]") {
  REQUIRE(true);
}

TEST_CASE("api_ivf_flat_index: init constructor", "[api_ivf_flat_index]") {
  SECTION("default") {
    auto a = IndexIVFFlat();
    CHECK(a.feature_type() == TILEDB_ANY);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.px_type() == TILEDB_UINT32);
    CHECK(dimension(a) == 0);
    CHECK(num_partitions(a) == 0);
  }

  SECTION("float uint32 uint32") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint32"},
         {"px_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.px_type() == TILEDB_UINT32);
    CHECK(dimension(a) == 0);
    CHECK(num_partitions(a) == 0);
  }

  SECTION("uint8 uint32 uint32") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint32"},
         {"px_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.px_type() == TILEDB_UINT32);
  }

  SECTION("float uint64 uint32") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint64"},
         {"px_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.px_type() == TILEDB_UINT32);
  }

  SECTION("float uint32 uint64") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint32"},
         {"px_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.px_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint64 uint32") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint64"},
         {"px_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.px_type() == TILEDB_UINT32);
  }

  SECTION("uint8 uint32 uint64") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint32"},
         {"px_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.px_type() == TILEDB_UINT64);
  }

  SECTION("float uint64 uint64") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint64"},
         {"px_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.px_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint64 uint64") {
    auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint64"},
         {"px_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.px_type() == TILEDB_UINT64);
  }
}

TEST_CASE("api_ivf_flat_index: api_ivf_flat_index write and read", "[api_ivf_flat_index]") {
  auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
      {{"feature_type", "float32"},
       {"id_type", "uint32"},
       {"px_type", "uint32"}}));
  auto ctx = tiledb::Context{};
  auto training_set = FeatureVectorArray(ctx, siftsmall_base_uri);

 // a.train(training_set, kmeans_init::random);
}
