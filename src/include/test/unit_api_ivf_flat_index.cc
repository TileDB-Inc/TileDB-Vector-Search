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
#include "test/utils/query_common.h"

TEST_CASE("init constructor", "[api_ivf_flat_index]") {
  SECTION("default") {
    auto a = IndexIVFFlat();
    CHECK(a.feature_type() == TILEDB_ANY);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.px_type() == TILEDB_UINT32);
    CHECK(dimensions(a) == 0);
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
    CHECK(dimensions(a) == 0);
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

TEST_CASE("infer feature type", "[api_ivf_flat_index]") {
  auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"px_type", "uint32"}}));
  auto ctx = tiledb::Context{};
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set, kmeans_init::random);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(a.id_type() == TILEDB_UINT32);
  CHECK(a.px_type() == TILEDB_UINT32);
}

TEST_CASE("infer dimension", "[api_ivf_flat_index]") {
  auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"px_type", "uint32"}}));
  auto ctx = tiledb::Context{};
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  CHECK(dimensions(a) == 0);
  a.train(training_set, kmeans_init::random);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(a.id_type() == TILEDB_UINT32);
  CHECK(a.px_type() == TILEDB_UINT32);
  CHECK(dimensions(a) == 128);
}

TEST_CASE("api_ivf_flat_index write and read", "[api_ivf_flat_index]") {
  auto ctx = tiledb::Context{};
  std::string api_ivf_flat_index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_flat_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(api_ivf_flat_index_uri)) {
    vfs.remove_dir(api_ivf_flat_index_uri);
  }

  auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
      {{"feature_type", "float32"},
       {"id_type", "uint32"},
       {"px_type", "uint32"}}));
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set, kmeans_init::random);
  a.add(training_set);
  a.write_index(ctx, api_ivf_flat_index_uri);

  auto b = IndexIVFFlat(ctx, api_ivf_flat_index_uri);

  CHECK(dimensions(a) == dimensions(b));
  CHECK(a.feature_type() == b.feature_type());
  CHECK(a.id_type() == b.id_type());
  CHECK(a.px_type() == b.px_type());
}

TEST_CASE("build index and query in place infinite", "[api_ivf_flat_index]") {
  auto ctx = tiledb::Context{};
  size_t k_nn = 10;
  size_t nprobe = GENERATE(8, 32);

  auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"px_type", "uint32"}}));
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);
  a.train(training_set, kmeans_init::random);
  a.add(training_set);

  SECTION("infinite, nprobe = " + std::to_string(nprobe)) {
    INFO("infinite, nprobe = " + std::to_string(nprobe));
    auto&& [s, t] = a.query_infinite_ram(query_set, k_nn, nprobe);

    auto intersections = count_intersections(t, groundtruth_set, k_nn);
    auto nt = num_vectors(t);
    auto recall = ((double)intersections) / ((double)nt * k_nn);
    if (nprobe == 32) {
      CHECK(recall >= 0.999);
    } else if (nprobe == 8) {
      CHECK(recall > 0.925);
    }
  }

// Catch2 isn't catching this exception
#if 0
  SECTION("finite, nprobe = std::to_string(nprobe) " + std::to_string(nprobe) + " throws") {
    INFO("finite, nprobe = std::to_string(nprobe)" + std::to_string(nprobe) + "throws");
    CHECK_THROWS(a.query_finite_ram(query_set, k_nn, nprobe));
  }
#endif
}

TEST_CASE("read index and query infinite and finite", "[api_ivf_flat_index]") {
  auto ctx = tiledb::Context{};
  size_t k_nn = 10;
  size_t nprobe = GENERATE(8, 32);
  size_t max_iter = GENERATE(4, 8);

  std::string api_ivf_flat_index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_flat_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(api_ivf_flat_index_uri)) {
    vfs.remove_dir(api_ivf_flat_index_uri);
  }

  auto a = IndexIVFFlat(std::make_optional<IndexOptions>(
      {{"feature_type", "float32"},
       {"id_type", "uint32"},
       {"px_type", "uint32"},
       {"max_iter", std::to_string(max_iter)}}));

  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set, kmeans_init::random);
  a.add(training_set);
  a.write_index(ctx, api_ivf_flat_index_uri);
  auto b = IndexIVFFlat(ctx, api_ivf_flat_index_uri);

  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);

  SECTION(
      "finite all in core, default, nprobe = std::to_string(nprobe)" +
      std::to_string(nprobe)) {
    INFO(
        "finite all in core, default, nprobe = std::to_string(nprobe)" +
        std::to_string(nprobe));
    auto&& [s, t] = a.query_infinite_ram(query_set, k_nn, nprobe);
    auto&& [u, v] = b.query_finite_ram(query_set, k_nn, nprobe);

    auto intersections_a = count_intersections(t, groundtruth_set, k_nn);
    auto intersections_b = count_intersections(v, groundtruth_set, k_nn);
    CHECK(intersections_a == intersections_b);
    auto nt = num_vectors(t);
    auto nv = num_vectors(v);
    CHECK(nt == nv);
    auto recall = ((double)intersections_a) / ((double)nt * k_nn);
    if (nprobe == 32) {
      CHECK(recall >= 0.998);
    } else if (nprobe == 8) {
      CHECK(recall > 0.925);
    }
  }

  SECTION(
      "finite all in core, 0, nprobe = std::to_string(nprobe)" +
      std::to_string(nprobe)) {
    INFO(
        "finite all in core, 0, nprobe = std::to_string(nprobe)" +
        std::to_string(nprobe));
    auto&& [s, t] = a.query_infinite_ram(query_set, k_nn, nprobe);
    auto&& [u, v] = b.query_finite_ram(query_set, k_nn, nprobe, 0);

    auto intersections_a = count_intersections(t, groundtruth_set, k_nn);
    auto intersections_b = count_intersections(v, groundtruth_set, k_nn);
    CHECK(intersections_a == intersections_b);
    auto nt = num_vectors(t);
    auto nv = num_vectors(v);
    CHECK(nt == nv);
    auto recall = ((double)intersections_a) / ((double)nt * k_nn);
    if (nprobe == 32) {
      CHECK(recall >= 0.998);
    } else if (nprobe == 8) {
      CHECK(recall > 0.925);
    }
  }

  SECTION(
      "finite out of core, 1000, nprobe = std::to_string(nprobe)" +
      std::to_string(nprobe)) {
    INFO(
        "finite out of core, 1000, nprobe = std::to_string(nprobe)" +
        std::to_string(nprobe));
    auto&& [s, t] = a.query_infinite_ram(query_set, k_nn, nprobe);
    auto&& [u, v] = b.query_finite_ram(query_set, k_nn, nprobe, 1000);

    auto intersections_a = count_intersections(t, groundtruth_set, k_nn);
    auto intersections_b = count_intersections(v, groundtruth_set, k_nn);
    CHECK(intersections_a == intersections_b);
    auto nt = num_vectors(t);
    auto nv = num_vectors(v);
    CHECK(nt == nv);
    auto recall = ((double)intersections_a) / ((double)nt * k_nn);
    if (nprobe == 32) {
      CHECK(recall >= 0.998);
    } else if (nprobe == 8) {
      CHECK(recall > 0.925);
    }
  }
}
