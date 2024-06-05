/**
 * @file   unit_api_ivf_pq_index.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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

#include "api/ivf_pq_index.h"
#include "catch2/catch_all.hpp"
#include "test/utils/query_common.h"

TEST_CASE("init constructor", "[api_ivf_pq_index]") {
  SECTION("default") {
    auto a = IndexIVFPQ();
    CHECK(a.feature_type() == TILEDB_ANY);
    CHECK(a.feature_type_string() == datatype_to_string(TILEDB_ANY));
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.id_type_string() == datatype_to_string(TILEDB_UINT32));
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
    CHECK(
        a.partitioning_index_type_string() ==
        datatype_to_string(TILEDB_UINT32));
    CHECK(dimensions(a) == 0);
  }

  SECTION("float uint32 uint32") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint32"},
         {"partitioning_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
    CHECK(dimensions(a) == 0);
  }

  SECTION("int8 uint32 uint32") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "int8"},
         {"id_type", "uint32"},
         {"partitioning_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("uint8 uint32 uint32") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint32"},
         {"partitioning_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("float uint64 uint32") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint64"},
         {"partitioning_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("float uint32 uint64") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint32"},
         {"partitioning_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("int8 uint64 uint32") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "int8"},
         {"id_type", "uint64"},
         {"partitioning_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("uint8 uint64 uint32") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint64"},
         {"partitioning_index_type", "uint32"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("int8 uint32 uint64") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "int8"},
         {"id_type", "uint32"},
         {"partitioning_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint32 uint64") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint32"},
         {"partitioning_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("float uint64 uint64") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "float32"},
         {"id_type", "uint64"},
         {"partitioning_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("int8 uint64 uint64") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "int8"},
         {"id_type", "uint64"},
         {"partitioning_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint64 uint64") {
    auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", "uint8"},
         {"id_type", "uint64"},
         {"partitioning_index_type", "uint64"}}));
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }
}

TEST_CASE("create empty index and then train and query", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  using feature_type_type = uint8_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  size_t dimensions = 3;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  {
    auto index = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", feature_type},
         {"id_type", id_type},
         {"partitioning_index_type", partitioning_index_type},
         {"num_subspaces", "1"}}));

    size_t num_vectors = 0;
    auto empty_training_vector_array =
        FeatureVectorArray(dimensions, num_vectors, feature_type, id_type);
    index.train(empty_training_vector_array);
    index.add(empty_training_vector_array);
    index.write_index(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
  }

  {
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto training = ColMajorMatrix<feature_type_type>{
        {3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};
    auto training_vector_array = FeatureVectorArray(training);
    index.train(training_vector_array);
    index.add(training_vector_array);
    index.write_index(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{
        {3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};
    auto query_vector_array = FeatureVectorArray(queries);
    auto&& [scores_vector_array, ids_vector_array] =
        index.query(QueryType::InfiniteRAM, query_vector_array, 1, 1);

    auto scores = std::span<float>(
        (float*)scores_vector_array.data(), scores_vector_array.num_vectors());
    auto ids = std::span<uint32_t>(
        (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());
    CHECK(std::equal(
        scores.begin(), scores.end(), std::vector<float>{0, 0, 0, 0}.begin()));
    CHECK(std::equal(
        ids.begin(), ids.end(), std::vector<int>{0, 1, 2, 3}.begin()));
  }
}

TEST_CASE(
    "create empty index and then train and query with external IDs",
    "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  size_t dimensions = 3;
  size_t num_subspaces = 1;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  {
    auto index = IndexIVFPQ(std::make_optional<IndexOptions>({
        {"feature_type", feature_type},
        {"id_type", id_type},
        {"partitioning_index_type", partitioning_index_type},
        {"dimensions", std::to_string(dimensions)},
        {"num_subspaces", std::to_string(num_subspaces)},
    }));

    size_t num_vectors = 0;
    auto empty_training_vector_array =
        FeatureVectorArray(dimensions, num_vectors, feature_type, id_type);
    index.train(empty_training_vector_array);
    index.add(empty_training_vector_array);
    index.write_index(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.dimensions() == dimensions);
    CHECK(index.num_subspaces() == num_subspaces);
  }

  {
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.dimensions() == dimensions);
    CHECK(index.num_subspaces() == num_subspaces);
    auto training = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
        {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}, {10, 11, 12, 13}};

    auto training_vector_array = FeatureVectorArray(training);
    index.train(training_vector_array);
    index.add(training_vector_array);
    index.write_index(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{
        {8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}};
    auto query_vector_array = FeatureVectorArray(queries);
    auto&& [scores_vector_array, ids_vector_array] =
        index.query(QueryType::InfiniteRAM, query_vector_array, 1, 1);

    auto scores = std::span<float>(
        (float*)scores_vector_array.data(), scores_vector_array.num_vectors());
    auto ids = std::span<uint32_t>(
        (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());
    CHECK(std::equal(
        scores.begin(), scores.end(), std::vector<float>{0, 0, 0, 0}.begin()));
    CHECK(std::equal(
        ids.begin(), ids.end(), std::vector<uint32_t>{10, 11, 12, 13}.begin()));
  }
}

TEST_CASE(
    "create empty index and then train and query with sift",
    "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  size_t k_nn = 10;
  auto feature_type = "float32";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  {
    auto index = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", feature_type},
         {"id_type", id_type},
         {"partitioning_index_type", partitioning_index_type},
         {"num_subspaces", std::to_string(siftsmall_dimensions / 4)}}));

    size_t num_vectors = 0;
    auto empty_training_vector_array = FeatureVectorArray(
        siftsmall_dimensions, num_vectors, feature_type, id_type);
    index.train(empty_training_vector_array);
    index.add(empty_training_vector_array);
    index.write_index(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
  }

  {
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
    index.train(training_set);
    index.add(training_set);
    index.write_index(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
    auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);
    auto&& [scores, ids] =
        index.query(QueryType::InfiniteRAM, query_set, k_nn, 5);
    auto intersections = count_intersections(ids, groundtruth_set, k_nn);
    auto num_ids = num_vectors(ids);
    auto recall = ((double)intersections) / ((double)num_ids * k_nn);
    CHECK(recall > 0.7);
  }
}

TEST_CASE("infer feature type", "[api_ivf_pq_index]") {
  auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"partitioning_index_type", "uint32"}}));
  auto ctx = tiledb::Context{};
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(a.id_type() == TILEDB_UINT32);
  CHECK(a.partitioning_index_type() == TILEDB_UINT32);
}

TEST_CASE("infer dimension", "[api_ivf_pq_index]") {
  auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"partitioning_index_type", "uint32"}}));
  auto ctx = tiledb::Context{};
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  CHECK(dimensions(a) == 0);
  a.train(training_set);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(a.id_type() == TILEDB_UINT32);
  CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  CHECK(dimensions(a) == 128);
}

TEST_CASE("write and read", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  std::string api_ivf_pq_index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(api_ivf_pq_index_uri)) {
    vfs.remove_dir(api_ivf_pq_index_uri);
  }

  auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
      {{"feature_type", "float32"},
       {"id_type", "uint32"},
       {"partitioning_index_type", "uint32"},
       {"num_subspaces", "1"}}));
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set);
  a.add(training_set);
  a.write_index(ctx, api_ivf_pq_index_uri);

  auto b = IndexIVFPQ(ctx, api_ivf_pq_index_uri);

  CHECK(dimensions(a) == dimensions(b));
  CHECK(a.feature_type() == b.feature_type());
  CHECK(a.id_type() == b.id_type());
  CHECK(a.partitioning_index_type() == b.partitioning_index_type());
}

TEST_CASE("build index and query", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  size_t k_nn = 10;
  size_t nprobe = GENERATE(8, 32);

  auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
      {{"id_type", "uint32"}, {"partitioning_index_type", "uint32"}}));
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);
  a.train(training_set);
  a.add(training_set);

  auto&& [s, t] = a.query(QueryType::InfiniteRAM, query_set, k_nn, 5);

  auto intersections = count_intersections(t, groundtruth_set, k_nn);
  auto nt = num_vectors(t);
  auto recall = ((double)intersections) / ((double)nt * k_nn);
  CHECK(recall > 0.6);
}

TEST_CASE("read index and query", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  tiledb::VFS vfs(ctx);

  size_t k_nn = 10;

  std::string api_ivf_pq_index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  if (vfs.is_dir(api_ivf_pq_index_uri)) {
    vfs.remove_dir(api_ivf_pq_index_uri);
  }

  auto a = IndexIVFPQ(std::make_optional<IndexOptions>(
      {{"feature_type", "float32"},
       {"id_type", "uint32"},
       {"partitioning_index_type", "uint32"},
       {"num_subspaces", std::to_string(sift_dimensions / 4)}}));

  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set);
  a.add(training_set);
  a.write_index(ctx, api_ivf_pq_index_uri);
  auto b = IndexIVFPQ(ctx, api_ivf_pq_index_uri);

  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);

  auto&& [s, t] = a.query(QueryType::InfiniteRAM, query_set, k_nn, 5);
  auto&& [u, v] = b.query(QueryType::InfiniteRAM, query_set, k_nn, 5);

  auto intersections_a = count_intersections(t, groundtruth_set, k_nn);
  auto intersections_b = count_intersections(v, groundtruth_set, k_nn);
  CHECK(intersections_a == intersections_b);
  auto nt = num_vectors(t);
  auto nv = num_vectors(v);
  CHECK(nt == nv);
  auto recall = ((double)intersections_a) / ((double)nt * k_nn);
  CHECK(recall > 0.7);
}

TEST_CASE("storage_version", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  size_t dimensions = 3;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  {
    // First we create the index with a storage_version.
    auto index = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", feature_type},
         {"id_type", id_type},
         {"partitioning_index_type", partitioning_index_type},
         {"num_subspaces", "1"}}));

    size_t num_vectors = 0;
    auto empty_training_vector_array =
        FeatureVectorArray(dimensions, num_vectors, feature_type, id_type);
    index.train(empty_training_vector_array);
    index.add(empty_training_vector_array);
    index.write_index(ctx, index_uri, std::nullopt, "0.3");

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
  }

  {
    // Now make sure if we try to write it again with a different
    // storage_version, we throw.
    auto index = IndexIVFPQ(ctx, index_uri);
    auto training = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
        {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}, {10, 11, 12, 13}};

    auto training_vector_array = FeatureVectorArray(training);
    index.train(training_vector_array);
    index.add(training_vector_array);

    // Throw with the wrong version.
    CHECK_THROWS_WITH(
        index.write_index(ctx, index_uri, std::nullopt, "0.4"),
        "Version mismatch. Requested 0.4 but found 0.3");
    // Succeed without a version.
    index.write_index(ctx, index_uri);
    // Succeed with the same version.
    index.write_index(ctx, index_uri, std::nullopt, "0.3");
  }
}

TEST_CASE("write and load index with timestamps", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  using partitioning_index_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  size_t dimensions = 3;
  size_t n_list = 1;
  size_t num_subspaces = 1;
  float convergence_tolerance = 0.00003f;
  size_t max_iterations = 3;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  // Create an empty index.
  {
    // We write the empty index at timestamp 0.
    auto index = IndexIVFPQ(std::make_optional<IndexOptions>(
        {{"feature_type", feature_type},
         {"id_type", id_type},
         {"partitioning_index_type", partitioning_index_type},
         {"n_list", std::to_string(n_list)},
         {"num_subspaces", std::to_string(num_subspaces)},
         {"convergence_tolerance", std::to_string(convergence_tolerance)},
         {"max_iterations", std::to_string(max_iterations)}}));

    size_t num_vectors = 0;
    auto empty_training_vector_array =
        FeatureVectorArray(dimensions, num_vectors, feature_type, id_type);
    index.train(empty_training_vector_array);
    index.add(empty_training_vector_array);
    index.write_index(ctx, index_uri, TemporalPolicy(TimeTravel, 0));

    CHECK(index.temporal_policy().timestamp_end() == 0);
    CHECK(index.dimensions() == dimensions);
    CHECK(index.n_list() == n_list);
    CHECK(index.num_subspaces() == num_subspaces);
    CHECK(index.convergence_tolerance() == convergence_tolerance);
    CHECK(index.max_iterations() == max_iterations);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(ctx, index_uri);
    CHECK(typed_index.group().get_dimensions() == dimensions);
    CHECK(typed_index.group().get_temp_size() == 0);
    CHECK(typed_index.group().get_history_index() == 0);

    CHECK(typed_index.group().get_base_size() == 0);
    CHECK(typed_index.group().get_ingestion_timestamp() == 0);

    CHECK(typed_index.group().get_all_num_partitions().size() == 1);
    CHECK(typed_index.group().get_all_base_sizes().size() == 1);
    CHECK(typed_index.group().get_all_ingestion_timestamps().size() == 1);

    CHECK(typed_index.group().get_all_num_partitions()[0] == n_list);
    CHECK(typed_index.group().get_all_base_sizes()[0] == 0);
    CHECK(typed_index.group().get_all_ingestion_timestamps()[0] == 0);
  }

  // Train it at timestamp 99.
  {
    // We then load this empty index and don't set a timestamp (which means
    // we'll read from 0 -> max uint64).
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.temporal_policy().timestamp_start() == 0);
    CHECK(
        index.temporal_policy().timestamp_end() ==
        std::numeric_limits<uint64_t>::max());
    CHECK(index.dimensions() == dimensions);
    CHECK(index.n_list() == n_list);
    CHECK(index.num_subspaces() == num_subspaces);
    // TODO(paris): Have ivf_pq_index store these in metadata so we don't lose
    // values on load by URI.
    // CHECK(index.convergence_tolerance() == convergence_tolerance);
    // CHECK(index.max_iterations() == max_iterations);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto training = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}, {1, 2, 3, 4}};

    auto training_vector_array = FeatureVectorArray(training);
    index.train(training_vector_array);
    index.add(training_vector_array);
    // We then write the index at timestamp 99.
    index.write_index(ctx, index_uri, TemporalPolicy(TimeTravel, 99));

    // This also updates the timestamp of the index - we're now at timestamp 99.
    CHECK(index.temporal_policy().timestamp_end() == 99);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{
        {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
    auto query_vector_array = FeatureVectorArray(queries);
    auto&& [scores_vector_array, ids_vector_array] =
        index.query(QueryType::InfiniteRAM, query_vector_array, 1, n_list);

    auto scores = std::span<float>(
        (float*)scores_vector_array.data(), scores_vector_array.num_vectors());
    auto ids = std::span<uint32_t>(
        (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());
    CHECK(std::equal(
        scores.begin(), scores.end(), std::vector<float>{0, 0, 0, 0}.begin()));
    CHECK(std::equal(
        ids.begin(), ids.end(), std::vector<int>{1, 2, 3, 4}.begin()));

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(ctx, index_uri);
    CHECK(typed_index.group().get_dimensions() == dimensions);
    CHECK(typed_index.group().get_temp_size() == 0);
    CHECK(typed_index.group().get_history_index() == 0);
    CHECK(typed_index.group().get_ingestion_timestamp() == 99);
    CHECK(typed_index.group().get_all_num_partitions().size() == 1);
    CHECK(typed_index.group().get_all_base_sizes().size() == 1);
    CHECK(typed_index.group().get_all_ingestion_timestamps().size() == 1);

    CHECK(typed_index.group().get_all_num_partitions()[0] > 0);
    CHECK(typed_index.group().get_all_base_sizes()[0] == 4);
    CHECK(typed_index.group().get_all_ingestion_timestamps()[0] == 99);
  }

  // Train it at timestamp 100.
  {
    // We then load the trained index and don't set a timestamp (which means
    // we'll load it at timestamp 99).
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.temporal_policy().timestamp_start() == 0);
    CHECK(
        index.temporal_policy().timestamp_end() ==
        std::numeric_limits<uint64_t>::max());
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto training = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
        {{11, 11, 11}, {22, 22, 22}, {33, 33, 33}, {44, 44, 44}, {55, 55, 55}},
        {11, 22, 33, 44, 55}};

    auto training_vector_array = FeatureVectorArray(training);
    index.train(training_vector_array);
    index.add(training_vector_array);
    // We then write the index at timestamp 100.
    index.write_index(ctx, index_uri, TemporalPolicy(TimeTravel, 100));

    // This also updates the timestamp of the index - we're now at timestamp
    // 100.
    CHECK(index.temporal_policy().timestamp_end() == 100);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{
        {11, 11, 11}, {22, 22, 22}, {33, 33, 33}, {44, 44, 44}, {55, 55, 55}};
    auto query_vector_array = FeatureVectorArray(queries);
    auto&& [scores_vector_array, ids_vector_array] =
        index.query(QueryType::InfiniteRAM, query_vector_array, 1, 1);

    auto scores = std::span<float>(
        (float*)scores_vector_array.data(), scores_vector_array.num_vectors());
    auto ids = std::span<uint32_t>(
        (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());
    CHECK(std::equal(
        scores.begin(), scores.end(), std::vector<int>{0, 0, 0, 0, 0}.begin()));
    CHECK(std::equal(
        ids.begin(), ids.end(), std::vector<int>{11, 22, 33, 44, 55}.begin()));

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(ctx, index_uri);
    CHECK(typed_index.group().get_dimensions() == dimensions);
    CHECK(typed_index.group().get_temp_size() == 0);
    CHECK(typed_index.group().get_history_index() == 1);

    CHECK(typed_index.group().get_base_size() == 5);
    CHECK(typed_index.group().get_ingestion_timestamp() == 100);

    CHECK(typed_index.group().get_all_num_partitions().size() == 2);
    CHECK(typed_index.group().get_all_base_sizes().size() == 2);
    CHECK(typed_index.group().get_all_ingestion_timestamps().size() == 2);

    CHECK(typed_index.group().get_all_num_partitions()[0] > 0);
    CHECK(typed_index.group().get_all_num_partitions()[1] > 0);
    auto all_base_sizes = typed_index.group().get_all_base_sizes();
    CHECK(std::equal(
        all_base_sizes.begin(),
        all_base_sizes.end(),
        std::vector<uint64_t>{4, 5}.begin()));
    auto all_ingestion_timestamps =
        typed_index.group().get_all_ingestion_timestamps();
    CHECK(std::equal(
        all_ingestion_timestamps.begin(),
        all_ingestion_timestamps.end(),
        std::vector<uint64_t>{99, 100}.begin()));
  }

  // Load it at timestamp 99 and make sure we can query it correctly.
  {
    auto temporal_policy = TemporalPolicy{TimeTravel, 99};
    auto index = IndexIVFPQ(ctx, index_uri, temporal_policy);

    CHECK(index.temporal_policy().timestamp_end() == 99);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{
        {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
    auto query_vector_array = FeatureVectorArray(queries);
    auto&& [scores_vector_array, ids_vector_array] =
        index.query(QueryType::InfiniteRAM, query_vector_array, 1, 1);

    auto scores = std::span<float>(
        (float*)scores_vector_array.data(), scores_vector_array.num_vectors());
    auto ids = std::span<uint32_t>(
        (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());
    CHECK(std::equal(
        scores.begin(), scores.end(), std::vector<float>{0, 0, 0, 0}.begin()));
    CHECK(std::equal(
        ids.begin(), ids.end(), std::vector<uint32_t>{1, 2, 3, 4}.begin()));

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(ctx, index_uri, temporal_policy);
    CHECK(typed_index.group().get_dimensions() == dimensions);
    CHECK(typed_index.group().get_temp_size() == 0);
    CHECK(typed_index.group().get_history_index() == 0);

    CHECK(typed_index.group().get_base_size() == 4);
    CHECK(typed_index.group().get_ingestion_timestamp() == 99);

    CHECK(typed_index.group().get_all_num_partitions().size() == 2);
    CHECK(typed_index.group().get_all_base_sizes().size() == 2);
    CHECK(typed_index.group().get_all_ingestion_timestamps().size() == 2);

    CHECK(typed_index.group().get_all_num_partitions()[0] > 0);
    CHECK(typed_index.group().get_all_num_partitions()[1] > 0);
    auto all_base_sizes = typed_index.group().get_all_base_sizes();
    CHECK(std::equal(
        all_base_sizes.begin(),
        all_base_sizes.end(),
        std::vector<uint64_t>{4, 5}.begin()));
    auto all_ingestion_timestamps =
        typed_index.group().get_all_ingestion_timestamps();
    CHECK(std::equal(
        all_ingestion_timestamps.begin(),
        all_ingestion_timestamps.end(),
        std::vector<uint64_t>{99, 100}.begin()));
  }

  // Load it at timestamp 5 (before ingestion) and make sure we can query and be
  // returned fill values.
  {
    auto temporal_policy = TemporalPolicy{TimeTravel, 0};
    auto index = IndexIVFPQ(ctx, index_uri, temporal_policy);

    CHECK(index.temporal_policy().timestamp_start() == 0);
    CHECK(index.temporal_policy().timestamp_end() == 0);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{{1, 1, 1}};
    auto query_vector_array = FeatureVectorArray(queries);
    {
      auto&& [scores_vector_array, ids_vector_array] =
          index.query(QueryType::FiniteRAM, query_vector_array, 1, 1);

      auto scores = std::span<float>(
          (float*)scores_vector_array.data(),
          scores_vector_array.num_vectors());
      auto ids = std::span<uint32_t>(
          (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());

      CHECK(std::equal(
          scores.begin(),
          scores.end(),
          std::vector<float>{std::numeric_limits<float>::max()}.begin()));
      CHECK(std::equal(
          ids.begin(),
          ids.end(),
          std::vector<uint32_t>{std::numeric_limits<uint32_t>::max()}.begin()));
    }
    {
      auto&& [scores_vector_array, ids_vector_array] =
          index.query(QueryType::InfiniteRAM, query_vector_array, 1, 1);

      auto scores = std::span<float>(
          (float*)scores_vector_array.data(),
          scores_vector_array.num_vectors());
      auto ids = std::span<uint32_t>(
          (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());
      debug_vector(scores, "scores");
      debug_vector(ids, "ids");

      CHECK(std::equal(
          scores.begin(),
          scores.end(),
          std::vector<float>{std::numeric_limits<float>::max()}.begin()));
      CHECK(std::equal(
          ids.begin(),
          ids.end(),
          std::vector<uint32_t>{std::numeric_limits<uint32_t>::max()}.begin()));
    }

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(ctx, index_uri, temporal_policy);
    CHECK(typed_index.group().get_dimensions() == dimensions);
    CHECK(typed_index.group().get_temp_size() == 0);
    CHECK(typed_index.group().get_history_index() == 0);

    CHECK(typed_index.group().get_base_size() == 4);
    CHECK(typed_index.group().get_ingestion_timestamp() == 99);

    CHECK(typed_index.group().get_all_num_partitions().size() == 2);
    CHECK(typed_index.group().get_all_base_sizes().size() == 2);
    CHECK(typed_index.group().get_all_ingestion_timestamps().size() == 2);

    CHECK(typed_index.group().get_all_num_partitions()[0] > 0);
    CHECK(typed_index.group().get_all_num_partitions()[1] > 0);
    auto all_base_sizes = typed_index.group().get_all_base_sizes();
    CHECK(std::equal(
        all_base_sizes.begin(),
        all_base_sizes.end(),
        std::vector<uint64_t>{4, 5}.begin()));
    auto all_ingestion_timestamps =
        typed_index.group().get_all_ingestion_timestamps();
    CHECK(std::equal(
        all_ingestion_timestamps.begin(),
        all_ingestion_timestamps.end(),
        std::vector<uint64_t>{99, 100}.begin()));
  }

  // Clear history for <= 99 and then load at 99, then make sure we cannot
  // query.
  {
    IndexIVFPQ::clear_history(ctx, index_uri, 99);

    auto temporal_policy = TemporalPolicy{TimeTravel, 99};
    auto index = IndexIVFPQ(ctx, index_uri, temporal_policy);

    CHECK(index.temporal_policy().timestamp_end() == 99);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{
        {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}};
    auto query_vector_array = FeatureVectorArray(queries);
    auto&& [scores_vector_array, ids_vector_array] =
        index.query(QueryType::InfiniteRAM, query_vector_array, 1, 1);

    auto scores = std::span<float>(
        (float*)scores_vector_array.data(), scores_vector_array.num_vectors());
    auto ids = std::span<uint32_t>(
        (uint32_t*)ids_vector_array.data(), ids_vector_array.num_vectors());
    CHECK(scores.size() == 4);
    CHECK(ids.size() == 4);
    auto default_score = std::numeric_limits<float>::max();
    auto default_id = std::numeric_limits<uint32_t>::max();
    CHECK(std::equal(
        scores.begin(),
        scores.end(),
        std::vector<float>{
            default_score, default_score, default_score, default_score}
            .begin()));
    CHECK(std::equal(
        ids.begin(),
        ids.end(),
        std::vector<uint32_t>{default_id, default_id, default_id, default_id}
            .begin()));

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(ctx, index_uri, temporal_policy);
    CHECK(typed_index.group().get_dimensions() == dimensions);
    CHECK(typed_index.group().get_temp_size() == 0);
    CHECK(typed_index.group().get_history_index() == 0);

    CHECK(typed_index.group().get_base_size() == 5);
    CHECK(typed_index.group().get_ingestion_timestamp() == 100);

    CHECK(typed_index.group().get_all_num_partitions().size() == 1);
    CHECK(typed_index.group().get_all_base_sizes().size() == 1);
    CHECK(typed_index.group().get_all_ingestion_timestamps().size() == 1);

    CHECK(typed_index.group().get_all_num_partitions()[0] > 0);
    auto all_base_sizes = typed_index.group().get_all_base_sizes();
    CHECK(std::equal(
        all_base_sizes.begin(),
        all_base_sizes.end(),
        std::vector<uint64_t>{5}.begin()));
    auto all_ingestion_timestamps =
        typed_index.group().get_all_ingestion_timestamps();
    CHECK(std::equal(
        all_ingestion_timestamps.begin(),
        all_ingestion_timestamps.end(),
        std::vector<uint64_t>{100}.begin()));
  }
}
