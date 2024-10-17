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
#include "test/utils/test_utils.h"

TEST_CASE("init constructor", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  std::string index_uri =
      (std::filesystem::temp_directory_path() / "ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  SECTION("default") {
    CHECK_THROWS(IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_ANY, TILEDB_UINT32, TILEDB_UINT32));
  }

  SECTION("float uint32 uint32") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
    CHECK(dimensions(a) == 16);
  }

  SECTION("int8 uint32 uint32") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_INT8, TILEDB_UINT32, TILEDB_UINT32);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("uint8 uint32 uint32") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_UINT8, TILEDB_UINT32, TILEDB_UINT32);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("float uint64 uint32") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("float uint32 uint64") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("int8 uint64 uint32") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_INT8, TILEDB_UINT64, TILEDB_UINT32);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("uint8 uint64 uint32") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_UINT8, TILEDB_UINT64, TILEDB_UINT32);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT32);
  }

  SECTION("int8 uint32 uint64") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_INT8, TILEDB_UINT32, TILEDB_UINT64);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint32 uint64") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_UINT8, TILEDB_UINT32, TILEDB_UINT64);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_UINT8);
    CHECK(a.id_type() == TILEDB_UINT32);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("float uint64 uint64") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_FLOAT32);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("int8 uint64 uint64") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_INT8, TILEDB_UINT64, TILEDB_UINT64);
    auto a = IndexIVFPQ(ctx, index_uri);
    CHECK(a.feature_type() == TILEDB_INT8);
    CHECK(a.id_type() == TILEDB_UINT64);
    CHECK(a.partitioning_index_type() == TILEDB_UINT64);
  }

  SECTION("uint8 uint64 uint64") {
    IndexIVFPQ::create(
        ctx, index_uri, 16, TILEDB_UINT8, TILEDB_UINT64, TILEDB_UINT64);
    auto a = IndexIVFPQ(ctx, index_uri);
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
  uint64_t dimensions = 3;
  size_t partitions = 1;
  uint32_t num_subspaces = 3;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  auto queries = ColMajorMatrix<feature_type_type>{
      {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}}};

  {
    IndexIVFPQ::create(
        ctx,
        index_uri,
        dimensions,
        TILEDB_UINT8,
        TILEDB_UINT32,
        TILEDB_UINT32,
        num_subspaces);
    auto index = IndexIVFPQ(ctx, index_uri);

    size_t num_vectors = 0;
    auto empty_training_vector_array =
        FeatureVectorArray(dimensions, num_vectors, feature_type, id_type);
    auto empty_feature_vector = FeatureVector(num_vectors, id_type);
    index.train(empty_training_vector_array, partitions);
    index.ingest(empty_training_vector_array);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.distance_metric() == DistanceMetric::SUM_OF_SQUARES);

    // Make sure we can query with k_factor > 1 on an empty index that is not
    // loaded by URI.
    size_t top_k = 1;
    size_t nprobe = 1;
    float k_factor = 2.f;
    auto&& [scores, ids] =
        index.query(FeatureVectorArray(queries), top_k, nprobe, k_factor);
    auto default_score = std::numeric_limits<float>::max();
    auto default_id = std::numeric_limits<uint32_t>::max();
    check_single_vector_equals(
        scores,
        ids,
        {default_score, default_score, default_score, default_score},
        {default_id, default_id, default_id, default_id});
  }

  // Check IndexLoadStrategy.
  {
    CHECK_THROWS(IndexIVFPQ(ctx, index_uri, IndexLoadStrategy::PQ_OOC, 0));
    CHECK_NOTHROW(IndexIVFPQ(ctx, index_uri, IndexLoadStrategy::PQ_OOC, 10));

    CHECK_NOTHROW(IndexIVFPQ(ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0));
    CHECK_THROWS(IndexIVFPQ(ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 10));

    CHECK_NOTHROW(IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX_AND_RERANKING_VECTORS, 0));
    CHECK_THROWS(IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX_AND_RERANKING_VECTORS, 10));
  }

  // We can open, train, and query an infinite index.
  {
    std::unique_ptr<IndexIVFPQ> index;
    SECTION("infinite") {
      index = std::make_unique<IndexIVFPQ>(ctx, index_uri);
    }
    SECTION("finite") {
      size_t upper_bound = 97;
      index = std::make_unique<IndexIVFPQ>(
          ctx, index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
      CHECK(index->upper_bound() == upper_bound);
    }

    CHECK(index->feature_type_string() == feature_type);
    CHECK(index->id_type_string() == id_type);
    CHECK(index->partitioning_index_type_string() == partitioning_index_type);

    auto training = ColMajorMatrix<feature_type_type>{
        {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}}};
    auto training_vector_array = FeatureVectorArray(training);
    index->train(training_vector_array);
    index->ingest(training_vector_array);
    CHECK(index->feature_type_string() == feature_type);
    CHECK(index->id_type_string() == id_type);
    CHECK(index->partitioning_index_type_string() == partitioning_index_type);

    size_t top_k = 1;
    size_t nprobe = 1;
    auto&& [scores, ids] =
        index->query(FeatureVectorArray(queries), top_k, nprobe);
    check_single_vector_equals(scores, ids, {0, 0, 0, 0}, {0, 1, 2, 3});
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
  size_t partitions = 5;
  uint64_t dimensions = 3;
  uint32_t num_subspaces = 1;
  uint32_t max_iterations = 2;
  float convergence_tolerance = 0.000025f;
  float reassign_ratio = 0.075f;
  auto distance_metric = DistanceMetric::L2;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  {
    IndexIVFPQ::create(
        ctx,
        index_uri,
        dimensions,
        TILEDB_UINT8,
        TILEDB_UINT32,
        TILEDB_UINT32,
        num_subspaces,
        max_iterations,
        convergence_tolerance,
        reassign_ratio,
        std::nullopt,
        distance_metric);
    auto index = IndexIVFPQ(ctx, index_uri);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.dimensions() == dimensions);
    CHECK(index.num_subspaces() == num_subspaces);
    CHECK(index.distance_metric() == distance_metric);
  }

  {
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.dimensions() == dimensions);
    CHECK(index.num_subspaces() == num_subspaces);
    CHECK(index.distance_metric() == distance_metric);
    auto training = ColMajorMatrix<feature_type_type>{
        {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}};
    auto training_ids =
        FeatureVector(std::vector<id_type_type>{10, 11, 12, 13});

    auto training_vector_array = FeatureVectorArray(training);
    index.train(training_vector_array, partitions);
    index.ingest(training_vector_array, training_ids);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.dimensions() == dimensions);
    CHECK(index.num_subspaces() == num_subspaces);
    CHECK(index.distance_metric() == distance_metric);

    auto queries = ColMajorMatrix<feature_type_type>{
        {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}};

    auto&& [scores, ids] = index.query(FeatureVectorArray(queries), 1, 1);
    check_single_vector_equals(scores, ids, {0, 0, 0, 0}, {10, 11, 12, 13});
  }

  {
    size_t top_k = 1;
    size_t nprobe = 1;

    std::unique_ptr<IndexIVFPQ> index;
    SECTION("infinite") {
      index = std::make_unique<IndexIVFPQ>(ctx, index_uri);
    }
    SECTION("finite") {
      size_t upper_bound = GENERATE(3, 4, 5, 100);
      index = std::make_unique<IndexIVFPQ>(
          ctx, index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
      CHECK(index->upper_bound() == upper_bound);
    }

    CHECK(index->feature_type_string() == feature_type);
    CHECK(index->id_type_string() == id_type);
    CHECK(index->partitioning_index_type_string() == partitioning_index_type);

    auto queries = ColMajorMatrix<feature_type_type>{
        {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}};
    auto&& [scores, ids] =
        index->query(FeatureVectorArray(queries), top_k, nprobe);
    check_single_vector_equals(scores, ids, {0, 0, 0, 0}, {10, 11, 12, 13});
  }
}

TEST_CASE(
    "query finite & infinite, with re-ranking and without",
    "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  size_t k_nn = 10;
  size_t partitions = 100;
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
    IndexIVFPQ::create(
        ctx,
        index_uri,
        siftsmall_dimensions,
        TILEDB_FLOAT32,
        TILEDB_UINT32,
        TILEDB_UINT32,
        siftsmall_dimensions / 4);
    auto index = IndexIVFPQ(ctx, index_uri);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.distance_metric() == DistanceMetric::SUM_OF_SQUARES);
  }

  {
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
    index.train(training_set, partitions);
    index.ingest(training_set, FeatureVector(0, id_type));

    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
    auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);
    auto&& [scores, ids] = index.query(query_set, k_nn, 5);
    auto intersections = count_intersections(ids, groundtruth_set, k_nn);
    CHECK((intersections / static_cast<double>(num_vectors(ids) * k_nn)) > 0.7);
  }

  {
    float k_factor = 20.f;
    auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
    auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);

    auto index = IndexIVFPQ(ctx, index_uri);
    auto index_finite =
        IndexIVFPQ(ctx, index_uri, IndexLoadStrategy::PQ_OOC, 450);

    for (auto [nprobe, expected_accuracy, expected_accuracy_with_reranking] :
         std::vector<std::tuple<int, float, float>>{
             {1, .4f, .44f},
             {2, .5f, .6f},
             {5, .7f, .7f},
             {10, .75f, .9f},
             {100, .78f, 1.f}}) {
      auto&& [distances, ids] = index.query(query_set, k_nn, nprobe);
      auto intersections = count_intersections(ids, groundtruth_set, k_nn);
      CHECK(
          (intersections / static_cast<double>(num_vectors(ids) * k_nn)) >=
          expected_accuracy);

      auto&& [distances_with_reranking, ids_with_reranking] =
          index.query(query_set, k_nn, nprobe, k_factor);
      auto intersections_with_reranking =
          count_intersections(ids_with_reranking, groundtruth_set, k_nn);
      CHECK(
          (intersections_with_reranking /
           static_cast<double>(num_vectors(ids_with_reranking) * k_nn)) >=
          expected_accuracy_with_reranking);

      auto&& [distances_finite, ids_finite] =
          index_finite.query(query_set, k_nn, nprobe);
      CHECK(are_equal(ids_finite, ids));
      CHECK(are_equal(distances_finite, distances));

      auto&& [distances_finite_with_reranking, ids_finite_with_reranking] =
          index_finite.query(query_set, k_nn, nprobe, k_factor);
      CHECK(are_equal(ids_finite_with_reranking, ids_with_reranking));
      CHECK(
          are_equal(distances_finite_with_reranking, distances_with_reranking));
    }
  }
}

TEST_CASE("write and read", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  IndexIVFPQ::create(
      ctx,
      index_uri,
      siftsmall_dimensions,
      TILEDB_FLOAT32,
      TILEDB_UINT32,
      TILEDB_UINT32,
      1);
  auto a = IndexIVFPQ(ctx, index_uri);
  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set);
  a.ingest(training_set);

  auto b = IndexIVFPQ(ctx, index_uri);

  CHECK(dimensions(a) == dimensions(b));
  CHECK(a.feature_type() == b.feature_type());
  CHECK(a.id_type() == b.id_type());
  CHECK(a.partitioning_index_type() == b.partitioning_index_type());
}

TEST_CASE("build index and query", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  size_t partitions = 10;
  size_t k_nn = 10;
  size_t nprobe = GENERATE(8, 32);

  IndexIVFPQ::create(
      ctx,
      index_uri,
      siftsmall_dimensions,
      TILEDB_FLOAT32,
      TILEDB_UINT32,
      TILEDB_UINT32);
  auto a = IndexIVFPQ(ctx, index_uri);

  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);
  a.train(training_set, partitions);
  a.ingest(training_set);

  auto&& [s, t] = a.query(query_set, k_nn, nprobe);

  auto intersections = count_intersections(t, groundtruth_set, k_nn);
  auto nt = num_vectors(t);
  auto recall = intersections / static_cast<double>(nt * k_nn);
  CHECK(recall > 0.6);
}

TEST_CASE("read index and query", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  tiledb::VFS vfs(ctx);

  size_t k_nn = 10;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  IndexIVFPQ::create(
      ctx,
      index_uri,
      sift_dimensions,
      TILEDB_FLOAT32,
      TILEDB_UINT32,
      TILEDB_UINT32,
      sift_dimensions / 4);
  auto a = IndexIVFPQ(ctx, index_uri);

  auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
  a.train(training_set, 100);
  a.ingest(training_set);

  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);

  std::unique_ptr<IndexIVFPQ> b;
  SECTION("infinite") {
    b = std::make_unique<IndexIVFPQ>(ctx, index_uri);
  }
  SECTION("finite") {
    size_t upper_bound = GENERATE(500, 1000);
    b = std::make_unique<IndexIVFPQ>(
        ctx, index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
    CHECK(b->upper_bound() == upper_bound);
  }

  // Make sure the same query results are returned for two different indexes.
  size_t nprobe = 5;
  auto&& [scores_1, ids_1] = a.query(query_set, k_nn, nprobe);
  auto&& [scores_2, ids_2] = b->query(query_set, k_nn, nprobe);

  CHECK(are_equal(scores_1, scores_2));
  CHECK(are_equal(ids_1, ids_2));

  auto intersections_1 = count_intersections(ids_1, groundtruth_set, k_nn);
  auto intersections_2 = count_intersections(ids_2, groundtruth_set, k_nn);
  CHECK(num_vectors(ids_1) == num_vectors(ids_2));
  auto recall =
      intersections_1 / static_cast<double>(num_vectors(ids_1) * k_nn);
  CHECK(recall > 0.7);
}

TEST_CASE("storage_version", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  size_t partitions = 5;
  uint64_t dimensions = 3;
  uint32_t num_subspaces = 1;
  uint32_t max_iterations = 2;
  float convergence_tolerance = 0.000025f;
  float reassign_ratio = 0.075f;
  auto distance_metric = DistanceMetric::L2;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  for (auto invalid_storage_version : {"0.1", "0.2", "invalid"}) {
    CHECK_THROWS(IndexIVFPQ::create(
        ctx,
        index_uri,
        dimensions,
        TILEDB_UINT8,
        TILEDB_UINT32,
        TILEDB_UINT32,
        num_subspaces,
        max_iterations,
        convergence_tolerance,
        reassign_ratio,
        std::nullopt,
        distance_metric,
        invalid_storage_version));
  }
}

TEST_CASE("clear history with an open index", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  uint64_t dimensions = 3;
  size_t partitions = 1;
  uint32_t num_subspaces = 1;
  float convergence_tolerance = 0.00003f;
  uint32_t max_iterations = 3;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  IndexIVFPQ::create(
      ctx,
      index_uri,
      dimensions,
      TILEDB_UINT8,
      TILEDB_UINT32,
      TILEDB_UINT32,
      num_subspaces,
      max_iterations,
      convergence_tolerance);
  auto index = IndexIVFPQ(ctx, index_uri);

  auto training = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
      {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}, {1, 2, 3, 4}};
  auto training_vector_array = FeatureVectorArray(training);
  index.train(
      training_vector_array, partitions, TemporalPolicy(TimeTravel, 99));
  index.ingest(training_vector_array);

  auto&& [scores, ids] = index.query(training_vector_array, 1, 1);

  auto second_index = IndexIVFPQ(ctx, index_uri);
  auto&& [scores_finite, ids_finite] =
      second_index.query(training_vector_array, 1, 1);

  // Here we check that we can clear_history() even with a index in memory. This
  // makes sure that every Array which IndexIVFPQ opens has been closed,
  // otherwise clear_history() will throw when it tries to call
  // delete_fragments() on the index Array's.
  IndexIVFPQ::clear_history(ctx, index_uri, 99);
}

TEST_CASE("write and load index with timestamps", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  using partitioning_index_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  uint64_t dimensions = 3;
  size_t partitions = 1;
  uint32_t num_subspaces = 1;
  uint32_t max_iterations = 3;
  float convergence_tolerance = 0.00003f;
  float reassign_ratio = 0.08f;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  // Create an empty index.
  {
    // We write the empty index at timestamp 0.
    IndexIVFPQ::create(
        ctx,
        index_uri,
        dimensions,
        TILEDB_UINT8,
        TILEDB_UINT32,
        TILEDB_UINT32,
        num_subspaces,
        max_iterations,
        convergence_tolerance,
        reassign_ratio);
    auto index = IndexIVFPQ(ctx, index_uri);

    CHECK(index.dimensions() == dimensions);
    CHECK(index.num_subspaces() == num_subspaces);
    CHECK(index.max_iterations() == max_iterations);
    CHECK(index.convergence_tolerance() == convergence_tolerance);
    CHECK(index.reassign_ratio() == reassign_ratio);
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

    CHECK(typed_index.group().get_all_num_partitions()[0] == 0);
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
    CHECK(index.num_subspaces() == num_subspaces);
    CHECK(index.max_iterations() == max_iterations);
    CHECK(index.convergence_tolerance() == convergence_tolerance);
    CHECK(index.reassign_ratio() == reassign_ratio);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    auto training = ColMajorMatrix<feature_type_type>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}};
    auto training_ids = FeatureVector(std::vector<id_type_type>{1, 2, 3, 4});

    auto training_vector_array = FeatureVectorArray(training);
    // We then write the index at timestamp 99.
    CHECK(index.temporal_policy().timestamp_end() != 99);
    index.train(
        training_vector_array, partitions, TemporalPolicy(TimeTravel, 99));
    CHECK(index.temporal_policy().timestamp_end() == 99);
    CHECK(index.partitions() == partitions);
    index.ingest(training_vector_array, training_ids);

    // This also updates the timestamp of the index - we're now at timestamp 99.
    CHECK(index.temporal_policy().timestamp_end() == 99);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.partitions() == partitions);

    size_t top_k = 1;
    size_t nprobe = 1;

    auto queries = ColMajorMatrix<feature_type_type>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}};
    auto&& [scores, ids] =
        index.query(FeatureVectorArray(queries), top_k, nprobe);
    check_single_vector_equals(scores, ids, {0, 0, 0, 0}, {1, 2, 3, 4});

    float k_factor = 2.f;
    auto&& [scores_with_reranking, ids_with_reranking] =
        index.query(FeatureVectorArray(queries), top_k, nprobe, k_factor);
    check_single_vector_equals(
        scores_with_reranking, ids_with_reranking, {0, 0, 0, 0}, {1, 2, 3, 4});

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

    // Check that we can do finite and infinite queries and then train + write
    // the index.
    {
      size_t top_k = 1;
      size_t nprobe = 1;
      auto queries = ColMajorMatrix<feature_type_type>{
          {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}};
      auto&& [scores, ids] =
          index.query(FeatureVectorArray(queries), top_k, nprobe);
      check_single_vector_equals(scores, ids, {0, 0, 0, 0}, {1, 2, 3, 4});

      {
        size_t upper_bound = 5;
        auto index_finite =
            IndexIVFPQ(ctx, index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
        auto&& [scores_finite, ids_finite] =
            index_finite.query(FeatureVectorArray(queries), top_k, nprobe);
        check_single_vector_equals(
            scores_finite, ids_finite, {0, 0, 0, 0}, {1, 2, 3, 4});
      }
    }

    CHECK(index.temporal_policy().timestamp_start() == 0);
    CHECK(
        index.temporal_policy().timestamp_end() ==
        std::numeric_limits<uint64_t>::max());
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.max_iterations() == max_iterations);
    CHECK(index.convergence_tolerance() == convergence_tolerance);
    CHECK(index.reassign_ratio() == reassign_ratio);

    auto training = ColMajorMatrix<feature_type_type>{
        {{11, 11, 11}, {22, 22, 22}, {33, 33, 33}, {44, 44, 44}, {55, 55, 55}},
    };
    auto training_ids =
        FeatureVector(std::vector<id_type_type>{11, 22, 33, 44, 55});

    auto training_vector_array = FeatureVectorArray(training);
    // We then write the index at timestamp 100.
    index.train(training_vector_array, 10, TemporalPolicy(TimeTravel, 100));
    index.ingest(training_vector_array, training_ids);

    // This also updates the timestamp of the index - we're now at timestamp
    // 100.
    CHECK(index.temporal_policy().timestamp_end() == 100);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    size_t top_k = 1;
    size_t nprobe = 1;
    auto queries = ColMajorMatrix<feature_type_type>{
        {{11, 11, 11}, {22, 22, 22}, {33, 33, 33}, {44, 44, 44}, {55, 55, 55}}};
    auto&& [scores, ids] =
        index.query(FeatureVectorArray(queries), top_k, nprobe);
    check_single_vector_equals(
        scores, ids, {0, 0, 0, 0, 0}, {11, 22, 33, 44, 55});

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

  // Load it at timestamp 99 and make sure we can query it correctly. Do this
  // with both a finite and infinite index.
  for (auto upper_bound : std::vector<size_t>{0, 4}) {
    auto temporal_policy = TemporalPolicy{TimeTravel, 99};
    auto load_strategy = upper_bound == 0 ? IndexLoadStrategy::PQ_INDEX :
                                            IndexLoadStrategy::PQ_OOC;
    auto index =
        IndexIVFPQ(ctx, index_uri, load_strategy, upper_bound, temporal_policy);
    CHECK(index.upper_bound() == upper_bound);

    CHECK(index.temporal_policy().timestamp_end() == 99);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.max_iterations() == max_iterations);
    CHECK(index.convergence_tolerance() == convergence_tolerance);
    CHECK(index.reassign_ratio() == reassign_ratio);

    size_t top_k = 1;
    size_t nprobe = 1;
    auto queries = ColMajorMatrix<feature_type_type>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}};

    auto&& [scores, ids] =
        index.query(FeatureVectorArray(queries), top_k, nprobe);
    check_single_vector_equals(scores, ids, {0, 0, 0, 0}, {1, 2, 3, 4});

    float k_factor = 2.f;
    auto&& [scores_with_reranking, ids_with_reranking] =
        index.query(FeatureVectorArray(queries), top_k, nprobe, k_factor);
    check_single_vector_equals(
        scores_with_reranking, ids_with_reranking, {0, 0, 0, 0}, {1, 2, 3, 4});

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(
        ctx, index_uri, load_strategy, upper_bound, temporal_policy);
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
  for (auto upper_bound : std::vector<size_t>{0, 4}) {
    auto temporal_policy = TemporalPolicy{TimeTravel, 0};
    auto load_strategy = upper_bound == 0 ? IndexLoadStrategy::PQ_INDEX :
                                            IndexLoadStrategy::PQ_OOC;

    auto index =
        IndexIVFPQ(ctx, index_uri, load_strategy, upper_bound, temporal_policy);
    CHECK(index.upper_bound() == upper_bound);

    CHECK(index.temporal_policy().timestamp_start() == 0);
    CHECK(index.temporal_policy().timestamp_end() == 0);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);
    CHECK(index.max_iterations() == max_iterations);
    CHECK(index.convergence_tolerance() == convergence_tolerance);
    CHECK(index.reassign_ratio() == reassign_ratio);

    size_t top_k = 1;
    size_t nprobe = 1;
    float k_factor = 1.9f;
    auto queries = ColMajorMatrix<feature_type_type>{{{1, 1, 1}}};
    auto&& [scores, ids] =
        index.query(FeatureVectorArray(queries), top_k, nprobe);
    check_single_vector_equals(
        scores,
        ids,
        {std::numeric_limits<float>::max()},
        {std::numeric_limits<uint32_t>::max()});
    auto&& [scores_with_reranking, ids_with_reranking] =
        index.query(FeatureVectorArray(queries), top_k, nprobe, k_factor);
    check_single_vector_equals(
        scores_with_reranking,
        ids_with_reranking,
        {std::numeric_limits<float>::max()},
        {std::numeric_limits<uint32_t>::max()});

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(
        ctx, index_uri, load_strategy, upper_bound, temporal_policy);
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

    CHECK(typed_index.group().get_max_iterations() == max_iterations);
    CHECK(
        typed_index.group().get_convergence_tolerance() ==
        convergence_tolerance);
    CHECK(typed_index.group().get_reassign_ratio() == reassign_ratio);
  }

  // Clear history for <= 99 and then load at 99, then make sure we cannot
  // query.
  IndexIVFPQ::clear_history(ctx, index_uri, 99);
  for (auto upper_bound : std::vector<size_t>{0, 3}) {
    auto temporal_policy = TemporalPolicy{TimeTravel, 99};
    auto load_strategy = upper_bound == 0 ? IndexLoadStrategy::PQ_INDEX :
                                            IndexLoadStrategy::PQ_OOC;

    auto index =
        IndexIVFPQ(ctx, index_uri, load_strategy, upper_bound, temporal_policy);
    CHECK(index.upper_bound() == upper_bound);

    CHECK(index.temporal_policy().timestamp_end() == 99);
    CHECK(index.feature_type_string() == feature_type);
    CHECK(index.id_type_string() == id_type);
    CHECK(index.partitioning_index_type_string() == partitioning_index_type);

    size_t top_k = 1;
    size_t nprobe = 1;
    float k_factor = 17.3f;
    auto queries = ColMajorMatrix<feature_type_type>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}};

    auto default_score = std::numeric_limits<float>::max();
    auto default_id = std::numeric_limits<uint32_t>::max();
    auto expected_scores = std::vector<float>{
        default_score, default_score, default_score, default_score};
    auto expected_ids =
        std::vector<uint32_t>{default_id, default_id, default_id, default_id};

    auto&& [scores, ids] =
        index.query(FeatureVectorArray(queries), top_k, nprobe);
    check_single_vector_equals(scores, ids, expected_scores, expected_ids);
    auto&& [scores_with_reranking, ids_with_reranking] =
        index.query(FeatureVectorArray(queries), top_k, nprobe, k_factor);
    check_single_vector_equals(
        scores_with_reranking,
        ids_with_reranking,
        expected_scores,
        expected_ids);

    auto typed_index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(
        ctx, index_uri, load_strategy, upper_bound, temporal_policy);
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

TEST_CASE("ingest_parts testing", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  using feature_type = float;

  uint64_t dimensions = 4;
  size_t partitions = 0;
  uint32_t num_subspaces = 2;

  IndexIVFPQ::create(
      ctx,
      index_uri,
      dimensions,
      TILEDB_FLOAT32,
      TILEDB_UINT32,
      TILEDB_UINT32,
      num_subspaces);

  auto temporal_policy = TemporalPolicy{TimeTravel, 100};
  auto vectors = ColMajorMatrix<feature_type>{
      {{1.0f, 1.1f, 1.2f, 1.3f}, {2.0f, 2.1f, 2.2f, 2.3f}}};

  std::string partial_write_array_dir = "temp";
  {
    auto index = IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0, temporal_policy);
    index.create_temp_data_group(partial_write_array_dir);
  }
  {
    auto index = IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0, temporal_policy);
    index.train(FeatureVectorArray(vectors), partitions, temporal_policy);
  }
  {
    auto index = IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0, temporal_policy);
    size_t part = 0;
    size_t part_end = 2;
    size_t part_id = 0;
    index.ingest_parts(
        FeatureVectorArray(vectors),
        FeatureVector(0, "uint64"),
        FeatureVector(0, "uint64"),
        part,
        part_end,
        part_id,
        partial_write_array_dir);
  }
  {
    // Here is where Python does compute_partition_indexes_udf. We simulate that
    // by copying the indices from temp to permanent.
    auto group = ivf_pq_group<ivf_pq_index<feature_type>>(
        ctx, index_uri, TILEDB_READ, temporal_policy);
    auto total_partitions = 2;
    auto indexes = read_vector<uint32_t>(
        ctx,
        group.feature_vectors_index_temp_uri(partial_write_array_dir),
        0,
        total_partitions,
        temporal_policy);
    write_vector(
        ctx,
        indexes,
        group.feature_vectors_index_uri(),
        0,
        false,
        temporal_policy);
  }

  {
    auto index = IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0, temporal_policy);
    size_t partitions = 1;
    size_t work_items = 1;
    size_t partition_id_start = 0;
    size_t partition_id_end = 1;
    size_t batch = 33554432;
    index.consolidate_partitions(
        partitions,
        work_items,
        partition_id_start,
        partition_id_end,
        batch,
        partial_write_array_dir);
  }

  {
    auto index = IndexIVFPQ(ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0);
    auto&& [scores, ids] = index.query(FeatureVectorArray(vectors), 1, 1);
    check_single_vector_equals(scores, ids, {0, 0}, {0, 1});
  }
}

TEST_CASE("train python", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  using feature_type = uint8_t;

  uint64_t dimensions = 6;
  uint32_t num_subspaces = dimensions;
  uint32_t max_iterations = 2;
  float convergence_tolerance = 0.000025f;
  float reassign_ratio = 0.075f;

  IndexIVFPQ::create(
      ctx,
      index_uri,
      dimensions,
      TILEDB_UINT8,
      TILEDB_UINT64,
      TILEDB_UINT64,
      num_subspaces,
      max_iterations,
      convergence_tolerance,
      reassign_ratio,
      std::make_optional<TemporalPolicy>(TemporalPolicy{TimeTravel, 0}));

  auto vectors = ColMajorMatrix<feature_type>{{
      {7, 6, 249, 3, 2, 2},
      {254, 249, 7, 0, 9, 3},
      {248, 255, 4, 0, 249, 0},
      {251, 249, 245, 3, 250, 252},
      {249, 248, 250, 0, 5, 251},
      {254, 7, 1, 4, 1, 2},
      {5, 254, 2, 0, 253, 255},
      {3, 5, 250, 245, 249, 0},
      {250, 252, 6, 7, 252, 5},
      {4, 5, 9, 9, 248, 252},
  }};
  {
    size_t partitions = 2;
    auto temporal_policy = TemporalPolicy{TimeTravel, 1};
    auto index = IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0, temporal_policy);
    index.train(FeatureVectorArray(vectors), partitions, temporal_policy);
    index.ingest(FeatureVectorArray(vectors));
  }

  {
    size_t partitions = 3;
    auto temporal_policy = TemporalPolicy{TimeTravel, 5};
    auto index = IndexIVFPQ(
        ctx, index_uri, IndexLoadStrategy::PQ_INDEX, 0, temporal_policy);
    index.train(FeatureVectorArray(vectors), partitions, temporal_policy);
  }
}
