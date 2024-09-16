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
    CHECK(a.distance_metric() == DistanceMetric::SUM_OF_SQUARES);
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
  uint64_t dimensions = 3;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  auto queries = ColMajorMatrix<feature_type_type>{
      {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}}};

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
    index->add(training_vector_array);
    index->write_index(ctx, index_uri);

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
  uint64_t dimensions = 3;
  uint32_t num_subspaces = 1;
  auto distance_metric = DistanceMetric::L2;

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
         {"dimensions", std::to_string(dimensions)},
         {"num_subspaces", std::to_string(num_subspaces)},
         {"distance_metric",
          std::to_string(static_cast<size_t>(distance_metric))}}));

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
    auto training = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
        {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}, {10, 11, 12, 13}};

    auto training_vector_array = FeatureVectorArray(training);
    index.train(training_vector_array);
    index.add(training_vector_array);
    index.write_index(ctx, index_uri);

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
  size_t n_list = 100;
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
         {"n_list", std::to_string(n_list)},
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
    CHECK(index.distance_metric() == DistanceMetric::SUM_OF_SQUARES);
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
             {1, .4f, .45f},
             {2, .5f, .6f},
             {5, .7f, .7f},
             {10, .75f, .9f},
             {100, .8f, 1.f}}) {
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

  auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
  auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);

  std::unique_ptr<IndexIVFPQ> b;
  SECTION("infinite") {
    b = std::make_unique<IndexIVFPQ>(ctx, api_ivf_pq_index_uri);
  }
  SECTION("finite") {
    size_t upper_bound = GENERATE(500, 1000);
    b = std::make_unique<IndexIVFPQ>(
        ctx, api_ivf_pq_index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
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
  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  uint64_t dimensions = 3;

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

TEST_CASE("clear history with an open index", "[api_ivf_pq_index]") {
  auto ctx = tiledb::Context{};
  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  uint64_t dimensions = 3;
  size_t n_list = 1;
  uint32_t num_subspaces = 1;
  float convergence_tolerance = 0.00003f;
  uint32_t max_iterations = 3;

  std::string index_uri =
      (std::filesystem::temp_directory_path() / "api_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  auto index = IndexIVFPQ(std::make_optional<IndexOptions>(
      {{"feature_type", feature_type},
       {"id_type", id_type},
       {"partitioning_index_type", partitioning_index_type},
       {"n_list", std::to_string(n_list)},
       {"num_subspaces", std::to_string(num_subspaces)},
       {"convergence_tolerance", std::to_string(convergence_tolerance)},
       {"max_iterations", std::to_string(max_iterations)}}));

  auto training = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
      {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}, {1, 2, 3, 4}};
  auto training_vector_array = FeatureVectorArray(training);
  index.train(training_vector_array);
  index.add(training_vector_array);
  index.write_index(ctx, index_uri, TemporalPolicy(TimeTravel, 99));

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
  size_t n_list = 1;
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
    auto index = IndexIVFPQ(std::make_optional<IndexOptions>({
        {"feature_type", feature_type},
        {"id_type", id_type},
        {"partitioning_index_type", partitioning_index_type},
        {"n_list", std::to_string(n_list)},
        {"num_subspaces", std::to_string(num_subspaces)},
        {"max_iterations", std::to_string(max_iterations)},
        {"convergence_tolerance", std::to_string(convergence_tolerance)},
        {"reassign_ratio", std::to_string(reassign_ratio)},
    }));

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
    CHECK(index.max_iterations() == max_iterations);
    CHECK(index.convergence_tolerance() == convergence_tolerance);
    CHECK(index.reassign_ratio() == reassign_ratio);
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
