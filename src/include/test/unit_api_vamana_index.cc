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
#include "index/vamana_index.h"

// TEST_CASE("api_vamana_index: test test", "[api_vamana_index]") {
//   REQUIRE(true);
// }

// TEST_CASE("api_vamana_index: init constructor", "[api_vamana_index]") {
//   SECTION("default") {
//     auto a = IndexVamana();
//     CHECK(a.feature_type() == TILEDB_ANY);
//     CHECK(a.feature_type_string() == datatype_to_string(TILEDB_ANY));
//     CHECK(a.id_type() == TILEDB_UINT32);
//     CHECK(a.id_type_string() == datatype_to_string(TILEDB_UINT32));
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
//     CHECK(
//         a.adjacency_row_index_type_string() ==
//         datatype_to_string(TILEDB_UINT32));
//     CHECK(dimension(a) == 0);
//   }

//   SECTION("float uint32 uint32") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "float32"},
//          {"id_type", "uint32"},
//          {"adjacency_row_index_type", "uint32"}}));
//     CHECK(a.feature_type() == TILEDB_FLOAT32);
//     CHECK(a.id_type() == TILEDB_UINT32);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
//     CHECK(dimension(a) == 0);
//   }

//   SECTION("uint8 uint32 uint32") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "uint8"},
//          {"id_type", "uint32"},
//          {"adjacency_row_index_type", "uint32"}}));
//     CHECK(a.feature_type() == TILEDB_UINT8);
//     CHECK(a.id_type() == TILEDB_UINT32);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
//   }

//   SECTION("float uint64 uint32") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "float32"},
//          {"id_type", "uint64"},
//          {"adjacency_row_index_type", "uint32"}}));
//     CHECK(a.feature_type() == TILEDB_FLOAT32);
//     CHECK(a.id_type() == TILEDB_UINT64);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
//   }

//   SECTION("float uint32 uint64") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "float32"},
//          {"id_type", "uint32"},
//          {"adjacency_row_index_type", "uint64"}}));
//     CHECK(a.feature_type() == TILEDB_FLOAT32);
//     CHECK(a.id_type() == TILEDB_UINT32);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
//   }

//   SECTION("uint8 uint64 uint32") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "uint8"},
//          {"id_type", "uint64"},
//          {"adjacency_row_index_type", "uint32"}}));
//     CHECK(a.feature_type() == TILEDB_UINT8);
//     CHECK(a.id_type() == TILEDB_UINT64);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
//   }

//   SECTION("uint8 uint32 uint64") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "uint8"},
//          {"id_type", "uint32"},
//          {"adjacency_row_index_type", "uint64"}}));
//     CHECK(a.feature_type() == TILEDB_UINT8);
//     CHECK(a.id_type() == TILEDB_UINT32);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
//   }

//   SECTION("float uint64 uint64") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "float32"},
//          {"id_type", "uint64"},
//          {"adjacency_row_index_type", "uint64"}}));
//     CHECK(a.feature_type() == TILEDB_FLOAT32);
//     CHECK(a.id_type() == TILEDB_UINT64);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
//   }

//   SECTION("uint8 uint64 uint64") {
//     auto a = IndexVamana(std::make_optional<IndexOptions>(
//         {{"feature_type", "uint8"},
//          {"id_type", "uint64"},
//          {"adjacency_row_index_type", "uint64"}}));
//     CHECK(a.feature_type() == TILEDB_UINT8);
//     CHECK(a.id_type() == TILEDB_UINT64);
//     CHECK(a.adjacency_row_index_type() == TILEDB_UINT64);
//   }
// }

// TEST_CASE("api_vamana_index: infer feature type", "[api_vamana_index]") {
//   auto a = IndexVamana(std::make_optional<IndexOptions>(
//       {{"id_type", "uint32"}, {"adjacency_row_index_type", "uint32"}}));
//   auto ctx = tiledb::Context{};
//   auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
//   a.train(training_set);
//   CHECK(a.feature_type() == TILEDB_FLOAT32);
//   CHECK(a.id_type() == TILEDB_UINT32);
//   CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
// }

// TEST_CASE("api_vamana_index: infer dimension", "[api_vamana_index]") {
//   auto a = IndexVamana(std::make_optional<IndexOptions>(
//       {{"id_type", "uint32"}, {"adjacency_row_index_type", "uint32"}}));
//   auto ctx = tiledb::Context{};
//   auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
//   CHECK(dimension(a) == 0);
//   a.train(training_set);
//   CHECK(a.feature_type() == TILEDB_FLOAT32);
//   CHECK(a.id_type() == TILEDB_UINT32);
//   CHECK(a.adjacency_row_index_type() == TILEDB_UINT32);
//   CHECK(dimension(a) == 128);
// }

// TEST_CASE(
//     "api_vamana_index: api_vamana_index write and read",
//     "[api_vamana_index]") {
//   auto ctx = tiledb::Context{};
//   std::string api_vamana_index_uri =
//       (std::filesystem::temp_directory_path() / "api_vamana_index").string();

//   auto a = IndexVamana(std::make_optional<IndexOptions>(
//       {{"feature_type", "float32"},
//        {"id_type", "uint32"},
//        {"adjacency_row_index_type", "uint32"}}));
//   auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
//   a.train(training_set);
//   a.add(training_set);
//   a.write_index(ctx, api_vamana_index_uri, true);

//   auto b = IndexVamana(ctx, api_vamana_index_uri);

//   CHECK(dimension(a) == dimension(b));
//   CHECK(a.feature_type() == b.feature_type());
//   CHECK(a.id_type() == b.id_type());
//   CHECK(a.adjacency_row_index_type() == b.adjacency_row_index_type());
// }

// TEST_CASE(
//     "api_vamana_index: build index and query",
//     "[api_vamana_index]") {
//   auto ctx = tiledb::Context{};
//   size_t k_nn = 10;
//   size_t nprobe = GENERATE(8, 32);

//   auto a = IndexVamana(std::make_optional<IndexOptions>(
//       {{"id_type", "uint32"}, {"adjacency_row_index_type", "uint32"}}));
//   auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
//   auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
//   auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);
//   a.train(training_set);
//   a.add(training_set);

//   auto&& [s, t] = a.query(query_set, k_nn);

//   auto intersections = count_intersections(t, groundtruth_set, k_nn);
//   auto nt = num_vectors(t);
//   auto recall = ((double)intersections) / ((double)nt * k_nn);
//   CHECK(recall == 1.0);
// }

// TEST_CASE(
//     "api_vamana_index: read index and query",
//     "[api_vamana_index][ci-skip]") {
//   auto ctx = tiledb::Context{};
//   size_t k_nn = 10;
//   size_t nprobe = GENERATE(8, 32);

//   std::string api_vamana_index_uri =
//       (std::filesystem::temp_directory_path() / "api_vamana_index").string();

//   auto a = IndexVamana(std::make_optional<IndexOptions>(
//       {{"feature_type", "float32"},
//        {"id_type", "uint32"},
//        {"adjacency_row_index_type", "uint32"}}));

//   auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
//   a.train(training_set);
//   a.add(training_set);
//   a.write_index(ctx, api_vamana_index_uri, true);
//   auto b = IndexVamana(ctx, api_vamana_index_uri);

//   auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
//   auto groundtruth_set = FeatureVectorArray(ctx, siftsmall_groundtruth_uri);

//   auto&& [s, t] = a.query(query_set, k_nn);
//   auto&& [u, v] = b.query(query_set, k_nn);

//   auto intersections_a = count_intersections(t, groundtruth_set, k_nn);
//   auto intersections_b = count_intersections(v, groundtruth_set, k_nn);
//   CHECK(intersections_a == intersections_b);
//   auto nt = num_vectors(t);
//   auto nv = num_vectors(v);
//   CHECK(nt == nv);
//   auto recall = ((double)intersections_a) / ((double)nt * k_nn);
//   CHECK(recall == 1.0);
// }

//TEST_CASE("api_vamana_index: create index", "[api_vamana_index]") {
//    tiledb::Context ctx;
//    tiledb::Config cfg;
//
//    std::string index_uri = (std::filesystem::temp_directory_path() / "api_vamana_index").string();
//    tiledb::VFS vfs(ctx);
//    if (vfs.is_dir(index_uri)) {
//        vfs.remove_dir(index_uri);
//    }
//
//    auto feature_type = "float32";
//    auto id_type = "uint32";
//    size_t dimensions = 3;
//    size_t num_vectors = 0;
//
//    auto type_erased_index = IndexVamana(std::make_optional<IndexOptions>(
//            {{"feature_type", feature_type},
//             {"id_type", id_type},
//             {"adjacency_row_index_type", "uint32"}}));
//    auto empty_training_set = FeatureVectorArray(dimensions, num_vectors, feature_type, id_type);
//    std::cout << "[unit_api_vamana_index] type_erased_index.train();" << std::endl;
//    type_erased_index.train(empty_training_set);
//    std::cout << "[unit_api_vamana_index] type_erased_index.add();" << std::endl;
//    type_erased_index.add(empty_training_set);
//    std::cout << "[unit_api_vamana_index] type_erased_index.write_index();" << std::endl;
//    type_erased_index.write_index(ctx, index_uri, true);
//
//    std::cout << "[unit_api_vamana_index] metadata fetch" << std::endl;
//    {
//        auto timestamp = 0;
//        auto index = vamana_index<float, uint32_t, uint32_t>(ctx, index_uri);
//        auto group = vamana_index_group<vamana_index<float, uint32_t, uint32_t>>(index, ctx, index_uri, TILEDB_READ, timestamp);
//        std::cout << "group\n";
//        group.dump();
//    }
//
//    std::cout << "--------------------------- SECOND TIME\n";
//    {
//        auto matrix = ColMajorMatrix<float>{{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};
//        auto training_set = FeatureVectorArray(matrix);
//        std::cout << "[unit_api_vamana_index] type_erased_index.train();" << std::endl;
//        type_erased_index.train(training_set);
//        std::cout << "[unit_api_vamana_index] type_erased_index.add();" << std::endl;
//        type_erased_index.add(training_set);
//        std::cout << "[unit_api_vamana_index] type_erased_index.write_index();" << std::endl;
//        type_erased_index.write_index(ctx, index_uri, true);
//
//        std::cout << "[unit_api_vamana_index] metadata fetch" << std::endl;
//        auto timestamp = 0;
//        auto index = vamana_index<float, uint32_t, uint32_t>(ctx, index_uri);
//        auto group = vamana_index_group<vamana_index<float, uint32_t, uint32_t>>(index, ctx, index_uri, TILEDB_READ, timestamp);
//        std::cout << "group\n";
//        group.dump();
//    }
//}

TEST_CASE("api_vamana_index: py index", "[api_vamana_index]") {
    auto ctx = tiledb::Context{};
//    auto uri_before_write_index = "/private/var/folders/jb/5gq49wh97wn0j7hj6zfn9pzh0000gn/T/pytest-of-parismorgan/pytest-469/test_vamana_index0/array";
//    auto uri_after_first_to_metatadata_lists = "/private/var/folders/jb/5gq49wh97wn0j7hj6zfn9pzh0000gn/T/pytest-of-parismorgan/pytest-472/test_vamana_index0/array";
//    auto uri_while_double_writing_ingestion_timestamps = "/private/var/folders/jb/5gq49wh97wn0j7hj6zfn9pzh0000gn/T/pytest-of-parismorgan/pytest-489/test_vamana_index0/array";
    auto uri = "/private/var/folders/jb/5gq49wh97wn0j7hj6zfn9pzh0000gn/T/pytest-of-parismorgan/pytest-491/test_vamana_index0/array";
    auto b = IndexVamana(ctx, uri);
}

//TEST_CASE("api_vamana_index: empty index", "[api_vamana_index]") {
//  auto ctx = tiledb::Context{};
//  size_t k_nn = 10;
//  size_t nprobe = GENERATE(8, 32);
//
//  // SECTION("Read from Python") {
//  //   std::string uri = "/tmp/vamana_index";
//  //   auto b = IndexVamana(ctx, uri);
//  // }
//
//  std::string api_vamana_index_uri =
//      (std::filesystem::temp_directory_path() / "api_vamana_index").string();
//  std::cout << "api_vamana_index_uri " << api_vamana_index_uri << std::endl;
//  auto feature_type = "float32";
//  auto id_type = "uint32";
//  SECTION("Create empty index") {
//    auto a = IndexVamana(std::make_optional<IndexOptions>(
//        {{"feature_type", feature_type},
//         {"id_type", id_type},
//         {"adjacency_row_index_type", "uint32"}}));
//
//    size_t dimensions = sift_dimension;
//    size_t num_vectors = 0;
//    auto empty_training_set =
//        FeatureVectorArray(dimensions, num_vectors, feature_type, id_type);
//    a.train(empty_training_set);
//    a.add(empty_training_set);
//    a.write_index(ctx, api_vamana_index_uri, true);
//
//    auto b = IndexVamana(ctx, api_vamana_index_uri);
//  }
//  //  SECTION("Read the empty index, retrain it, and query it") {
//  //      auto b = IndexVamana(ctx, api_vamana_index_uri);
//  //      auto training_set = FeatureVectorArray(ctx, siftsmall_inputs_uri);
//  //      b.train(training_set);
//  //      b.add(training_set);
//  //      b.write_index(ctx, api_vamana_index_uri, true);
//  //
//  //      auto query_set = FeatureVectorArray(ctx, siftsmall_query_uri);
//  //      auto groundtruth_set = FeatureVectorArray(ctx,
//  //      siftsmall_groundtruth_uri); auto&& [scores, ids] = b.query(query_set,
//  //      k_nn); auto intersections = count_intersections(ids, groundtruth_set,
//  //      k_nn); auto num_ids = num_vectors(ids); auto recall =
//  //      ((double)intersections) / ((double)num_ids * k_nn); CHECK(recall
//  //      == 1.0);
//  //  }
//}
