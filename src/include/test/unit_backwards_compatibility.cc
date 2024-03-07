/**
 * @file   unit_backwards_compatibility.cc
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
 * Test that the current codebase can read and query old indexes.
 *
 */

#include <catch2/catch_all.hpp>
#include <iostream>
#include "api/feature_vector_array.h"
#include "api/flat_l2_index.h"
#include "api/ivf_flat_index.h"
#include "api/vamana_index.h"
#include "array_defs.h"
#include "detail/linalg/matrix.h"
#include "index/flat_l2_index.h"
#include "index/ivf_flat_index.h"
#include "index/vamana_index.h"
#include "mdspan/mdspan.hpp"
#include "utils/print_types.h"

TEST_CASE("backwards_compatibility: test test", "[backwards_compatibility]") {
  REQUIRE(true);
}

const float MINIMUM_ACCURACY = 0.85;

//   ColMajorMatrix<siftsmall_feature_type> groundTruthMatrix;
// //  std::copy(query_indices.begin(), query_indices.end(),
// groundTruthMatrix.data());
//   //    std::cout << "queries" << std::endl;
//   //    for (auto vec: queries) {
//   //        std::cout << "[";
//   //        for (auto el: vec) {
//   //            std::cout << el << ",";
//   //        }
//   //        std::cout << "]" << std::endl;
//   //    }
// std::filesystem::path backwards_compatibility_path =
// std::filesystem::current_path().parent_path().parent_path().parent_path() /
// "backwards-compatibility-data"; std::cout << backwards_compatibility_path
// << std::endl; return;
//    std::vector<std::vector<float>> base = {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}};

TEST_CASE(
    "backwards_compatibility: test_query_old_indices",
    "[backwards_compatibility]") {
  tiledb::Context ctx;
  std::string datasets_path = backwards_compatibility_root / "data";
  auto base =
      read_bin_local<siftsmall_feature_type>(ctx, siftmicro_inputs_file);

  std::vector<size_t> query_indices = {0,  3,  4,  8,  10, 19, 28, 31,
                                       39, 40, 41, 47, 49, 50, 56, 64,
                                       68, 70, 71, 79, 82, 89, 90, 94};
  std::vector<std::span<siftsmall_feature_type>> queries;
  for (size_t idx : query_indices) {
    queries.push_back(base[idx]);
  }

  //  auto num_rows = query_indices.size();
  //  auto num_cols = queries[0].size();
  //  std::cout << "num_rows " << num_rows << std::endl;
  //  std::cout << "num_cols " << num_cols << std::endl;
  auto queries_matrix = ColMajorMatrix<siftsmall_feature_type>(
      queries[0].size(), query_indices.size());
  for (size_t i = 0; i < query_indices.size(); ++i) {
    for (size_t j = 0; j < queries[0].size(); ++j) {
      queries_matrix(j, i) = queries[i][j];
    }
  }
  debug(queries_matrix, "after");
  auto queries_feature_vector_array = FeatureVectorArray(queries_matrix);
  std::cout << "queries_feature_vector_array.num_vectors() "
            << queries_feature_vector_array.num_vectors() << std::endl;
  std::cout << "queries_feature_vector_array.dimension() "
            << queries_feature_vector_array.dimension() << std::endl;
  std::cout << "extents(queries_feature_vector_array)[0] "
            << extents(queries_feature_vector_array)[0] << std::endl;
  std::cout << "extents(queries_feature_vector_array)[1] "
            << extents(queries_feature_vector_array)[1] << std::endl;
  std::cout << "feature_type_string "
            << queries_feature_vector_array.feature_type_string() << std::endl;

  for (const auto& directory_name :
       std::filesystem::directory_iterator(datasets_path)) {
    std::string version_path = directory_name.path();
    if (!std::filesystem::is_directory(version_path)) {
      continue;
    }

    for (const auto& index_name :
         std::filesystem::directory_iterator(version_path)) {
      std::string index_uri = index_name.path();
      if (!std::filesystem::is_directory(index_uri)) {
        continue;
      }
      // TODO(paris): Fix bug where we can't load old indexes b/c of a storage
      // version mismatch in metadata.
      if (index_uri.find("0.0.10") != std::string::npos ||
          index_uri.find("0.0.17") != std::string::npos ||
          index_uri.find("0.0.21") != std::string::npos) {
        continue;
      }
      std::cout << "index_uri: " << index_uri << std::endl;
      std::vector<float> expected_distances(query_indices.size(), 0.0);
      if (index_uri.find("ivf_flat") != std::string::npos) {
        std::cout << "  ivf" << std::endl;
        // auto index = ivf_flat_index<float, uint64_t, uint64_t>(ctx,
        // index_uri); auto&& [scores, ids] =
        // index.query_infinite_ram(queries_matrix, 1, 10);
        auto index = IndexIVFFlat(ctx, index_uri);
        auto&& [scores, ids] =
            index.query_infinite_ram(queries_feature_vector_array, 1, 10);
        std::cout << "scores.num_vectors() " << scores.num_vectors()
                  << std::endl;
        std::cout << "scores.dimension() " << scores.dimension() << std::endl;
        std::cout << "extents(scores)[0] " << extents(scores)[0] << std::endl;
        std::cout << "extents(scores)[1] " << extents(scores)[1] << std::endl;
        std::cout << "feature_type_string " << scores.feature_type_string()
                  << std::endl;

        std::cout << "ids.num_vectors() " << ids.num_vectors() << std::endl;
        std::cout << "ids.dimension() " << ids.dimension() << std::endl;
        std::cout << "extents(ids)[0] " << extents(ids)[0] << std::endl;
        std::cout << "extents(ids)[1] " << extents(ids)[1] << std::endl;
        std::cout << "feature_type_string " << ids.feature_type_string()
                  << std::endl;

        auto scores_span =
            MatrixView<siftsmall_feature_type, stdx::layout_left>{
                (siftsmall_feature_type*)scores.data(),
                extents(scores)[0],
                extents(scores)[1]};
        debug(scores_span, "scores_span");

        auto ids_span = MatrixView<siftsmall_ids_type, stdx::layout_left>{
            (siftsmall_ids_type*)ids.data(), extents(ids)[0], extents(ids)[1]};
        debug(ids_span, "ids_span");

        for (size_t i = 0; i < query_indices.size(); ++i) {
          CHECK(ids_span[0][i] == query_indices[i]);
          CHECK(scores_span[0][i] == 0);
          std::cout << "id: " << ids_span[0][i]
                    << " score: " << scores_span[0][i] << " " << std::endl;
        }
        //          count_intersections(scores, )
        //                  CHECK(std::equal(expected_distances.begin(),
        //                  expected_distances.end(), scores.data()));
        //                  CHECK(std::equal(query_indices.begin(),
        //                  query_indices.end(), ids.data()));
      } else if (index_uri.find("flat") != std::string::npos) {
        // TODO(paris): Fix flat_l2_index and re-enable. Right now it just tries
        // to load the URI as a tdbMatrix.
        // auto index = flat_l2_index<float>(ctx, index_uri);
        //  auto&& [scores, ids] = index.query(queries_matrix, 1);
        //  CHECK(std::equal(expected_distances.begin(),
        //  expected_distances.end(), scores.data()));
        //  CHECK(std::equal(query_indices.begin(), query_indices.end(),
        //  ids.data()));
      } else if (index_uri.find("vamana") != std::string::npos) {
        auto index = vamana_index<float, uint64_t>(ctx, index_uri);
        //  auto&& [scores, ids] = index.query(queries_matrix, 1);
        //  CHECK(std::equal(expected_distances.begin(),
        //  expected_distances.end(), scores.data()));
        //  CHECK(std::equal(query_indices.begin(), query_indices.end(),
        //  ids.data()));
      } else {
        REQUIRE(false);
      }
    }
  }
}
