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
#include "detail/linalg/matrix.h"
#include "index/flat_l2_index.h"
#include "index/ivf_flat_index.h"
#include "index/vamana_index.h"
#include "mdspan/mdspan.hpp"
#include "test/utils/array_defs.h"
#include "utils/print_types.h"

TEST_CASE("test_query_old_indices", "[backwards_compatibility]") {
  tiledb::Context ctx;
  tiledb::Config cfg;

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

  auto queries_matrix = ColMajorMatrix<siftsmall_feature_type>(
      queries[0].size(), query_indices.size());
  for (size_t i = 0; i < query_indices.size(); ++i) {
    for (size_t j = 0; j < queries[0].size(); ++j) {
      queries_matrix(j, i) = queries[i][j];
    }
  }
  auto queries_feature_vector_array = FeatureVectorArray(queries_matrix);

  for (const auto& directory_name :
       std::filesystem::directory_iterator(datasets_path)) {
    std::string version_path = directory_name.path().string();
    if (!std::filesystem::is_directory(version_path)) {
      continue;
    }

    for (const auto& index_name :
         std::filesystem::directory_iterator(version_path)) {
      std::string index_uri = index_name.path().string();
      if (!std::filesystem::is_directory(index_uri)) {
        continue;
      }
      // TODO(paris): Fix bug where we can't load old indexes b/c of a storage
      // version mismatch in metadata.
      if (index_uri.find("0.0.10") != std::string::npos ||
          index_uri.find("0.0.17") != std::string::npos) {
        continue;
      }

      auto read_group = tiledb::Group(ctx, index_uri, TILEDB_READ, cfg);
      std::vector<float> expected_distances(query_indices.size(), 0.0);
      if (index_uri.find("ivf_flat") != std::string::npos) {
        // First check that we can query the index.
        auto index = IndexIVFFlat(ctx, index_uri);
        auto&& [scores, ids] =
            index.query_infinite_ram(queries_feature_vector_array, 1, 10);
        auto scores_span =
            MatrixView<siftsmall_feature_type, stdx::layout_left>{
                (siftsmall_feature_type*)scores.data(),
                extents(scores)[0],
                extents(scores)[1]};

        auto ids_span = MatrixView<siftsmall_ids_type, stdx::layout_left>{
            (siftsmall_ids_type*)ids.data(), extents(ids)[0], extents(ids)[1]};

        for (size_t i = 0; i < query_indices.size(); ++i) {
          CHECK(ids_span[0][i] == query_indices[i]);
          CHECK(scores_span[0][i] == 0);
        }

        // Next check that we can load the metadata.
        auto metadata = ivf_flat_index_metadata();
        metadata.load_metadata(read_group);
      } else if (index_uri.find("flat") != std::string::npos) {
        // TODO(paris): Fix flat_l2_index and re-enable. Right now it just tries
        // to load the URI as a tdbMatrix.
      } else if (index_uri.find("vamana") != std::string::npos) {
        // First check that we can query the index.
        auto index = IndexVamana(ctx, index_uri);
        auto&& [scores, ids] = index.query(queries_feature_vector_array, 1);
        auto scores_span =
            MatrixView<siftsmall_feature_type, stdx::layout_left>{
                (siftsmall_feature_type*)scores.data(),
                extents(scores)[0],
                extents(scores)[1]};

        auto ids_span = MatrixView<siftsmall_ids_type, stdx::layout_left>{
            (siftsmall_ids_type*)ids.data(), extents(ids)[0], extents(ids)[1]};

        for (size_t i = 0; i < query_indices.size(); ++i) {
          CHECK(ids_span[0][i] == query_indices[i]);
          CHECK(scores_span[0][i] == 0);
        }

        // Next check that we can load the metadata.
        auto metadata = vamana_index_metadata();
        metadata.load_metadata(read_group);
      } else {
        REQUIRE(false);
      }
    }
  }
}
