/**
 * @file   unit_array_defs.cc
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
 *
 */

#include <catch2/catch_all.hpp>
#include <filesystem>
#include <tiledb/tiledb>
#include <vector>
#include "test/utils/array_defs.h"

#include "detail/flat/qv.h"

TEST_CASE("test array root uris", "[array_defs]") {
  for (const auto& uri : {siftsmall_root, bigann10k_root}) {
    REQUIRE(std::filesystem::is_directory(uri));
  }
  for (const auto& uri : {siftsmall_files_root}) {
    REQUIRE(std::filesystem::is_directory(uri));
  }
}

std::vector<std::string> siftsmall_array_uris{
    siftsmall_inputs_uri,
    siftsmall_centroids_uri,
    siftsmall_index_uri,
    siftsmall_ids_uri,
    siftsmall_parts_uri,
    siftsmall_query_uri,
    siftsmall_groundtruth_uri,
};

// Note that we don't have a canonical IVF index for bigann10k yet, so some
// of these URIs are placeholders
std::vector<std::string> bigann10k_array_uris{
    bigann10k_inputs_uri,
    bigann10k_centroids_uri,
    bigann10k_index_uri,
    bigann10k_ids_uri,
    bigann10k_parts_uri,
    bigann10k_query_uri,
    bigann10k_groundtruth_uri,
};

std::vector<std::string> siftsmall_files{
    siftsmall_inputs_file,
    siftsmall_query_file,
    siftsmall_groundtruth_file,
};

TEST_CASE("test array uris", "[array_defs]") {
  bool debug = false;

  for (const auto& test : {siftsmall_array_uris, bigann10k_array_uris}) {
    for (const auto& uri : test) {
      if (debug) {
        std::cout << uri << " "
                  << (std::filesystem::exists(uri) ? "exists" :
                                                     "does not exist");
        std::cout << " and is directory is "
                  << std::filesystem::is_directory(uri) << " ";
        std::cout << std::endl;
      }
      // CHECK(std::filesystem::exists(uri));

      CHECK(std::filesystem::is_directory(uri));
    }
  }

  for (const auto& file : siftsmall_files) {
    if (debug) {
      std::cout << file << " "
                << (std::filesystem::exists(file) ? "exists" : "does not exist")
                << std::endl;
    }
    CHECK(std::filesystem::is_regular_file(file));
  }
}

TEST_CASE("compare siftsmall arrays and files", "[array_defs]") {
  tiledb::Context ctx;

  auto array_inputs = tdbColMajorPreLoadMatrix<siftsmall_feature_type>(
      ctx, siftsmall_inputs_uri);
  auto array_queries = tdbColMajorPreLoadMatrix<siftsmall_feature_type>(
      ctx, siftsmall_query_uri);
  auto array_groundtruth = tdbColMajorPreLoadMatrix<siftsmall_groundtruth_type>(
      ctx, siftsmall_groundtruth_uri);

  auto file_inputs =
      read_bin_local<siftsmall_feature_type>(ctx, siftsmall_inputs_file);
  auto file_queries =
      read_bin_local<siftsmall_feature_type>(ctx, siftsmall_query_file);
  auto file_groundtruth =
      read_bin_local<uint32_t>(ctx, siftsmall_groundtruth_file);

  auto file_groundtruth_64 = ColMajorMatrix<siftsmall_groundtruth_type>(
      file_groundtruth.num_rows(), file_groundtruth.num_cols());

  std::copy(
      file_groundtruth.raveled().begin(),
      file_groundtruth.raveled().end(),
      file_groundtruth_64.raveled().begin());

  CHECK(file_groundtruth_64 == file_groundtruth);

  CHECK(array_inputs == file_inputs);
  CHECK(array_queries == file_queries);

  size_t intersections00 =
      count_intersections(file_groundtruth_64, array_groundtruth, 100);
  CHECK(intersections00 != 0);
  size_t expected00 = array_groundtruth.num_cols() * 100;
  CHECK(intersections00 == expected00);
}
