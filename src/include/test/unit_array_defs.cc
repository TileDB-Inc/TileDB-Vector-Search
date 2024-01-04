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
#include "array_defs.h"

TEST_CASE("array_defs: test test", "[array_defs]") {
  REQUIRE(true);
}

std::vector<std::filesystem::path> test_array_roots{
    sift_root,
    siftsmall_root,
    bigann1M_root,
    bigann10k_root,
    fmnist_root,
    diskann_root,
};

std::vector<std::filesystem::path> test_file_roots{
    siftsmall_files_root,
};

TEST_CASE("array_defs: test array root uris", "[array_defs]") {
  for (auto& uri : {siftsmall_root, bigann10k_root}) {
    REQUIRE(std::filesystem::is_directory(uri));
  }
  for (auto& uri : {siftsmall_files_root}) {
    REQUIRE(std::filesystem::is_directory(uri));
  }
}

std::vector<std::filesystem::path> sift_array_uris{
    sift_inputs_uri,
    sift_centroids_uri,
    sift_index_uri,
    sift_ids_uri,
    sift_parts_uri,
    sift_query_uri,
    sift_groundtruth_uri,
};

std::vector<std::filesystem::path> siftsmall_array_uris{
    siftsmall_inputs_uri,
    siftsmall_centroids_uri,
    siftsmall_index_uri,
    siftsmall_ids_uri,
    siftsmall_parts_uri,
    siftsmall_query_uri,
    siftsmall_groundtruth_uri,
};

std::vector<std::filesystem::path> bigann10k_array_uris{
    bigann10k_inputs_uri,
    bigann10k_centroids_uri,
    bigann10k_index_uri,
    bigann10k_ids_uri,
    bigann10k_parts_uri,
    bigann10k_query_uri,
    bigann10k_groundtruth_uri,
};

std::vector<std::filesystem::path> bigann1M_array_uris{
    bigann1M_inputs_uri,
    bigann1M_centroids_uri,
    bigann1M_index_uri,
    bigann1M_ids_uri,
    bigann1M_parts_uri,
    bigann1M_query_uri,
    bigann1M_groundtruth_uri,
};

std::vector<std::filesystem::path> fmnist_array_uris{
    // Note that we don't have a canonical IVF index for fmnist yet, so some
    // of these URIs are placeholders
    fmnist_inputs_uri,
    fmnist_centroids_uri,
    fmnist_index_uri,
    fmnist_ids_uri,
    fmnist_parts_uri,
    fmnist_query_uri,
    fmnist_groundtruth_uri,
};

std::vector<std::filesystem::path> siftsmall_files{
    siftsmall_inputs_file,
    siftsmall_query_file,
    siftsmall_groundtruth_file,
};

TEST_CASE("array_defs: test array uris", "[array_defs]") {
  bool debug = true;

  for (auto& uri : siftsmall_array_uris) {
    if (debug) {
      std::cout << uri << " "
                << (std::filesystem::exists(uri) ? "exists" : "does not exist")
                << std::endl;
    }
    // CHECK(std::filesystem::exists(uri));
    CHECK(std::filesystem::is_directory(uri));
  }
  for (auto& uri : bigann10k_array_uris) {
    if (debug) {
      std::cout << uri << " "
                << (std::filesystem::exists(uri) ? "exists" : "does not exist")
                << std::endl;
    }
    CHECK(std::filesystem::is_directory(uri));
  }
  for (auto& file : siftsmall_files) {
    if (debug) {
      std::cout << file << " "
                << (std::filesystem::exists(file) ? "exists" : "does not exist")
                << std::endl;
    }
    CHECK(std::filesystem::is_regular_file(file));
  }
}