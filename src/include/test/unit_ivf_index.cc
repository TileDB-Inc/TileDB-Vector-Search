/**
 * @file   unit_ivf_index.cc
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

#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#include "../defs.h"
#include "../ivf_index.h"
#include "../linalg.h"

bool global_debug = false;

TEST_CASE("ivf_index: test test", "[ivf_index]") {
  REQUIRE(true);
}

TEST_CASE("ivf_index: test kmeans initializations", "[ivf_index]") {
  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);
  std::copy(begin(data), end(data), training_data.data());

  auto index = kmeans_index<float, uint32, uint32>(4, 3, 10, 1e-4, 1);

  SECTION("random") {
    index.kmeans_random_init(training_data);
  }

  SECTION("kmeans++") {
    index.kmeans_pp(training_data);
  }

  CHECK(index.get_centroids().num_cols() == 3);
  CHECK(index.get_centroids().num_rows() == 4);

#if 0
  for (size_t j = 0; j < index.get_centroids().num_rows(); ++j) {
    for (size_t i = 0; i < index.get_centroids().num_cols(); ++i) {
      std::cout << index.get_centroids()(j,i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  for (size_t i = 0; i < index.get_centroids().num_cols() - 1; ++i) {
    for (size_t j = i + 1; j < index.get_centroids().num_cols(); ++j) {
      CHECK(!std::equal(
          index.get_centroids()[i].begin(),
          index.get_centroids()[i].end(),
          index.get_centroids()[j].begin()));
    }
  }

  size_t outer_counts = 0;
  for (size_t i = 0; i < index.get_centroids().num_cols(); ++i) {
    size_t inner_counts = 0;
    for (size_t j = 0; j < training_data.num_cols(); ++j) {
      inner_counts += std::equal(
          index.get_centroids()[i].begin(),
          index.get_centroids()[i].end(),
          training_data[j].begin());
    }
    CHECK(inner_counts == 1);
    outer_counts += inner_counts;
  }
  CHECK(outer_counts == index.get_centroids().num_cols());
}

void debug_centroids(auto& index) {
  for (size_t j = 0; j < index.get_centroids().num_rows(); ++j) {
    for (size_t i = 0; i < index.get_centroids().num_cols(); ++i) {
      std::cout << index.get_centroids()(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

TEST_CASE("ivf_index: test kmeans", "[ivf_index]") {
  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);
  std::copy(begin(data), end(data), training_data.data());

  auto index = kmeans_index<float, size_t, size_t>(4, 3, 10, 1e-4, 1);

  SECTION("random") {
    index.kmeans_random_init(training_data);
    index.train_no_init(training_data);
  }

  SECTION("kmeans++") {
    index.kmeans_pp(training_data);
    index.train_no_init(training_data);
  }

  // Test???

  // debug_centroids(index);
}

TEST_CASE("ivf_index: not a unit test per se", "[ivf_index]") {
  tiledb::Context ctx;
  //  auto A = tdbColMajorMatrix<float>(ctx,
  //  "s3://tiledb-andrew/sift/siftsmall_base");
  auto A = tdbColMajorMatrix<float>(
      ctx,
      "/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/sift/"
      "sift_base");

  CHECK(A.num_rows() == 128);
  CHECK(A.num_cols() == 10'000);
  auto index =
      kmeans_index<float, uint32_t, size_t>(A.num_rows(), 1000, 10, 1e-4, 8);

  SECTION("kmeans++") {
    index.kmeans_pp(A);
  }
  SECTION("random") {
    index.kmeans_random_init(A);
  }

  index.train_no_init(A);
}
#if 0
TEST_CASE("ivf_index: also not a unit test per se", "[ivf_index]") {
  tiledb::Context ctx;
  //  auto A = tdbColMajorMatrix<float>(ctx, "s3://tiledb-andrew/sift/siftsmall_base");
  auto A = tdbColMajorMatrix<float>(ctx, "/Users/lums/TileDB/feature-vector-prototype/external/data/arrays/sift/sift_base");

  CHECK(A.num_rows() == 128);
  CHECK(A.num_cols() == 10'000);
  auto index = kmeans_index<float>(A.num_rows(), 1000, 10, 1e-4, 8);

  //SECTION("kmeans++") {
  //  index.kmeans_pp(A);
  //}
  //SECTION("random") {
//    index.kmeans_random_init(A);
//  }

//  index.train_no_init(A);
//  index.add(A);

}
#endif
