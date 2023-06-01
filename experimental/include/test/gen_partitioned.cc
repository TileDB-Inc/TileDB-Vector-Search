/**
 * @file   gen_partitioned.cc
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
 * Generate test cases for partitioned vector database.
 */

#include <catch2/catch_all.hpp>
#include <string>
#include "flat_query.h"
#include "ivf_query.h"
#include "linalg.h"

bool global_debug = false;
std::string global_region = "us-east-1";

TEST_CASE("getn_partitioned: test test", "[gen_partitioned]") {
  REQUIRE(true);
}

TEST_CASE("gen_partitioned: even odd", "[gen_partitioned]") {
  size_t dimension{128};
  size_t n{10};
  std::vector<uint32_t> ids = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
  std::vector<uint32_t> index = {0, 5, 10};

  CHECK(size(ids) == n);

  auto part_mat = ColMajorMatrix<float>(dimension, n);
  for (size_t i = 0; i < n / 2; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      part_mat(j, i) = 2 * i * dimension + j;
    }
  }
  for (size_t i = 1; i < n / 2; ++i) {
    for (size_t j = 0; j < dimension; ++j) {
      part_mat(j, i + 5) = 2 * i * dimension + j;
    }
  }
  auto centroid_mat = ColMajorMatrix<float>(dimension, 2);
  for (size_t i = 0; i < dimension; ++i) {
    centroid_mat(i, 0) = part_mat(i, 0);
    centroid_mat(i, 9) = part_mat(i, 9);
  }

  write_matrix(part_mat, "even_odd_parts");
  write_vector(ids, "even_odd_ids");
  write_vector(index, "even_odd_index");
  write_matrix(centroid_mat, "even_odd_centroids");
  write_matrix(centroid_mat, "even_odd_queries");
}

TEST_CASE("gen_partitioned: 3D", "[gen_partitioned]") {
  auto v = std::vector<float>{-64, 40, 82,  77,  -85, -65, -53, 17,
                              -41, 73, -46, -66, 50,  74,  -85, 74,
                              -58, 59, -50, 80,  32,  100, 45,  -68};

  auto m = ColMajorMatrix<float>(3, 8);
  std::copy(v.begin(), v.end(), m.data());

  auto q = ColMajorMatrix<float>(3, 1);
  for (size_t i = 0; i < 3; ++i) {
    q(i, 0) = m(i, 4);
  }

  auto top_k = vq_query_heap(m, q, 3, 1);

  CHECK(top_k(0, 0) == 4);
  for (size_t i = 0; i < top_k.num_rows(); ++i) {
    for (size_t j = 0; j < top_k.num_cols(); ++j) {
      std::cout << top_k(i, j) << " ";
    }
    std::cout << std::endl;
  }
}