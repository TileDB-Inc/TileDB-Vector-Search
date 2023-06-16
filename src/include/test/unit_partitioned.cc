/**
 * @file   unit_partitioned.cc
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
 * Test correctness of partitioned vector database.
 */

#include <catch2/catch_all.hpp>
#include <set>
#include <vector>
#include "linalg.h"
#include "partitioned.h"

bool global_debug = false;
std::string global_region = "us-east-1";

TEST_CASE("partitioned: test test", "[partitioned]") {
  REQUIRE(true);
}

TEST_CASE("partitioned: even odd", "[partitioned][ci-skip]") {
  tiledb::Context ctx;

  std::string parts_uri{"even_odd_parts"};
  std::string index_uri{"even_odd_index"};
  std::string ids_uri{"even_odd_ids"};
  std::string centroids_uri{"even_odd_centroids"};
  std::string queries_uri{"even_odd_queries"};

  auto parts_mat = tdbColMajorMatrix<float>(ctx, parts_uri);
  auto index = read_vector<uint32_t>(ctx, index_uri);
  auto ids = read_vector<uint32_t>(ctx, ids_uri);
  auto centroids_mat = tdbColMajorMatrix<float>(ctx, centroids_uri);
  auto queries_mat = tdbColMajorMatrix<float>(ctx, queries_uri);

  auto partitioned = tdbPartitionedMatrix<float>(
      ctx, parts_uri, centroids_mat, queries_mat, index, ids, 2, 2);
}
