/**
 * @file   unit_tdb_partitioned_matrix.cc
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

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <vector>
#include "cpos.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "mdspan/mdspan.hpp"
#include "test/test_utils.h"

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

TEST_CASE("tdb_partitioned_matrix: test test", "[tdb_partitioned_matrix]") {
  REQUIRE(true);
}

TEST_CASE("tdb_partitioned_matrix: uri constructor", "[tdb_partitioned_matrix]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  using feature_type = int;
  using id_type = int;
  using part_index_type = int;

  std::string partitioned_vectors_uri = (std::filesystem::temp_directory_path() / "partitioned_vectors").string();
  std::string ids_uri = (std::filesystem::temp_directory_path() / "ids").string();
  // Setup data.
  {
    if (vfs.is_dir(partitioned_vectors_uri)) {
      vfs.remove_dir(partitioned_vectors_uri);
    }
    if (vfs.is_dir(ids_uri)) {
      vfs.remove_dir(ids_uri);
    }

    auto partitioned_vectors = ColMajorMatrix<feature_type>{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};
    write_matrix(ctx, partitioned_vectors, partitioned_vectors_uri);
    std::vector<id_type> ids = {1, 2, 3, 4, 5};
    write_vector(ctx, ids, ids_uri);
  }

  // Test if we have two parts we can load them both.
  {
    std::vector<part_index_type> indices = {0, 3, 5};
    std::vector<part_index_type> relevant_parts = {0, 1};

    auto matrix = tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
      ctx, 
      partitioned_vectors_uri, 
      indices, 
      ids_uri, 
      relevant_parts, 
      0);
    matrix.load();

    CHECK(matrix.num_vectors() == 5);
    CHECK(matrix.num_partitions() == 2);
    CHECK(std::equal(matrix.data(), matrix.data() + matrix.num_vectors() * _cpo::dimension(matrix), 
      std::vector<feature_type>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5}.begin()));
    CHECK(std::equal(matrix.ids().begin(), matrix.ids().end(), std::vector<part_index_type>{1, 2, 3, 4, 5}.begin()));
    CHECK(std::equal(matrix.indices().begin(), matrix.indices().end(), indices.begin()));
  }

  // Test if we have two parts we can load just the first.
  {
    std::vector<part_index_type> indices = {0, 3, 5};
    std::vector<part_index_type> relevant_parts = {0};

    auto matrix = tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
      ctx, 
      partitioned_vectors_uri, 
      indices, 
      ids_uri, 
      relevant_parts, 
      0);
    matrix.load();

    CHECK(matrix.num_vectors() == 3);
    CHECK(matrix.num_partitions() == 1);
    CHECK(std::equal(matrix.data(), matrix.data() + matrix.num_vectors() * _cpo::dimension(matrix), 
      std::vector<feature_type>{1, 1, 2, 2, 3, 3}.begin()));
    CHECK(std::equal(matrix.ids().begin(), matrix.ids().end(), std::vector<part_index_type>{1, 2, 3}.begin()));
    CHECK(std::equal(matrix.indices().begin(), matrix.indices().end(), indices.begin()));
  }

  // Test if we have two parts we can load none.
  {
    std::vector<part_index_type> indices = {0, 3, 5};
    std::vector<part_index_type> relevant_parts = {};

    auto matrix = tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
        ctx,
        partitioned_vectors_uri,
        indices,
        ids_uri,
        relevant_parts,
        0);
    matrix.load();

    CHECK(matrix.num_vectors() == 0);
    CHECK(matrix.num_partitions() == 0);
    CHECK(std::equal(matrix.data(), matrix.data() + matrix.num_vectors() * _cpo::dimension(matrix), std::vector<feature_type>{}.begin()));
    CHECK(std::equal(matrix.ids().begin(), matrix.ids().end(), std::vector<part_index_type>{}.begin()));
    CHECK(std::equal(matrix.indices().begin(), matrix.indices().end(), indices.begin()));

    debug_partitioned_matrix(matrix);
  }
}
