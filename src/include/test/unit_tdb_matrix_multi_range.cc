/**
 * @file unit_tdb_matrix_multi_range.cc
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

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <vector>
#include "cpos.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/tdb_matrix_multi_range.h"
#include "mdspan/mdspan.hpp"
#include "test/utils/test_utils.h"

TEMPLATE_TEST_CASE(
    "constructors",
    "[tdb_matrix_multi_range]",
    float,
    double,
    int,
    char,
    size_t,
    uint32_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  int offset = 13;
  size_t dimensions = 200;
  size_t num_vectors = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(dimensions, num_vectors);
  std::iota(X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  std::vector<size_t> column_indices(num_vectors);
  std::iota(column_indices.begin(), column_indices.end(), 0);

  auto Y = tdbColMajorMatrixMultiRange<TestType>(
      ctx, tmp_matrix_uri, dimensions, column_indices, num_vectors);
  CHECK(Y.load() == true);
  for (int i = 0; i < 5; ++i) {
    CHECK(Y.load() == false);
  }

  auto Z = tdbColMajorMatrixMultiRange<TestType>(std::move(Y));

  CHECK(::num_vectors(Y) == ::num_vectors(X));
  CHECK(::dimensions(Y) == ::dimensions(X));

  CHECK(::num_vectors(Z) == ::num_vectors(X));
  CHECK(::dimensions(Z) == ::dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), Z.data()));
  for (size_t c = 0; c < num_vectors; ++c) {
    for (size_t r = 0; r < dimensions; ++r) {
      CHECK(X(r, c) == Z(r, c));
    }
  }
}

TEMPLATE_TEST_CASE(
    "assign to matrix", "[tdb_matrix_multi_range]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();

  int offset = 13;

  size_t dimensions = 200;
  size_t num_vectors = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(dimensions, num_vectors);
  std::iota(X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  auto B = ColMajorMatrix<TestType>(0, 0);
  {
    std::vector<size_t> column_indices(num_vectors);
    std::iota(column_indices.begin(), column_indices.end(), 0);
    auto Y = tdbColMajorMatrixMultiRange<TestType>(
        ctx, tmp_matrix_uri, dimensions, column_indices, num_vectors);
    Y.load();
    B = std::move(Y);
  }

  {
    std::vector<size_t> column_indices(num_vectors);
    std::iota(column_indices.begin(), column_indices.end(), 0);
    auto Y = tdbColMajorMatrixMultiRange<TestType>(
        tdbColMajorMatrixMultiRange<TestType>(
            ctx, tmp_matrix_uri, dimensions, column_indices, num_vectors));
  }

  std::vector<size_t> column_indices(num_vectors);
  std::iota(column_indices.begin(), column_indices.end(), 0);
  auto Y = tdbColMajorMatrixMultiRange<TestType>(
      ctx, tmp_matrix_uri, dimensions, column_indices, num_vectors);
  Y.load();

  CHECK(::num_vectors(Y) == ::num_vectors(X));
  CHECK(::dimensions(Y) == ::dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), Y.data()));
  for (size_t c = 0; c < num_vectors; ++c) {
    for (size_t r = 0; r < dimensions; ++r) {
      CHECK(X(r, c) == Y(r, c));
    }
  }

  // Check that we can assign to a matrix
  auto Z = ColMajorMatrix<TestType>(0, 0);
  Z = std::move(Y);

  CHECK(::num_vectors(Z) == ::num_vectors(X));
  CHECK(::dimensions(Z) == ::dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), Z.data()));
  for (size_t c = 0; c < num_vectors; ++c) {
    for (size_t r = 0; r < dimensions; ++r) {
      CHECK(X(r, c) == Z(r, c));
    }
  }

  auto A = ColMajorMatrix<TestType>(0, 0);
  A = std::move(Z);
  CHECK(::num_vectors(A) == ::num_vectors(X));
  CHECK(::dimensions(A) == ::dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), A.data()));
  for (size_t c = 0; c < num_vectors; ++c) {
    for (size_t r = 0; r < dimensions; ++r) {
      CHECK(X(r, c) == A(r, c));
    }
  }

  CHECK(::num_vectors(B) == ::num_vectors(X));
  CHECK(::dimensions(B) == ::dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), B.data()));
  for (size_t c = 0; c < num_vectors; ++c) {
    for (size_t r = 0; r < dimensions; ++r) {
      CHECK(X(r, c) == B(r, c));
    }
  }
}

TEST_CASE("limit column_indices", "[tdb_matrix_multi_range]") {
  tiledb::Context ctx;
  int offset = 13;

  size_t dimensions = 200;
  size_t num_vectors = 500;

  using T = float;
  using I = size_t;
  using LayoutPolicy = stdx::layout_left;

  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();

  auto X = ColMajorMatrix<T>(dimensions, num_vectors);
  fill_and_write_matrix(
      ctx, X, tmp_matrix_uri, dimensions, num_vectors, offset);
  CHECK(::num_vectors(X) == num_vectors);
  CHECK(::dimensions(X) == dimensions);

  std::vector<size_t> column_indices = {
      0, 1, 2, 3, 10, 100, 15, 299, 309, 4, 100};
  auto Y = tdbBlockedMatrixMultiRange<T, LayoutPolicy, I>(
      ctx, tmp_matrix_uri, dimensions, column_indices, 0);
  Y.load();
  CHECK(::num_vectors(Y) == column_indices.size());
  CHECK(::dimensions(Y) == ::dimensions(X));
  for (size_t c = 0; c < column_indices.size(); ++c) {
    for (size_t r = 0; r < dimensions; ++r) {
      CHECK(X(r, column_indices[c]) == Y(r, c));
    }
  }
}

TEST_CASE("empty matrix", "[tdb_matrix_multi_range]") {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  size_t matrix_dimension{128};
  int32_t matrix_domain{1000};
  int32_t tile_extent{100};

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  create_empty_for_matrix<float, stdx::layout_left>(
      ctx,
      tmp_matrix_uri,
      matrix_dimension,
      matrix_domain,
      matrix_dimension,
      tile_extent,
      TILEDB_FILTER_NONE);

  {
    // No dimensions and no num_vectors.
    auto X = tdbColMajorMatrixMultiRange<float>(
        ctx,
        tmp_matrix_uri,
        0,
        std::vector<size_t>{},
        0,
        TemporalPolicy{TimeTravel, 50});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(::num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(::dimensions(X) == 0);
  }

  {
    // All dimensions and no num_vectors.
    auto X = tdbColMajorMatrixMultiRange<float>(
        ctx,
        tmp_matrix_uri,
        0,
        std::vector<size_t>{},
        0,
        TemporalPolicy{TimeTravel, 50});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(::num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(::dimensions(X) == 0);
  }

  {
    // No dimensions and all num_vectors.
    auto X = tdbColMajorMatrixMultiRange<float>(
        ctx,
        tmp_matrix_uri,
        0,
        std::vector<size_t>{},
        0,
        TemporalPolicy{TimeTravel, 50});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(::num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(::dimensions(X) == 0);
  }

  {
    // No constraints.
    auto X = tdbColMajorMatrixMultiRange<float>(
        ctx,
        tmp_matrix_uri,
        0,
        std::vector<size_t>{},
        0,
        TemporalPolicy{TimeTravel, 50});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(::num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(::dimensions(X) == 0);
  }
}

TEST_CASE("time travel", "[tdb_matrix_multi_range]") {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  int offset = 13;

  size_t dimensions = 20;
  size_t num_vectors = 50;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<int>(dimensions, num_vectors);
  std::iota(X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri, 0, true, TemporalPolicy{TimeTravel, 50});

  std::vector<size_t> column_indices(num_vectors);
  std::iota(column_indices.begin(), column_indices.end(), 0);

  {
    // We can load the matrix at the creation timestamp.
    auto Y = tdbColMajorMatrixMultiRange<int>(
        ctx, tmp_matrix_uri, dimensions, column_indices, 0);
    CHECK(Y.load());
    CHECK(::num_vectors(Y) == ::num_vectors(X));
    CHECK(::dimensions(Y) == ::dimensions(X));
    CHECK(std::equal(
        X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), Y.data()));
    for (size_t c = 0; c < num_vectors; ++c) {
      for (size_t r = 0; r < dimensions; ++r) {
        CHECK(X(r, c) == Y(r, c));
      }
    }
  }

  {
    // We can load the matrix at a later timestamp.
    auto Y = tdbColMajorMatrixMultiRange<int>(
        ctx,
        tmp_matrix_uri,
        dimensions,
        column_indices,
        num_vectors,
        TemporalPolicy{TimeTravel, 100});
    CHECK(Y.load());
    CHECK(::num_vectors(Y) == ::num_vectors(X));
    CHECK(::dimensions(Y) == ::dimensions(X));
    CHECK(std::equal(
        X.data(), X.data() + ::dimensions(X) * ::num_vectors(X), Y.data()));
    for (size_t c = 0; c < num_vectors; ++c) {
      for (size_t r = 0; r < dimensions; ++r) {
        CHECK(X(r, c) == Y(r, c));
      }
    }
  }

  {
    // We get no data if we load the matrix at an earlier timestamp.
    auto Y = tdbColMajorMatrixMultiRange<int>(
        ctx,
        tmp_matrix_uri,
        dimensions,
        column_indices,
        num_vectors,
        TemporalPolicy{TimeTravel, 5});
    CHECK(Y.load());
    // Note that even though there are no vectors to load, we still try to load
    // all the vectors, so we will have filled the matrix with fill values.
    CHECK(::num_vectors(Y) == column_indices.size());
    CHECK(::dimensions(Y) == dimensions);
  }
}
