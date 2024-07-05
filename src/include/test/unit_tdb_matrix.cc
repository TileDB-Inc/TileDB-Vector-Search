/**
 * @file   unit_tdb_matrix.cc
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
#include "detail/linalg/tdb_matrix.h"
#include "mdspan/mdspan.hpp"
#include "test/utils/test_utils.h"

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

TEMPLATE_TEST_CASE("constructors", "[tdb_matrix]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  int offset = 13;
  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimensions(X) * num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  auto Y = tdbColMajorMatrix<TestType>(ctx, tmp_matrix_uri);
  Y.load();

  auto Z = tdbColMajorMatrix<TestType>(std::move(Y));

  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimensions(Y) == dimensions(X));

  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimensions(Z) == dimensions(X));

  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }
}

TEMPLATE_TEST_CASE("assign to matrix", "[tdb_matrix]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();

  int offset = 13;

  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimensions(X) * num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  auto B = ColMajorMatrix<TestType>(0, 0);
  {
    auto Y = tdbColMajorMatrix<TestType>(ctx, tmp_matrix_uri);
    Y.load();
    B = std::move(Y);
  }

  {
    auto Y = tdbColMajorMatrix<TestType>(
        tdbColMajorMatrix<TestType>(ctx, tmp_matrix_uri));
  }

  auto Y = tdbColMajorMatrix<TestType>(ctx, tmp_matrix_uri);
  Y.load();

  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimensions(Y) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }

  // Check that we can assign to a matrix
  auto Z = ColMajorMatrix<TestType>(0, 0);
  Z = std::move(Y);

  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimensions(Z) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }

  auto A = ColMajorMatrix<TestType>(0, 0);
  A = std::move(Z);
  CHECK(num_vectors(A) == num_vectors(X));
  CHECK(dimensions(A) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), A.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == A(i, j));
    }
  }

  CHECK(num_vectors(B) == num_vectors(X));
  CHECK(dimensions(B) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), B.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == B(i, j));
    }
  }
}

TEMPLATE_TEST_CASE("preload", "[tdb_matrix]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  int offset = 13;

  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimensions(X) * num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  auto Y = tdbPreLoadMatrix<TestType, stdx::layout_left>(ctx, tmp_matrix_uri);
  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimensions(Y) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }

  auto Z = ColMajorMatrix<TestType>(0, 0);
  Z = std::move(Y);

  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimensions(Z) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }
}

TEST_CASE("MatrixBase template parameter", "[tdb_matrix]") {
  // Load data.
  tiledb::Context ctx;
  int offset = 13;

  size_t Mrows = 200;
  size_t Ncols = 500;

  using T = float;
  using I = size_t;
  using LayoutPolicy = stdx::layout_left;
  using IdsType = uint8_t;

  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();

  // 1. Use Matrix as the MatrixBase template parameter.
  {
    auto X = ColMajorMatrix<T>(Mrows, Ncols);
    fill_and_write_matrix(ctx, X, tmp_matrix_uri, Mrows, Ncols, offset);

    auto Y = tdbBlockedMatrix<T, LayoutPolicy, I, Matrix<T, LayoutPolicy, I>>(
        ctx, tmp_matrix_uri);
    Y.load();
    CHECK(num_vectors(Y) == num_vectors(X));
    CHECK(dimensions(Y) == dimensions(X));
    CHECK(num_vectors(Y) == num_vectors(X));
    CHECK(dimensions(Y) == dimensions(X));
    CHECK(std::equal(
        X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 5; ++j) {
        CHECK(X(i, j) == Y(i, j));
      }
    }
  }

  // 2. Use MatrixWithIds as the MatrixBase template parameter.
  {
    auto X = ColMajorMatrixWithIds<T, IdsType, I>(Mrows, Ncols);
    fill_and_write_matrix(
        ctx, X, tmp_matrix_uri, tmp_ids_uri, Mrows, Ncols, offset);

    auto Y = tdbBlockedMatrix<
        T,
        LayoutPolicy,
        I,
        MatrixWithIds<T, IdsType, LayoutPolicy, I>>(ctx, tmp_matrix_uri);
    Y.load();
    CHECK(num_vectors(Y) == num_vectors(X));
    CHECK(dimensions(Y) == dimensions(X));
    CHECK(num_vectors(Y) == num_vectors(X));
    CHECK(dimensions(Y) == dimensions(X));
    CHECK(std::equal(
        X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
    for (size_t i = 0; i < 5; ++i) {
      for (size_t j = 0; j < 5; ++j) {
        CHECK(X(i, j) == Y(i, j));
      }
    }
  }
}

TEST_CASE("empty matrix", "[tdb_matrix]") {
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
    // No rows and no cols.
    auto X =
        tdbColMajorMatrix<float>(ctx, tmp_matrix_uri, 0, 0, 0, 0, 10000, {});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(dimensions(X) == 0);
  }

  {
    // All rows and no cols.
    auto X = tdbColMajorMatrix<float>(
        ctx, tmp_matrix_uri, 0, std::nullopt, 0, 0, 10000, {});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(dimensions(X) == 0);
  }

  {
    // No rows and all cols.
    auto X = tdbColMajorMatrix<float>(
        ctx, tmp_matrix_uri, 0, 0, 0, std::nullopt, 10000, {});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(dimensions(X) == 0);
  }

  {
    // No constraints.
    auto X = tdbColMajorMatrix<float>(ctx, tmp_matrix_uri);
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(dimensions(X) == 0);
  }
}

TEST_CASE("time travel", "[tdb_matrix]") {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  int offset = 13;

  size_t Mrows = 20;
  size_t Ncols = 50;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<int>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimensions(X) * num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri, 0, true, TemporalPolicy{TimeTravel, 50});

  {
    // We can load the matrix at the creation timestamp.
    auto Y = tdbPreLoadMatrix<int, stdx::layout_left>(
        ctx, tmp_matrix_uri, 0, TemporalPolicy{TimeTravel, 50});
    CHECK(num_vectors(Y) == num_vectors(X));
    CHECK(dimensions(Y) == dimensions(X));
    CHECK(std::equal(
        X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
    for (size_t i = 0; i < Mrows; ++i) {
      for (size_t j = 0; j < Ncols; ++j) {
        CHECK(X(i, j) == Y(i, j));
      }
    }
  }

  {
    // We can load the matrix at a later timestamp.
    auto Y = tdbPreLoadMatrix<int, stdx::layout_left>(
        ctx, tmp_matrix_uri, 0, TemporalPolicy{TimeTravel, 100});
    CHECK(num_vectors(Y) == num_vectors(X));
    CHECK(dimensions(Y) == dimensions(X));
    CHECK(std::equal(
        X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
    for (size_t i = 0; i < Mrows; ++i) {
      for (size_t j = 0; j < Ncols; ++j) {
        CHECK(X(i, j) == Y(i, j));
      }
    }
  }

  {
    // We get no data if we load the matrix at an earlier timestamp.
    auto Y = tdbPreLoadMatrix<int, stdx::layout_left>(
        ctx, tmp_matrix_uri, 0, TemporalPolicy{TimeTravel, 5});
    CHECK(num_vectors(Y) == 0);
    CHECK(dimensions(Y) == 0);
  }

  {
    // We get no data if we load the matrix at an earlier timestamp, even if we
    // specify we want to read 4 rows and 2 cols.
    auto Y = tdbPreLoadMatrix<int, stdx::layout_left>(
        ctx, tmp_matrix_uri, 4, 2, 0, TemporalPolicy{TimeTravel, 5});
    CHECK(num_vectors(Y) == 0);
    CHECK(dimensions(Y) == 0);
    CHECK(Y.size() == 0);
  }
}
