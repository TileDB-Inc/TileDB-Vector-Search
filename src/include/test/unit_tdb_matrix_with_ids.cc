/**
 * @file   unit_tdb_matrix_with_matrix.cc
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
#include "detail/linalg/matrix_with_ids.h"
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/tdb_matrix_with_ids.h"
#include "test/utils/array_defs.h"
#include "test/utils/test_utils.h"

TEMPLATE_TEST_CASE(
    "constructors",
    "[tdb_matrix_with_ids]",
    float,
    double,
    int,
    char,
    size_t,
    uint32_t) {
  tiledb::Context ctx;

  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();
  int offset = 13;
  size_t Mrows = 200;
  size_t Ncols = 500;
  auto X = ColMajorMatrixWithIds<TestType, TestType, size_t>(Mrows, Ncols);
  fill_and_write_matrix(
      ctx, X, tmp_matrix_uri, tmp_ids_uri, Mrows, Ncols, offset);
  CHECK(X.ids()[0] == offset + 0);
  CHECK(X.ids()[1] == offset + 1);
  CHECK(X.ids()[10] == offset + 10);

  auto Y = tdbColMajorMatrixWithIds<TestType, TestType>(
      ctx, tmp_matrix_uri, tmp_ids_uri);
  Y.load();
  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimensions(Y) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < X.num_rows(); ++i) {
    for (size_t j = 0; j < X.num_cols(); ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }
  CHECK(size(Y.raveled_ids()) == Y.num_ids());
  CHECK(size(X.raveled_ids()) == size(Y.raveled_ids()));
  CHECK(X.num_ids() == Y.num_ids());
  CHECK(std::equal(X.ids(), X.ids() + X.num_ids(), Y.ids()));
  for (size_t i = 0; i < X.num_ids(); ++i) {
    CHECK(X.ids()[i] == Y.ids()[i]);
  }

  auto Z = tdbColMajorMatrixWithIds<TestType, TestType>(std::move(Y));
  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimensions(Z) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }

  CHECK(size(Z.raveled_ids()) == Z.num_ids());
  CHECK(size(X.raveled_ids()) == size(Z.raveled_ids()));
  CHECK(X.num_ids() == Z.num_ids());
  CHECK(std::equal(X.ids(), X.ids() + X.num_ids(), Z.ids()));
  for (size_t i = 0; i < X.num_ids(); ++i) {
    CHECK(X.ids()[i] == Z.ids()[i]);
  }
}

TEST_CASE("different types", "[tdb_matrix_with_ids]") {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();
  int offset = 13;
  size_t Mrows = 200;
  size_t Ncols = 500;
  using DataType = float;
  using IdsType = uint64_t;

  auto X = ColMajorMatrixWithIds<DataType, IdsType, size_t>(Mrows, Ncols);
  fill_and_write_matrix(
      ctx, X, tmp_matrix_uri, tmp_ids_uri, Mrows, Ncols, offset);
  CHECK(X.ids()[0] == offset + 0);
  CHECK(X.ids()[1] == offset + 1);
  CHECK(X.ids()[10] == offset + 10);

  auto Y = tdbColMajorMatrixWithIds<DataType, IdsType>(
      ctx, tmp_matrix_uri, tmp_ids_uri);
  Y.load();
  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimensions(Y) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < X.num_rows(); ++i) {
    for (size_t j = 0; j < X.num_cols(); ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }
  CHECK(Y.num_ids() == Y.num_ids());
  CHECK(size(Y.raveled_ids()) == Y.num_ids());
  CHECK(size(X.raveled_ids()) == size(Y.raveled_ids()));
  CHECK(X.num_ids() == Y.num_ids());
  CHECK(std::equal(X.ids(), X.ids() + X.num_ids(), Y.ids()));
  for (size_t i = 0; i < X.num_ids(); ++i) {
    CHECK(X.ids()[i] == Y.ids()[i]);
  }
}

TEMPLATE_TEST_CASE(
    "assign to matrix", "[tdb_matrix_with_ids]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();
  int offset = 123;
  size_t Mrows = 200;
  size_t Ncols = 500;

  auto X = ColMajorMatrixWithIds<TestType, TestType, size_t>(Mrows, Ncols);
  fill_and_write_matrix(
      ctx, X, tmp_matrix_uri, tmp_ids_uri, Mrows, Ncols, offset);
  CHECK(X.ids()[0] == offset + 0);
  CHECK(X.ids()[1] == offset + 1);
  CHECK(X.ids()[10] == offset + 10);
  CHECK(size(X.raveled_ids()) == Ncols);

  auto B = ColMajorMatrixWithIds<TestType, TestType, size_t>(0, 0);
  {
    auto Y = tdbColMajorMatrixWithIds<TestType, TestType>(
        ctx, tmp_matrix_uri, tmp_ids_uri);
    Y.load();
    B = std::move(Y);
  }

  auto Y = tdbColMajorMatrixWithIds<TestType, TestType>(
      ctx, tmp_matrix_uri, tmp_ids_uri);
  Y.load();
  CHECK(size(Y.raveled_ids()) == size(X.raveled_ids()));
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
  auto Z = ColMajorMatrixWithIds<TestType, TestType, size_t>(0, 0);
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

  auto A = ColMajorMatrixWithIds<TestType, TestType, size_t>(0, 0);
  A = std::move(Z);
  CHECK(size(A.raveled_ids()) == size(X.raveled_ids()));
  CHECK(num_vectors(A) == num_vectors(X));
  CHECK(dimensions(A) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), A.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == A(i, j));
    }
  }

  CHECK(size(B.raveled_ids()) == size(X.raveled_ids()));
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

TEST_CASE("load from uri", "[tdb_matrix_with_ids]") {
  tiledb::Context ctx;

  auto ck = tdbColMajorMatrixWithIds<float>(ctx, sift_inputs_uri, sift_ids_uri);
  ck.load();
  CHECK(ck.num_ids() == num_sift_vectors);

  auto num_queries = 10;
  auto qk = tdbColMajorMatrixWithIds<float>(
      ctx, sift_query_uri, sift_ids_uri, num_queries);
  load(qk);
  CHECK(qk.num_ids() == num_queries);
}

TEST_CASE("empty matrix", "[tdb_matrix_with_ids]") {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();
  size_t matrix_dimension{128};
  int32_t matrix_domain{1000};
  int32_t tile_extent{100};

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }
  if (vfs.is_dir(tmp_ids_uri)) {
    vfs.remove_dir(tmp_ids_uri);
  }

  create_empty_for_matrix<float, stdx::layout_left>(
      ctx,
      tmp_matrix_uri,
      matrix_dimension,
      matrix_domain,
      matrix_dimension,
      tile_extent,
      TILEDB_FILTER_NONE);
  create_empty_for_vector<uint64_t>(
      ctx, tmp_ids_uri, matrix_domain, tile_extent, TILEDB_FILTER_NONE);

  {
    // Empty.
    auto X = tdbColMajorMatrixWithIds<float>(
        ctx,
        tmp_matrix_uri,
        tmp_ids_uri,
        0,
        0,
        0,
        0,
        10000,
        TemporalPolicy{TimeTravel, 0});
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(dimensions(X) == 0);

    CHECK(X.num_ids() == 0);
    CHECK(X.raveled_ids().size() == 0);
  }

  {
    // No constraints.
    auto X = tdbColMajorMatrixWithIds<float>(ctx, tmp_matrix_uri, tmp_ids_uri);
    X.load();
    CHECK(X.num_cols() == 0);
    CHECK(num_vectors(X) == 0);
    CHECK(X.num_rows() == 0);
    CHECK(dimensions(X) == 0);
    CHECK(X.num_ids() == 0);
    CHECK(X.raveled_ids().size() == 0);
  }
}

TEMPLATE_TEST_CASE("preload", "[tdb_matrix_with_ids]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();
  int offset = 13;
  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrixWithIds<TestType, TestType>(Mrows, Ncols);
  fill_and_write_matrix(
      ctx, X, tmp_matrix_uri, tmp_ids_uri, Mrows, Ncols, offset);
  CHECK(X.ids()[0] == offset + 0);
  CHECK(X.ids()[1] == offset + 1);
  CHECK(X.ids()[10] == offset + 10);

  auto Y = tdbPreLoadMatrixWithIds<TestType, TestType, stdx::layout_left>(
      ctx, tmp_matrix_uri, tmp_ids_uri);
  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimensions(Y) == dimensions(X));
  CHECK(std::equal(
      X.data(), X.data() + dimensions(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }

  CHECK(size(Y.raveled_ids()) == Y.num_ids());
  CHECK(size(X.raveled_ids()) == X.num_ids());
  CHECK(X.num_ids() == Y.num_ids());
  CHECK(std::equal(X.ids(), X.ids() + X.num_ids(), Y.ids()));
  for (size_t i = 0; i < X.num_ids(); ++i) {
    CHECK(X.ids()[i] == Y.ids()[i]);
  }

  auto Z = ColMajorMatrixWithIds<TestType, TestType>(0, 0);
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

  CHECK(size(Z.raveled_ids()) == Z.num_ids());
  CHECK(X.num_ids() == Z.num_ids());
  CHECK(std::equal(X.ids(), X.ids() + X.num_ids(), Z.ids()));
  for (size_t i = 0; i < X.num_ids(); ++i) {
    CHECK(X.ids()[i] == Z.ids()[i]);
  }
}

TEST_CASE("time travel", "[tdb_matrix_with_ids]") {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_ids_vector").string();

  int offset = 13;

  size_t Mrows = 40;
  size_t Ncols = 20;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }
  if (vfs.is_dir(tmp_ids_uri)) {
    vfs.remove_dir(tmp_ids_uri);
  }

  auto X = ColMajorMatrixWithIds<float, uint64_t, size_t>(Mrows, Ncols);
  fill_and_write_matrix(
      ctx,
      X,
      tmp_matrix_uri,
      tmp_ids_uri,
      Mrows,
      Ncols,
      offset,
      TemporalPolicy{TimeTravel, 50});

  {
    // We can load the matrix at the creation timestamp.
    auto Y = tdbColMajorPreLoadMatrixWithIds<float, uint64_t, size_t>(
        ctx, tmp_matrix_uri, tmp_ids_uri, 0, TemporalPolicy{TimeTravel, 50});
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
    auto Y = tdbColMajorPreLoadMatrixWithIds<float, uint64_t, size_t>(
        ctx, tmp_matrix_uri, tmp_ids_uri, 0, TemporalPolicy{TimeTravel, 100});
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
    auto Y = tdbColMajorPreLoadMatrixWithIds<float, uint64_t, size_t>(
        ctx, tmp_matrix_uri, tmp_ids_uri, 0, TemporalPolicy{TimeTravel, 5});
    CHECK(num_vectors(Y) == 0);
    CHECK(dimensions(Y) == 0);
    CHECK(Y.size() == 0);
  }

  {
    // We get no data if we load the matrix at an earlier timestamp, even if we
    // specify we want to read 4 rows and 2 cols.
    auto Y = tdbColMajorPreLoadMatrixWithIds<float, uint64_t, size_t>(
        ctx,
        tmp_matrix_uri,
        tmp_ids_uri,
        4,
        2,
        0,
        TemporalPolicy{TimeTravel, 5});
    CHECK(num_vectors(Y) == 0);
    CHECK(dimensions(Y) == 0);
    CHECK(Y.size() == 0);
  }
}
