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

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

TEST_CASE("tdb_matrix: test test", "[tdb_matrix]") {
  REQUIRE(true);
}

TEMPLATE_TEST_CASE("tdb_matrix: constructors", "[tdb_matrix]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri = "/tmp/tmp_tdb_matrix";
  int offset = 13;
  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimension(X) * num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  auto Y = tdbColMajorMatrix<TestType>(ctx, tmp_matrix_uri);
  Y.load();

  auto Z = tdbColMajorMatrix<TestType>(std::move(Y));

  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimension(Y) == dimension(X));

  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimension(Z) == dimension(X));

  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }
}

TEMPLATE_TEST_CASE(
    "tdb_matrix: assign to matrix", "[tdb_matrix]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri = "/tmp/tmp_tdb_matrix";
  int offset = 13;

  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimension(X) * num_vectors(X), offset);
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
  CHECK(dimension(Y) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }

  // Check that we can assign to a matrix
  auto Z = ColMajorMatrix<TestType>(0, 0);
  Z = std::move(Y);

  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimension(Z) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }

  auto A = ColMajorMatrix<TestType>(0, 0);
  A = std::move(Z);
  CHECK(num_vectors(A) == num_vectors(X));
  CHECK(dimension(A) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), A.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == A(i, j));
    }
  }

  CHECK(num_vectors(B) == num_vectors(X));
  CHECK(dimension(B) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), B.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == B(i, j));
    }
  }
}

TEMPLATE_TEST_CASE("tdb_matrix: preload", "[tdb_matrix]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri = "/tmp/tmp_tdb_matrix";
  int offset = 13;

  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimension(X) * num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  auto Y = tdbPreLoadMatrix<TestType, stdx::layout_left>(ctx, tmp_matrix_uri);
  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimension(Y) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }

  auto Z = ColMajorMatrix<TestType>(0, 0);
  Z = std::move(Y);

  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimension(Z) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }
}
