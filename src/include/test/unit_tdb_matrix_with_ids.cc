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
#include "test/array_defs.h"
#include "test/test_utils.h"

TEST_CASE("tdb_matrix_with_ids: test test", "[tdb_matrix_with_ids]") {
  REQUIRE(true);
}

TEMPLATE_TEST_CASE(
    "tdb_matrix_with_ids: constructors",
    "[tdb_matrix_with_ids]",
    float,
    double,
    int,
    char,
    size_t,
    uint32_t) {
  tiledb::Context ctx;

  std::string tmp_matrix_uri = "/tmp/tmp_tdb_matrix";
  std::string tmp_ids_uri = "/tmp/tmp_tdb_ids_matrix";
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
  CHECK(dimension(Y) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < X.num_rows(); ++i) {
    for (size_t j = 0; j < X.num_cols(); ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }
  CHECK(size(Y.ids()) == Y.num_ids());
  CHECK(size(X.ids()) == size(Y.ids()));
  CHECK(X.num_ids() == Y.num_ids());
  CHECK(std::equal(X.ids().begin(), X.ids().end(), Y.ids().begin()));
  for (size_t i = 0; i < X.num_ids(); ++i) {
    CHECK(X.ids()[i] == Y.ids()[i]);
  }

  auto Z = tdbColMajorMatrixWithIds<TestType, TestType>(std::move(Y));
  CHECK(num_vectors(Z) == num_vectors(X));
  CHECK(dimension(Z) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Z.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Z(i, j));
    }
  }

  CHECK(size(Z.ids()) == Z.num_ids());
  CHECK(size(X.ids()) == size(Z.ids()));
  CHECK(X.num_ids() == Z.num_ids());
  CHECK(std::equal(X.ids().begin(), X.ids().end(), Z.ids().begin()));
  for (size_t i = 0; i < X.num_ids(); ++i) {
    CHECK(X.ids()[i] == Z.ids()[i]);
  }
}

TEMPLATE_TEST_CASE(
    "tdb_matrix_with_ids: assign to matrix",
    "[tdb_matrix_with_ids]",
    float,
    uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri = "/tmp/tmp_tdb_matrix";
  std::string tmp_ids_uri = "/tmp/tmp_tdb_ids_matrix";
  int offset = 123;
  size_t Mrows = 200;
  size_t Ncols = 500;

  auto X = ColMajorMatrixWithIds<TestType, TestType, size_t>(Mrows, Ncols);
  fill_and_write_matrix(
      ctx, X, tmp_matrix_uri, tmp_ids_uri, Mrows, Ncols, offset);
  CHECK(X.ids()[0] == offset + 0);
  CHECK(X.ids()[1] == offset + 1);
  CHECK(X.ids()[10] == offset + 10);
  CHECK(size(X.ids()) == Ncols);

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
  CHECK(size(Y.ids()) == size(X.ids()));
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
  auto Z = ColMajorMatrixWithIds<TestType, TestType, size_t>(0, 0);
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

  auto A = ColMajorMatrixWithIds<TestType, TestType, size_t>(0, 0);
  A = std::move(Z);
  CHECK(size(A.ids()) == size(X.ids()));
  CHECK(num_vectors(A) == num_vectors(X));
  CHECK(dimension(A) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), A.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == A(i, j));
    }
  }

  CHECK(size(B.ids()) == size(X.ids()));
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

TEST_CASE("tdb_matrix_with_ids: load from uri", "[tdb_matrix_with_ids]") {
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
