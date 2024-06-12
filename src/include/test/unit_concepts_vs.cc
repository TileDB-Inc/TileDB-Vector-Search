/**
 * @file   unit_concepts.cc
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
 * Test application of concepts with TileDB-Vector-Search types
 *
 */

#include <catch2/catch_all.hpp>

#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/matrix_with_ids.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_matrix_with_ids.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "detail/linalg/vector.h"

// Concepts:
// 1. subscriptable_range
// 2. callable
// 3. callable_range
// 4. dimensionable
// 5. vectorable
// 6. partitionable
// 7. feature_vector
// 8. feature_vector_array
// 9. contiguous_feature_vector_range
// 10. partitioned_feature_vector_array
// 11. contiguous_partitioned_feature_vector_array
// 12. matrix_with_ids

TEST_CASE("Vector", "[concepts_vs]") {
  CHECK(subscriptable_range<Vector<int>>);
  CHECK(subscriptable_range<Vector<double>>);
  CHECK(subscriptable_range<Vector<bool>>);

  CHECK(callable_range<Vector<int>>);
  CHECK(callable_range<Vector<double>>);
  CHECK(callable_range<Vector<bool>>);

  CHECK(dimensionable<Vector<int>>);
  CHECK(dimensionable<Vector<double>>);
  CHECK(dimensionable<Vector<bool>>);

  CHECK(vectorable<Vector<int>>);
  CHECK(vectorable<Vector<double>>);
  CHECK(vectorable<Vector<bool>>);

  CHECK(feature_vector<Vector<int>>);
  CHECK(feature_vector<Vector<double>>);

  // @note: There is an open question about whether we should consider Vector of
  // bool bo be a contiguous range or not. It is not contiguous in the sense
  // that it is not a contiguous array of bools, but it is contiguous in the
  // sense that it is a contiguous array of bits.  For now, if it passes the
  // concept check, we'll let it be.
  CHECK(feature_vector<Vector<bool>>);

  CHECK(query_vector<Vector<int>>);
  CHECK(query_vector<Vector<double>>);
  CHECK(query_vector<Vector<bool>>);

  CHECK(!feature_vector_array<Vector<int>>);
  CHECK(!feature_vector_array<Vector<double>>);
  CHECK(!feature_vector_array<Vector<bool>>);

  CHECK(!contiguous_feature_vector_array<Vector<int>>);
  CHECK(!contiguous_feature_vector_array<Vector<double>>);
  CHECK(!contiguous_feature_vector_array<Vector<bool>>);

  CHECK(!partitioned_feature_vector_array<Vector<int>>);
  CHECK(!partitioned_feature_vector_array<Vector<double>>);
  CHECK(!partitioned_feature_vector_array<Vector<bool>>);

  CHECK(!contiguous_partitioned_feature_vector_array<Vector<int>>);
  CHECK(!contiguous_partitioned_feature_vector_array<Vector<double>>);
  CHECK(!contiguous_partitioned_feature_vector_array<Vector<bool>>);
}

template <dimensionable D>
auto _dimensionable(const D& d) {
  return dimensions(d);
}

auto test_dimensionable(const Matrix<int>& d) {
  return _dimensionable(d);
}

template <feature_vector_array D>
auto _feature_vector_range(const D& d) {
  return num_vectors(d);
}
auto test_feature_vector_range(const Matrix<int>& d) {
  return _feature_vector_range(d);
}

template <contiguous_feature_vector_array D>
auto _contiguous_feature_vector_array(const D& d) {
  return num_vectors(d);
}
auto test_contiguous_feature_vector_array(const Matrix<int>& d) {
  return _contiguous_feature_vector_array(d);
}

TEST_CASE("Matrix", "[concepts_vs]") {
  CHECK(!subscriptable_range<Matrix<int>>);
  CHECK(!subscriptable_range<Matrix<double>>);
  CHECK(!subscriptable_range<Matrix<bool>>);

  CHECK(!callable_range<Matrix<int>>);
  CHECK(!callable_range<Matrix<double>>);
  CHECK(!callable_range<Matrix<bool>>);

  CHECK(dimensionable<Matrix<int>>);
  CHECK(dimensionable<Matrix<double>>);
  CHECK(dimensionable<Matrix<bool>>);

  CHECK(vectorable<Matrix<int>>);
  CHECK(vectorable<Matrix<double>>);
  CHECK(vectorable<Matrix<bool>>);

  CHECK(!feature_vector<Matrix<int>>);
  CHECK(!feature_vector<Matrix<double>>);
  CHECK(!feature_vector<Matrix<bool>>);

  CHECK(!query_vector<Matrix<int>>);
  CHECK(!query_vector<Matrix<double>>);
  CHECK(!query_vector<Matrix<bool>>);

  CHECK(feature_vector_array<Matrix<int>>);
  CHECK(feature_vector_array<Matrix<double>>);
  CHECK(feature_vector_array<Matrix<bool>>);

  CHECK(!feature_vector_array_with_ids<Matrix<int>>);
  CHECK(!feature_vector_array_with_ids<Matrix<double>>);
  CHECK(!feature_vector_array_with_ids<Matrix<bool>>);

  CHECK(contiguous_feature_vector_array<Matrix<int>>);
  CHECK(contiguous_feature_vector_array<Matrix<double>>);
  CHECK(contiguous_feature_vector_array<Matrix<bool>>);

  CHECK(!partitioned_feature_vector_array<Matrix<int>>);
  CHECK(!partitioned_feature_vector_array<Matrix<double>>);
  CHECK(!partitioned_feature_vector_array<Matrix<bool>>);

  CHECK(!contiguous_partitioned_feature_vector_array<Matrix<int>>);
  CHECK(!contiguous_partitioned_feature_vector_array<Matrix<double>>);
  CHECK(!contiguous_partitioned_feature_vector_array<Matrix<bool>>);
}

TEST_CASE("MatrixWithIds", "[concepts_vs]") {
  CHECK(!subscriptable_range<MatrixWithIds<int>>);
  CHECK(!subscriptable_range<MatrixWithIds<double>>);
  CHECK(!subscriptable_range<MatrixWithIds<bool>>);

  CHECK(!callable_range<MatrixWithIds<int>>);
  CHECK(!callable_range<MatrixWithIds<double>>);
  CHECK(!callable_range<MatrixWithIds<bool>>);

  CHECK(dimensionable<MatrixWithIds<int>>);
  CHECK(dimensionable<MatrixWithIds<double>>);
  CHECK(dimensionable<MatrixWithIds<bool>>);

  CHECK(vectorable<MatrixWithIds<int>>);
  CHECK(vectorable<MatrixWithIds<double>>);
  CHECK(vectorable<MatrixWithIds<bool>>);

  CHECK(!feature_vector<MatrixWithIds<int>>);
  CHECK(!feature_vector<MatrixWithIds<double>>);
  CHECK(!feature_vector<MatrixWithIds<bool>>);

  CHECK(!query_vector<MatrixWithIds<int>>);
  CHECK(!query_vector<MatrixWithIds<double>>);
  CHECK(!query_vector<MatrixWithIds<bool>>);

  CHECK(feature_vector_array<MatrixWithIds<int>>);
  CHECK(feature_vector_array<MatrixWithIds<double>>);
  CHECK(feature_vector_array<MatrixWithIds<bool>>);

  CHECK(feature_vector_array_with_ids<MatrixWithIds<int>>);
  CHECK(feature_vector_array_with_ids<MatrixWithIds<double>>);
  CHECK(feature_vector_array_with_ids<MatrixWithIds<bool>>);

  CHECK(contiguous_feature_vector_array<MatrixWithIds<int>>);
  CHECK(contiguous_feature_vector_array<MatrixWithIds<double>>);
  CHECK(contiguous_feature_vector_array<MatrixWithIds<bool>>);

  CHECK(!partitioned_feature_vector_array<MatrixWithIds<int>>);
  CHECK(!partitioned_feature_vector_array<MatrixWithIds<double>>);
  CHECK(!partitioned_feature_vector_array<MatrixWithIds<bool>>);

  CHECK(!contiguous_partitioned_feature_vector_array<MatrixWithIds<int>>);
  CHECK(!contiguous_partitioned_feature_vector_array<MatrixWithIds<double>>);
  CHECK(!contiguous_partitioned_feature_vector_array<MatrixWithIds<bool>>);
}

TEST_CASE("tdbMatrixWithIds", "[concepts_vs]") {
  CHECK(!subscriptable_range<tdbMatrixWithIds<int>>);
  CHECK(!subscriptable_range<tdbMatrixWithIds<double>>);
  CHECK(!subscriptable_range<tdbMatrixWithIds<bool>>);

  CHECK(!callable_range<tdbMatrixWithIds<int>>);
  CHECK(!callable_range<tdbMatrixWithIds<double>>);
  CHECK(!callable_range<tdbMatrixWithIds<bool>>);

  CHECK(dimensionable<tdbMatrixWithIds<int>>);
  CHECK(dimensionable<tdbMatrixWithIds<double>>);
  CHECK(dimensionable<tdbMatrixWithIds<bool>>);

  CHECK(vectorable<tdbMatrixWithIds<int>>);
  CHECK(vectorable<tdbMatrixWithIds<double>>);
  CHECK(vectorable<tdbMatrixWithIds<bool>>);

  CHECK(!feature_vector<tdbMatrixWithIds<int>>);
  CHECK(!feature_vector<tdbMatrixWithIds<double>>);
  CHECK(!feature_vector<tdbMatrixWithIds<bool>>);

  CHECK(!query_vector<tdbMatrixWithIds<int>>);
  CHECK(!query_vector<tdbMatrixWithIds<double>>);
  CHECK(!query_vector<tdbMatrixWithIds<bool>>);

  CHECK(feature_vector_array<tdbMatrixWithIds<int>>);
  CHECK(feature_vector_array<tdbMatrixWithIds<double>>);
  CHECK(feature_vector_array<tdbMatrixWithIds<bool>>);

  CHECK(feature_vector_array_with_ids<tdbMatrixWithIds<int>>);
  CHECK(feature_vector_array_with_ids<tdbMatrixWithIds<double>>);
  CHECK(feature_vector_array_with_ids<tdbMatrixWithIds<bool>>);

  CHECK(contiguous_feature_vector_array<tdbMatrixWithIds<int>>);
  CHECK(contiguous_feature_vector_array<tdbMatrixWithIds<double>>);
  CHECK(contiguous_feature_vector_array<tdbMatrixWithIds<bool>>);

  CHECK(!partitioned_feature_vector_array<tdbMatrixWithIds<int>>);
  CHECK(!partitioned_feature_vector_array<tdbMatrixWithIds<double>>);
  CHECK(!partitioned_feature_vector_array<tdbMatrixWithIds<bool>>);

  CHECK(!contiguous_partitioned_feature_vector_array<tdbMatrixWithIds<int>>);
  CHECK(!contiguous_partitioned_feature_vector_array<tdbMatrixWithIds<double>>);
  CHECK(!contiguous_partitioned_feature_vector_array<tdbMatrixWithIds<bool>>);
}
