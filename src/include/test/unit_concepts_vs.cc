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
#include "detail/linalg/vector.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_partitioned_matrix.h"


TEST_CASE("concepts_vs: test test", "[concepts_vs]") {
  REQUIRE(true);
}

// Concepts:
// 1. subscriptable_container
// 2. callable
// 3. callable_range
// 4. dimensionable
// 5. vectorable
// 6. partitionable
// 7. feature_vector
// 8. feature_vector_range
// 9. contiguous_feature_vector_range

TEST_CASE("concepts_vs: Vector", "[concepts_vs]") {
  CHECK(subscriptable_container<Vector<int>>);
  CHECK(subscriptable_container<Vector<double>>);
  CHECK(subscriptable_container<Vector<bool>>);

  CHECK(callable_range<Vector<int>>);
  CHECK(callable_range<Vector<double>>);
  CHECK(callable_range<Vector<bool>>);

  CHECK(dimensionable<Vector<int>>);
  CHECK(dimensionable<Vector<double>>);
  CHECK(dimensionable<Vector<bool>>);

  CHECK(!vectorable<Vector<int>>);
  CHECK(!vectorable<Vector<double>>);
  CHECK(!vectorable<Vector<bool>>);

  CHECK(feature_vector<Vector<int>>);
  CHECK(feature_vector<Vector<double>>);
  CHECK(feature_vector<Vector<bool>>);  // @todo ? Vector of bool is not contiguous?

  CHECK(query_vector<Vector<int>>);
  CHECK(query_vector<Vector<double>>);
  CHECK(query_vector<Vector<bool>>);   // @todo ? Vector of bool is not contiguous?

  CHECK(!feature_vector_range<Vector<int>>);
  CHECK(!feature_vector_range<Vector<double>>);
  CHECK(!feature_vector_range<Vector<bool>>);

  CHECK(!contiguous_feature_vector_range<Vector<int>>);
  CHECK(!contiguous_feature_vector_range<Vector<double>>);
  CHECK(!contiguous_feature_vector_range<Vector<bool>>);

  CHECK(!partitioned_feature_vector_range<Vector<int>>);
  CHECK(!partitioned_feature_vector_range<Vector<double>>);
  CHECK(!partitioned_feature_vector_range<Vector<bool>>);

  CHECK(!contiguous_partitioned_feature_vector_range<Vector<int>>);
  CHECK(!contiguous_partitioned_feature_vector_range<Vector<double>>);
  CHECK(!contiguous_partitioned_feature_vector_range<Vector<bool>>);
}

template <dimensionable D>
auto _dimensionable(const D& d) {
  return dimension(d);
}

auto test_dimensionable(const Matrix<int>& d) {
  return _dimensionable(d);
}

template <feature_vector_range D>
auto _feature_vector_range(const D& d) {
  return num_vectors(d);
}
auto test_feature_vector_range(const Matrix<int>& d) {
  return _feature_vector_range(d);
}

TEST_CASE("concepts_vs: Matrix", "[concepts_vs]") {
  CHECK(!subscriptable_container<Matrix<int>>);
  CHECK(!subscriptable_container<Matrix<double>>);
  CHECK(!subscriptable_container<Matrix<bool>>);

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
  CHECK(!feature_vector<Matrix<bool>>);  // @todo ? Matrix of bool is not contiguous?

  CHECK(!query_vector<Matrix<int>>);
  CHECK(!query_vector<Matrix<double>>);
  CHECK(!query_vector<Matrix<bool>>);   // @todo ? Matrix of bool is not contiguous?

  CHECK(feature_vector_range<Matrix<int>>);
  CHECK(feature_vector_range<Matrix<double>>);
  CHECK(feature_vector_range<Matrix<bool>>);

  CHECK(contiguous_feature_vector_range<Matrix<int>>);
  CHECK(contiguous_feature_vector_range<Matrix<double>>);
  CHECK(contiguous_feature_vector_range<Matrix<bool>>);

  CHECK(!partitioned_feature_vector_range<Matrix<int>>);
  CHECK(!partitioned_feature_vector_range<Matrix<double>>);
  CHECK(!partitioned_feature_vector_range<Matrix<bool>>);

  CHECK(!contiguous_partitioned_feature_vector_range<Matrix<int>>);
  CHECK(!contiguous_partitioned_feature_vector_range<Matrix<double>>);
  CHECK(!contiguous_partitioned_feature_vector_range<Matrix<bool>>);
}

TEST_CASE("concepts_vs: tdbMatrix", "[concepts_vs]") {
  REQUIRE(true);
}

TEST_CASE("concepts_vs: tdbPartitionedMatrix", "[concepts_vs]") {
  REQUIRE(true);
}

