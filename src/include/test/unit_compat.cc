/**
* @file   unit_compat.cc
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
* Tests for compatibility wrappers.
*
*/

#include <catch2/catch_all.hpp>
#include "detail/linalg/compat.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/partitioned_matrix.h"

TEST_CASE("compat: test test", "[compat]") {
  REQUIRE(true);
}

TEST_CASE("compat: is feature vector array", "[compat]") {
  REQUIRE(feature_vector_array<Matrix<int>>);
  REQUIRE(feature_vector_array<ColMajorMatrix<int>>);
  REQUIRE(feature_vector_array<RowMajorMatrix<int>>);

  REQUIRE(partitioned_feature_vector_array<PartitionedMatrix<int, int, int>>);
  REQUIRE(partitioned_feature_vector_array<ColMajorPartitionedMatrix<int, int, int>>);
  REQUIRE(partitioned_feature_vector_array<RowMajorPartitionedMatrix<int, int, int>>);

  auto foo = ::dimension(PartitionedMatrix<int, int, int>{});

  REQUIRE(partitioned_feature_vector_array<PartitionedMatrixWrapper<int, int, int>>);
  REQUIRE(partitioned_feature_vector_array<ColMajorPartitionedMatrixWrapper<int, int, int>>);
  REQUIRE(partitioned_feature_vector_array<RowMajorPartitionedMatrixWrapper<int, int, int>>);
}

