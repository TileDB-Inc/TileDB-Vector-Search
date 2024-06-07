/**
 * @file   unit_matrix_with_ids.cc
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
#include <type_traits>
#include <typeinfo>
#include <vector>
#include "cpos.h"
#include "detail/linalg/matrix_with_ids.h"
#include "mdspan/mdspan.hpp"

TEMPLATE_TEST_CASE(
    "template arguments", "[matrix_with_ids]", char, float, int32_t, int64_t) {
  auto vectors = std::unique_ptr<float[]>(new float[100]);
  auto ids = std::unique_ptr<TestType[]>(new TestType[100]);
  auto matrix = MatrixWithIds<float, TestType>{
      std::move(vectors), std::move(ids), 100, 1};
  CHECK(typeid(decltype(matrix.ids()[0])) == typeid(TestType));
}

TEMPLATE_TEST_CASE(
    "move constructor",
    "[matrix_with_ids]",
    stdx::layout_right,
    stdx::layout_left) {
  size_t rows = 5;
  size_t cols = 10;
  auto vectors = std::unique_ptr<float[]>(new float[rows * cols]);
  auto expectedNumVectors =
      std::is_same<TestType, stdx::layout_right>::value ? rows : cols;
  auto expectedDimension =
      std::is_same<TestType, stdx::layout_right>::value ? cols : rows;
  auto ids = std::unique_ptr<size_t[]>(new size_t[expectedNumVectors]);
  auto matrix = MatrixWithIds<float, size_t, TestType>{
      std::move(vectors), std::move(ids), rows, cols};
  CHECK(matrix.num_rows() == rows);
  CHECK(matrix.num_cols() == cols);
  CHECK(dimensions(matrix) == expectedDimension);
  CHECK(num_vectors(matrix) == expectedNumVectors);
  CHECK(size(matrix.raveled_ids()) == expectedNumVectors);
  CHECK(matrix.num_ids() == expectedNumVectors);
}

TEMPLATE_TEST_CASE(
    "size constructor", "[matrix_with_ids]", char, float, int32_t, int64_t) {
  auto row_matrix =
      MatrixWithIds<TestType, TestType, stdx::layout_right, size_t>{2, 5};
  CHECK(row_matrix.num_rows() == 2);
  CHECK(row_matrix.num_cols() == 5);
  CHECK(dimensions(row_matrix) == 5);
  CHECK(num_vectors(row_matrix) == 2);
  CHECK(row_matrix.num_ids() == 2);
  CHECK(size(row_matrix.raveled_ids()) == 2);

  auto col_matrix =
      MatrixWithIds<TestType, TestType, stdx::layout_left, size_t>{2, 5};
  CHECK(col_matrix.num_rows() == 2);
  CHECK(col_matrix.num_cols() == 5);
  CHECK(dimensions(col_matrix) == 2);
  CHECK(num_vectors(col_matrix) == 5);
  CHECK(col_matrix.num_ids() == 5);
  CHECK(size(col_matrix.raveled_ids()) == 5);
}

TEMPLATE_TEST_CASE(
    "initializer list",
    "[matrix_with_ids]",
    stdx::layout_right,
    stdx::layout_left) {
  auto A = MatrixWithIds<float, size_t, TestType>{
      {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}}, {1, 2, 3, 4}};

  auto a = std::vector<float>{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
  auto idsData = std::vector<size_t>{1, 2, 3, 4};
  auto raveled = A.raveled();
  auto ids = A.raveled_ids();

  CHECK(A.num_rows() * A.num_cols() == a.size());
  CHECK(
      std::equal(A.data(), A.data() + A.num_rows() * A.num_cols(), a.begin()));
  CHECK(std::equal(raveled.begin(), raveled.end(), a.begin()));

  CHECK(A.num_ids() == idsData.size());
  CHECK(std::equal(A.ids(), A.ids() + A.num_ids(), idsData.begin()));
  CHECK(std::equal(ids.begin(), ids.end(), idsData.begin()));
  CHECK(typeid(decltype(A.ids()[0])) == typeid(size_t));
  CHECK(A.ids()[0] == 1);
  CHECK(A.ids()[1] == 2);
  CHECK(A.ids()[2] == 3);
  CHECK(A.ids()[3] == 4);
}

TEMPLATE_TEST_CASE(
    "copy", "[matrix_with_ids]", stdx::layout_right, stdx::layout_left) {
  auto A = MatrixWithIds<float, float, TestType>{
      {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}}, {1, 2, 3, 4}};

  auto aptr = A.data();
  auto ptrIds = A.ids();

  auto B{std::move(A)};
  auto raveled = B.raveled();
  auto ids = B.raveled_ids();

  CHECK(aptr == B.data());
  CHECK(A.data() == nullptr);
  CHECK(ptrIds == B.ids());
  CHECK(A.raveled_ids().size() == 4);

  auto matrixData = std::vector<float>{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
  CHECK(std::equal(
      B.data(), B.data() + B.num_rows() * B.num_cols(), matrixData.begin()));
  CHECK(std::equal(raveled.begin(), raveled.end(), matrixData.begin()));

  auto idsData = std::vector<float>{1, 2, 3, 4};
  CHECK(B.num_ids() == idsData.size());
  CHECK(std::equal(B.ids(), B.ids() + B.num_ids(), idsData.begin()));
  CHECK(std::equal(ids.begin(), ids.end(), idsData.begin()));
  CHECK(B.ids()[0] == 1);
  CHECK(B.ids()[3] == 4);
}

TEMPLATE_TEST_CASE(
    "assign", "[matrix_with_ids]", stdx::layout_right, stdx::layout_left) {
  auto A = MatrixWithIds<float, float, TestType>{
      {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}, {0, 1, 2, 3}};
  auto a = std::vector<float>{8, 6, 7, 5, 3, 0, 9, 5, 0, 2, 7, 3};
  auto ids = std::vector<float>{0, 1, 2, 3};

  auto B = MatrixWithIds<float, float, TestType>{
      {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}}, {100, 101, 102, 103}};

  auto aptr = A.data();
  auto aptrIds = A.ids();

  B = std::move(A);

  CHECK(
      std::equal(B.data(), B.data() + B.num_rows() * B.num_cols(), a.begin()));
  CHECK(B.num_ids() == ids.size());
  CHECK(std::equal(B.ids(), B.ids() + B.num_ids(), ids.begin()));

  CHECK(aptr == B.data());
  CHECK(aptrIds == B.ids());
  CHECK(A.data() == nullptr);
}

TEMPLATE_TEST_CASE(
    "assign to matrix",
    "[matrix_with_ids]",
    stdx::layout_right,
    stdx::layout_left) {
  auto A = MatrixWithIds<float, float, TestType, size_t>{
      {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}}, {0, 1, 2, 3}};
  auto a = std::vector<float>{8, 6, 7, 5, 3, 0, 9, 5, 0, 2, 7, 3};
  auto ids = std::vector<float>{0, 1, 2, 3};
  auto aptr = A.data();
  auto aptrIds = A.ids();

  auto B = Matrix<float, TestType, size_t>{
      {{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}}};
  B = std::move(A);
  auto bptr = A.data();
  CHECK(
      std::equal(B.data(), B.data() + B.num_rows() * B.num_cols(), a.begin()));
  CHECK(aptr == B.data());
  CHECK(A.data() == nullptr);
}

TEMPLATE_TEST_CASE(
    "vector of matrix",
    "[matrix_with_ids]",
    stdx::layout_right,
    stdx::layout_left) {
  std::vector<MatrixWithIds<float, float, TestType>> v;

  auto numIds = 3;
  auto A = MatrixWithIds<float, float, TestType>{
      {{8, 6, 7}, {5, 3, 0}, {9, 5, 0}}, {0, 1, 2}};
  auto aptr = A.data();
  auto aptrIds = A.ids();

  SECTION("push_back and emplace_back") {
    SECTION("push_back") {
      // v.push_back(A);  // Error: no matching construct_at
      v.push_back(std::move(A));
    }

    SECTION("emplace_back") {
      // v.emplace_back(A);  // Error: no matching construct_at
      v.emplace_back(std::move(A));
    }

    SECTION("reserve and push_back") {
      v.reserve(10);
      // v.push_back(A);  // Error: no matching construct_at
      v.push_back(std::move(A));
    }

    SECTION("reserve and emplace_back") {
      v.reserve(10);
      // v.push_back(A);  // Error: no matching construct_at
      v.emplace_back(std::move(A));
    }
    CHECK(v.size() == 1);
    CHECK(v[0].data() == aptr);
    CHECK(A.data() == nullptr);
    CHECK(v[0].num_ids() == numIds);
    CHECK(v[0].ids() == aptrIds);
    CHECK(A.num_ids() == 3);
    CHECK(A.raveled_ids().size() == 3);
  }

  SECTION("operator[]") {
    std::vector<MatrixWithIds<float, float, TestType>> x;

    SECTION("operator[]") {
      std::vector<MatrixWithIds<float, float, TestType>> w;
      w.reserve(10);
      w.emplace_back();
      // w[0] = A; // Error: no matching construct_at
      w[0] = std::move(A);
      CHECK(w[0].data() == aptr);
      CHECK(w[0].num_ids() == numIds);
      CHECK(w[0].ids() == aptrIds);
    }
    SECTION("resize and operator[]") {
      x.resize(10);
      x[0] = std::move(A);
      CHECK(x[0].data() == aptr);
      CHECK(x[0].num_ids() == numIds);
      CHECK(x[0].ids() == aptrIds);
    }
    CHECK(A.data() == nullptr);
    CHECK(A.num_ids() == 3);
    CHECK(A.raveled_ids().size() == 3);
  }
}

TEMPLATE_TEST_CASE("view", "[matrix_with_ids]", char, float, int32_t, int64_t) {
  size_t major = 7;
  size_t minor = 13;
  auto v = std::vector<TestType>(major * minor);
  TestType* t = v.data();
  std::iota(v.begin(), v.end(), 0);

  auto mda = Kokkos::mdspan(t, major, minor);
  CHECK(mda.extent(0) == major);
  CHECK(mda.extent(1) == minor);
  CHECK(mda(0, 0) == 0);
  CHECK(mda(0, 1) == 1);

  auto a =
      Kokkos::mdspan<TestType, stdx::dextents<size_t, 2>, Kokkos::layout_right>(
          t, major, minor);
  CHECK(a.extent(0) == major);
  CHECK(a.extent(1) == minor);
  CHECK(a(0, 0) == 0);
  CHECK(a(0, 1) == 1);
  CHECK(a(1, 0) == 13);

  auto b =
      Kokkos::mdspan<TestType, stdx::dextents<size_t, 2>, Kokkos::layout_left>(
          t, major, minor);
  CHECK(b.extent(0) == major);
  CHECK(b.extent(1) == minor);
  CHECK(b(0, 0) == 0);
  CHECK(b(1, 0) == 1);
  CHECK(b(0, 1) == 7);

  auto ids = std::vector<TestType>(major);
  std::iota(ids.begin(), ids.end(), 0);
  auto c = RowMajorMatrixWithIds<TestType, TestType>(major, minor);
  std::copy(v.begin(), v.end(), c.data());
  std::copy(ids.begin(), ids.end(), c.ids());
  CHECK(c(0, 0) == 0);
  CHECK(
      c(1, 0) == 13);  // 0 + 1 * 13 => j + i * extents(1) => minor = extents(1)
  CHECK(c(0, 1) == 1);  // 1 + 0 * 13
  CHECK(c.ids()[0] == 0);
  CHECK(c.ids()[1] == 1);
  CHECK(c.ids()[5] == 5);
  CHECK(c.num_ids() == ids.size());
  CHECK(c.num_rows() == major);
  CHECK(c.num_cols() == minor);
  CHECK(num_vectors(c) == major);
  CHECK(dimensions(c) == minor);
  CHECK(typeid(decltype(c.ids()[2])) == typeid(TestType));

  CHECK(std::equal(c.ids(), c.ids() + c.num_ids(), ids.begin()));

  auto mc =
      Kokkos::mdspan<TestType, stdx::dextents<size_t, 2>, Kokkos::layout_right>(
          t, major, minor);
  CHECK(c.extent(0) == major);
  CHECK(c.extent(1) == minor);
  CHECK(mc(0, 0) == 0);
  CHECK(
      mc(0, 1) ==
      1);  // j + extents(1) * i => j + minor * i => j + num_cols * i
  CHECK(mc(1, 0) == 13);  // 0 + 1 * 13
  CHECK(num_vectors(mc) == major);
  CHECK(dimensions(mc) == minor);

  auto x = c[1];
  CHECK(x[0] == 13);  // == mc(i, 0)  = data() + i * minor
  CHECK(x[1] == 14);
  CHECK(size(x) == minor);

  auto cv = MatrixView(
      (TestType*)mc.data_handle(), (size_t)mc.extent(0), (size_t)mc.extent(1));
  CHECK(cv[0][0] == 0);
  CHECK(cv[1][0] == 13);
  CHECK(cv[1][1] == 14);
  CHECK(cv.num_rows() == major);
  CHECK(cv.num_cols() == minor);
  CHECK(num_vectors(cv) == major);
  CHECK(dimensions(cv) == minor);

  auto fv = MatrixView(mc);
  auto fz = fv[1];
  CHECK(fz[0] == 13);
  CHECK(fz[1] == 14);
  CHECK(fv.num_rows() == major);
  CHECK(fv.num_cols() == minor);
  CHECK(num_vectors(cv) == major);
  CHECK(dimensions(cv) == minor);

  ids = std::vector<TestType>(minor);
  std::iota(ids.begin(), ids.end(), 0);
  auto d = ColMajorMatrixWithIds<TestType, TestType>(major, minor);
  std::copy(v.begin(), v.end(), d.data());
  std::copy(ids.begin(), ids.end(), d.ids());
  CHECK(d.num_ids() == ids.size());
  CHECK(d(0, 0) == 0);
  CHECK(d(0, 1) == 7);  // 0 + 1 * 7
  CHECK(d(1, 0) == 1);  // 1 + 0 * 7 => i + j * extents(0) => major = extents(0)
  CHECK(d.ids()[0] == 0);
  CHECK(d.ids()[10] == 10);
  CHECK(d.num_rows() == major);
  CHECK(d.num_cols() == minor);
  CHECK(num_vectors(cv) == major);
  CHECK(dimensions(cv) == minor);
  CHECK(typeid(decltype(d.ids()[5])) == typeid(TestType));

  CHECK(std::equal(d.ids(), d.ids() + d.num_ids(), ids.begin()));

  // Column major
  auto md =
      Kokkos::mdspan<TestType, stdx::dextents<size_t, 2>, Kokkos::layout_left>(
          t, major, minor);
  CHECK(md.extent(0) == major);
  CHECK(md.extent(1) == minor);
  CHECK(
      md(0, 0) ==
      0);  // i + major * j => i + num_rows * j => i + extents(0) * j
  CHECK(md(0, 1) == 7);
  CHECK(md(1, 0) == 1);
  CHECK(d.num_rows() == major);
  CHECK(d.num_cols() == minor);
  CHECK(d.num_ids() == minor);
  CHECK(num_vectors(cv) == major);
  CHECK(dimensions(cv) == minor);

  auto y = d[1];
  CHECK(y[0] == 7);  // == md(0, i)  = data() + i * major
  CHECK(y[1] == 8);

  auto dv = MatrixView<TestType, Kokkos::layout_left, size_t>(
      (TestType*)md.data_handle(), (size_t)md.extent(0), (size_t)md.extent(1));
  CHECK(dv[0][0] == 0);
  CHECK(dv[1][0] == 7);
  CHECK(dv[1][1] == 8);

  auto ev = MatrixView(md);
  auto ez = ev[1];
  CHECK(ez[0] == 7);
  CHECK(ez[1] == 8);
}
