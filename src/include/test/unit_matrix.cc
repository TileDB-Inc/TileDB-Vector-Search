/**
 * @file   unit_linalg.cc
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
#include "mdspan/mdspan.hpp"

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

/*
 * Many tests remain in linalg.cc from before the refactor.
 */
TEST_CASE("initializer list", "[matrix]") {
  auto A = Matrix<float>{{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};
  auto a = std::vector<float>{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
  CHECK(
      std::equal(A.data(), A.data() + A.num_rows() * A.num_cols(), a.begin()));
}

TEST_CASE("copy", "[matrix]") {
  auto A = Matrix<float>{{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};
  auto a = std::vector<float>{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
  auto aptr = A.data();
  //  auto B = A;  // Error: copy constructor is deleted

  auto C{std::move(A)};  // move constructor
  auto cptr = C.data();

  CHECK(
      std::equal(C.data(), C.data() + C.num_rows() * C.num_cols(), a.begin()));
  CHECK(aptr == cptr);
  CHECK(A.data() == nullptr);
}

TEST_CASE("assign", "[matrix]") {
  auto A = Matrix<float>{{8, 6, 7}, {5, 3, 0}, {9, 5, 0}, {2, 7, 3}};
  auto a = std::vector<float>{8, 6, 7, 5, 3, 0, 9, 5, 0, 2, 7, 3};

  auto B = Matrix<float>{{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};

  auto aptr = A.data();

  // B = A;  // Error: copy assignment is deleted

  B = std::move(A);  // move assignment
  CHECK(
      std::equal(B.data(), B.data() + B.num_rows() * B.num_cols(), a.begin()));

  auto bptr = B.data();
  CHECK(aptr == bptr);
  CHECK(A.data() == nullptr);
}

TEST_CASE("vector of matrix", "[matrix]") {
  std::vector<Matrix<float>> v;

  auto A = Matrix<float>{{8, 6, 7}, {5, 3, 0}, {9, 5, 0}};
  auto aptr = A.data();

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
  }

  SECTION("operator[]") {
    std::vector<Matrix<float>> x;

    SECTION("operator[]") {
      std::vector<Matrix<float>> w(1);
      w.reserve(10);
      // w[0] = A; // Error: no matching construct_at
      w[0] = std::move(A);
      CHECK(w[0].data() == aptr);
    }
    SECTION("resize and operator[]") {
      x.resize(10);
      x[0] = std::move(A);
      CHECK(x[0].data() == aptr);
    }
    CHECK(A.data() == nullptr);
  }
}

TEMPLATE_TEST_CASE("view", "[matrix]", char, float, int32_t, int64_t) {
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
  CHECK(b(0, 0) == 0);  // M(0, 0)
  CHECK(b(1, 0) == 1);  // M(1,0)
  CHECK(b(0, 1) == 7);  // M(1,0)

#if 0
  // Extents are major and minor -- 7, 13 -- each row is 13 elements long
  for (ptrdiff_t i = 0; i < s.extent(0); ++i) {
    for (ptrdiff_t j = 0; j < s.extent(1); ++j) {
      sum += s(i, j);
    }
  }
  for (ptrdiff_t i = 0; i < x; ++i) {
    for (ptrdiff_t j = 0; j < y; ++j){
      sum += s_ptr[j + i*y];
    }
  }
#endif

  // row idx = j + i * extents(1)
  // col idx = i + j * extents(0)

  auto c = RowMajorMatrix<TestType>(major, minor);
  std::copy(v.begin(), v.end(), c.data());
  CHECK(c(0, 0) == 0);
  CHECK(
      c(1, 0) == 13);  // 0 + 1 * 13 => j + i * extents(1) => minor = extents(1)
  CHECK(c(0, 1) == 1);  // 1 + 0 * 13
  CHECK(c.num_rows() == major);
  CHECK(c.num_cols() == minor);
  CHECK(num_vectors(c) == major);
  CHECK(dimensions(c) == minor);

  auto mc =
      Kokkos::mdspan<TestType, stdx::dextents<size_t, 2>, Kokkos::layout_right>(
          t, major, minor);
  CHECK(mc.extent(0) == major);
  CHECK(mc.extent(1) == minor);
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

  auto d = ColMajorMatrix<TestType>(major, minor);
  std::copy(v.begin(), v.end(), d.data());
  CHECK(d(0, 0) == 0);
  CHECK(d(0, 1) == 7);  // 0 + 1 * 7
  CHECK(d(1, 0) == 1);  // 1 + 0 * 7 => i + j * extents(0) => major = extents(0)
  CHECK(d.num_rows() == major);
  CHECK(d.num_cols() == minor);
  CHECK(num_vectors(cv) == major);
  CHECK(dimensions(cv) == minor);

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
