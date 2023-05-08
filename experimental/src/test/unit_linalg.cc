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

#include <catch2/catch_all.hpp>
#include <tuple>
#include "../linalg.h"


using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

TEST_CASE("linalg: test test", "[linalg]") {
  REQUIRE(true);
}


TEMPLATE_LIST_TEST_CASE("linalg: test Vector constructor", "[linalg]", TestTypes) {
  auto a = Vector<TestType>(7);
  auto v = a.data();
  std::iota(v, v + 7, 1);

  SECTION("dims") {
    CHECK(a.num_rows() == 7);
  }
  SECTION("values") {
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK(a(i) == i + 1);
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      a(i) *= a[i];
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK(a(i) == (i + 1)*(i + 1));
    }
  }
  SECTION("values, copy") {
    auto b = std::move(a);
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK(b(i) == i + 1);
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      b(i) *= b[i];
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK(b(i) == (i + 1)*(i + 1));
    }
  }
}

TEMPLATE_LIST_TEST_CASE(
    "linalg: test Matrix constructor, default oriented", "[linalg]", TestTypes) {
  auto a = Matrix<TestType>(3, 2);
  auto v = a.data();
  std::iota(v, v + 6, 1);

  SECTION("dims") {
    CHECK(a.num_rows() == 3);
    CHECK(a.num_cols() == 2);
  }
  SECTION("values") {
    CHECK(a(0, 0) == 1);
    CHECK(a(0, 1) == 2);
    CHECK(a(1, 0) == 3);
    CHECK(a(1, 1) == 4);
    CHECK(a(2, 0) == 5);
    CHECK(a(2, 1) == 6);
  }
  SECTION("ravel") {
    auto b = raveled(a);
    CHECK(std::equal(b.begin(), b.end(), a.data()));
  }
  SECTION("operator[]") {
    auto b = a[0];
    CHECK(size(b) == 2);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
    CHECK(b[3] == 4);
    CHECK(b[4] == 5);
    CHECK(b[5] == 6);
  }
}


TEMPLATE_LIST_TEST_CASE(
    "linalg: test Matrix constructor, row oriented", "[linalg]", TestTypes) {
  auto a = Matrix<TestType, Kokkos::layout_right>(3, 2);
  auto v = a.data();
  std::iota(v, v + 6, 1);

  SECTION("dims") {
    CHECK(a.num_rows() == 3);
    CHECK(a.num_cols() == 2);
  }
  SECTION("values") {
    CHECK(a(0, 0) == 1);
    CHECK(a(0, 1) == 2);
    CHECK(a(1, 0) == 3);
    CHECK(a(1, 1) == 4);
    CHECK(a(2, 0) == 5);
    CHECK(a(2, 1) == 6);
  }
  SECTION("ravel") {
    auto b = raveled(a);
    CHECK(std::equal(b.begin(), b.end(), a.data()));
  }
  SECTION("operator[]") {
    auto b = a[0];
    CHECK(size(b) == 2);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);
    CHECK(b[3] == 4);
    CHECK(b[4] == 5);
    CHECK(b[5] == 6);
  }
}

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;
TEMPLATE_LIST_TEST_CASE(
    "linalg: test Matrix constructor, column oriented", "[linalg]", TestTypes) {
  auto a = Matrix<TestType, Kokkos::layout_left>(3, 2);
  auto v = a.data();
  std::iota(v, v + 6, 1);

  SECTION("dims") {
    CHECK(a.num_rows() == 3);
    CHECK(a.num_cols() == 2);
  }
  SECTION("values") {
    CHECK(a(0, 0) == 1);
    CHECK(a(0, 1) == 4);
    CHECK(a(1, 0) == 2);
    CHECK(a(1, 1) == 5);
    CHECK(a(2, 0) == 3);
    CHECK(a(2, 1) == 6);
  }
  SECTION("ravel") {
    auto b = raveled(a);
    CHECK(std::equal(b.begin(), b.end(), a.data()));
  }
  SECTION("operator[]") {
    auto b = a[0];
    CHECK(size(b) == 3);
    CHECK(b[0] == 1);
    CHECK(b[1] == 2);
    CHECK(b[2] == 3);

    auto c = a[1];
    CHECK(size(c) == 3);
    CHECK(c[0] == 4);
    CHECK(c[1] == 5);
    CHECK(c[2] == 6);

    CHECK(a[0][0] == 1);
    CHECK(a[0][1] == 2);
    CHECK(a[0][2] == 3);
    CHECK(a[1][0] == 4);
    CHECK(a[1][1] == 5);
    CHECK(a[1][2] == 6);
  }
}

template <class TestType>
auto make_matrix(size_t num_rows, size_t num_cols) {
  auto a = Matrix<TestType, Kokkos::layout_right>(3, 2);
  auto v = a.data();
  std::iota(v, v + 6, 1);
  //return std::make_tuple(a, a.data(), v);
  return std::make_tuple(std::move(a), v);
};

TEST_CASE("linalg: test Matrix copy constructor, row oriented", "[linalg]") {
  auto&& [a, v] = make_matrix<float>(3, 2);
  CHECK(a.data() == v);
  auto b = a[0];
  CHECK(size(b) == 2);
  CHECK(b[0] == 1);
  CHECK(b[1] == 2);
  CHECK(b[2] == 3);
  CHECK(b[3] == 4);
  CHECK(b[4] == 5);
  CHECK(b[5] == 6);
}

TEST_CASE("linalg: test tdbMatrix constructor, row", "[linalg]") {
// d1, d2, val_1
//  data = np.array([
//    [8, 6, 7, 5, 3, 1, 4, 1],
//    [3, 0, 9, 9, 5, 9, 2, 7],
//    [9, 8, 6, 7, 2, 6, 4, 3],
//    [5, 3, 0, 9, 4, 2, 2, 4]], dtype=np.int32)

  try {
    auto a = tdbMatrix<int32_t>("array_dense_1");

    CHECK(a.num_rows() == 4);
    CHECK(a.num_cols() == 8);

    CHECK(a(0, 0) == 8);
    CHECK(a(0, 1) == 6);
    CHECK(a(0, 2) == 7);
    CHECK(a(0, 3) == 5);
    CHECK(a(1, 0) == 3);
    CHECK(a(3, 7) == 4);

  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  }
}

TEST_CASE("linalg: test tdbMatrix constructor, column", "[linalg]") {

  auto a = tdbMatrix<int32_t, Kokkos::layout_left>("array_dense_1");

  CHECK(a.num_rows() == 4);
  CHECK(a.num_cols() == 8);

  CHECK(a(0, 0) == 8);
  CHECK(a(1, 0) == 6);
  CHECK(a(2, 0) == 7);
  CHECK(a(3, 0) == 5);
  CHECK(a(0, 1) == 3);
  CHECK(a(3, 7) == 4);
}
