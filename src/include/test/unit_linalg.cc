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
#include <cstdio>
#include <filesystem>
#include <tuple>
#include "linalg.h"
#include "utils/utils.h"

using TestTypes =
    std::tuple<float, uint8_t, double, int, char, size_t, uint32_t>;

TEMPLATE_LIST_TEST_CASE("test mdspan", "[linalg][mdspan]", TestTypes) {
  size_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  size_t N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  TestType* t = nullptr;
  auto m = Kokkos::mdspan(t, M, N);
  CHECK(m.size() == M * N);
  CHECK(m.rank() == 2);
}

TEMPLATE_LIST_TEST_CASE("test span", "[linalg][span]", TestTypes) {
}

TEMPLATE_LIST_TEST_CASE(
    "test Vector constructor", "[linalg][vector][create]", TestTypes) {
  auto a = Vector<TestType>(7);
  auto v = a.data();
  std::iota(v, v + 7, 1);

  SECTION("dims") {
    CHECK(a.num_rows() == 7);
  }
  SECTION("values") {
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK((size_t)a(i) == i + 1);
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      a(i) *= a[i];
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK((size_t)a(i) == (i + 1) * (i + 1));
    }
  }
  SECTION("values, copy") {
    auto b = std::move(a);
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK((size_t)b(i) == i + 1);
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      b(i) *= b[i];
    }
    for (size_t i = 0; i < a.num_rows(); ++i) {
      CHECK((size_t)b(i) == (i + 1) * (i + 1));
    }
  }
}

TEMPLATE_LIST_TEST_CASE(
    "test Matrix constructor, default oriented",
    "[linalg][matrix][create][default]",
    TestTypes) {
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
    "test Matrix constructor, row oriented",
    "[linalg][matrix][create][row]",
    TestTypes) {
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

TEMPLATE_LIST_TEST_CASE(
    "test Matrix constructor, column oriented",
    "[linalg][matrixx][create][column]",
    TestTypes) {
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

TEMPLATE_LIST_TEST_CASE(
    "test Matrix initializer_list constructor, row oriented",
    "[linalg][matrix][create][row]",
    TestTypes) {
  auto a = Matrix<TestType, Kokkos::layout_right>{{1, 2}, {3, 4}, {5, 6}};
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
    "test Matrix initializer list constructor, column oriented",
    "[linalg][matrixx][create][column]",
    TestTypes) {
  auto a = Matrix<TestType, Kokkos::layout_left>{{1, 2, 3}, {4, 5, 6}};
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
  // return std::make_tuple(a, a.data(), v);
  return std::make_tuple(std::move(a), v);
};

TEST_CASE(
    "test Matrix copy constructor, row oriented",
    "[linalg][matrix][copy-create][row]") {
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

#ifdef TDB_ROW_MATRIX
TEST_CASE(
    "test tdbMatrix constructor, row", "[linalg][tdbmatrix][create][row]") {
  // d1, d2, val_1
  //  data = np.array([
  //    [8, 6, 7, 5, 3, 1, 4, 1],
  //    [3, 0, 9, 9, 5, 9, 2, 7],
  //    [9, 8, 6, 7, 2, 6, 4, 3],
  //    [5, 3, 0, 9, 4, 2, 2, 4]], dtype=np.int32)

  try {
    tiledb::Context ctx;
    std::vector<int32_t> data = {8, 6, 7, 5, 3, 1, 4, 1, 3, 0, 9,
                                 9, 5, 9, 2, 7, 9, 8, 6, 7, 2, 6,
                                 4, 3, 5, 3, 0, 9, 4, 2, 2, 4};  // OMG
    auto a = tdbMatrix<int32_t>(ctx, "array_dense_1");

    CHECK(a.num_rows() == 4);
    CHECK(a.num_cols() == 8);
    CHECK(std::equal(data.begin(), data.end(), a.data()));

    CHECK(a(0, 0) == 8);
    CHECK(a(0, 1) == 6);
    CHECK(a(0, 2) == 7);
    CHECK(a(0, 3) == 5);
    CHECK(a(0, 4) == 3);
    CHECK(a(0, 5) == 1);
    CHECK(a(0, 6) == 4);
    CHECK(a(0, 7) == 1);

    auto b = a[0];
    CHECK(size(b) == 8);
    CHECK(b[0] == 8);
    CHECK(b[1] == 6);
    CHECK(b[2] == 7);
    CHECK(b[3] == 5);
    CHECK(b[4] == 3);
    CHECK(b[5] == 1);
    CHECK(b[6] == 4);
    CHECK(b[7] == 1);

    CHECK(a(1, 0) == 3);
    CHECK(a(1, 1) == 0);
    CHECK(a(1, 2) == 9);
    CHECK(a(1, 3) == 9);
    CHECK(a(1, 4) == 5);
    CHECK(a(1, 5) == 9);
    CHECK(a(1, 6) == 2);
    CHECK(a(1, 7) == 7);

    auto c = a[1];
    CHECK(size(c) == 8);
    CHECK(c[0] == 3);
    CHECK(c[1] == 0);
    CHECK(c[2] == 9);
    CHECK(c[3] == 9);
    CHECK(c[4] == 5);
    CHECK(c[5] == 9);
    CHECK(c[6] == 2);
    CHECK(c[7] == 7);

    CHECK(a(2, 0) == 9);
    CHECK(a(2, 1) == 8);
    CHECK(a(2, 2) == 6);
    CHECK(a(2, 3) == 7);
    CHECK(a(2, 4) == 2);
    CHECK(a(2, 5) == 6);
    CHECK(a(2, 6) == 4);
    CHECK(a(2, 7) == 3);

    auto d = a[2];
    CHECK(size(d) == 8);
    CHECK(d[0] == 9);
    CHECK(d[1] == 8);
    CHECK(d[2] == 6);
    CHECK(d[3] == 7);
    CHECK(d[4] == 2);
    CHECK(d[5] == 6);
    CHECK(d[6] == 4);
    CHECK(d[7] == 3);

    CHECK(a(3, 0) == 5);
    CHECK(a(3, 1) == 3);
    CHECK(a(3, 2) == 0);
    CHECK(a(3, 3) == 9);
    CHECK(a(3, 4) == 4);
    CHECK(a(3, 5) == 2);
    CHECK(a(3, 6) == 2);
    CHECK(a(3, 7) == 4);

    auto e = a[3];
    CHECK(size(e) == 8);
    CHECK(e[0] == 5);
    CHECK(e[1] == 3);
    CHECK(e[2] == 0);
    CHECK(e[3] == 9);
    CHECK(e[4] == 4);
    CHECK(e[5] == 2);
    CHECK(e[6] == 2);
    CHECK(e[7] == 4);

  } catch (const std::exception& e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  }
}
#endif

TEST_CASE("print cwd", "[linalg][cwd]") {
  std::filesystem::path currentPath = std::filesystem::current_path();
  std::cout << "Current Working Directory: " << currentPath << std::endl;
}

TEST_CASE(
    "test tdbMatrix constructor, column",
    "[linalg][tdbmatrix][create][column]") {
  std::vector<float> data = {
      8, 6, 7, 5, 3, 1, 4, 1, 3, 0, 9, 9, 5, 9, 2, 7,
      9, 8, 6, 7, 2, 6, 4, 3, 5, 3, 0, 9, 4, 2, 2, 4};  // OMG

  REQUIRE(local_array_exists("array_dense_1"));
  tiledb::Context ctx;
  auto a = tdbMatrix<float, Kokkos::layout_left>(ctx, "array_dense_2");
  a.load();

  CHECK(a.num_rows() == 8);
  CHECK(a.num_cols() == 4);
  CHECK(std::equal(data.begin(), data.end(), a.data()));

  CHECK(a(0, 0) == 8);
  CHECK(a(1, 0) == 6);
  CHECK(a(2, 0) == 7);
  CHECK(a(3, 0) == 5);
  CHECK(a(4, 0) == 3);
  CHECK(a(5, 0) == 1);
  CHECK(a(6, 0) == 4);
  CHECK(a(7, 0) == 1);

  auto b = a[0];
  CHECK(size(b) == 8);
  CHECK(b[0] == 8);
  CHECK(b[1] == 6);
  CHECK(b[2] == 7);
  CHECK(b[3] == 5);
  CHECK(b[4] == 3);
  CHECK(b[5] == 1);
  CHECK(b[6] == 4);
  CHECK(b[7] == 1);

  CHECK(a(0, 1) == 3);
  CHECK(a(1, 1) == 0);
  CHECK(a(2, 1) == 9);
  CHECK(a(3, 1) == 9);
  CHECK(a(4, 1) == 5);
  CHECK(a(5, 1) == 9);
  CHECK(a(6, 1) == 2);
  CHECK(a(7, 1) == 7);

  auto c = a[1];
  CHECK(size(c) == 8);
  CHECK(c[0] == 3);
  CHECK(c[1] == 0);
  CHECK(c[2] == 9);
  CHECK(c[3] == 9);
  CHECK(c[4] == 5);
  CHECK(c[5] == 9);
  CHECK(c[6] == 2);
  CHECK(c[7] == 7);

  CHECK(a(0, 2) == 9);
  CHECK(a(1, 2) == 8);
  CHECK(a(2, 2) == 6);
  CHECK(a(3, 2) == 7);
  CHECK(a(4, 2) == 2);
  CHECK(a(5, 2) == 6);
  CHECK(a(6, 2) == 4);
  CHECK(a(7, 2) == 3);

  auto d = a[2];
  CHECK(size(d) == 8);
  CHECK(d[0] == 9);
  CHECK(d[1] == 8);
  CHECK(d[2] == 6);
  CHECK(d[3] == 7);
  CHECK(d[4] == 2);
  CHECK(d[5] == 6);
  CHECK(d[6] == 4);
  CHECK(d[7] == 3);

  CHECK(a(0, 3) == 5);
  CHECK(a(1, 3) == 3);
  CHECK(a(2, 3) == 0);
  CHECK(a(3, 3) == 9);
  CHECK(a(4, 3) == 4);
  CHECK(a(5, 3) == 2);
  CHECK(a(6, 3) == 2);
  CHECK(a(7, 3) == 4);

  auto e = a[3];
  CHECK(size(e) == 8);
  CHECK(e[0] == 5);
  CHECK(e[1] == 3);
  CHECK(e[2] == 0);
  CHECK(e[3] == 9);
  CHECK(e[4] == 4);
  CHECK(e[5] == 2);
  CHECK(e[6] == 2);
  CHECK(e[7] == 4);
}

#ifdef TILEDB_ROW_MATRIX
TEST_CASE(
    "test partitioned tdbMatrix constructor, row",
    "[linalg][partitioned][tdbmatrix][create][row]") {
  size_t part = GENERATE(0, 1, 2, 3, 4);

  tiledb::Context ctx;
  auto a = tdbMatrix<int32_t, Kokkos::layout_right>(ctx, "array_dense_1", part);

  CHECK(a.num_rows() == (part == 0 ? 4 : part));
  CHECK(a.num_cols() == 8);

  if (part > 0) {
    CHECK(a(0, 0) == 8);
    CHECK(a(0, 1) == 6);
    CHECK(a(0, 2) == 7);
    CHECK(a(0, 3) == 5);
    CHECK(a(0, 4) == 3);
    CHECK(a(0, 5) == 1);
    CHECK(a(0, 6) == 4);
    CHECK(a(0, 7) == 1);
  }
  if (part > 1) {
    CHECK(a(1, 0) == 3);
    CHECK(a(1, 1) == 0);
    CHECK(a(1, 2) == 9);
    CHECK(a(1, 3) == 9);
    CHECK(a(1, 4) == 5);
    CHECK(a(1, 5) == 9);
    CHECK(a(1, 6) == 2);
    CHECK(a(1, 7) == 7);
  }
  if (part > 2) {
    CHECK(a(2, 0) == 9);
    CHECK(a(2, 1) == 8);
    CHECK(a(2, 2) == 6);
    CHECK(a(2, 3) == 7);
    CHECK(a(2, 4) == 2);
    CHECK(a(2, 5) == 6);
    CHECK(a(2, 6) == 4);
    CHECK(a(2, 7) == 3);

    auto d = a[2];
    CHECK(size(d) == 8);
    CHECK(d[0] == 9);
    CHECK(d[1] == 8);
    CHECK(d[2] == 6);
    CHECK(d[3] == 7);
    CHECK(d[4] == 2);
    CHECK(d[5] == 6);
    CHECK(d[6] == 4);
    CHECK(d[7] == 3);
  }
  if (part > 3 || part == 0) {
    CHECK(a(3, 0) == 5);
    CHECK(a(3, 1) == 3);
    CHECK(a(3, 2) == 0);
    CHECK(a(3, 3) == 9);
    CHECK(a(3, 4) == 4);
    CHECK(a(3, 5) == 2);
    CHECK(a(3, 6) == 2);
    CHECK(a(3, 7) == 4);
  }
}
#endif

TEST_CASE(
    "test partitioned tdbMatrix constructor, column",
    "[linalg][partitioned][tdbmatrix][create][column]") {
  REQUIRE(local_array_exists("array_dense_1"));
  size_t part = GENERATE(0, 1, 2, 3, 4);

  tiledb::Context ctx;
  auto a = tdbMatrix<float, Kokkos::layout_left>(ctx, "array_dense_2", part);
  a.load();

  if (part == 0) {
    part = 4;
  }
  CHECK(a.num_rows() == 8);
  CHECK(a.num_cols() == part);

  if (part > 0) {
    CHECK(a(0, 0) == 8);
    CHECK(a(1, 0) == 6);
    CHECK(a(2, 0) == 7);
    CHECK(a(3, 0) == 5);
    CHECK(a(4, 0) == 3);
    CHECK(a(5, 0) == 1);
    CHECK(a(6, 0) == 4);
    CHECK(a(7, 0) == 1);
  }

  if (part > 1) {
    CHECK(a(0, 1) == 3);
    CHECK(a(1, 1) == 0);
    CHECK(a(2, 1) == 9);
    CHECK(a(3, 1) == 9);
    CHECK(a(4, 1) == 5);
    CHECK(a(5, 1) == 9);
    CHECK(a(6, 1) == 2);
    CHECK(a(7, 1) == 7);
  }

  if (part > 2) {
    CHECK(a(0, 2) == 9);
    CHECK(a(1, 2) == 8);
    CHECK(a(2, 2) == 6);
    CHECK(a(3, 2) == 7);
    CHECK(a(4, 2) == 2);
    CHECK(a(5, 2) == 6);
    CHECK(a(6, 2) == 4);
    CHECK(a(7, 2) == 3);
  }

  if (part > 3) {
    CHECK(a(0, 3) == 5);
    CHECK(a(1, 3) == 3);
    CHECK(a(2, 3) == 0);
    CHECK(a(3, 3) == 9);
    CHECK(a(4, 3) == 4);
    CHECK(a(5, 3) == 2);
    CHECK(a(6, 3) == 2);
    CHECK(a(7, 3) == 4);
  }
}

#ifdef TILEDB_ROW_MATRIX
TEST_CASE("test advance, row major", "[linalg][tdbmatrix][advance][row]") {
  tiledb::Context ctx;
  auto a = tdbMatrix<float, Kokkos::layout_right>(ctx, "array_dense_1", 2);
  a.load();

  CHECK(a.num_rows() == 2);
  CHECK(a.num_cols() == 8);

  CHECK(a(0, 0) == 8);
  CHECK(a(0, 1) == 6);
  CHECK(a(0, 2) == 7);
  CHECK(a(0, 3) == 5);
  CHECK(a(0, 4) == 3);
  CHECK(a(0, 5) == 1);
  CHECK(a(0, 6) == 4);
  CHECK(a(0, 7) == 1);

  CHECK(a(1, 0) == 3);
  CHECK(a(1, 1) == 0);
  CHECK(a(1, 2) == 9);
  CHECK(a(1, 3) == 9);
  CHECK(a(1, 4) == 5);
  CHECK(a(1, 5) == 9);
  CHECK(a(1, 6) == 2);
  CHECK(a(1, 7) == 7);

  a.load();

  CHECK(a(0, 0) == 9);
  CHECK(a(0, 1) == 8);
  CHECK(a(0, 2) == 6);
  CHECK(a(0, 3) == 7);
  CHECK(a(0, 4) == 2);
  CHECK(a(0, 5) == 6);
  CHECK(a(0, 6) == 4);
  CHECK(a(0, 7) == 3);

  CHECK(a(1, 0) == 5);
  CHECK(a(1, 1) == 3);
  CHECK(a(1, 2) == 0);
  CHECK(a(1, 3) == 9);
  CHECK(a(1, 4) == 4);
  CHECK(a(1, 5) == 2);
  CHECK(a(1, 6) == 2);
  CHECK(a(1, 7) == 4);
}
#endif

TEST_CASE("test advance, column", "[linalg][tdbmatrix][advance][column]") {
  REQUIRE(local_array_exists("array_dense_1"));
  tiledb::Context ctx;
  auto a = tdbMatrix<float, Kokkos::layout_left>(ctx, "array_dense_2", 2);
  a.load();

  CHECK(a.num_rows() == 8);
  CHECK(a.num_cols() == 2);

  CHECK(a(0, 0) == 8);
  CHECK(a(1, 0) == 6);
  CHECK(a(2, 0) == 7);
  CHECK(a(3, 0) == 5);
  CHECK(a(4, 0) == 3);
  CHECK(a(5, 0) == 1);
  CHECK(a(6, 0) == 4);
  CHECK(a(7, 0) == 1);

  CHECK(a(0, 1) == 3);
  CHECK(a(1, 1) == 0);
  CHECK(a(2, 1) == 9);
  CHECK(a(3, 1) == 9);
  CHECK(a(4, 1) == 5);
  CHECK(a(5, 1) == 9);
  CHECK(a(6, 1) == 2);
  CHECK(a(7, 1) == 7);

  a.load();

  CHECK(a(0, 0) == 9);
  CHECK(a(1, 0) == 8);
  CHECK(a(2, 0) == 6);
  CHECK(a(3, 0) == 7);
  CHECK(a(4, 0) == 2);
  CHECK(a(5, 0) == 6);
  CHECK(a(6, 0) == 4);
  CHECK(a(7, 0) == 3);

  CHECK(a(0, 1) == 5);
  CHECK(a(1, 1) == 3);
  CHECK(a(2, 1) == 0);
  CHECK(a(3, 1) == 9);
  CHECK(a(4, 1) == 4);
  CHECK(a(5, 1) == 2);
  CHECK(a(6, 1) == 2);
  CHECK(a(7, 1) == 4);
}

TEMPLATE_LIST_TEST_CASE(
    "test write/read std::vector", "[linalg][read-write][vector]", TestTypes) {
  auto length = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  tiledb::Context ctx;

  auto a = std::vector<TestType>(length);
  std::iota(begin(a), end(a), 17);

  auto v = std::vector<TestType>(length);
  std::copy(begin(a), end(a), begin(v));

  auto tmpfilename = std::string(tmpnam(nullptr));
  auto tempDir = std::filesystem::temp_directory_path();
  auto uri = (tempDir / tmpfilename).string();
  write_vector(ctx, v, uri);
  auto w = read_vector<TestType>(ctx, uri);
  CHECK(std::equal(begin(v), end(v), begin(w)));
  std::filesystem::remove_all(uri);
}

using LayoutTypes = std::tuple<Kokkos::layout_right, Kokkos::layout_left>;

TEMPLATE_LIST_TEST_CASE(
    "test write/read Matrix",
    "[linalg][tdbmatrix][read-write][matrix]",
    TestTypes) {
  size_t M = GENERATE(1, 2, 13, 1440, 1441);
  size_t N = GENERATE(1, 2, 5, 1440, 1441);

  auto tmpfilename = std::string(tmpnam(nullptr));
  auto tempDir = std::filesystem::temp_directory_path();
  auto uri = (tempDir / tmpfilename).string();

  tiledb::Context ctx;

#ifdef TILEDB_ROW_MATRIX
  SECTION("right") {
    auto A = Matrix<TestType, Kokkos::layout_right>(M, N);
    std::iota(A.data(), A.data() + M * N, 17);

    write_matrix(ctx, A, uri);
    auto B = tdbMatrix<TestType, Kokkos::layout_right>(ctx, uri);
    B.load();

    CHECK(A.num_rows() == M);
    CHECK(A.num_cols() == N);
    CHECK(A.num_rows() == B.num_rows());
    CHECK(A.num_cols() == B.num_cols());
    CHECK(
        std::equal(A.data(), A.data() + A.num_rows() * A.num_cols(), B.data()));
  }
#endif

  SECTION("left") {
    auto A = Matrix<TestType, Kokkos::layout_left>(M, N);
    std::iota(A.data(), A.data() + M * N, 17);

    write_matrix(ctx, A, uri);
    auto B = tdbMatrix<TestType, Kokkos::layout_left>(ctx, uri);
    B.load();

    SECTION("sizes") {
      CHECK(A.num_rows() == M);
      CHECK(A.num_cols() == N);
      CHECK(A.num_rows() == B.num_rows());
      CHECK(A.num_cols() == B.num_cols());
    }
    SECTION("contents") {
      REQUIRE(std::equal(
          A.data(), A.data() + A.num_rows() * A.num_cols(), B.data()));
    }
    SECTION("complete contents") {
      for (size_t i = 0; i < M * N; ++i) {
        REQUIRE(A.data()[i] == B.data()[i]);
      }
    }
  }
}
