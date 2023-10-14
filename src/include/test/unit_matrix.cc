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
#include "detail/linalg/matrix.h"

#include "utils/print_types.h"

using TestTypes = std::tuple<float, double, int, char, size_t, uint32_t>;

TEST_CASE("matrix: test test", "[matrix]") {
  REQUIRE(true);
}

/*
 * Many tests remain in linalg.cc from before the refactor.
 */
TEST_CASE("matrix: initializer list", "[matrix]") {
  auto A = Matrix<float>{{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};
  auto a = std::vector<float>{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
  CHECK(
      std::equal(A.data(), A.data() + A.num_rows() * A.num_cols(), a.begin()));
}

TEST_CASE("matrix: copy", "[matrix]") {
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

TEST_CASE("matrix: assign", "[matrix]") {
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

TEST_CASE("matrix: vector of matrix", "[matrix]") {
  std::vector<Matrix<float>> v;
  std::vector<Matrix<float>> w(10);

  auto A = Matrix<float>{{8, 6, 7}, {5, 3, 0}, {9, 5, 0}};
  auto B = Matrix<float>{{3, 1, 4}, {1, 5, 9}, {2, 6, 5}, {3, 5, 8}};
  auto aptr = A.data();
  auto bptr = B.data();

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


TEST_CASE("matrix: sub matrix view", "[matrix]") {
  auto m = RowMajorMatrix<int>(11, 17);
  auto n = ColMajorMatrix<int>(11, 17);

  std::iota(m.data(), m.data() + m.num_rows() * m.num_cols(), 17);
  std::iota(n.data(), n.data() + n.num_rows() * n.num_cols(), 17);

  SECTION("check access") {
    CHECK(m(0, 0) == 17);
    CHECK(m(0, 1) == 18);
    CHECK(m(1, 0) == 34);
    CHECK(n(0, 0) == 17);
    CHECK(n(1, 0) == 18);
    CHECK(n(0, 1) == 28);
  }

    SECTION("submatrix, row") {
      auto sbm = SubMatrix(m, std::pair{3, 5}, std::pair{2, 5});
      CHECK(sbm(0, 0) == 70);
      CHECK(sbm(0, 1) == 71);
      CHECK(sbm(1, 0) == 87);
      CHECK(sbm(1, 1) == 88);

      CHECK(sbm.extent(0) == 2);
      CHECK(sbm.extent(1) == 3);

    }

  SECTION("submatrix, row, operator[]") {
    auto k = 5;

    auto sbm = SubMatrix(m, std::pair{1, m.num_rows() - 1}, std::pair{0, k});
    auto sbn = SubMatrix(n, std::pair{1, n.num_rows() - 1}, std::pair{0, k});

    for (size_t i = 1; i < m.num_rows()-1; ++i) {
      for (size_t h = 0; h < k; ++h) {
        CHECK(sbm(i-1, h) == m(i, h));
//        print_types(sbm, sbn);
//        CHECK(sbm[i-1][h] == m[i][h]);
//        std::cout << sbm(i-1, h) << " " << m(i, h) << " " ;
//        std::cout << sbm[i-1][h] << " " << m[i][h] << std::endl;
      }

//      CHECK(std::equal(begin(sbm[i-1]), end(sbm[i-1]), begin(m[i])));
    }
  }
}