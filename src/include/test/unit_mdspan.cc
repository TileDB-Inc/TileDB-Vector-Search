/**
 * @file   unit_mdspan.cc
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
#include <mdspan/mdspan.hpp>

namespace stdx {
using namespace Kokkos;
using namespace Kokkos::Experimental;
}  // namespace stdx

TEST_CASE("mdspan: test test", "[mdspan]") {
  REQUIRE(true);
}

TEST_CASE("mdspan: basic construction", "[mdspan]") {
  // using a = stdx::mdspan<int, stdx::extents<>>;
  // using a = Kokkos::mdspan<int, Kokkos::extents<size_t,
  // stdx::dynamic_extent>>;
  std::vector<int> v(100);
  std::iota(v.begin(), v.end(), 17);
  stdx::mdspan<
      int,
      stdx::extents<size_t, stdx::dynamic_extent, stdx::dynamic_extent>>
      m(v.data(), 10, 10);
  stdx::mdspan<
      int,
      stdx::extents<size_t, stdx::dynamic_extent, stdx::dynamic_extent>,
      stdx::layout_left>
      n(v.data(), 10, 10);

  SECTION("check extents") {
    CHECK(m.extent(0) == 10);
    CHECK(m.extent(1) == 10);
    CHECK(n.extent(0) == 10);
    CHECK(n.extent(1) == 10);
  }
  SECTION("check access") {
    CHECK(m(0, 0) == 17);
    CHECK(m(0, 1) == 18);
    CHECK(m(1, 0) == 27);
    CHECK(n(0, 0) == 17);
    CHECK(n(1, 0) == 18);
    CHECK(n(0, 1) == 27);
  }
  SECTION("submdspan") {
    auto sbm = stdx::submdspan(m, std::pair{1, 3}, std::pair{2, 4});
    CHECK(sbm(0, 0) == 29);
    CHECK(sbm(0, 1) == 30);
    CHECK(sbm(1, 0) == 39);
    CHECK(sbm(1, 1) == 40);
  }
}


TEST_CASE("mdspan: rectangular", "[mdspan]") {
  std::vector<int> v(187);
  std::iota(v.begin(), v.end(), 17);

  // Row-oriented
  stdx::mdspan<
      int,
      stdx::extents<size_t, stdx::dynamic_extent, stdx::dynamic_extent>>
      m(v.data(), 11, 17);

  // Column-oriented
  stdx::mdspan<
      int,
      stdx::extents<size_t, stdx::dynamic_extent, stdx::dynamic_extent>,
      stdx::layout_left>
      n(v.data(), 11, 17);

  SECTION("check extents") {
    CHECK(m.extent(0) == 11);
    CHECK(m.extent(1) == 17);
    CHECK(n.extent(0) == 11);
    CHECK(n.extent(1) == 17);
  }

  SECTION("check access") {
    CHECK(m(0, 0) == 17);
    CHECK(m(0, 1) == 18);
    CHECK(m(1, 0) == 34);
    CHECK(n(0, 0) == 17);
    CHECK(n(1, 0) == 18);
    CHECK(n(0, 1) == 28);
  }

  SECTION("submdspan, row") {
    auto sbm = stdx::submdspan(m, std::pair{3, 5}, std::pair{2, 5});
    CHECK(sbm(0, 0) == 70);
    CHECK(sbm(0, 1) == 71);
    CHECK(sbm(1, 0) == 87);
    CHECK(sbm(1, 1) == 88);

    CHECK(sbm.extent(0) == 2);
    CHECK(sbm.extent(1) == 3);

  }

  SECTION("submdspan, col") {
    auto sbn = stdx::submdspan(n, std::pair{3, 5}, std::pair{2, 5});
    CHECK(sbn(0, 0) == 42);
    CHECK(sbn(0, 1) == 53);
    CHECK(sbn(1, 0) == 43);
    CHECK(sbn(1, 1) == 54);

    CHECK(sbn.extent(0) == 2);
    CHECK(sbn.extent(1) == 3);
  }
}
