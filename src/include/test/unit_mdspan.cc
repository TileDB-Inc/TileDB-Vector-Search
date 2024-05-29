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
}  // namespace stdx

TEST_CASE("basic construction", "[mdspan]") {
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

TEST_CASE("rectangular", "[mdspan]") {
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

  SECTION("submdspan, slices") {
    auto k = 5;

    [[maybe_unused]] auto sbm =
        stdx::submdspan(m, std::pair{1, m.extent(0) - 1}, std::pair{0, k});
    [[maybe_unused]] auto sbn =
        stdx::submdspan(n, std::pair{0, k}, std::pair{1, n.extent(1) - 1});

    [[maybe_unused]] auto sbe = stdx::submdspan(m, stdx::full_extent, 0);
    [[maybe_unused]] auto sbf = stdx::submdspan(n, 0, stdx::full_extent);

    auto sbv = stdx::submdspan(m, 0, stdx::full_extent);  // right
    auto sbw = stdx::submdspan(m, 0, std::tuple{0, k});   // right
    auto sbx = stdx::submdspan(n, stdx::full_extent, 0);  // left
    auto sby = stdx::submdspan(n, std::tuple{0, k}, 0);   // left

    CHECK(std::is_same_v<
          typename decltype(sbv)::layout_type,
          stdx::layout_right>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbw)::layout_type,
          stdx::layout_right>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbx)::layout_type,
          stdx::layout_left>);  // true
    CHECK(std::is_same_v<
          typename decltype(sby)::layout_type,
          stdx::layout_left>);  // true

    auto sbvv =
        stdx::submdspan(m, std::tuple{0, k}, stdx::full_extent);  // right
    auto sbww =
        stdx::submdspan(m, std::tuple{0, k}, std::tuple{0, k});  // stride
    auto sbxx =
        stdx::submdspan(n, stdx::full_extent, std::tuple{0, k});  // left
    auto sbyy =
        stdx::submdspan(n, std::tuple{0, k}, std::tuple{0, k});  // stride

    CHECK(std::is_same_v<
          typename decltype(sbvv)::layout_type,
          stdx::layout_right>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbww)::layout_type,
          stdx::layout_stride>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbxx)::layout_type,
          stdx::layout_left>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbyy)::layout_type,
          stdx::layout_stride>);  // true

    auto sbsss = stdx::submdspan(n, 0, 0);  // left
    //  print_types(sbsss);
    auto sbttt = stdx::submdspan(n, 0, stdx::full_extent);  // stride
    // print_types(sbttt);
    auto sbuuu = stdx::submdspan(n, stdx::full_extent, 0);  // left
    // print_types(sbuuu);
    auto sbvvv = stdx::submdspan(n, 0, std::tuple{0, k});  // stride
    // print_types(sbvvv);
    auto sbwww = stdx::submdspan(n, std::tuple{0, k}, k);  // left
    // print_types(sbwww);
    auto sbxxx =
        stdx::submdspan(n, std::tuple{0, k}, stdx::full_extent);  // stride
    // print_types(sbxxx);
    auto sbyyy =
        stdx::submdspan(n, stdx::full_extent, std::tuple{0, k});  // left
    // print_types(sbyyy);
    auto sbzzz =
        stdx::submdspan(n, std::tuple{0, k}, std::tuple{0, k});  // stride
    // print_types(sbzzz);

    CHECK(std::is_same_v<
          typename decltype(sbsss)::layout_type,
          stdx::layout_left>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbttt)::layout_type,
          stdx::layout_stride>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbuuu)::layout_type,
          stdx::layout_left>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbvvv)::layout_type,
          stdx::layout_stride>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbwww)::layout_type,
          stdx::layout_left>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbxxx)::layout_type,
          stdx::layout_stride>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbyyy)::layout_type,
          stdx::layout_left>);  // true
    CHECK(std::is_same_v<
          typename decltype(sbzzz)::layout_type,
          stdx::layout_stride>);  // true
  }
}
