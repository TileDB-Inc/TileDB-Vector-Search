/**
 * @file   unit_adj_list.cc
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
#include "detail/graph/adj_list.h"
#include "test/utils/tiny_graphs.h"

TEMPLATE_TEST_CASE(
    "index_adj_list initializer_list",
    "[adj_list]",
    int,
    unsigned,
    long,
    size_t,
    uint8_t) {
  auto A = detail::graph::index_adj_list<TestType>{tiny_index_adj_list};

  CHECK(A.size() == 6);
  CHECK(A[0].size() == 1);
  CHECK(A[1].size() == 2);
  CHECK(A[2].size() == 2);
  CHECK(A[3].size() == 2);
  CHECK(A[4].size() == 1);
  CHECK(A[5].size() == 1);
}

TEMPLATE_TEST_CASE(
    "pseudo copy constructor",
    "[adj_list]",
    (std::tuple<float, int>),
    (std::tuple<float, size_t>),
    (std::tuple<double, uint8_t>)) {
  auto A = detail::graph::adj_list<
      std::tuple_element_t<0, TestType>,
      std::tuple_element_t<1, TestType>>{tiny_adj_list};

  CHECK(A.size() == 6);
  CHECK(A[0].size() == 1);
  CHECK(A[1].size() == 2);
  CHECK(A[2].size() == 2);
  CHECK(A[3].size() == 2);
  CHECK(A[4].size() == 1);
  CHECK(A[5].size() == 1);

  auto&& [sc00, id00] = A[0].front();
  auto&& [sc01, id01] = A[1].front();
  auto&& [sc11, id11] = A[1].back();
  auto&& [sc02, id02] = A[2].front();
  auto&& [sc12, id12] = A[2].back();
  auto&& [sc03, id03] = A[3].front();
  auto&& [sc13, id13] = A[3].back();
  auto&& [sc04, id04] = A[4].front();
  auto&& [sc05, id05] = A[5].front();
  CHECK(sc00 == 8.0);
  CHECK(id00 == 1);
  CHECK(std::abs(sc01 - 6.7) < 1e-6);
  CHECK(id01 == 2);
  CHECK(std::abs(sc11 - 5.0) < 1e-6);
  CHECK(id11 == 3);
  CHECK(sc02 == 1.0);
  CHECK(id02 == 3);
  CHECK(sc12 == 31.0);
  CHECK(id12 == 6);
  CHECK(sc03 == 4.0);
  CHECK(id03 == 5);
  CHECK(sc13 == 15.0);
  CHECK(id13 == 6);
  CHECK(std::abs(sc04 - 9.9) < 1e-6);
  CHECK(id04 == 3);
  CHECK(sc05 == 99.0);
  CHECK(id05 == 4);
}
