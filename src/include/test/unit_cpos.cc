/**
 * @file   unit_cpos.cc
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
#include <iostream>
#include "cpos.h"
#include "mdspan/mdspan.hpp"
#include "utils/print_types.h"

template <class I = size_t>
using matrix_extents = stdx::dextents<I, 2>;

TEMPLATE_TEST_CASE("test num_rows", "[cpos]", float, uint8_t) {
  std::vector<TestType> v0(200);
  std::vector<TestType> v1(200);

  // rows, cols
  auto m0 = stdx::mdspan<TestType, matrix_extents<size_t>, stdx::layout_right>{
      v0.data(), 10, 20};
  CHECK(num_vectors(m0) == 10);
  CHECK(dimensions(m0) == 20);
  CHECK(data(m0) == v0.data());

  CHECK(extents(m0)[0] == 10);
  CHECK(extents(m0)[1] == 20);

  auto m1 = stdx::mdspan<TestType, matrix_extents<size_t>, stdx::layout_left>{
      v1.data(), 10, 20};
  CHECK(num_vectors(m1) == 20);
  CHECK(dimensions(m1) == 10);
  CHECK(data(m1) == v1.data());

  CHECK(extents(m1)[0] == 10);
  CHECK(extents(m1)[1] == 20);
}
