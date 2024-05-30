/**
 * @file   unit_flat_vq.cc
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
#include "detail/flat/vq.h"
#include "test/utils/query_common.h"

// @todo: test with tdbMatrix
TEST_CASE("flat vq all or nothing", "[flat vq]") {
  auto ids = std::vector<size_t>(sift_base.num_cols());
  std::iota(ids.rbegin(), ids.rend(), 9);

  auto&& [D00, I00] = detail::flat::vq_query_heap(sift_base, sift_query, 3, 1);
  auto&& [D01, I01] =
      detail::flat::vq_query_heap_tiled(sift_base, sift_query, 3, 1);
  auto&& [D02, I02] =
      detail::flat::vq_query_heap_2(sift_base, sift_query, 3, 1);

  CHECK(std::equal(D00.data(), D00.data() + D00.size(), D01.data()));
  CHECK(std::equal(I00.data(), I00.data() + I00.size(), I01.data()));
  CHECK(std::equal(D00.data(), D00.data() + D00.size(), D02.data()));
  CHECK(std::equal(I00.data(), I00.data() + I00.size(), I02.data()));

  auto&& [D10, I10] =
      detail::flat::vq_query_heap(sift_base, sift_query, ids, 3, 1);
  auto&& [D11, I11] =
      detail::flat::vq_query_heap_tiled(sift_base, sift_query, ids, 3, 1);
  auto&& [D12, I12] =
      detail::flat::vq_query_heap_2(sift_base, sift_query, ids, 3, 1);

  CHECK(!std::equal(I00.data(), I00.data() + I00.size(), I10.data()));

  CHECK(std::equal(D10.data(), D10.data() + D10.size(), D11.data()));
  CHECK(std::equal(I10.data(), I10.data() + I10.size(), I11.data()));
  CHECK(std::equal(D10.data(), D10.data() + D10.size(), D12.data()));
  CHECK(std::equal(I10.data(), I10.data() + I10.size(), I12.data()));
}
