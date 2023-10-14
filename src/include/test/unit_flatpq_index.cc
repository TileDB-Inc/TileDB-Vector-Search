/**
 * @file   unit_flatpq_index.cc
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

#include "flatpq_index.h"
#include "query_common.h"
#include "detail/flat/qv.h"
#include "scoring.h"

TEST_CASE("flatpq_index: test test", "[flatpq_index]") {
  REQUIRE(true);
}

TEST_CASE("flatpq_index: flat qv_partition with sub distance", "[flatpq_index]") {
  {
    auto sub_sift_base = ColMajorMatrix<float>(2, num_vectors(sift_base));
    for (size_t i = 0; i < num_vectors(sift_base); i++) {
      for (size_t j = 0; j < 2; j++) {
        sub_sift_base(j, i) = sift_base(j, i);
      }
    }
    auto sub_sift_query = ColMajorMatrix<float>(2, num_vectors(sift_query));
    for (size_t i = 0; i < num_vectors(sift_query); i++) {
      for (size_t j = 0; j < 2; j++) {
        sub_sift_query(j, i) = sift_query(j, i);
      }
    }

    auto aa = detail::flat::qv_partition(sub_sift_base, sub_sift_query, 1);
    auto bb = detail::flat::qv_partition(
        sub_sift_base, sub_sift_query, 1, sum_of_squares_distance{});
    auto cc = detail::flat::qv_partition(
        sift_base, sift_query, 1, sub_sum_of_squares_distance{0, 2});
    CHECK(std::equal(aa.begin(), aa.end(), bb.begin()));
    CHECK(std::equal(aa.begin(), aa.end(), cc.begin()));
  }

  {
    auto sub_sift_base = ColMajorMatrix<float>(2, num_vectors(sift_base));
    for (size_t i = 0; i < num_vectors(sift_base); i++) {
      for (size_t j = 1; j < 3; j++) {
        sub_sift_base(j-1, i) = sift_base(j, i);
      }
    }
    auto sub_sift_query = ColMajorMatrix<float>(2, num_vectors(sift_query));
    for (size_t i = 0; i < num_vectors(sift_query); i++) {
      for (size_t j = 1; j < 3; j++) {
        sub_sift_query(j-1, i) = sift_query(j, i);
      }
    }

    auto aa = detail::flat::qv_partition(sub_sift_base, sub_sift_query, 1);
    auto bb = detail::flat::qv_partition(
        sub_sift_base, sub_sift_query, 1, sum_of_squares_distance{});
    auto cc = detail::flat::qv_partition(
        sift_base, sift_query, 1, sub_sum_of_squares_distance{1, 3});
    CHECK(std::equal(aa.begin(), aa.end(), bb.begin()));
    CHECK(std::equal(aa.begin(), aa.end(), cc.begin()));
  }

}
