/**
 * @file   unit_api.cc
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
#include "tdb_defs.h"


TEST_CASE("api: test test", "[api]") {
  REQUIRE(true);
}

TEST_CASE("api: types", "[types]") {
  CHECK(tiledb::impl::type_to_tiledb<float>::name == std::string("FLOAT32"));
  CHECK(tiledb::impl::type_to_tiledb<int>::name == std::string("INT32"));
  CHECK(tiledb::impl::type_to_tiledb<unsigned>::name == std::string("UINT32"));
  CHECK(tiledb::impl::type_to_tiledb<int64_t>::name == std::string("INT64"));
  CHECK(tiledb::impl::type_to_tiledb<uint64_t>::name == std::string("UINT64"));

  CHECK(tiledb::impl::type_to_tiledb<long>::name != std::string("INT64"));
  CHECK(
      tiledb::impl::type_to_tiledb<unsigned long>::name !=
      std::string("UINT64"));
  CHECK(tiledb::impl::type_to_tiledb<size_t>::name != std::string("UINT64"));
}
