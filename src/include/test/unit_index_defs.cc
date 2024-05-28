/**
 * @file   unit_index.cc
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
#include <type_traits>
#include "index/index_defs.h"
#include "tdb_defs.h"

TEST_CASE("tiledb_to_type", "[index_defs]") {
  CHECK(std::is_same_v<tiledb_to_type_t<TILEDB_FLOAT32>, float>);
  CHECK(std::is_same_v<tiledb_to_type_t<TILEDB_UINT64>, uint64_t>);
  CHECK(std::is_same_v<tiledb_to_type_t<TILEDB_UINT32>, uint32_t>);
  CHECK(std::is_same_v<tiledb_to_type_t<TILEDB_INT64>, int64_t>);
  CHECK(std::is_same_v<tiledb_to_type_t<TILEDB_INT32>, int32_t>);
  CHECK(std::is_same_v<tiledb_to_type_t<TILEDB_INT8>, int8_t>);
  CHECK(std::is_same_v<tiledb_to_type_t<TILEDB_UINT8>, uint8_t>);
}

TEST_CASE("type_to_tiledb", "[index_defs]") {
  CHECK(type_to_tiledb_t<float> == TILEDB_FLOAT32);
  CHECK(type_to_tiledb_t<uint64_t> == TILEDB_UINT64);
  CHECK(type_to_tiledb_t<uint32_t> == TILEDB_UINT32);
  CHECK(type_to_tiledb_t<int64_t> == TILEDB_INT64);
  CHECK(type_to_tiledb_t<int32_t> == TILEDB_INT32);
  CHECK(type_to_tiledb_t<int8_t> == TILEDB_INT8);
  CHECK(type_to_tiledb_t<uint8_t> == TILEDB_UINT8);
}

TEST_CASE("string_to_datatype", "[index_defs]") {
  CHECK(string_to_datatype("float32") == TILEDB_FLOAT32);
  CHECK(string_to_datatype("int8") == TILEDB_INT8);
  CHECK(string_to_datatype("uint8") == TILEDB_UINT8);
  CHECK(string_to_datatype("int32") == TILEDB_INT32);
  CHECK(string_to_datatype("uint32") == TILEDB_UINT32);
  CHECK(string_to_datatype("int64") == TILEDB_INT64);
  CHECK(string_to_datatype("uint64") == TILEDB_UINT64);
}

TEST_CASE("datatype_to_string", "[index_defs]") {
  CHECK(datatype_to_string(TILEDB_FLOAT32) == "float32");
  CHECK(datatype_to_string(TILEDB_INT8) == "int8");
  CHECK(datatype_to_string(TILEDB_UINT8) == "uint8");
  CHECK(datatype_to_string(TILEDB_INT32) == "int32");
  CHECK(datatype_to_string(TILEDB_UINT32) == "uint32");
  CHECK(datatype_to_string(TILEDB_INT64) == "int64");
  CHECK(datatype_to_string(TILEDB_UINT64) == "uint64");
}
