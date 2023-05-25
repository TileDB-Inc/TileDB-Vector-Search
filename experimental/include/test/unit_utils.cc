/**
 * @file   unit_utils.cc
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


#include "../utils.h"
#include <catch2/catch_all.hpp>

TEST_CASE("utils: test test", "[utils]") {
  REQUIRE(true);
}

TEST_CASE("utils: test", "[utils]") {
  CHECK(is_http_address("http://www.tiledb.com"));
  CHECK(is_http_address("http://www.tiledb.com/index.html"));
  CHECK(is_http_address("https://www.tiledb.com"));
  CHECK(!is_http_address("s3://www.tiledb.com"));
  CHECK(!is_http_address("file://www.tiledb.com"));
  CHECK(!is_http_address("www.tiledb.com"));

  CHECK(is_s3_container("s3://www.tiledb.com"));
  CHECK(is_s3_container("s3://www.tiledb.com/index"));
  CHECK(!is_s3_container("http://www.tiledb.com"));
  CHECK(!is_s3_container("https://www.tiledb.com"));
  CHECK(!is_s3_container("file://www.tiledb.com"));
  CHECK(!is_s3_container("www.tiledb.com"));

  CHECK(is_local_file("unit_utils"));
  CHECK(!is_local_file("unit_utils_bad_path"));
  CHECK(is_local_file("../../src/test/unit_utils"));
  CHECK(is_local_file("../test/unit_utils"));
  CHECK(is_local_file("file://unit_utils"));
  CHECK(!is_local_file("file://unit_utils_bad_path"));
  CHECK(is_local_file("file://../../src/test/unit_utils"));
  CHECK(is_local_file("file://../test/unit_utils"));
  CHECK(!is_local_file("s3://www.tiledb.com/index"));
  CHECK(!is_local_file("http://www.tiledb.com"));
  CHECK(!is_local_file("https://www.tiledb.com"));
  CHECK(!is_local_file("http://www.tiledb.com/index.html"));
  CHECK(!is_local_file("https://www.tiledb.com/index.html"));

  CHECK(is_local_directory("./"));
  CHECK(is_local_directory(".."));
  CHECK(is_local_directory("../.."));
  CHECK(is_local_directory("../../src"));
  CHECK(is_local_directory("../../src/test"));
  CHECK(is_local_directory("array_dense_1"));
  CHECK(is_local_directory("./array_dense_1"));
  CHECK(!is_local_directory("../../src/test/unit_utils"));
  CHECK(!is_local_directory("../../src/test/unit_utils_bad_path"));
  CHECK(!is_local_directory("unit_utils"));
  CHECK(!is_local_directory("unit_utils_bad_path"));
  CHECK(!is_local_directory("file://unit_utils"));
  CHECK(!is_local_directory("file://unit_utils_bad_path"));
  CHECK(!is_local_directory("s3://www.tiledb.com/index"));
  CHECK(!is_local_directory("http://www.tiledb.com"));
  CHECK(!is_local_directory("https://www.tiledb.com"));
  CHECK(!is_local_directory("http://www.tiledb.com/index.html"));
  CHECK(!is_local_directory("https://www.tiledb.com/index.html"));

  CHECK(!is_local_array("./"));
  CHECK(!is_local_array(".."));
  CHECK(!is_local_array("../.."));
  CHECK(!is_local_array("../../src"));
  CHECK(!is_local_array("../../src/test"));
  CHECK(is_local_array("array_dense_1"));
  CHECK(is_local_array("./array_dense_1"));
  CHECK(!is_local_array("../../src/test/unit_utils"));
  CHECK(!is_local_array("../../src/test/unit_utils_bad_path"));
  CHECK(!is_local_array("unit_utils"));
  CHECK(!is_local_array("unit_utils_bad_path"));
  CHECK(!is_local_array("file://unit_utils"));
  CHECK(!is_local_array("file://unit_utils_bad_path"));
  CHECK(!is_local_array("s3://www.tiledb.com/index"));
  CHECK(!is_local_array("http://www.tiledb.com"));
  CHECK(!is_local_array("https://www.tiledb.com"));
  CHECK(!is_local_array("http://www.tiledb.com/index.html"));
  CHECK(!is_local_array("https://www.tiledb.com/index.html"));
}
