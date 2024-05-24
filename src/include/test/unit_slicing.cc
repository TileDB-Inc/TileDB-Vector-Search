/**
 * @file   unit_slicing.cc
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
 * Test various slicing constructors for tdbMatrix
 *
 */

#include <catch2/catch_all.hpp>
#include "../linalg.h"
#include "test/utils/array_defs.h"

std::string global_region = "us-east-1";

TEST_CASE("slice", "[linalg]") {
  const bool debug = false;

  tiledb::Context ctx_;

  std::vector<int> data_(288);
  std::vector<int> data2_(288);
  std::vector<float> value_(288);

  auto array_ = tiledb_helpers::open_array(
      tdb_func__, ctx_, sift_inputs_uri, TILEDB_READ);
  tiledb::ArraySchema schema_{array_->schema()};
  tiledb::Query query(ctx_, *array_);

  tiledb::Subarray subarray(ctx_, *array_);
  subarray.add_range(0, 0, 5).add_range(1, 88, 100).add_range(0, 10, 13);

  //      .add_range(1, col_0_start, col_0_end);
  query.set_subarray(subarray);

  query.set_subarray(subarray)
      .set_layout(TILEDB_COL_MAJOR)
      .set_data_buffer("cols", data2_.data(), 288)
      .set_data_buffer("rows", data_.data(), 288)
      .set_data_buffer("values", value_.data(), 288);

  tiledb_helpers::submit_query(tdb_func__, sift_inputs_uri, query);

  if (debug) {
    for (int i = 0; i < 135; i++) {
      std::cout << data_[i] << ", " << data2_[i] << ": " << value_[i]
                << std::endl;
    }
  }
}
