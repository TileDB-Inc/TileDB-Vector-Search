/**
 * @file   unit_abstraction_penalty.cc
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
 * Tests to measure abstraction penalty of data structures and algorithms.
 *
 */

#include <tiledb/tiledb>
#include "utils/timer.h"

#if 0
// Quick and dirty program to read pytest generated arrays
int main(){
  tiledb::Context ctx;
  for (size_t i = 0; i < 25; ++i) {
    auto uri = std::string("/tmp/test_vector_") + std::to_string(i);
    auto array =  tiledb::Array(ctx, uri, TILEDB_READ);
    auto schema = array.schema();
    std::cout << "=============================" << std::endl;
    std::cout << uri << std::endl;
    schema.dump();
  }
}
#endif