/**
 * @file   tdb_vector.h
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

#ifndef TILEDB_TDB_VECTOR_H
#define TILEDB_TDB_VECTOR_H

#include <tiledb/tiledb>
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/vector.h"

/**
 * @brief Placeholder for now
 *
 * @tparam T
 */
template <class T>
class tdbVector : public Vector<T> {
  using Base = Vector<T>;
  using Base::Base;

 public:
  tdbVector(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t start,
      size_t end,
      TemporalPolicy temporal_policy)
      : Base(
            (start == 0 && end == 0) ?
                read_vector<T>(ctx, uri, temporal_policy) :
                read_vector<T>(ctx, uri, start, end, temporal_policy)) {
  }
};

#endif  // TILEDB_TDB_VECTOR_H
