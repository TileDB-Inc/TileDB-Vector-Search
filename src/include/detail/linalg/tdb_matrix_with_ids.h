/**
 * @file   tdb_matrix.h
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
 * Class that provides a matrix interface to a TileDB array and also holds an ID
 * for each TODO row or col.
 *
 */

#ifndef TDB_MATRIX_H
#define TDB_MATRIX_H

#include <future>

#include <tiledb/tiledb>

#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "tdb_defs.h"

template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class tdbBlockedMatrixWithIds : tdbBlockedMatrix {
  tdbBlockedMatrixWithIds(
      const string& uri, const string& ids_uri, const string& num_vectors)

      std::unique_ptr<tiledb::Array> arrayIds_;
  std::unique_ptr<T[]> storageIds_;

  auto id(index_type i) const {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&Base::id()(i, 0), num_cols_);
    } else {
      return std::span(&Base::id()(0, i), num_rows_);
    }
  }

  auto ids_data() {
    return storage_->get();
  }

  auto ids_data() const {
    return storage_->get();
  }

  auto ids() {
    return TODO;
  }

  auto ids() const {
    return TODO;
  }

  void load() {
    // load ids into storageIds_
  }
}

cpos.h concepts
    .h

        Then there is also one that inherits from matrix.h
