/**
 * @file   api.h
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
 * Nascent C++ API (including concepts).
 *
 */

#ifndef TDB_API_H
#define TDB_API_H

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------

#include <vector>

#include <memory>
#include <vector>

#include "concepts.h"
#include "cpos.h"

// ----------------------------------------------------------------------------
// Prototype of type erased FeatureVectorArray
// ----------------------------------------------------------------------------
class ProtoFeatureVectorArray {
 public:
  template <typename T>
  ProtoFeatureVectorArray(T&& obj)
      : vector_array(
            std::make_shared<vector_array_impl<T>>(std::forward<T>(obj))) {
  }

  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(*vector_array);
  }

  [[nodiscard]] auto num_vectors() const {
    return _cpo::num_vectors(*vector_array);
  }

  struct vector_array_base {
    virtual ~vector_array_base() = default;
    [[nodiscard]] virtual size_t dimension() const = 0;
    [[nodiscard]] virtual size_t num_vectors() const = 0;
  };

  template <typename T>
  struct vector_array_impl : vector_array_base {
    explicit vector_array_impl(const T& t)
        : vector_array(t) {
    }
    [[nodiscard]] size_t dimension() const override {
      return _cpo::dimension(vector_array);
    }
    [[nodiscard]] size_t num_vectors() const override {
      return _cpo::num_vectors(vector_array);
    }

   private:
    T vector_array;
  };

  std::shared_ptr<const vector_array_base> vector_array;
};

#include <tiledb/tiledb>
#include "detail/linalg/tdb_helpers.h"

class MatrixWrapper {
 public:
  MatrixWrapper(const tiledb::Context& ctx, const std::string& uri) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    auto schema = array.schema();
    auto attr = schema.attribute(0);  // @todo Is there a better way to look up
                                      // the attribute we're interested in?
    tiledb_datatype_t attr_type = attr.type();

    array.close();  // @todo create Matrix constructor that takes opened array
  }
  // open uri
  // get schema
  // create matrix based on that type
  // wrap matrix in ProtoFeatureVectorArray
  // return
};

#endif