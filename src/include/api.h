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

#include <memory>
#include <vector>

#include "concepts.h"
#include "cpos.h"

/**
 * Basic wrapper to type-erase a class.  It has a shared pointer to
 * a non-template base class that has virtual methods for accessing member
 * functions of the wrapped type.  It provides data(), dimension(), and
 * num_vectors() methods that delegate to the wrapped type.
 *
 * @todo Instrument carefully to make sure we are not doing any copying.
 */
class ProtoFeatureVectorArray {
 public:
  template <typename T>
  ProtoFeatureVectorArray(T&& obj)
      : vector_array(
            std::make_shared<vector_array_impl<T>>(std::move(std::forward<T>(obj)))) {
  }

  [[nodiscard]] auto data() const {
    return _cpo::data(*vector_array);
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
    [[nodiscard]] virtual void* data() const = 0;
  };

  // @todo Create move constructors for Matrix and tdbMatrix
  template <typename T>
  struct vector_array_impl : vector_array_base {
    explicit vector_array_impl(T&& t)
        : vector_array(std::move(t)) {
    }
    [[nodiscard]] void* data() const override {
      return _cpo::data(vector_array);
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

// Put these here for now to enforce separation between matrix aware
// and matrix unaware code.  We could (and probably should) merge these
// into a single class.  Although we could use the above for
// wrapping feature vectors as well (we would just use a concept
// to elide num_vectors).  The separation may also be useful for
// dealing with in-memory arrays vs arrays on disk.

#include <tiledb/tiledb>
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"

class FeatureVectorArray {
 public:

  ProtoFeatureVectorArray open(const tiledb::Context& ctx, const std::string& uri) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    auto schema = array.schema();
    auto attr = schema.attribute(0);  // @todo Is there a better way to look up
                                      // the attribute we're interested in?
    datatype_ = attr.type();

    array.close();  // @todo create Matrix constructor that takes opened array

    /**
     * Row and column orientation are kind of irrelevant?  We could dispatch
     * on the layout in the schema, but that might not be necessary.  What is
     * important is that the vectors are along the major axis, which should
     * happen with either orientation, and so will work at the other end with
     * either orientation since we are just passing a pointer to the data.
     */
    switch(datatype_) {
      case TILEDB_FLOAT32:
        return ProtoFeatureVectorArray(std::move(tdbColMajorMatrix<float>(ctx, uri)));
      case TILEDB_UINT8:
        return ProtoFeatureVectorArray(std::move(tdbColMajorMatrix<uint8_t>(ctx, uri)));
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }

  FeatureVectorArray(const tiledb::Context& ctx, const std::string& uri) : vector_array{open(ctx, uri)} {
  }

  [[nodiscard]] tiledb_datatype_t datatype() const {
    return datatype_;
  }

  [[nodiscard]] void* data() const {
    return (void*)_cpo::data(vector_array);
  }

  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(vector_array);
  }

  [[nodiscard]] auto num_vectors() const {
    return _cpo::dimension(vector_array);
  }

  tiledb_datatype_t datatype_;
  ProtoFeatureVectorArray vector_array;
};

#endif