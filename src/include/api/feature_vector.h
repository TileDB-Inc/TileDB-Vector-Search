/**
 * @file   api/feature_vector.h
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
 * This file defines the `FeatureVector` class, which is a type-erased wrapper
 * of `tdbVector` that allows for runtime polymorphism of the `tdbVector`
 * class template.
 *
 * See IVF.md for details on type erasure.
 *
 */

#ifndef TILEDB_API_FEATURE_VECTOR_H
#define TILEDB_API_FEATURE_VECTOR_H

#include "api_defs.h"
#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_vector.h"
#include "tdb_defs.h"

class FeatureVector {
 public:
  /**
   * @brief Construct from a class meeting the requirements of feature_vector.
   * @tparam T
   * @param vec
   */
  template <feature_vector T>
  explicit FeatureVector(T&& vec)
      : vector_(std::make_unique<vector_impl<T>>(std::forward<T>(vec))) {
    feature_type_ = tiledb::impl::type_to_tiledb<
        typename std::remove_cvref_t<T>::value_type>::tiledb_type;
  }

  /**
   * @brief Construct from a dtype string and size.
   * @param size
   * @param dtype
   */
  FeatureVector(size_t N, const std::string& dtype) {
    feature_type_ = string_to_datatype(dtype);
    vector_from_datatype(N);
  }

  /**
   * @brief Construct from a dtype string and size.
   * @param size
   * @param dtype
   */
  FeatureVector(size_t N, void*, const std::string& dtype) {
    feature_type_ = string_to_datatype(dtype);

    // Need constructor that takes void* and size and aliases the void*
    vector_from_datatype(N);
  }

  /**
   * @brief Constructs a feature vector from an array URI.
   * @param ctx
   * @param uri
   */
  FeatureVector(const tiledb::Context& ctx, const std::string& uri) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);

    feature_type_ = get_array_datatype(*array);
    array->close();

    tdb_vector_from_datatype(ctx, uri);
  }

  /*
   * Dispatch to the appropriate concrete class based on the datatype.
   */
  void vector_from_datatype(size_t N) {
    switch (feature_type_) {
      case TILEDB_FLOAT32:
        vector_ = std::make_unique<vector_impl<Vector<float>>>(N);
        break;
      case TILEDB_INT8:
        vector_ = std::make_unique<vector_impl<Vector<int8_t>>>(N);
        break;
      case TILEDB_UINT8:
        vector_ = std::make_unique<vector_impl<Vector<uint8_t>>>(N);
        break;
      case TILEDB_INT32:
        vector_ = std::make_unique<vector_impl<Vector<int32_t>>>(N);
        break;
      case TILEDB_UINT32:
        vector_ = std::make_unique<vector_impl<Vector<uint32_t>>>(N);
        break;
      case TILEDB_UINT64:
        vector_ = std::make_unique<vector_impl<Vector<uint64_t>>>(N);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }
  /*
   * Dispatch to the appropriate concrete class based on the datatype.
   */
  void tdb_vector_from_datatype(
      const tiledb::Context& ctx, const std::string& uri) {
    switch (feature_type_) {
      case TILEDB_FLOAT32:
        vector_ = std::make_unique<vector_impl<tdbVector<float>>>(ctx, uri);
        break;
      case TILEDB_INT8:
        vector_ = std::make_unique<vector_impl<tdbVector<int8_t>>>(ctx, uri);
        break;
      case TILEDB_UINT8:
        vector_ = std::make_unique<vector_impl<tdbVector<uint8_t>>>(ctx, uri);
        break;
      case TILEDB_INT32:
        vector_ = std::make_unique<vector_impl<tdbVector<int32_t>>>(ctx, uri);
        break;
      case TILEDB_UINT32:
        vector_ = std::make_unique<vector_impl<tdbVector<uint32_t>>>(ctx, uri);
        break;
      case TILEDB_UINT64:
        vector_ = std::make_unique<vector_impl<tdbVector<uint64_t>>>(ctx, uri);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }

  /**
   * @brief Returns a pointer to the underlying data.
   * @return
   */
  [[nodiscard]] auto data() const {
    // return _cpo::data(*vector);
    return vector_->data();
  }

  /**
   * @brief Returns the dimension (number of elements) of the vector
   * @return
   */
  [[nodiscard]] auto dimensions() const {
    return _cpo::dimensions(*vector_);
  }

  /**
   * @brief Returns the TileDB datatype of the vector
   * @return
   */
  [[nodiscard]] tiledb_datatype_t feature_type() const {
    return feature_type_;
  }

  [[nodiscard]] std::string feature_type_string() const {
    return datatype_to_string(feature_type_);
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct vector_base {
    virtual ~vector_base() = default;
    [[nodiscard]] virtual size_t dimensions() const = 0;
    //[[nodiscard]] virtual void* data() = 0;
    [[nodiscard]] virtual void* data() const = 0;
  };

  /**
   * @brief Type-parameterize implementation class.
   * @tparam T Type of the concrete class that is being type-erased.
   */
  template <typename T>
  struct vector_impl : vector_base {
    explicit vector_impl(T&& t)
        : vector_(std::forward<T>(t)) {
    }
    explicit vector_impl(size_t size)
        : vector_(size) {
    }
    vector_impl(const tiledb::Context& ctx, const std::string& uri)
        : vector_(ctx, uri) {
    }
    //[[nodiscard]] void* data() override {
    //  return _cpo::data(vector_);
    // return vector_.data();
    //}
    [[nodiscard]] void* data() const override {
      return (void*)::data(vector_);
      // return vector_.data();
    }
    [[nodiscard]] size_t dimensions() const override {
      return _cpo::dimensions(vector_);
    }

   private:
    T vector_;
  };

 private:
  tiledb_datatype_t feature_type_{TILEDB_ANY};
  std::unique_ptr<const vector_base> vector_;
};

using QueryVector = FeatureVector;
using IdVector = FeatureVector;

#endif  // TILEDB_API_FEATURE_VECTOR_H
