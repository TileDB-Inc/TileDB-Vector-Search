/**
 * * @file   api/feature_vector_array.h
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
 * This file defines the `FeatureVectorArray` class, which is a type-erased
 * wrapper of `tdbMatrix` that allows for runtime polymorphism of the
 * `tdbMatrix` class template.
 *
 * See README.md for details on type erasure.
 *
 */

#ifndef TILEDB_API_FEATURE_VECTOR_ARRAY_H
#define TILEDB_API_FEATURE_VECTOR_ARRAY_H

#include <unordered_set>
#include "api_defs.h"
#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "scoring.h"
#include "tdb_defs.h"

#include <type_traits>
#include "utils/print_types.h"

// TODO(paris): Move this somewhere specific if we decide to use this.

// template <tiledb_datatype_t Enum>
// struct BaseTypeFromEnum {
//     using type = void; // Placeholder, will be specialized
// };
//
//// Specialized TypeFromEnum structs inherit from BaseTypeFromEnum
// template <tiledb_datatype_t Enum>
// struct TypeFromEnum : BaseTypeFromEnum<Enum> {};
//
//// // Define a function to map from enum values to types
//// template <tiledb_datatype_t Enum>
//// struct TypeFromEnum;
//
// template <>
// struct TypeFromEnum<TILEDB_FLOAT32> {
//  using type = float;
//};
//
// template <>
// struct TypeFromEnum<TILEDB_UINT8> {
//  using type = uint8_t;
//};
//
// template <>
// struct TypeFromEnum<TILEDB_INT32> {
//  using type = int32_t;
//};
//
// template <>
// struct TypeFromEnum<TILEDB_UINT32> {
//  using type = uint32_t;
//};
//
// template <>
// struct TypeFromEnum<TILEDB_INT64> {
//  using type = int64_t;
//};
//
// template <>
// struct TypeFromEnum<TILEDB_UINT64> {
//  using type = uint64_t;
//};
//
// auto typeFromEnum(tiledb_datatype_t enumValue) {
//  switch (enumValue) {
//    case TILEDB_FLOAT32:
//      return static_cast<void *>(&TypeFromEnum<TILEDB_FLOAT32>{});
//    case TILEDB_UINT8:
//      return static_cast<void *>(&TypeFromEnum<TILEDB_UINT8>{});
//    case TILEDB_INT32:
//      return static_cast<void *>(&TypeFromEnum<TILEDB_INT32>{});
//    case TILEDB_UINT32:
//      return static_cast<void *>(&TypeFromEnum<TILEDB_UINT32>{});
//    case TILEDB_INT64:
//      return static_cast<void *>(&TypeFromEnum<TILEDB_INT64>{});
//    case TILEDB_UINT64:
//      return static_cast<void *>(&TypeFromEnum<TILEDB_UINT64>{});
//    default:
//      throw std::runtime_error("Unsupported attribute type");
//  }
//}

// template <tiledb_datatype_t A>
// auto typeFromEnum(tiledb_datatype_t enumValue) {
//  switch (enumValue) {
//    case TILEDB_FLOAT32:
//      return TypeFromEnum<TILEDB_FLOAT32>{};
//    case TILEDB_UINT8:
//      return TypeFromEnum<TILEDB_UINT8>{};
//    case TILEDB_INT32:
//      return TypeFromEnum<TILEDB_INT32>{};
//    case TILEDB_UINT32:
//      return TypeFromEnum<TILEDB_UINT32>{};
//    case TILEDB_INT64:
//      return TypeFromEnum<TILEDB_INT64>{};
//    case TILEDB_UINT64:
//      return TypeFromEnum<TILEDB_UINT64>{};
//    default:
//      throw std::runtime_error("Unsupported attribute type");
//  }
// }

// template<tiledb_datatype_t A>
// typename TypeFromEnum<A>::type typeFromEnum() { return
// TypeFromEnum<A>::value; }

// struct Converter {
//   template <tiledb_datatype_t EnumValue>
//   struct ToType {};

//   template <>
//   struct ToType<TILEDB_UINT64> {
//     using type = uint64_t;
//   };

//   template <>
//   struct ToType<TILEDB_INT64> {
//     using type = int64_t;
//   };

//   // Add more specializations for other enum values as needed
// };

static const std::unordered_set<tiledb_datatype_t> supported_types_ = {
    TILEDB_FLOAT32,
    TILEDB_UINT8,
    TILEDB_INT32,
    TILEDB_UINT32,
    TILEDB_INT64,
    TILEDB_UINT64};

using VariantType =
    std::variant<float, uint8_t, int32_t, uint32_t, int64_t, uint64_t>;
VariantType typeFromEnum(tiledb_datatype_t enumValue) {
  VariantType v;
  switch (enumValue) {
    case TILEDB_FLOAT32:
      v = static_cast<float>(0);
      break;
    case TILEDB_UINT8:
      v = static_cast<uint8_t>(0);
      break;
    case TILEDB_INT32:
      v = static_cast<int32_t>(0);
      break;
    case TILEDB_UINT32:
      v = static_cast<uint32_t>(0);
      break;
    case TILEDB_INT64:
      v = static_cast<int64_t>(0);
      break;
    case TILEDB_UINT64:
      v = static_cast<uint64_t>(0);
      break;
    default:
      throw std::runtime_error("Unsupported attribute type");
  }
  return v;
}

class FeatureVectorArray {
 public:
  FeatureVectorArray(const FeatureVectorArray& other) = delete;
  FeatureVectorArray& operator=(const FeatureVectorArray& other) = delete;
  FeatureVectorArray() = delete;
  virtual ~FeatureVectorArray() = default;

  FeatureVectorArray(FeatureVectorArray&& other) = default;
  FeatureVectorArray& operator=(FeatureVectorArray&& other) = default;

  template <feature_vector_array T>
  explicit FeatureVectorArray(T&& obj)
      : vector_array(
            std::make_unique<vector_array_impl<T>>(std::forward<T>(obj))) {
    feature_type_ = tiledb::impl::type_to_tiledb<
        typename std::remove_cvref_t<T>::value_type>::tiledb_type;
    feature_size_ = datatype_to_size(feature_type_);

    //    TODO(paris): Get this working.
    //    has_ids_ = obj.has_ids();
    //    if (has_ids_) {
    //      ids_type_ = tiledb::impl::type_to_tiledb<typename
    //      std::remove_cvref_t<T>::value_type>::tiledb_type; feature_size_ =
    //      datatype_to_size(feature_type_);
    //    }
  }

  FeatureVectorArray(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri = "",
      size_t num_vectors = 0) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    feature_type_ = get_array_datatype(*array);
    array->close();  // @todo create Matrix constructor that takes opened array
    feature_size_ = datatype_to_size(feature_type_);

    /**
     * Row and column orientation are kind of irrelevant?  We could dispatch
     * on the layout in the schema, but that might not be necessary.  What is
     * important is that the vectors are along the major axis, which should
     * happen with either orientation, and so will work at the other end with
     * either orientation since we are just passing a pointer to the data.
     */
    auto feature_type_variant = typeFromEnum(feature_type_);
    if (ids_uri.empty()) {
      vector_array = std::make_unique<vector_array_impl<
          tdbColMajorMatrix<decltype(feature_type_variant.index())>>>(
          ctx, uri, num_vectors, ids_uri);
      (void)vector_array->load();
    } else {
      has_ids_ = true;
      auto array =
          tiledb_helpers::open_array(tdb_func__, ctx, ids_uri, TILEDB_READ);
      ids_type_ = get_array_datatype(*array);
      array->close();
      ids_size_ = datatype_to_size(ids_type_);

      auto ids_type_variant = typeFromEnum(ids_type_);
      vector_array =
          std::make_unique<vector_array_impl<tdbColMajorMatrixWithIds<
              decltype(feature_type_variant.index()),
              size_t,
              decltype(ids_type_variant.index())>>>(ctx, uri, num_vectors, ids_uri);
      (void)vector_array->load();
    }
  }

  FeatureVectorArray(
      size_t rows,
      size_t cols,
      const std::string& type_string,
      const std::string& ids_type_string = "") {
    feature_type_ = string_to_datatype(type_string);
    feature_size_ = datatype_to_size(feature_type_);
    if (supported_types_.find(feature_type_) == supported_types_.end()) {
      throw std::runtime_error("Unsupported attribute type");
    }

    auto feature_type_variant = typeFromEnum(feature_type_);
    if (ids_type_string.empty()) {
      vector_array = std::make_unique<vector_array_impl<
          ColMajorMatrix<decltype(feature_type_variant.index())>>>(rows, cols);
    } else {
      has_ids_ = true;
      ids_type_ = string_to_datatype(ids_type_string);
      ids_size_ = datatype_to_size(ids_type_);
      if (supported_types_.find(ids_type_) == supported_types_.end()) {
        throw std::runtime_error("Unsupported attribute type for ids");
      }
      auto ids_type_variant = typeFromEnum(ids_type_);
      vector_array =
          std::make_unique<vector_array_impl<ColMajorMatrixWithIds<
              decltype(feature_type_variant.index()),
              size_t,
              decltype(ids_type_variant.index())>>>(rows, cols);
    }
  }

  // A FeatureVectorArray is always loaded
#if 0
  auto load() const {
    // return _cpo::load(*vector_array);
    return vector_array->load();
  }
#endif

  // @todo fix so niebloids work
  [[nodiscard]] auto data() const {
    // return _cpo::data(*vector_array);
    return vector_array->data();
  }

  [[nodiscard]] auto ids() const {
    // return _cpo::data(*vector_array);
    return vector_array->ids();
  }

  [[nodiscard]] auto extents() const {
    return _cpo::extents(*vector_array);
  }

  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(*vector_array);
  }

  [[nodiscard]] auto num_vectors() const {
    return _cpo::num_vectors(*vector_array);
  }

  [[nodiscard]] auto has_ids() const {
    return has_ids_;
  }

  [[nodiscard]] tiledb_datatype_t feature_type() const {
    return feature_type_;
  }

  [[nodiscard]] std::string feature_type_string() const {
    return datatype_to_string(feature_type_);
  }

  [[nodiscard]] size_t feature_size() const {
    return feature_size_;
  }

  [[nodiscard]] tiledb_datatype_t ids_type() const {
    return ids_type_;
  }

  [[nodiscard]] std::string ids_type_string() const {
    return datatype_to_string(ids_type_);
  }

  [[nodiscard]] size_t ids_size() const {
    return ids_size_;
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct vector_array_base {
    virtual ~vector_array_base() = default;
    [[nodiscard]] virtual size_t dimension() const = 0;
    [[nodiscard]] virtual size_t num_vectors() const = 0;
    [[nodiscard]] virtual void* data() const = 0;
    [[nodiscard]] virtual void* ids() const = 0;
    [[nodiscard]] virtual std::vector<size_t> extents() const = 0;
    [[nodiscard]] virtual bool load() = 0;
  };

  // @todo Create move constructors for Matrix and tdbMatrix?
  template <typename T>
  struct vector_array_impl : vector_array_base {
    explicit vector_array_impl(T&& t)
        : impl_vector_array(std::forward<T>(t)) {
      // explicit vector_array_impl(const T& t)
      //     : impl_vector_array(t) {
    }
    vector_array_impl(
        const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, const std::string& ids_uri)
        : impl_vector_array(ctx, uri, num_vectors, 0, ids_uri) {
    }
    vector_array_impl(size_t rows, size_t cols)
        : impl_vector_array(rows, cols) {
    }
    [[nodiscard]] void* data() const override {
      return _cpo::data(impl_vector_array);
    }
    [[nodiscard]] void* ids() const override {
      return _cpo::ids(impl_vector_array);
    }
    [[nodiscard]] size_t dimension() const override {
      return _cpo::dimension(impl_vector_array);
    }
    [[nodiscard]] size_t num_vectors() const override {
      return _cpo::num_vectors(impl_vector_array);
    }
    [[nodiscard]] std::vector<size_t> extents() const override {
      return _cpo::extents(impl_vector_array);
    }
    bool load() override {
      return _cpo::load(impl_vector_array);
    }

   private:
    T impl_vector_array;
  };

 private:
  tiledb_datatype_t feature_type_{TILEDB_ANY};
  size_t feature_size_{0};

  bool has_ids_{false};
  tiledb_datatype_t ids_type_{TILEDB_ANY};
  size_t ids_size_{0};

  // @todo const????
  std::unique_ptr</*const*/ vector_array_base> vector_array;
};

using QueryVectorArray = FeatureVectorArray;

bool validate_top_k(const FeatureVectorArray& a, const FeatureVectorArray& b) {
  // assert(a.datatype() == b.datatype());

  auto proc_b = [&b](auto& aview) {
    switch (b.feature_type()) {
      case TILEDB_INT32: {
        auto bview = MatrixView<int32_t, stdx::layout_left>{
            (int32_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_UINT32: {
        auto bview = MatrixView<uint32_t, stdx::layout_left>{
            (uint32_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_INT64: {
        auto bview = MatrixView<int64_t, stdx::layout_left>{
            (int64_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_UINT64: {
        auto bview = MatrixView<uint64_t, stdx::layout_left>{
            (uint64_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  };

  switch (a.feature_type()) {
    case TILEDB_INT32: {
      auto aview = MatrixView<int32_t, stdx::layout_left>{
          (int32_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT32: {
      auto aview = MatrixView<uint32_t, stdx::layout_left>{
          (uint32_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_INT64: {
      auto aview = MatrixView<int64_t, stdx::layout_left>{
          (int64_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT64: {
      auto aview = MatrixView<uint64_t, stdx::layout_left>{
          (uint64_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    default:
      throw std::runtime_error("Unsupported attribute type");
  }
}

/**
 * @brief Count the number of intersections between two sets of feature vectors.
 * This dispatches to the appropriate type-specific implementation.
 * @param a FeatureVectorArray to be compared
 * @param b FeatureVectorArray to be compared
 * @param k_nn Number of nearest neighbors to consider
 * @return Number of intersections found
 */
auto count_intersections(
    const FeatureVectorArray& a, const FeatureVectorArray& b, size_t k_nn) {
  auto proc_b = [&b, k_nn](auto& aview) {
    switch (b.feature_type()) {
      case TILEDB_INT32: {
        auto bview = MatrixView<int32_t, stdx::layout_left>{
            (int32_t*)b.data(), extents(b)[0], extents(b)[1]};
        return count_intersections(aview, bview, k_nn);
      }
      case TILEDB_UINT32: {
        auto bview = MatrixView<uint32_t, stdx::layout_left>{
            (uint32_t*)b.data(), extents(b)[0], extents(b)[1]};
        return count_intersections(aview, bview, k_nn);
      }
      case TILEDB_INT64: {
        auto bview = MatrixView<int64_t, stdx::layout_left>{
            (int64_t*)b.data(), extents(b)[0], extents(b)[1]};
        return count_intersections(aview, bview, k_nn);
      }
      case TILEDB_UINT64: {
        auto bview = MatrixView<uint64_t, stdx::layout_left>{
            (uint64_t*)b.data(), extents(b)[0], extents(b)[1]};
        return count_intersections(aview, bview, k_nn);
      }
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  };

  switch (a.feature_type()) {
    case TILEDB_INT32: {
      auto aview = MatrixView<int32_t, stdx::layout_left>{
          (int32_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT32: {
      auto aview = MatrixView<uint32_t, stdx::layout_left>{
          (uint32_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_INT64: {
      auto aview = MatrixView<int64_t, stdx::layout_left>{
          (int64_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT64: {
      auto aview = MatrixView<uint64_t, stdx::layout_left>{
          (uint64_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    default:
      throw std::runtime_error("Unsupported attribute type");
  }
}

#endif  // TILEDB_API_FEATURE_VECTOR_ARRAY_H
