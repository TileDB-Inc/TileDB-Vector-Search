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
 * See IVF.md for details on type erasure.
 *
 */

#ifndef TILEDB_API_FEATURE_VECTOR_ARRAY_H
#define TILEDB_API_FEATURE_VECTOR_ARRAY_H

#include <unordered_set>
#include "api_defs.h"
#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/matrix_with_ids.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_matrix_with_ids.h"
#include "scoring.h"
#include "tdb_defs.h"

#include <type_traits>
#include "utils/print_types.h"

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

    if constexpr (feature_vector_array_with_ids<
                      std::remove_cvref_t<decltype(obj)>>) {
      ids_type_ = tiledb::impl::type_to_tiledb<
          typename std::remove_cvref_t<T>::ids_type>::tiledb_type;
      ids_size_ = datatype_to_size(ids_type_);
    }
  }

  FeatureVectorArray(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri = "",
      size_t num_vectors = 0,
      std::optional<TemporalPolicy> temporal_policy_input = std::nullopt) {
    auto temporal_policy = temporal_policy_input.value_or(TemporalPolicy{});
    auto array = tiledb_helpers::open_array(
        tdb_func__, ctx, uri, TILEDB_READ, temporal_policy);
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
    if (ids_uri.empty()) {
      if (tdb_col_major_matrix_dispatch_table.find(feature_type_) ==
          tdb_col_major_matrix_dispatch_table.end()) {
        throw std::runtime_error("Unsupported features attribute type");
      }
      vector_array = tdb_col_major_matrix_dispatch_table.at(feature_type_)(
          ctx, uri, num_vectors, temporal_policy);
    } else {
      auto ids_array = tiledb_helpers::open_array(
          tdb_func__, ctx, ids_uri, TILEDB_READ, temporal_policy);
      ids_type_ = get_array_datatype(*ids_array);
      array->close();
      ids_size_ = datatype_to_size(ids_type_);

      auto type = std::tuple{feature_type_, ids_type_};
      if (tdb_col_major_matrix_with_ids_dispatch_table.find(type) ==
          tdb_col_major_matrix_with_ids_dispatch_table.end()) {
        throw std::runtime_error(
            "Unsupported attribute type for feature vector with ids");
      }
      vector_array = tdb_col_major_matrix_with_ids_dispatch_table.at(type)(
          ctx, uri, ids_uri, num_vectors, temporal_policy);
    }
    (void)vector_array->load();
  }

  FeatureVectorArray(
      size_t rows,
      size_t cols,
      const std::string& type_string,
      const std::string& ids_type_string = "") {
    feature_type_ = string_to_datatype(type_string);
    feature_size_ = datatype_to_size(feature_type_);

    if (ids_type_string.empty()) {
      if (col_major_matrix_dispatch_table.find(feature_type_) ==
          col_major_matrix_dispatch_table.end()) {
        throw std::runtime_error("Unsupported features attribute type");
      }
      vector_array =
          col_major_matrix_dispatch_table.at(feature_type_)(rows, cols);
    } else {
      ids_type_ = string_to_datatype(ids_type_string);
      ids_size_ = datatype_to_size(ids_type_);

      auto type = std::tuple{feature_type_, ids_type_};
      if (col_major_matrix_with_ids_dispatch_table.find(type) ==
          col_major_matrix_with_ids_dispatch_table.end()) {
        throw std::runtime_error(
            "Unsupported attribute type for feature vector with ids");
      }
      vector_array =
          col_major_matrix_with_ids_dispatch_table.at(type)(rows, cols);
    }
  }

  // @todo fix so niebloids work
  [[nodiscard]] auto data() const {
    // return _cpo::data(*vector_array);
    return vector_array->data();
  }

  [[nodiscard]] auto ids() const {
    return vector_array->ids();
  }

  [[nodiscard]] auto extents() const {
    return _cpo::extents(*vector_array);
  }

  [[nodiscard]] auto dimensions() const {
    return _cpo::dimensions(*vector_array);
  }

  [[nodiscard]] auto num_vectors() const {
    return _cpo::num_vectors(*vector_array);
  }

  [[nodiscard]] auto num_ids() const {
    return _cpo::num_ids(*vector_array);
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
    [[nodiscard]] virtual size_t dimensions() const = 0;
    [[nodiscard]] virtual size_t num_vectors() const = 0;
    [[nodiscard]] virtual void* data() const = 0;
    [[nodiscard]] virtual size_t num_ids() const = 0;
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
        const tiledb::Context& ctx,
        const std::string& uri,
        size_t num_vectors,
        TemporalPolicy temporal_policy)
        : impl_vector_array(ctx, uri, num_vectors, temporal_policy) {
    }
    vector_array_impl(
        const tiledb::Context& ctx,
        const std::string& uri,
        const std::string& ids_uri,
        size_t num_vectors,
        TemporalPolicy temporal_policy)
        : impl_vector_array(ctx, uri, ids_uri, num_vectors, temporal_policy) {
    }
    vector_array_impl(size_t rows, size_t cols)
        : impl_vector_array(rows, cols) {
    }
    [[nodiscard]] void* data() const override {
      return _cpo::data(impl_vector_array);
    }
    [[nodiscard]] size_t num_ids() const override {
      return _cpo::num_ids(impl_vector_array);
    }
    [[nodiscard]] void* ids() const override {
      return _cpo::ids(impl_vector_array);
    }
    [[nodiscard]] size_t dimensions() const override {
      return _cpo::dimensions(impl_vector_array);
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
  // clang-format off
  using col_major_matrix_constructor_function = std::function<std::unique_ptr<vector_array_base>(size_t, size_t)>;
  using col_major_matrix_table_type = std::map<tiledb_datatype_t, col_major_matrix_constructor_function>;
  static const col_major_matrix_table_type col_major_matrix_dispatch_table;

  using tdb_col_major_matrix_constructor_function = std::function<std::unique_ptr<vector_array_base>(const tiledb::Context&, const std::string&, size_t, TemporalPolicy)>;
  using tdb_col_major_matrix_table_type = std::map<tiledb_datatype_t, tdb_col_major_matrix_constructor_function>;
  static const tdb_col_major_matrix_table_type tdb_col_major_matrix_dispatch_table;

  using col_major_matrix_with_ids_constructor_function = std::function<std::unique_ptr<vector_array_base>(size_t, size_t)>;
  using col_major_matrix_with_ids_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t>, col_major_matrix_with_ids_constructor_function>;
  static const col_major_matrix_with_ids_table_type col_major_matrix_with_ids_dispatch_table;

  using tdb_col_major_matrix_with_ids_constructor_function = std::function<std::unique_ptr<vector_array_base>(const tiledb::Context&, const std::string&, const std::string&, size_t, TemporalPolicy)>;
  using tdb_col_major_matrix_with_ids_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t>, tdb_col_major_matrix_with_ids_constructor_function>;
  static const tdb_col_major_matrix_with_ids_table_type tdb_col_major_matrix_with_ids_dispatch_table;
  // clang-format on

  tiledb_datatype_t feature_type_{TILEDB_ANY};
  size_t feature_size_{0};

  tiledb_datatype_t ids_type_{TILEDB_ANY};
  size_t ids_size_{0};

  // @todo const????
  std::unique_ptr</*const*/ vector_array_base> vector_array;
};

// clang-format off
const FeatureVectorArray::col_major_matrix_table_type FeatureVectorArray::col_major_matrix_dispatch_table = {
  {TILEDB_FLOAT32, [](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrix<float   >>>(rows, cols);}},
  {TILEDB_INT8,    [](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrix<int8_t  >>>(rows, cols);}},
  {TILEDB_UINT8,   [](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrix<uint8_t >>>(rows, cols);}},
  {TILEDB_INT32,   [](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrix<int32_t >>>(rows, cols);}},
  {TILEDB_UINT32,  [](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrix<uint32_t>>>(rows, cols);}},
  {TILEDB_INT64,   [](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrix<int64_t >>>(rows, cols);}},
  {TILEDB_UINT64,  [](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrix<uint64_t>>>(rows, cols);}},
};

const FeatureVectorArray::tdb_col_major_matrix_table_type FeatureVectorArray::tdb_col_major_matrix_dispatch_table = {
  {TILEDB_FLOAT32, [](const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrix<float   >>>(ctx, uri, num_vectors, temporal_policy); }},
  {TILEDB_INT8,    [](const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrix<int8_t  >>>(ctx, uri, num_vectors, temporal_policy); }},
  {TILEDB_UINT8,   [](const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrix<uint8_t >>>(ctx, uri, num_vectors, temporal_policy); }},
  {TILEDB_INT32,   [](const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrix<int32_t >>>(ctx, uri, num_vectors, temporal_policy); }},
  {TILEDB_UINT32,  [](const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrix<uint32_t>>>(ctx, uri, num_vectors, temporal_policy); }},
  {TILEDB_INT64,   [](const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrix<int64_t >>>(ctx, uri, num_vectors, temporal_policy); }},
  {TILEDB_UINT64,  [](const tiledb::Context& ctx, const std::string& uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrix<uint64_t>>>(ctx, uri, num_vectors, temporal_policy); }},
};

const FeatureVectorArray::col_major_matrix_with_ids_table_type FeatureVectorArray::col_major_matrix_with_ids_dispatch_table = {
  {{TILEDB_FLOAT32, TILEDB_UINT32},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<float,    uint32_t>>>(rows, cols); }},
  {{TILEDB_INT8,    TILEDB_UINT32},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<int8_t,   uint32_t>>>(rows, cols); }},
  {{TILEDB_UINT8,   TILEDB_UINT32},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<uint8_t,  uint32_t>>>(rows, cols); }},
  {{TILEDB_INT32,   TILEDB_UINT32},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<int32_t,  uint32_t>>>(rows, cols); }},
  {{TILEDB_UINT32,  TILEDB_UINT32},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<uint32_t, uint32_t>>>(rows, cols); }},
  {{TILEDB_INT64,   TILEDB_UINT32},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<int64_t,  uint32_t>>>(rows, cols); }},
  {{TILEDB_UINT64,  TILEDB_UINT32},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<uint64_t, uint32_t>>>(rows, cols); }},

  {{TILEDB_FLOAT32, TILEDB_UINT64},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<float,    uint64_t>>>(rows, cols); }},
  {{TILEDB_INT8,    TILEDB_UINT64},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<int8_t,   uint64_t>>>(rows, cols); }},
  {{TILEDB_UINT8,   TILEDB_UINT64},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<uint8_t,  uint64_t>>>(rows, cols); }},
  {{TILEDB_INT32,   TILEDB_UINT64},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<int32_t,  uint64_t>>>(rows, cols); }},
  {{TILEDB_UINT32,  TILEDB_UINT64},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<uint32_t, uint64_t>>>(rows, cols); }},
  {{TILEDB_INT64,   TILEDB_UINT64},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<int64_t,  uint64_t>>>(rows, cols); }},
  {{TILEDB_UINT64,  TILEDB_UINT64},[](size_t rows, size_t cols) { return std::make_unique<FeatureVectorArray::vector_array_impl<ColMajorMatrixWithIds<uint64_t, uint64_t>>>(rows, cols); }},
};

const FeatureVectorArray::tdb_col_major_matrix_with_ids_table_type FeatureVectorArray::tdb_col_major_matrix_with_ids_dispatch_table = {
  {{TILEDB_FLOAT32, TILEDB_UINT32},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<float,    uint32_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_INT8,    TILEDB_UINT32},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<int8_t,   uint32_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_UINT8,   TILEDB_UINT32},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<uint8_t,  uint32_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_INT32,   TILEDB_UINT32},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<int32_t,  uint32_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_UINT32,  TILEDB_UINT32},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<uint32_t, uint32_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_INT64,   TILEDB_UINT32},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<int64_t,  uint32_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_UINT64,  TILEDB_UINT32},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<uint64_t, uint32_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},

  {{TILEDB_FLOAT32, TILEDB_UINT64},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<float,    uint64_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_INT8,    TILEDB_UINT64},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<int8_t,   uint64_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_UINT8,   TILEDB_UINT64},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) { return std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<uint8_t,  uint64_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_INT32,   TILEDB_UINT64},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<int32_t,  uint64_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_UINT32,  TILEDB_UINT64},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<uint32_t, uint64_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_INT64,   TILEDB_UINT64},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<int64_t,  uint64_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
  {{TILEDB_UINT64,  TILEDB_UINT64},[](const tiledb::Context& ctx, const std::string& uri, const std::string& ids_uri, size_t num_vectors, TemporalPolicy temporal_policy) {return  std::make_unique<FeatureVectorArray::vector_array_impl<tdbColMajorMatrixWithIds<uint64_t, uint64_t>>>(ctx, uri, ids_uri, num_vectors, temporal_policy);}},
};
// clang-format on

using QueryVectorArray = FeatureVectorArray;

bool validate_top_k(const FeatureVectorArray& a, const FeatureVectorArray& b) {
  if (a.feature_type() != b.feature_type()) {
    throw std::runtime_error(
        "[feature_vector_array@validate_top_k] Feature types do not match: " +
        a.feature_type_string() + " vs " + b.feature_type_string());
  }

  auto proc_b = [&b](auto& aview) {
    switch (b.feature_type()) {
      case TILEDB_FLOAT32: {
        auto bview = MatrixView<float, stdx::layout_left>{
            (float*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_INT8: {
        auto bview = MatrixView<int8_t, stdx::layout_left>{
            (int8_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_UINT8: {
        auto bview = MatrixView<uint8_t, stdx::layout_left>{
            (uint8_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
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
    case TILEDB_FLOAT32: {
      auto aview = MatrixView<float, stdx::layout_left>{
          (float*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_INT8: {
      auto aview = MatrixView<int8_t, stdx::layout_left>{
          (int8_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT8: {
      auto aview = MatrixView<uint8_t, stdx::layout_left>{
          (uint8_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
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
      case TILEDB_FLOAT32: {
        auto bview = MatrixView<float, stdx::layout_left>{
            (float*)b.data(), extents(b)[0], extents(b)[1]};
        return count_intersections(aview, bview, k_nn);
      }
      case TILEDB_INT8: {
        auto bview = MatrixView<int8_t, stdx::layout_left>{
            (int8_t*)b.data(), extents(b)[0], extents(b)[1]};
        return count_intersections(aview, bview, k_nn);
      }
      case TILEDB_UINT8: {
        auto bview = MatrixView<uint8_t, stdx::layout_left>{
            (uint8_t*)b.data(), extents(b)[0], extents(b)[1]};
        return count_intersections(aview, bview, k_nn);
      }
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
    case TILEDB_FLOAT32: {
      auto aview = MatrixView<float, stdx::layout_left>{
          (float*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_INT8: {
      auto aview = MatrixView<int8_t, stdx::layout_left>{
          (int8_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT8: {
      auto aview = MatrixView<uint8_t, stdx::layout_left>{
          (uint8_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
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
