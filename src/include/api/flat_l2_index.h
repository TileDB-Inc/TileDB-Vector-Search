/**
 * * @file   api/flat_l2_index.h
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
 * This file defines the `IndexFlatL2` class, which is a type-erased
 * wrapper of `index_flat_l2` that allows for runtime polymorphism of the
 * `index_flat_l2` class template.
 *
 * See IVF.md for details on type erasure.
 */

#ifndef TILEDB_API_FLAT_L2_INDEX_H
#define TILEDB_API_FLAT_L2_INDEX_H

#include "api/feature_vector.h"
#include "api/feature_vector_array.h"
#include "api_defs.h"
#include "detail/linalg/tdb_helpers.h"
#include "index/flat_l2_index.h"

/**
 * A type-erased index class for C++ ivf_flat_l2_index.
 */
class IndexFlatL2 {
 public:
  // @todo Use group metadata to determine index type and associated array types
  IndexFlatL2(
      const tiledb::Context& ctx,
      const URI& index_uri,
      const std::optional<IndexOptions>& config = std::nullopt) {
    auto array =
        tiledb_helpers::open_array(tdb_func__, ctx, index_uri, TILEDB_READ);
    feature_datatype_ = get_array_datatype(*array);
    array->close();

    switch (feature_datatype_) {
      case TILEDB_FLOAT32:
        index_ = std::make_unique<index_impl<flat_l2_index<float>>>(
            ctx, index_uri, config);
        break;
      case TILEDB_UINT8:
        index_ = std::make_unique<index_impl<flat_l2_index<uint8_t>>>(
            ctx, index_uri, config);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  };

  void add() const {
    // @todo Is implementation more than no-op needed?
  }

  void add_with_ids() const {
    // @todo Is implementation more than no-op needed?
  }

  void train() const {
    // @todo Is implementation more than no-op needed?
  }

  void save() const {
    // @todo Implement
  }

  // @todo query() or search() -- or both?
  [[nodiscard]] auto query(
      const QueryVectorArray& vectors, size_t top_k) const {
    if (!index_) {
      throw std::runtime_error("Cannot query() because there is no index.");
    }
    return index_->query(vectors, top_k);
  }

  void update(
      const FeatureVectorArray& vectors,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    if (!index_) {
      throw std::runtime_error("Cannot update() because there is no index.");
    }
    index_->update(vectors, ids, options);
  }

  void update(
      const URI& vectors_uri,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    if (!index_) {
      throw std::runtime_error("Cannot update() because there is no index.");
    }
    index_->update(vectors_uri, ids, options);
  }

  void remove(const IdVector& ids) const {
    if (!index_) {
      throw std::runtime_error("Cannot remove() because there is no index.");
    }
    index_->remove(ids);
  }

  auto dimensions() {
    return _cpo::dimensions(*index_);
  }

  auto num_vectors() {
    return _cpo::num_vectors(*index_);
  }

  constexpr auto feature_type() const {
    return feature_datatype_;
  }

  auto feature_type_string() const {
    return datatype_to_string(feature_datatype_);
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct index_base {
    virtual ~index_base() = default;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query(const QueryVectorArray& vectors, size_t top_k) const = 0;

    virtual void update(
        const FeatureVectorArray&,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void remove(const IdVector& ids) const = 0;

    virtual size_t dimensions() const = 0;

    virtual size_t num_vectors() const = 0;
  };

  /**
   * @brief Type-parameterize implementation class.
   * @tparam T Type of the concrete class that is being type-erased.
   */
  template <typename T>
  struct index_impl : index_base {
    explicit index_impl(T&& t)
        : impl_index_(std::forward<T>(t)) {
    }

    index_impl(
        const tiledb::Context& ctx,
        const URI& index_uri,
        const std::optional<StringMap>& config = std::nullopt)
        : impl_index_(ctx, index_uri) {
    }

    template <feature_vector_array V>
    index_impl(
        const URI& index_uri,
        const V& vectors,
        const IndexOptions& options,
        const std::optional<StringMap>& config = std::nullopt)
        : impl_index_(index_uri, vectors, options, config) {
    }

    // Create from input URI
    index_impl(
        const URI& index_uri,
        const URI& vectors_uri,
        const IndexOptions& options,
        std::optional<StringMap> config = std::nullopt)
        : impl_index_(index_uri, vectors_uri, options, config) {
    }

    [[nodiscard]] auto query(
        tiledb::Context ctx, const URI& uri, size_t top_k) const {
      return impl_index_.query(ctx, uri, top_k);
    }

    /**
     * @brief Query the index with the given vectors.  The concrete query
     * function returns a tuple of arrays, which are type erased and returned as
     * a tuple of FeatureVectorArrays.
     * @param vectors
     * @param k_nn
     * @return
     *
     * @todo Make sure the extents of the returned arrays are used correctly.
     */
    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query(
        const QueryVectorArray& vectors, size_t k_nn) const override {
      // @todo using index_type = size_t; ?

      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(), extents(vectors)[0], extents(vectors)[1]};
          auto [s, t] = impl_index_.query(qspan, k_nn);

          auto& ss = s;
          auto& tt = t;
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};
          auto [s, t] = impl_index_.query(qspan, k_nn);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    // WIP
    void update(
        const FeatureVectorArray& vectors,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const override {
      //      index_.update(vectors, ids, options);
    }

    // WIP
    void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const override {
      //      index_.update(vectors_uri, ids, options);
    }

    // WIP
    void remove(const IdVector& ids) const override {
      //      index_.remove(ids);
    }

    size_t dimensions() const override {
      return _cpo::dimensions(impl_index_);
    }

    size_t num_vectors() const override {
      return _cpo::num_vectors(impl_index_);
    }

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_type_{TILEDB_ANY};
  tiledb_datatype_t ptx_type_{TILEDB_ANY};
  std::unique_ptr<const index_base> index_;
};

#endif  // TILEDB_API_FLAT_L2_INDEX_H
