/**
* * @file   api/ivf_flat_index.h
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

#ifndef TILEDB_API_IVF_FLAT_INDEX_H
#define TILEDB_API_IVF_FLAT_INDEX_H

#include "api_defs.h"
#include "index/ivf_flat_index.h"
#include "api/feature_vector.h"
#include "api/feature_vector_array.h"
#include <tiledb/tiledb>
#include <tiledb/group_experimental.h>


/*******************************************************************************
 * IndexIVFFlat
 ******************************************************************************/
// OK -- copilot filled this in completely, which really weirded me out

/**
 * A type-erased IVF flat index class. An index class is provides
 *   - URI-based constructor
 *   - Array-based constructor
 *   - A train method
 *   - An add method
 *   - A query method
 *   - An update method
 *   - A remove method
 */
class IndexIVFFlat {
 public:

  /**
   * Create empty index
   */
  // IndexIVFFlat() = delete;
  IndexIVFFlat(const IndexIVFFlat&) = delete;
  IndexIVFFlat(IndexIVFFlat&&) = default;
  IndexIVFFlat& operator=(const IndexIVFFlat&) = delete;
  IndexIVFFlat& operator=(IndexIVFFlat&&) = default;


  explicit IndexIVFFlat(const std::optional<IndexOptions>& config = std::nullopt) {

    size_t nlist = 0;
    size_t max_iter = 2;
    float tolerance = 1e-4;
    std::optional<size_t> num_threads = std::nullopt;
    feature_datatype_ = TILEDB_FLOAT32;
    id_datatype_ = TILEDB_UINT32;
    px_datatype_ = TILEDB_UINT32;

    if (config) {
      for (auto&& c : *config) {
        auto key = c.first;
        auto value = c.second;
        if (key == "nlist") {
          nlist = std::stol(value);
        } else if (key == "max_iter") {
          max_iter = std::stol(value);
        } else if (key == "tolerance") {
          tolerance = std::stof(value);
        } else if (key == "num_threads") {
          num_threads = std::make_optional<size_t>(std::stol(value));
        } else if (key == "feature_type") {
          feature_datatype_ = string_to_datatype(value);
        } else if (key == "id_type") {
          id_datatype_ = string_to_datatype(value);
        } else if (key == "px_type") {
          px_datatype_ = string_to_datatype(value);
        } else {
          throw std::runtime_error("Invalid index config key: " + key);
        }
      }
    }
    /**
     * We support all combinations of the following types for feature,
     * id, and px datatypes:
     *   feature_type: uint8 or float
     *   id_type: uint32 or uint64
     *   px_type: uint32 or uint64
     */
    if (feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint32_t, uint32_t>>>(
              nlist, max_iter, tolerance, num_threads);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint32_t, uint32_t>>>(
              nlist, max_iter, tolerance, num_threads);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint32_t, uint64_t>>>(
              nlist, max_iter, tolerance, num_threads);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint32_t, uint64_t>>>(
              nlist, max_iter, tolerance, num_threads);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint64_t, uint32_t>>>(
              nlist, max_iter, tolerance, num_threads);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint64_t, uint32_t>>>(
              nlist, max_iter, tolerance, num_threads);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint64_t, uint64_t>>>(
              nlist, max_iter, tolerance, num_threads);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint64_t, uint64_t>>>(
              nlist, max_iter, tolerance, num_threads);
    }
  }

  // @todo Who owns the context?
  IndexIVFFlat(
      const URI& group_uri,
      const std::optional<IndexOptions>& config = std::nullopt)
      : IndexIVFFlat(tiledb::Context{}, group_uri, config) {
  }

  /**
   * @brief Open an existing index.
   *
   * @note This will be able to infer all of its types using the group metadata
   * to create the internal ivf_flat_index object.
   *
   * @param ctx
   * @param group_uri
   * @param config
   */
  IndexIVFFlat(
      const tiledb::Context& ctx,
      const URI& group_uri,
      const std::optional<IndexOptions>& config = std::nullopt)
      : ctx_{ctx} {
    using metadata_element = std::tuple<std::string, void*, tiledb_datatype_t>;
    std::vector<metadata_element> metadata{
        {"feature_datatype", &feature_datatype_, TILEDB_UINT32},
        {"id_datatype", &id_datatype_, TILEDB_UINT32},
        {"px_datatype", &px_datatype_, TILEDB_UINT32}};

    tiledb::Config cfg;
    tiledb::Group read_group(ctx_, group_uri, TILEDB_READ, cfg);

    for (auto& [name, value, datatype] : metadata) {
      if (!read_group.has_metadata(name, &datatype)) {
        throw std::runtime_error("Missing metadata: " + name);
      }
      uint32_t count;
      void* addr;
      read_group.get_metadata(name, &datatype, &count, (const void**)&addr);
      if (datatype == TILEDB_UINT32) {
        *reinterpret_cast<uint32_t*>(value) =
            *reinterpret_cast<uint32_t*>(addr);
      } else {
        throw std::runtime_error("Unsupported datatype for metadata: " + name);
      }
    }

    /**
     * We support all combinations of the following types for feature,
     * id, and px datatypes:
     *   feature_type: uint8 or float
     *   id_type: uint32 or uint64
     *   px_type: uint32 or uint64
     */
    if (feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint32_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint32_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint32_t, uint64_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint32_t, uint64_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint64_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint64_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<uint8_t, uint64_t, uint64_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_flat_index<float, uint64_t, uint64_t>>>(
              ctx_, group_uri, config);
    }
  }

  template <feature_vector_array V>
  IndexIVFFlat(
      const URI& group_uri,
      const V& vectors,
      const std::optional<IndexOptions>& config = std::nullopt) {
  }

  // Create from input URI
  IndexIVFFlat(
      const URI& group_uri,
      const URI& vectors_uri,
      const std::optional<IndexOptions>& config = std::nullopt) {
    // @todo
  }

  void add() const {
    // @todo
  }

  void add_with_ids() const {
    // @todo
  }

  void train() const {
    // @todo
  }

  void save(const std::string& group_uri, bool overwrite) const {
    index_->write_index(group_uri, overwrite);
  }

  // todo query() or search() -- or both?
  [[nodiscard]] auto query_infinite_ram (
      const QueryVectorArray& vectors, size_t top_k, size_t nprobe)  const {
    return index_->query_infinite_ram(vectors, top_k, nprobe);
  }

  [[nodiscard]] auto query_finite_ram(
      const QueryVectorArray& vectors, size_t top_k, size_t nprobe) const  {
    return index_->query_finite_ram(vectors, top_k, nprobe);
  }

  void update(
      const FeatureVectorArray& vectors,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors, ids, options);
  }

  void update(
      const URI& vectors_uri,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors_uri, ids, options);
  }

  virtual void remove(const IdVector& ids) const {
    index_->remove(ids);
  }

  constexpr auto dimension() const {
    return ::dimension(*index_);
  }

  constexpr auto num_partitions() const {
    return ::num_partitions(*index_);
  }

// Don't think we need thi
#if 0
  constexpr size_t ntotal() const {
    // @todo
    return 0;
  }

  constexpr auto num_vectors() const {
    return _cpo::num_vectors(*index_);
  }
#endif

  constexpr auto feature_type() const {
    return feature_datatype_;
  }

  constexpr auto id_type() const {
    return id_datatype_;
  }

  constexpr auto px_type() const {
    return px_datatype_;
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct index_base {
    virtual ~index_base() = default;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_infinite_ram(const QueryVectorArray& vectors, size_t top_k, size_t nprobe)  = 0;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_finite_ram(const QueryVectorArray& vectors, size_t top_k, size_t nprobe)  = 0;

    virtual void update(
        const FeatureVectorArray&,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void write_index(const std::string& group_uri, bool overwrite) const = 0;

    virtual void remove(const IdVector& ids) const = 0;

    [[nodiscard]] virtual size_t dimension() const = 0;

    [[nodiscard]] virtual size_t num_partitions() const = 0;

// Don't think we need these
#if 0
    virtual size_t ntotal() const = 0;

    virtual size_t num_vectors() const = 0;
#endif
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
        size_t nlist,
        size_t max_iter,
        float tolerance,
        std::optional<size_t> num_threads) : impl_index_(nlist, max_iter, tolerance, num_threads) {
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

    [[nodiscard]] auto query_infinite_ram(
        tiledb::Context ctx, const URI& uri, size_t top_k, size_t nprobe)  {
      return impl_index_.query_infinite_ram(ctx, uri, top_k, nprobe);
    }

    [[nodiscard]] auto query_finite_ram(
        tiledb::Context ctx, const URI& uri, size_t top_k, size_t nprobe)  {
      return impl_index_.query_finite_ram(ctx, uri, top_k, nprobe);
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
    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query_infinite_ram(
        const QueryVectorArray& vectors, size_t k_nn, size_t nprobe)  override {
      // @todo using index_type = size_t;

      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_infinite_ram(qspan, k_nn, nprobe);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_infinite_ram(qspan, k_nn, nprobe);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query_finite_ram(
        const QueryVectorArray& vectors, size_t k_nn, size_t nprobe)  override {
      // @todo using index_type = size_t;

      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_finite_ram(qspan, k_nn, nprobe);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_finite_ram(qspan, k_nn, nprobe);
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

    void write_index(const std::string& group_uri, bool overwrite) const override {
      impl_index_.write_index(group_uri, overwrite);
    }

    // WIP
    void remove(const IdVector& ids) const override {
      //      index_.remove(ids);
    }

    size_t dimension() const override {
      return ::dimension(impl_index_);
    }

    size_t num_partitions() const override {
      return ::num_partitions(impl_index_);
    }

    // Don't think we need these
#if 0
    size_t ntotal() const override {
      return ::num_vectors(impl_index_);
    }

    size_t num_vectors() const override {
      return ::num_vectors(impl_index_);
    }
#endif

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  // @todo Who should own the context?
  tiledb::Context ctx_{};
  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};
  tiledb_datatype_t px_datatype_{TILEDB_ANY};
  std::unique_ptr</* const */ index_base> index_;
};


#endif // TILEDB_API_IVF_FLAT_INDEX_H