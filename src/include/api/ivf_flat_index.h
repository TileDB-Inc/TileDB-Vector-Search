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
 * This file defines the `IndexIVFFlat` class, which is a type-erased
 * wrapper of `index_ivf_flat` that allows for runtime polymorphism of the
 * `index_ivf_flat` class template.
 *
 * See IVF.md for details on type erasure.
 */

#ifndef TILEDB_API_IVF_FLAT_INDEX_H
#define TILEDB_API_IVF_FLAT_INDEX_H

#include <tiledb/tiledb>

#include "api/feature_vector.h"
#include "api/feature_vector_array.h"
#include "api_defs.h"
#include "index/ivf_flat_group.h"
#include "index/ivf_flat_index.h"

/*******************************************************************************
 * IndexIVFFlat
 ******************************************************************************/
// OK -- copilot filled this in completely, which really weirded me out

/**
 * A type-erased IVF flat index class. This is a type-erased wrapper around
 * the `ivf_flat_index` class in index/ivf_flat_index.h.
 *
 * An index class is provides
 *   - URI-based constructor
 *   - Array-based constructor
 *   - A train method
 *   - An add method
 *   - A query method
 *   - An update method
 *   - A remove method
 *
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

  /**
   * @brief Create an index with the given configuration.  The index in this
   * state is ready to be trained.  The sequence for creating an index in this
   * fashion is:
   *  - Create an IndexIVFFlat object with the desired configuration (using this
   *  constructor
   *  - Call train() with the training data
   *  - Call add() to add a set of vectors to the index (often the same as the
   *  training data)
   *  Either (or both)
   *    - Perform an "infinite ram" query (since the index is all in RAM at this
   *      point, only "infinite ram" queries make sense)
   *    - Call write_index() to write the index to disk
   * @param config A map of configuration parameters, as pairs of strings
   * containing config parameters and values.
   */
  explicit IndexIVFFlat(
      const std::optional<IndexOptions>& config = std::nullopt) {
    feature_datatype_ = TILEDB_ANY;
    id_datatype_ = TILEDB_UINT32;
    px_datatype_ = TILEDB_UINT32;

    if (config) {
      for (auto&& c : *config) {
        auto key = c.first;
        auto value = c.second;
        if (key == "nlist") {
          nlist_ = std::stol(value);
        } else if (key == "dimensions") {
          dimensions_ = std::stol(value);
        } else if (key == "max_iter") {
          max_iter_ = std::stol(value);
        } else if (key == "tolerance") {
          tolerance_ = std::stof(value);
        } else if (key == "num_threads") {
          num_threads_ = std::make_optional<size_t>(std::stol(value));
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
  }

  /**
   * @brief Open an existing index.
   *
   * @note This will be able to infer all of its types using the group metadata
   * to create the internal ivf_flat_index object.
   *
   * @param ctx
   * @param group_uri TileDB group containing all the arrays comprising the
   * index.

   */
  IndexIVFFlat(
      const tiledb::Context& ctx,
      const URI& group_uri,
      const std::optional<IndexOptions>& config = std::nullopt) {
    // Holds {metadata_name, address, datatype, array_key}
    using metadata_element =
        std::tuple<std::string, void*, tiledb_datatype_t, std::string>;
    std::vector<metadata_element> metadata{
        {"feature_datatype",
         &feature_datatype_,
         TILEDB_UINT32,
         "parts_array_name"},
        {"id_datatype", &id_datatype_, TILEDB_UINT32, "ids_array_name"},
        {"px_datatype", &px_datatype_, TILEDB_UINT32, "index_array_name"}};

    tiledb::Group read_group(ctx, group_uri, TILEDB_READ, ctx.config());

    // Get the storage_version in case the metadata is not present on read_group
    // and we need to read the individual arrays.
    std::string storage_version = current_storage_version;
    tiledb_datatype_t v_type;
    if (read_group.has_metadata("storage_version", &v_type)) {
      uint32_t v_num;
      const void* v;
      read_group.get_metadata("storage_version", &v_type, &v_num, &v);
      if (v_type == TILEDB_STRING_ASCII || v_type == TILEDB_STRING_UTF8) {
        storage_version = std::string(static_cast<const char*>(v), v_num);
      }
    }

    for (auto& [name, value, datatype, array_key] : metadata) {
      // We first try to read metadata from the group.
      if (read_group.has_metadata(name, &datatype)) {
        uint32_t count;
        void* addr;
        read_group.get_metadata(name, &datatype, &count, (const void**)&addr);
        if (datatype == TILEDB_UINT32) {
          *reinterpret_cast<uint32_t*>(value) =
              *reinterpret_cast<uint32_t*>(addr);
        } else {
          throw std::runtime_error(
              "Unsupported datatype for metadata: " + name);
        }
      } else {
        // If it is not present then fallback to checking the type on the array
        // directly.
        auto uri = array_name_to_uri(
            group_uri,
            array_key_to_array_name_from_map(
                storage_formats[storage_version], array_key));
        tiledb::ArraySchema schema(ctx, uri);
        *reinterpret_cast<uint32_t*>(value) = schema.attribute(0).type();
      }
    }

    /**
     * We support all combinations of the following types for feature,
     * id, and px datatypes:
     *   feature_type (partitioned_vectors_feature_type): uint8 or float
     *   id_type (partitioned_ids_type): uint32 or uint64
     *   px_type (partitioning_index_type): uint32 or uint64
     *
     *   @todo Unify the type-based switch-case statements in a manner
     *   similar to what was done in query_condition
     */
    if (feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint32_t, uint32_t>>>(
          ctx, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint32_t, uint32_t>>>(
          ctx, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint32_t, uint64_t>>>(
          ctx, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint32_t, uint64_t>>>(
          ctx, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint64_t, uint32_t>>>(
          ctx, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint64_t, uint32_t>>>(
          ctx, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint64_t, uint64_t>>>(
          ctx, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint64_t, uint64_t>>>(
          ctx, group_uri, config);
    } else {
      throw std::runtime_error("Unsupported datatype combination");
    }
    if (dimensions_ != 0 && dimensions_ != index_->dimensions()) {
      throw std::runtime_error(
          "Dimensions mismatch: " + std::to_string(dimensions_) +
          " != " + std::to_string(index_->dimensions()));
    }
    dimensions_ = index_->dimensions();
    if (nlist_ != 0 && nlist_ != index_->num_partitions()) {
      throw std::runtime_error(
          "nlist mismatch: " + std::to_string(nlist_) +
          " != " + std::to_string(index_->num_partitions()));
    }
    nlist_ = index_->num_partitions();
  }

  /**
   * @brief Train the index based on the given training set.
   * @param training_set
   * @param init
   */
  // @todo -- infer feature type from input
  void train(
      const FeatureVectorArray& training_set,
      kmeans_init init = kmeans_init::random) {
    if (feature_datatype_ == TILEDB_ANY) {
      feature_datatype_ = training_set.feature_type();
    } else if (feature_datatype_ != training_set.feature_type()) {
      throw std::runtime_error(
          "Feature datatype mismatch: " +
          datatype_to_string(feature_datatype_) +
          " != " + datatype_to_string(training_set.feature_type()));
    }

    /**
     * We support all combinations of the following types for feature,
     * id, and px datatypes:
     *   feature_type (partitioned_vectors_feature_type): uint8 or float
     *   id_type (partitioned_ids_type): uint32 or uint64
     *   px_type (partitioning_index_type): uint32 or uint64
     */
    if (feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint32_t, uint32_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint32_t, uint32_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint32_t, uint64_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint32_t, uint64_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint64_t, uint32_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT32) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint64_t, uint32_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<uint8_t, uint64_t, uint64_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        px_datatype_ == TILEDB_UINT64) {
      index_ = std::make_unique<
          index_impl<ivf_flat_index<float, uint64_t, uint64_t>>>(
          nlist_, max_iter_, tolerance_, num_threads_);
    }

    index_->train(training_set, init);

    if (dimensions_ != 0 && dimensions_ != index_->dimensions()) {
      throw std::runtime_error(
          "Dimensions mismatch: " + std::to_string(dimensions_) +
          " != " + std::to_string(index_->dimensions()));
    }
    dimensions_ = index_->dimensions();

    if (nlist_ != 0 && nlist_ != index_->num_partitions()) {
      throw std::runtime_error(
          "nlist mismatch: " + std::to_string(nlist_) +
          " != " + std::to_string(index_->num_partitions()));
    }
    nlist_ = index_->num_partitions();
  }

  /**
   * @brief Add a set of vectors to a trained index.
   * @param data_set
   */
  void add(const FeatureVectorArray& data_set) {
    if (feature_datatype_ != data_set.feature_type()) {
      throw std::runtime_error(
          "Feature datatype mismatch: " +
          datatype_to_string(feature_datatype_) +
          " != " + datatype_to_string(data_set.feature_type()));
    }
    if (!index_) {
      throw std::runtime_error("Cannot add() because there is no index.");
    }
    index_->add(data_set);
  }

  void add_with_ids() const {
    // @todo
  }

  // todo query() or search() -- or both?
  [[nodiscard]] auto query_infinite_ram(
      const QueryVectorArray& vectors, size_t top_k, size_t nprobe) const {
    if (!index_) {
      throw std::runtime_error(
          "Cannot query_infinite_ram() because there is no index.");
    }
    return index_->query_infinite_ram(vectors, top_k, nprobe);
  }

  [[nodiscard]] auto query_finite_ram(
      const QueryVectorArray& vectors,
      size_t top_k,
      size_t nprobe,
      size_t upper_bound = 0) const {
    if (!index_) {
      throw std::runtime_error(
          "Cannot query_finite_ram() because there is no index.");
    }
    return index_->query_finite_ram(vectors, top_k, nprobe, upper_bound);
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

  void write_index(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      const std::string& storage_version = "") const {
    if (!index_) {
      throw std::runtime_error(
          "Cannot write_index() because there is no index.");
    }
    index_->write_index(ctx, group_uri, storage_version);
  }

  constexpr auto dimensions() const {
    return dimensions_;  //::dimensions(*index_);
  }

  constexpr auto num_partitions() const {
    return nlist_;  // ::num_partitions(*index_);
  }

  constexpr auto feature_type() const {
    return feature_datatype_;
  }

  inline auto feature_type_string() const {
    return datatype_to_string(feature_datatype_);
  }

  constexpr auto id_type() const {
    return id_datatype_;
  }

  inline auto id_type_string() const {
    return datatype_to_string(id_datatype_);
  }

  constexpr auto px_type() const {
    return px_datatype_;
  }

  inline auto px_type_string() const {
    return datatype_to_string(px_datatype_);
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct index_base {
    virtual ~index_base() = default;

    virtual void train(
        const FeatureVectorArray& training_set,
        kmeans_init init = kmeans_init::random) = 0;

    virtual void add(const FeatureVectorArray& data_set) = 0;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_infinite_ram(
        const QueryVectorArray& vectors, size_t top_k, size_t nprobe) = 0;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_finite_ram(
        const QueryVectorArray& vectors,
        size_t top_k,
        size_t nprobe,
        size_t upper_bound) = 0;

    virtual void update(
        const FeatureVectorArray&,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void remove(const IdVector& ids) const = 0;

    virtual void write_index(
        const tiledb::Context& ctx,
        const std::string& group_uri,
        const std::string& storage_version) const = 0;

    [[nodiscard]] virtual size_t dimensions() const = 0;

    [[nodiscard]] virtual size_t num_partitions() const = 0;
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
        std::optional<size_t> num_threads)
        : impl_index_(nlist, max_iter, tolerance) {
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

    void train(
        const FeatureVectorArray& training_set,
        kmeans_init init = kmeans_init::random) override {
      using feature_type = typename T::value_type;
      auto fspan = MatrixView<feature_type, stdx::layout_left>{
          (feature_type*)training_set.data(),
          extents(training_set)[0],
          extents(training_set)[1]};
      impl_index_.train(fspan, init);
    }

    void add(const FeatureVectorArray& data_set) override {
      using feature_type = typename T::value_type;
      auto fspan = MatrixView<feature_type, stdx::layout_left>{
          (feature_type*)data_set.data(),
          extents(data_set)[0],
          extents(data_set)[1]};
      impl_index_.add(fspan);
    }

    [[nodiscard]] auto query_infinite_ram(
        const tiledb::Context& ctx,
        const URI& uri,
        size_t top_k,
        size_t nprobe) {
      return impl_index_.query_infinite_ram(ctx, uri, top_k, nprobe);
    }

    [[nodiscard]] auto query_finite_ram(
        const tiledb::Context& ctx,
        const URI& uri,
        size_t top_k,
        size_t nprobe) {
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
    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_infinite_ram(
        const QueryVectorArray& vectors, size_t k_nn, size_t nprobe) override {
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

    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_finite_ram(
        const QueryVectorArray& vectors,
        size_t k_nn,
        size_t nprobe,
        size_t upper_bound = 0) override {
      // @todo using index_type = size_t;

      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] =
              impl_index_.query_finite_ram(qspan, k_nn, nprobe, upper_bound);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] =
              impl_index_.query_finite_ram(qspan, k_nn, nprobe, upper_bound);
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

    void write_index(
        const tiledb::Context& ctx,
        const std::string& group_uri,
        const std::string& storage_version) const override {
      impl_index_.write_index(ctx, group_uri, storage_version);
    }

    // WIP
    void remove(const IdVector& ids) const override {
      //      index_.remove(ids);
    }

    size_t dimensions() const override {
      return ::dimensions(impl_index_);
    }

    size_t num_partitions() const override {
      return ::num_partitions(impl_index_);
    }

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  size_t dimensions_ = 0;
  size_t nlist_ = 0;
  size_t max_iter_ = 2;
  float tolerance_ = 1e-4;
  std::optional<size_t> num_threads_ = std::nullopt;
  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};
  tiledb_datatype_t px_datatype_{TILEDB_ANY};
  std::unique_ptr</* const */ index_base> index_;
};

#endif  // TILEDB_API_IVF_FLAT_INDEX_H
