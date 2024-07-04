/**
 * * @file   api/vamana_index.h
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
 * This file defines the `IndexVamana` class, which is a type-erased
 * wrapper of `index_vamana` that allows for runtime polymorphism of the
 * `index_vamana` class template.
 *
 * See IVF.md for details on type erasure.
 */

#ifndef TILEDB_API_VAMANA_INDEX_H
#define TILEDB_API_VAMANA_INDEX_H

#include <tiledb/tiledb>

#include "api/feature_vector.h"
#include "api/feature_vector_array.h"
#include "api_defs.h"
#include "index/vamana_index.h"

/*******************************************************************************
 * IndexVamana
 ******************************************************************************/
/**
 * A type-erased Vamana index class. This is a type-erased wrapper around
 * the `vamana_index` class in detail/graph/vamana.h.
 *
 * An index class is provides
 *   - URI-based constructor
 *   - Array-based constructor
 *   - A train method
 *   - An add method
 *   - A query method
 *
 * We support all combinations of the following types for feature, id, and px
 * datatypes:
 *   - feature_type: uint8 or float
 *   - id_type: uint32 or uint64
 *   - adjacency_row_index_type: uint32 or uint64
 */
class IndexVamana {
 public:
  IndexVamana(const IndexVamana&) = delete;
  IndexVamana(IndexVamana&&) = default;
  IndexVamana& operator=(const IndexVamana&) = delete;
  IndexVamana& operator=(IndexVamana&&) = default;

  /**
   * @brief Create an index with the given configuration. The index in this
   * state must next be trained. The sequence for creating an index in this
   * fashion is:
   *  - Create an IndexVamana object with the desired configuration (using this
   *  constructor
   *  - Call train() with the training data
   *  - Call add() to add a set of vectors to the index (often the same as the
   *  training data)
   *  Either (or both)
   *    - Perform a query
   *    - Call write_index() to write the index to disk
   * @param config A map of configuration parameters, as pairs of strings
   * containing config parameters and values.
   */
  explicit IndexVamana(
      const std::optional<IndexOptions>& config = std::nullopt) {
    feature_datatype_ = TILEDB_ANY;
    id_datatype_ = TILEDB_UINT32;

    if (config) {
      for (auto&& c : *config) {
        auto key = c.first;
        auto value = c.second;
        if (key == "dimensions") {
          dimensions_ = std::stol(value);
        } else if (key == "l_build") {
          l_build_ = std::stol(value);
        } else if (key == "r_max_degree") {
          r_max_degree_ = std::stol(value);
        } else if (key == "feature_type") {
          feature_datatype_ = string_to_datatype(value);
        } else if (key == "id_type") {
          id_datatype_ = string_to_datatype(value);
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
   * to create the internal vamana_index object.
   *
   * @param ctx
   * @param group_uri TileDB group containing all the arrays comprising the
   * index.
   */
  IndexVamana(
      const tiledb::Context& ctx,
      const URI& group_uri,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt) {
    read_types(ctx, group_uri, &feature_datatype_, &id_datatype_);

    auto type = std::tuple{
        feature_datatype_, id_datatype_, adjacency_row_index_datatype_};
    if (uri_dispatch_table.find(type) == uri_dispatch_table.end()) {
      throw std::runtime_error("Unsupported datatype combination");
    }
    index_ = uri_dispatch_table.at(type)(ctx, group_uri, temporal_policy);
    l_build_ = index_->l_build();
    r_max_degree_ = index_->r_max_degree();

    if (dimensions_ != 0 && dimensions_ != index_->dimensions()) {
      throw std::runtime_error(
          "Dimensions mismatch: " + std::to_string(dimensions_) +
          " != " + std::to_string(index_->dimensions()));
    }
    dimensions_ = index_->dimensions();
  }

  /**
   * @brief Train the index based on the given training set.
   * @param training_set
   * @param init
   */
  // @todo -- infer feature type from input
  void train(const FeatureVectorArray& training_set) {
    if (feature_datatype_ == TILEDB_ANY) {
      feature_datatype_ = training_set.feature_type();
    } else if (feature_datatype_ != training_set.feature_type()) {
      throw std::runtime_error(
          "Feature datatype mismatch: " +
          datatype_to_string(feature_datatype_) +
          " != " + datatype_to_string(training_set.feature_type()));
    }

    auto type = std::tuple{
        feature_datatype_, id_datatype_, adjacency_row_index_datatype_};
    if (dispatch_table.find(type) == dispatch_table.end()) {
      throw std::runtime_error("Unsupported datatype combination");
    }

    // Create a new index. Note that we may have already loaded an existing
    // index by URI. In that case, we have updated our local state (i.e.
    // l_build_, r_max_degree_), but we should also use the
    // timestamp from that already loaded index.
    index_ = dispatch_table.at(type)(
        training_set.num_vectors(),
        l_build_,
        r_max_degree_,
        index_ ? std::make_optional<TemporalPolicy>(index_->temporal_policy()) :
                 std::nullopt);

    index_->train(training_set);

    if (dimensions_ != 0 && dimensions_ != index_->dimensions()) {
      throw std::runtime_error(
          "Dimensions mismatch: " + std::to_string(dimensions_) +
          " != " + std::to_string(index_->dimensions()));
    }
    dimensions_ = index_->dimensions();
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

  // todo query() or search() -- or both?
  [[nodiscard]] auto query(
      const QueryVectorArray& vectors,
      size_t top_k,
      std::optional<size_t> l_search = std::nullopt) {
    if (!index_) {
      throw std::runtime_error("Cannot query() because there is no index.");
    }
    return index_->query(vectors, top_k, l_search);
  }

  void write_index(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt,
      const std::string& storage_version = "") {
    if (!index_) {
      throw std::runtime_error(
          "Cannot write_index() because there is no index.");
    }
    index_->write_index(ctx, group_uri, temporal_policy, storage_version);
  }

  static void clear_history(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      uint64_t timestamp) {
    tiledb_datatype_t feature_datatype{TILEDB_ANY};
    tiledb_datatype_t id_datatype{TILEDB_ANY};
    read_types(ctx, group_uri, &feature_datatype, &id_datatype);

    auto type = std::tuple{
        feature_datatype, id_datatype, adjacency_row_index_datatype_};
    if (clear_history_dispatch_table.find(type) ==
        clear_history_dispatch_table.end()) {
      throw std::runtime_error("Unsupported datatype combination");
    }
    clear_history_dispatch_table.at(type)(ctx, group_uri, timestamp);
  }

  auto temporal_policy() const {
    if (!index_) {
      throw std::runtime_error(
          "Cannot get temporal_policy() because there is no index.");
    }
    return index_->temporal_policy();
  }

  constexpr auto dimensions() const {
    return dimensions_;
  }

  constexpr auto l_build() const {
    return l_build_;
  }

  constexpr auto r_max_degree() const {
    return r_max_degree_;
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

 private:
  static void read_types(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      tiledb_datatype_t* feature_datatype,
      tiledb_datatype_t* id_datatype) {
    using metadata_element =
        std::tuple<std::string, tiledb_datatype_t*, tiledb_datatype_t>;
    std::vector<metadata_element> metadata{
        {"feature_datatype", feature_datatype, TILEDB_UINT32},
        {"id_datatype", id_datatype, TILEDB_UINT32}};

    tiledb::Group read_group(ctx, group_uri, TILEDB_READ, ctx.config());

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
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct index_base {
    virtual ~index_base() = default;

    virtual void train(const FeatureVectorArray& training_set) = 0;

    virtual void add(const FeatureVectorArray& data_set) = 0;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query(
        const QueryVectorArray& vectors,
        size_t top_k,
        std::optional<size_t> l_search) = 0;

    virtual void write_index(
        const tiledb::Context& ctx,
        const std::string& group_uri,
        std::optional<TemporalPolicy> temporal_policy,
        const std::string& storage_version) = 0;

    [[nodiscard]] virtual size_t dimensions() const = 0;
    [[nodiscard]] virtual size_t l_build() const = 0;
    [[nodiscard]] virtual size_t r_max_degree() const = 0;
    [[nodiscard]] virtual TemporalPolicy temporal_policy() const = 0;
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
        size_t num_vectors,
        size_t l_build,
        size_t r_max_degree,
        std::optional<TemporalPolicy> temporal_policy)
        : impl_index_(num_vectors, l_build, r_max_degree, temporal_policy) {
    }

    index_impl(
        const tiledb::Context& ctx,
        const URI& index_uri,
        std::optional<TemporalPolicy> temporal_policy)
        : impl_index_(ctx, index_uri, temporal_policy) {
    }

    void train(const FeatureVectorArray& training_set) override {
      using feature_type = typename T::feature_type;
      auto fspan = MatrixView<feature_type, stdx::layout_left>{
          (feature_type*)training_set.data(),
          extents(training_set)[0],
          extents(training_set)[1]};

      using id_type = typename T::id_type;
      if (num_ids(training_set) > 0) {
        auto ids = std::span<id_type>(
            (id_type*)training_set.ids(), training_set.num_vectors());
        impl_index_.train(fspan, ids);
      } else {
        auto ids = std::vector<id_type>(::num_vectors(training_set));
        std::iota(ids.begin(), ids.end(), 0);
        impl_index_.train(fspan, ids);
      }
    }

    void add(const FeatureVectorArray& data_set) override {
      using feature_type = typename T::feature_type;
      auto fspan = MatrixView<feature_type, stdx::layout_left>{
          (feature_type*)data_set.data(),
          extents(data_set)[0],
          extents(data_set)[1]};
      impl_index_.add(fspan);
    }

    /**
     * @brief Query the index with the given vectors.  The concrete query
     * function returns a tuple of arrays, which are type erased and returned as
     * a tuple of FeatureVectorArrays.
     * @param vectors
     * @param top_k
     * @return
     *
     * @todo Make sure the extents of the returned arrays are used correctly.
     */
    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query(
        const QueryVectorArray& vectors,
        size_t top_k,
        std::optional<size_t> l_search) override {
      // @todo using index_type = size_t;
      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(qspan, top_k, l_search);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(qspan, top_k, l_search);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    void write_index(
        const tiledb::Context& ctx,
        const std::string& group_uri,
        std::optional<TemporalPolicy> temporal_policy,
        const std::string& storage_version) override {
      impl_index_.write_index(ctx, group_uri, temporal_policy, storage_version);
    }

    size_t dimensions() const override {
      return ::dimensions(impl_index_);
    }

    size_t l_build() const override {
      return impl_index_.l_build();
    }

    size_t r_max_degree() const override {
      return impl_index_.r_max_degree();
    }

    TemporalPolicy temporal_policy() const override {
      return impl_index_.temporal_policy();
    }

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  // clang-format off
  using constructor_function = std::function<std::unique_ptr<index_base>(size_t, size_t, size_t, std::optional<TemporalPolicy>)>;
  using table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, constructor_function>;
  static const table_type dispatch_table;

  using uri_constructor_function = std::function<std::unique_ptr<index_base>(const tiledb::Context&, const std::string&, std::optional<TemporalPolicy>)>;
  using uri_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, uri_constructor_function>;
  static const uri_table_type uri_dispatch_table;

  using clear_history_constructor_function = std::function<void(const tiledb::Context&, const std::string&, uint64_t)>;
  using clear_history_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, clear_history_constructor_function>;
  static const clear_history_table_type clear_history_dispatch_table;
  // clang-format on

  size_t dimensions_ = 0;
  size_t l_build_ = 100;
  size_t r_max_degree_ = 64;
  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};
  static constexpr tiledb_datatype_t adjacency_row_index_datatype_{
      TILEDB_UINT64};
  std::unique_ptr<index_base> index_;
};

// clang-format off
const IndexVamana::table_type IndexVamana::dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint32_t, uint32_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint32_t, uint32_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint32_t, uint32_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint32_t, uint64_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint32_t, uint64_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint32_t, uint64_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint64_t, uint32_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint64_t, uint32_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint64_t, uint32_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint64_t, uint64_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint64_t, uint64_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](size_t num_vectors, size_t l_build, size_t r_max_degree, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint64_t, uint64_t>>>(num_vectors, l_build, r_max_degree, temporal_policy); }},
};

const IndexVamana::uri_table_type IndexVamana::uri_dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint32_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint32_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint32_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint32_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint32_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint32_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint64_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint64_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint64_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<int8_t,  uint64_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<uint8_t, uint64_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<vamana_index<float,   uint64_t, uint64_t>>>(ctx, uri, temporal_policy); }},
};

const IndexVamana::clear_history_table_type IndexVamana::clear_history_dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<int8_t,  uint32_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<uint8_t, uint32_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<float,   uint32_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<int8_t,  uint32_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<uint8_t, uint32_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<float,   uint32_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<int8_t,  uint64_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<uint8_t, uint64_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<float,   uint64_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<int8_t,  uint64_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<uint8_t, uint64_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return vamana_index<float,   uint64_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
};
// clang-format on

#endif  // TILEDB_API_VAMANA_INDEX_H
