/**
 * * @file   api/ivf_pq_index.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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
 * This file defines the `IndexIVFPQ` class, which is a type-erased
 * wrapper of `index_ivf_pq` that allows for runtime polymorphism of the
 * `index_ivf_pq` class template.
 *
 * See IVF.md for details on type erasure.
 */

#ifndef TILEDB_API_ivf_pq_index_H
#define TILEDB_API_ivf_pq_index_H

#include <tiledb/tiledb>

#include "api/feature_vector.h"
#include "api/feature_vector_array.h"
#include "api_defs.h"
#include "index/index_defs.h"
#include "index/ivf_pq_index.h"

/*******************************************************************************
 * IndexIVFPQ
 ******************************************************************************/
/**
 * A type-erased IVFPQ index class. This is a type-erased wrapper around
 * the `ivf_pq_index` class.
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
 *   - feature_type: uint8, int8, or float
 *   - id_type: uint32 or uint64
 *   - partitioning_index_type: uint32 or uint64
 */
class IndexIVFPQ {
 public:
  IndexIVFPQ(const IndexIVFPQ&) = delete;
  IndexIVFPQ(IndexIVFPQ&&) = default;
  IndexIVFPQ& operator=(const IndexIVFPQ&) = delete;
  IndexIVFPQ& operator=(IndexIVFPQ&&) = default;

  /**
   * @brief Create an index with the given configuration. The index in this
   * state must next be trained. The sequence for creating an index in this
   * fashion is:
   *  - Create an IndexIVFPQ object with the desired configuration (using this
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
  explicit IndexIVFPQ(
      const std::optional<IndexOptions>& config = std::nullopt) {
    feature_datatype_ = TILEDB_ANY;
    id_datatype_ = TILEDB_UINT32;
    partitioning_index_datatype_ = TILEDB_UINT32;

    if (config) {
      for (auto&& c : *config) {
        auto key = c.first;
        auto value = c.second;
        if (key == "dimensions") {
          dimensions_ = std::stol(value);
        } else if (key == "n_list") {
          n_list_ = std::stol(value);
        } else if (key == "num_subspaces") {
          num_subspaces_ = std::stol(value);
        } else if (key == "max_iterations") {
          max_iterations_ = std::stol(value);
        } else if (key == "convergence_tolerance") {
          convergence_tolerance_ = std::stof(value);
        } else if (key == "feature_type") {
          feature_datatype_ = string_to_datatype(value);
        } else if (key == "id_type") {
          id_datatype_ = string_to_datatype(value);
        } else if (key == "partitioning_index_type") {
          partitioning_index_datatype_ = string_to_datatype(value);
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
   * to create the internal ivf_pq_index object.
   *
   * @param ctx
   * @param group_uri TileDB group containing all the arrays comprising the
   * index.
   */
  IndexIVFPQ(
      const tiledb::Context& ctx,
      const URI& group_uri,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt) {
    read_types(
        ctx,
        group_uri,
        &feature_datatype_,
        &id_datatype_,
        &partitioning_index_datatype_);

    auto type = std::tuple{
        feature_datatype_, id_datatype_, partitioning_index_datatype_};
    if (uri_dispatch_table.find(type) == uri_dispatch_table.end()) {
      throw std::runtime_error("Unsupported datatype combination");
    }
    index_ = uri_dispatch_table.at(type)(ctx, group_uri, temporal_policy);
    n_list_ = index_->nlist();
    num_subspaces_ = index_->num_subspaces();
    max_iterations_ = index_->max_iterations();
    convergence_tolerance_ = index_->convergence_tolerance();

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
        feature_datatype_, id_datatype_, partitioning_index_datatype_};
    if (dispatch_table.find(type) == dispatch_table.end()) {
      throw std::runtime_error("Unsupported datatype combination");
    }

    // Create a new index. Note that we may have already loaded an existing
    // index by URI. In that case, we have updated our local state (i.e.
    // num_subspaces_, etc.), but we should also use the timestamp from that
    // already loaded index.
    index_ = dispatch_table.at(type)(
        n_list_,
        num_subspaces_,
        max_iterations_,
        convergence_tolerance_,
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

  [[nodiscard]] auto query(
      QueryType queryType,
      const QueryVectorArray& vectors,
      size_t top_k,
      size_t nprobe) {
    if (!index_) {
      throw std::runtime_error(
          "Cannot query_infinite_ram() because there is no index.");
    }
    return index_->query(queryType, vectors, top_k, nprobe);
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
    tiledb_datatype_t partitioning_index_datatype{TILEDB_ANY};
    read_types(
        ctx,
        group_uri,
        &feature_datatype,
        &id_datatype,
        &partitioning_index_datatype);

    auto type =
        std::tuple{feature_datatype, id_datatype, partitioning_index_datatype};
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

  constexpr auto n_list() const {
    return n_list_;
  }

  constexpr auto num_subspaces() const {
    return num_subspaces_;
  }

  constexpr auto max_iterations() const {
    return max_iterations_;
  }

  constexpr auto convergence_tolerance() const {
    return convergence_tolerance_;
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

  constexpr auto partitioning_index_type() const {
    return partitioning_index_datatype_;
  }

  inline auto partitioning_index_type_string() const {
    return datatype_to_string(partitioning_index_datatype_);
  }

 private:
  static void read_types(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      tiledb_datatype_t* feature_datatype,
      tiledb_datatype_t* id_datatype,
      tiledb_datatype_t* partitioning_index_datatype) {
    using metadata_element =
        std::tuple<std::string, tiledb_datatype_t*, tiledb_datatype_t>;
    std::vector<metadata_element> metadata{
        {"feature_datatype", feature_datatype, TILEDB_UINT32},
        {"id_datatype", id_datatype, TILEDB_UINT32},
        {"px_datatype", partitioning_index_datatype, TILEDB_UINT32}};

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
        QueryType queryType,
        const QueryVectorArray& vectors,
        size_t top_k,
        size_t nprobe) = 0;

    virtual void write_index(
        const tiledb::Context& ctx,
        const std::string& group_uri,
        std::optional<TemporalPolicy> temporal_policy,
        const std::string& storage_version) = 0;

    [[nodiscard]] virtual size_t dimensions() const = 0;
    [[nodiscard]] virtual TemporalPolicy temporal_policy() const = 0;
    [[nodiscard]] virtual uint64_t nlist() const = 0;
    [[nodiscard]] virtual uint64_t num_subspaces() const = 0;
    [[nodiscard]] virtual uint64_t max_iterations() const = 0;
    [[nodiscard]] virtual float convergence_tolerance() const = 0;
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
        size_t n_list,
        size_t num_subspaces,
        size_t max_iterations,
        size_t convergence_tolerance,
        std::optional<TemporalPolicy> temporal_policy)
        : impl_index_(
              n_list,
              num_subspaces,
              max_iterations,
              convergence_tolerance,
              temporal_policy) {
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

      using id_type = typename T::id_type;
      if (num_ids(data_set) > 0) {
        auto ids = std::span<id_type>(
            (id_type*)data_set.ids(), data_set.num_vectors());
        impl_index_.add(fspan, ids);
      } else {
        auto ids = std::vector<id_type>(::num_vectors(data_set));
        std::iota(ids.begin(), ids.end(), 0);
        impl_index_.add(fspan, ids);
      }
    }

    /**
     * @brief Query the index with the given vectors.  The concrete query
     * function returns a tuple of arrays, which are type erased and returned as
     * a tuple of FeatureVectorArrays.
     *
     * @param queryType Whether to use finite or infinite RAM.
     * @param vectors The query vectors.
     * @param top_k The number of results to return for each query vector.
     * @param nprobe The number of clusters to search in the index.
     * @return A tuple of FeatureVectorArrays, one for the scores and one for
     * the distances.
     *
     * @todo Make sure the extents of the returned arrays are used correctly.
     */
    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query(
        QueryType queryType,
        const QueryVectorArray& vectors,
        size_t top_k,
        size_t nprobe) override {
      // @todo using index_type = size_t;
      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(queryType, qspan, top_k, nprobe);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(queryType, qspan, top_k, nprobe);
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

    TemporalPolicy temporal_policy() const override {
      return impl_index_.temporal_policy();
    }

    uint64_t nlist() const override {
      return impl_index_.nlist();
    }

    uint64_t num_subspaces() const override {
      return impl_index_.num_subspaces();
    }

    uint64_t max_iterations() const override {
      return impl_index_.max_iterations();
    }

    float convergence_tolerance() const override {
      return impl_index_.convergence_tolerance();
    }

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  // clang-format off
  using constructor_function = std::function<std::unique_ptr<index_base>(size_t, size_t, size_t, float, std::optional<TemporalPolicy>)>;
  using table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, constructor_function>;
  static const table_type dispatch_table;

  using uri_constructor_function = std::function<std::unique_ptr<index_base>(const tiledb::Context&, const std::string&, std::optional<TemporalPolicy>)>;
  using uri_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, uri_constructor_function>;
  static const uri_table_type uri_dispatch_table;

  using clear_history_constructor_function = std::function<void(const tiledb::Context&, const std::string&, uint64_t)>;
  using clear_history_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, clear_history_constructor_function>;
  static const clear_history_table_type clear_history_dispatch_table;
  // clang-format on

  size_t dimensions_{0};
  size_t n_list_{0};
  size_t num_subspaces_{16};
  size_t max_iterations_{2};
  float convergence_tolerance_{0.000025f};
  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};
  tiledb_datatype_t partitioning_index_datatype_{TILEDB_ANY};
  std::unique_ptr<index_base> index_;
};

// clang-format off
const IndexIVFPQ::table_type IndexIVFPQ::dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint32_t, uint32_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint32_t, uint32_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint32_t, uint32_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint32_t, uint64_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint32_t, uint64_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint32_t, uint64_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint64_t, uint32_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint64_t, uint32_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint64_t, uint32_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint64_t, uint64_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint64_t, uint64_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](size_t nlist, size_t num_subspaces, size_t max_iterations, float convergence_tolerance, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint64_t, uint64_t>>>(nlist, num_subspaces, max_iterations, convergence_tolerance, temporal_policy); }},
};

const IndexIVFPQ::uri_table_type IndexIVFPQ::uri_dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint32_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint32_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint32_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint32_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint32_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint32_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint64_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint64_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint64_t, uint32_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint64_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint64_t, uint64_t>>>(ctx, uri, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint64_t, uint64_t>>>(ctx, uri, temporal_policy); }},
};

const IndexIVFPQ::clear_history_table_type IndexIVFPQ::clear_history_dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<int8_t,  uint32_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<uint8_t, uint32_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<float,   uint32_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<int8_t,  uint32_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<uint8_t, uint32_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<float,   uint32_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<int8_t,  uint64_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<uint8_t, uint64_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<float,   uint64_t, uint32_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<int8_t,  uint64_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<uint8_t, uint64_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, uint64_t timestamp) { return ivf_pq_index<float,   uint64_t, uint64_t>::clear_history(ctx, uri, timestamp); }},
};
// clang-format on

#endif  // TILEDB_API_ivf_pq_index_H
