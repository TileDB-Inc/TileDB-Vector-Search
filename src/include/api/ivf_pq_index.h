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
#include "index/index_metadata.h"
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
  static void create(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      uint64_t dimensions,
      tiledb_datatype_t feature_datatype = TILEDB_UINT32,
      tiledb_datatype_t id_datatype = TILEDB_UINT32,
      tiledb_datatype_t partitioning_index_datatype = TILEDB_UINT32,
      uint32_t num_subspaces = 16,
      uint32_t max_iterations = 2,
      float convergence_tolerance = 0.000025f,
      float reassign_ratio = 0.075f,
      std::optional<TemporalPolicy> optional_temporal_policy = std::nullopt,
      DistanceMetric distance_metric = DistanceMetric::SUM_OF_SQUARES,
      const std::string& storage_version = "") {
    if (distance_metric != DistanceMetric::SUM_OF_SQUARES &&
        distance_metric != DistanceMetric::L2) {
      throw std::runtime_error(
          "Invalid distance metric value, only SUM_OF_SQUARES and L2 are "
          "supported");
    }
    auto type =
        std::tuple{feature_datatype, id_datatype, partitioning_index_datatype};
    if (clear_history_dispatch_table.find(type) ==
        clear_history_dispatch_table.end()) {
      throw std::runtime_error("Unsupported datatype combination");
    }
    create_dispatch_table.at(type)(
        ctx,
        group_uri,
        dimensions,
        num_subspaces,
        max_iterations,
        convergence_tolerance,
        reassign_ratio,
        optional_temporal_policy,
        distance_metric,
        storage_version);
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
      IndexLoadStrategy index_load_strategy = IndexLoadStrategy::PQ_INDEX,
      size_t upper_bound = 0,
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
    index_ = uri_dispatch_table.at(type)(
        ctx, group_uri, index_load_strategy, upper_bound, temporal_policy);
    partitions_ = index_->partitions();
    num_subspaces_ = index_->num_subspaces();
    max_iterations_ = index_->max_iterations();
    convergence_tolerance_ = index_->convergence_tolerance();
    reassign_ratio_ = index_->reassign_ratio();
    distance_metric_ = index_->distance_metric();
    upper_bound_ = index_->upper_bound();

    if (dimensions_ != 0 && dimensions_ != index_->dimensions()) {
      throw std::runtime_error(
          "Dimensions mismatch: " + std::to_string(dimensions_) +
          " != " + std::to_string(index_->dimensions()));
    }
    dimensions_ = index_->dimensions();
  }

  void create_temp_data_group(const std::string& partial_write_array_dir) {
    if (!index_) {
      throw std::runtime_error(
          "Cannot create_temp_data_group() because there is no index.");
    }
    index_->create_temp_data_group(partial_write_array_dir);
  }

  /**
   * @brief Train the index based on the given training set.
   * @param training_set The training input vectors.
   * @param partitions The number of clusters to use in the index. Can be passed
   * to override the value we used when we first created the index.
   */
  // @todo -- infer feature type from input
  void train(
      const FeatureVectorArray& training_set,
      size_t partitions = 0,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt) {
    if (feature_datatype_ == TILEDB_ANY) {
      feature_datatype_ = training_set.feature_type();
    } else if (feature_datatype_ != training_set.feature_type()) {
      throw std::runtime_error(
          "[ivf_pq_index@train] Feature datatype mismatch: " +
          datatype_to_string(feature_datatype_) +
          " != " + datatype_to_string(training_set.feature_type()));
    }

    auto type = std::tuple{
        feature_datatype_, id_datatype_, partitioning_index_datatype_};
    if (create_dispatch_table.find(type) == create_dispatch_table.end()) {
      throw std::runtime_error("Unsupported datatype combination");
    }

    index_->train(training_set, partitions, temporal_policy);
    partitions_ = index_->partitions();

    if (dimensions_ != 0 && dimensions_ != index_->dimensions()) {
      throw std::runtime_error(
          "[ivf_pq_index@train] Dimensions mismatch: " +
          std::to_string(dimensions_) +
          " != " + std::to_string(index_->dimensions()));
    }
    dimensions_ = index_->dimensions();
  }

  /**
   * @brief Add a set of vectors to a trained index.
   * @param input_vectors The set of vectors to add to the index.
   * @param external_ids The ids of vectors to add to the index.
   * @param deleted_ids The ids of vectors to delete from the index.
   *
   */
  void ingest_parts(
      const FeatureVectorArray& input_vectors,
      const FeatureVector& external_ids,
      const FeatureVector& deleted_ids,
      size_t start,
      size_t end,
      size_t partition_start,
      const std::string& partial_write_array_dir) {
    if (feature_datatype_ != input_vectors.feature_type()) {
      throw std::runtime_error(
          "[ivf_pq_index@ingest_parts] Feature datatype mismatch: " +
          datatype_to_string(feature_datatype_) +
          " != " + datatype_to_string(input_vectors.feature_type()));
    }
    if (!index_) {
      throw std::runtime_error(
          "Cannot ingest_parts() because there is no index.");
    }
    index_->ingest_parts(
        input_vectors,
        external_ids,
        deleted_ids,
        start,
        end,
        partition_start,
        partial_write_array_dir);
  }

  void ingest(
      const FeatureVectorArray& input_vectors,
      const FeatureVector& external_ids = FeatureVector(0, "float32")) {
    if (feature_datatype_ != input_vectors.feature_type()) {
      throw std::runtime_error(
          "[ivf_pq_index@ingest] Feature datatype mismatch: " +
          datatype_to_string(feature_datatype_) +
          " != " + datatype_to_string(input_vectors.feature_type()));
    }
    if (!index_) {
      throw std::runtime_error("Cannot ingest() because there is no index.");
    }
    index_->ingest(input_vectors, external_ids);
  }

  void consolidate_partitions(
      size_t partitions,
      size_t work_items,
      size_t partition_id_start,
      size_t partition_id_end,
      size_t batch,
      const std::string& partial_write_array_dir) {
    if (!index_) {
      throw std::runtime_error(
          "[ivf_pq_index@consolidate_partitions] Cannot "
          "consolidate_partitions() because there is no index.");
    }
    index_->consolidate_partitions(
        partitions,
        work_items,
        partition_id_start,
        partition_id_end,
        batch,
        partial_write_array_dir);
  }

  [[nodiscard]] auto query(
      const QueryVectorArray& vectors,
      size_t top_k,
      size_t nprobe,
      float k_factor = 1.f) {
    if (!index_) {
      throw std::runtime_error(
          "[ivf_pq_index@query] Cannot query() because there is no index.");
    }
    return index_->query(vectors, top_k, nprobe, k_factor);
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

  TemporalPolicy temporal_policy() const {
    if (!index_) {
      throw std::runtime_error(
          "Cannot get temporal_policy() because there is no index.");
    }
    return index_->temporal_policy();
  }

  constexpr uint64_t dimensions() const {
    return dimensions_;
  }

  constexpr size_t upper_bound() const {
    return upper_bound_;
  }

  constexpr auto partitions() const {
    return partitions_;
  }

  constexpr uint32_t num_subspaces() const {
    return num_subspaces_;
  }

  constexpr uint32_t max_iterations() const {
    return max_iterations_;
  }

  constexpr float convergence_tolerance() const {
    return convergence_tolerance_;
  }

  constexpr float reassign_ratio() const {
    return reassign_ratio_;
  }

  constexpr DistanceMetric distance_metric() const {
    return distance_metric_;
  }

  constexpr tiledb_datatype_t feature_type() const {
    return feature_datatype_;
  }

  inline std::string feature_type_string() const {
    return datatype_to_string(feature_datatype_);
  }

  constexpr tiledb_datatype_t id_type() const {
    return id_datatype_;
  }

  inline std::string id_type_string() const {
    return datatype_to_string(id_datatype_);
  }

  constexpr tiledb_datatype_t partitioning_index_type() const {
    return partitioning_index_datatype_;
  }

  inline std::string partitioning_index_type_string() const {
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

    virtual void create_temp_data_group(
        const std::string& partial_write_array_dir) = 0;

    virtual void train(
        const FeatureVectorArray& training_set,
        size_t partitions,
        std::optional<TemporalPolicy> temporal_policy) = 0;

    virtual void ingest_parts(
        const FeatureVectorArray& input_vectors,
        const FeatureVector& external_ids,
        const FeatureVector& deleted_ids,
        size_t start,
        size_t end,
        size_t partition_start,
        const std::string& partial_write_array_dir) = 0;

    virtual void ingest(
        const FeatureVectorArray& input_vectors,
        const FeatureVector& external_ids) = 0;

    virtual void consolidate_partitions(
        size_t partitions,
        size_t work_items,
        size_t partition_id_start,
        size_t partition_id_end,
        size_t batch,
        const std::string& partial_write_array_dir) = 0;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query(
        const QueryVectorArray& vectors,
        size_t top_k,
        size_t nprobe,
        float k_factor) = 0;

    [[nodiscard]] virtual uint64_t dimensions() const = 0;
    [[nodiscard]] virtual size_t upper_bound() const = 0;
    [[nodiscard]] virtual TemporalPolicy temporal_policy() const = 0;
    [[nodiscard]] virtual uint64_t partitions() const = 0;
    [[nodiscard]] virtual uint32_t num_subspaces() const = 0;
    [[nodiscard]] virtual uint32_t max_iterations() const = 0;
    [[nodiscard]] virtual float convergence_tolerance() const = 0;
    [[nodiscard]] virtual float reassign_ratio() const = 0;
    [[nodiscard]] virtual DistanceMetric distance_metric() const = 0;
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
        size_t partitions,
        uint32_t num_subspaces,
        uint32_t max_iterations,
        float convergence_tolerance,
        float reassign_ratio,
        std::optional<TemporalPolicy> temporal_policy,
        DistanceMetric distance_metric)
        : impl_index_(
              partitions,
              num_subspaces,
              max_iterations,
              convergence_tolerance,
              reassign_ratio,
              temporal_policy,
              distance_metric) {
    }

    index_impl(
        const tiledb::Context& ctx,
        const URI& index_uri,
        IndexLoadStrategy index_load_strategy,
        size_t upper_bound,
        std::optional<TemporalPolicy> temporal_policy)
        : impl_index_(
              ctx,
              index_uri,
              index_load_strategy,
              upper_bound,
              temporal_policy) {
    }

    void create_temp_data_group(
        const std::string& partial_write_array_dir) override {
      impl_index_.create_temp_data_group(partial_write_array_dir);
    }

    void train(
        const FeatureVectorArray& training_set,
        size_t partitions,
        std::optional<TemporalPolicy> temporal_policy) override {
      using feature_type = typename T::feature_type;
      auto fspan = MatrixView<feature_type, stdx::layout_left>{
          (feature_type*)training_set.data(),
          extents(training_set)[0],
          extents(training_set)[1]};
      impl_index_.train(fspan, partitions, temporal_policy);
    }

    void ingest_parts(
        const FeatureVectorArray& input_vectors,
        const FeatureVector& external_ids,
        const FeatureVector& deleted_ids,
        size_t start,
        size_t end,
        size_t partition_start,
        const std::string& partial_write_array_dir) override {
      using feature_type = typename T::feature_type;
      using id_type = typename T::id_type;
      auto fspan = MatrixView<feature_type, stdx::layout_left>{
          (feature_type*)input_vectors.data(),
          extents(input_vectors)[0],
          extents(input_vectors)[1]};
      auto deleted_ids_span = std::span<id_type>(
          (id_type*)deleted_ids.data(), deleted_ids.dimensions());
      if (external_ids.dimensions() == 0) {
        auto ids = std::vector<id_type>(::num_vectors(input_vectors));
        std::iota(ids.begin(), ids.end(), start);
        impl_index_.ingest_parts(
            fspan,
            ids,
            deleted_ids_span,
            start,
            end,
            partition_start,
            partial_write_array_dir);
      } else {
        auto external_ids_span = std::span<id_type>(
            (id_type*)external_ids.data(), external_ids.dimensions());
        impl_index_.ingest_parts(
            fspan,
            external_ids_span,
            deleted_ids_span,
            start,
            end,
            partition_start,
            partial_write_array_dir);
      }
    }

    void ingest(
        const FeatureVectorArray& input_vectors,
        const FeatureVector& external_ids) override {
      using feature_type = typename T::feature_type;
      using id_type = typename T::id_type;
      auto fspan = MatrixView<feature_type, stdx::layout_left>{
          (feature_type*)input_vectors.data(),
          extents(input_vectors)[0],
          extents(input_vectors)[1]};
      if (external_ids.dimensions() == 0) {
        auto ids = std::vector<id_type>(::num_vectors(input_vectors));
        std::iota(ids.begin(), ids.end(), 0);
        impl_index_.ingest(fspan, ids);
      } else {
        auto ids = std::span<id_type>(
            (id_type*)external_ids.data(), external_ids.dimensions());
        impl_index_.ingest(fspan, ids);
      }
    }

    void consolidate_partitions(
        size_t partitions,
        size_t work_items,
        size_t partition_id_start,
        size_t partition_id_end,
        size_t batch,
        const std::string& partial_write_array_dir) override {
      impl_index_.consolidate_partitions(
          partitions,
          work_items,
          partition_id_start,
          partition_id_end,
          batch,
          partial_write_array_dir);
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
        const QueryVectorArray& vectors,
        size_t top_k,
        size_t nprobe,
        float k_factor) override {
      // @todo using index_type = size_t;
      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(qspan, top_k, nprobe, k_factor);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(qspan, top_k, nprobe, k_factor);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    uint64_t dimensions() const override {
      return ::dimensions(impl_index_);
    }

    size_t upper_bound() const override {
      return impl_index_.upper_bound();
    }

    TemporalPolicy temporal_policy() const override {
      return impl_index_.temporal_policy();
    }

    uint64_t partitions() const override {
      return impl_index_.partitions();
    }

    uint32_t num_subspaces() const override {
      return impl_index_.num_subspaces();
    }

    uint32_t max_iterations() const override {
      return impl_index_.max_iterations();
    }

    float convergence_tolerance() const override {
      return impl_index_.convergence_tolerance();
    }

    float reassign_ratio() const override {
      return impl_index_.reassign_ratio();
    }

    DistanceMetric distance_metric() const override {
      return impl_index_.distance_metric();
    }

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  // clang-format off
  using create_constructor_function = std::function<void(const tiledb::Context&, const std::string &, uint64_t, uint32_t, uint32_t, float, float, std::optional<TemporalPolicy>, DistanceMetric, const std::string &)>;
  using create_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, create_constructor_function>;
  static const create_table_type create_dispatch_table;

  using uri_constructor_function = std::function<std::unique_ptr<index_base>(const tiledb::Context&, const std::string&, IndexLoadStrategy, size_t, std::optional<TemporalPolicy>)>;
  using uri_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, uri_constructor_function>;
  static const uri_table_type uri_dispatch_table;

  using clear_history_constructor_function = std::function<void(const tiledb::Context&, const std::string&, uint64_t)>;
  using clear_history_table_type = std::map<std::tuple<tiledb_datatype_t, tiledb_datatype_t, tiledb_datatype_t>, clear_history_constructor_function>;
  static const clear_history_table_type clear_history_dispatch_table;
  // clang-format on

  uint64_t dimensions_{0};
  size_t upper_bound_{0};
  size_t partitions_{0};
  uint32_t num_subspaces_{16};
  uint32_t max_iterations_{2};
  float convergence_tolerance_{0.000025f};
  float reassign_ratio_{0.075f};
  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};
  tiledb_datatype_t partitioning_index_datatype_{TILEDB_ANY};
  std::unique_ptr<index_base> index_;
  DistanceMetric distance_metric_{DistanceMetric::SUM_OF_SQUARES};
};

// clang-format off
const IndexIVFPQ::create_table_type IndexIVFPQ::create_dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<int8_t,  uint32_t, uint32_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<uint8_t, uint32_t, uint32_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<float,   uint32_t, uint32_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<int8_t,  uint32_t, uint64_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<uint8_t, uint32_t, uint64_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<float,   uint32_t, uint64_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<int8_t,  uint64_t, uint32_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<uint8_t, uint64_t, uint32_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<float,   uint64_t, uint32_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<int8_t,  uint64_t, uint64_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<uint8_t, uint64_t, uint64_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string &group_uri, uint64_t dimensions, uint32_t num_subspaces, uint32_t max_iterations, float convergence_tolerance, float reassign_ratio, std::optional<TemporalPolicy> temporal_policy, DistanceMetric distance_metric, const std::string &storage_version) { return ivf_pq_index<float,   uint64_t, uint64_t>::create(ctx, group_uri, dimensions, num_subspaces, max_iterations, convergence_tolerance, reassign_ratio, temporal_policy, distance_metric, storage_version); }},
};

const IndexIVFPQ::uri_table_type IndexIVFPQ::uri_dispatch_table = {
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint32_t, uint32_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint32_t, uint32_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint32_t, uint32_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint32_t, uint64_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint32_t, uint64_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT32, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint32_t, uint64_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint64_t, uint32_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint64_t, uint32_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT32}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint64_t, uint32_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_INT8,    TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<int8_t,  uint64_t, uint64_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_UINT8,   TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<uint8_t, uint64_t, uint64_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
  {{TILEDB_FLOAT32, TILEDB_UINT64, TILEDB_UINT64}, [](const tiledb::Context& ctx, const std::string& uri, IndexLoadStrategy index_load_strategy, size_t upper_bound, std::optional<TemporalPolicy> temporal_policy) { return std::make_unique<index_impl<ivf_pq_index<float,   uint64_t, uint64_t>>>(ctx, uri, index_load_strategy, upper_bound, temporal_policy); }},
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
