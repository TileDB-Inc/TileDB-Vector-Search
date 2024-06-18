/**
 * @file   ivf_pq_group.h
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
 *
 * @todo This duplicates huge amounts of code from both ivf_flat and flat_pq,
 * but there are enough differences that immediate reuse isn't obvious.  This
 * should be refactored at some point.
 */

#ifndef TILEDB_IVF_PQ_GROUP_H
#define TILEDB_IVF_PQ_GROUP_H

#include "index/index_defs.h"
#include "index/index_group.h"
#include "index/ivf_pq_metadata.h"

// flat_ivf_centroids, pq_ivf_centroids,
// partitioned_pq_vectors, partitioned_pq_index, partitioned_pq_ids
// cluster_centroids, distance_tables,

[[maybe_unused]] static StorageFormat ivf_pq_storage_formats = {
    {"0.3",
     {
         // @todo Should these be kept consistent with ivf_flat?
         {"cluster_centroids_array_name", "pq_cluster_centroids"},
         {"flat_ivf_centroids_array_name", "uncompressed_centroids"},
         {"pq_ivf_centroids_array_name", "partition_centroids"},

         {"pq_ivf_indices_array_name", "partition_indexes"},
         {"pq_ivf_vectors_array_name", "shuffled_vectors"},

         {"distance_tables_array_name", "pq_symmetric_distance_tables"},
     }}};

template <class index_type>
class ivf_pq_group : public base_index_group<index_type> {
  using Base = base_index_group<index_type>;

  using Base::array_key_to_array_name_;
  using Base::array_name_to_uri_;
  using Base::cached_ctx_;
  using Base::group_uri_;
  using Base::metadata_;
  using Base::valid_array_keys_;
  using Base::valid_array_names_;
  using Base::version_;

  // @todo These should be defined in some common place and passed as parameters
  static const int32_t default_domain{std::numeric_limits<int32_t>::max() - 1};
  static const int32_t default_tile_extent{100'000};
  static const int32_t tile_size_bytes{64 * 1024 * 1024};

 public:
  ivf_pq_group(
      const tiledb::Context& ctx,
      const std::string& uri,
      tiledb_query_type_t rw = TILEDB_READ,
      TemporalPolicy temporal_policy = TemporalPolicy{TimeTravel, 0},
      const std::string& version = std::string{""},
      uint64_t dimensions = 0,
      size_t num_clusters = 0,
      size_t num_subspaces = 0)
      : Base(ctx, uri, rw, temporal_policy, version, dimensions) {
    if (rw == TILEDB_WRITE && !this->exists()) {
      if (num_clusters == 0) {
        throw std::invalid_argument(
            "num_clusters must be specified when creating a new group.");
      }
      if (num_subspaces == 0) {
        throw std::invalid_argument(
            "num_subspaces must be specified when creating a new group.");
      }
    }
    set_num_clusters(num_clusters);
    set_num_subspaces(num_subspaces);
    Base::load();
  }

  void append_valid_array_names_impl() {
    for (auto&& [array_key, array_name] : ivf_pq_storage_formats[version_]) {
      if (array_key == "distance_tables_array_name") {
        for (size_t i = 0; i < this->get_num_subspaces(); ++i) {
          valid_array_keys_.insert(array_key + "_" + std::to_string(i));
          valid_array_names_.insert(array_name + "_" + std::to_string(i));
          array_key_to_array_name_[array_key + "_" + std::to_string(i)] =
              array_name + "_" + std::to_string(i);
          array_name_to_uri_[array_name] =
              array_name_to_uri(group_uri_, array_name);
        }
      }
      valid_array_keys_.insert(array_key);
      valid_array_names_.insert(array_name);
      array_key_to_array_name_[array_key] = array_name;
      array_name_to_uri_[array_name] =
          array_name_to_uri(group_uri_, array_name);
    }
  }

  void clear_history_impl(uint64_t timestamp) {
    tiledb::Array::delete_fragments(
        cached_ctx_, cluster_centroids_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(
        cached_ctx_, flat_ivf_centroids_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(
        cached_ctx_, pq_ivf_centroids_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(
        cached_ctx_, pq_ivf_indices_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(cached_ctx_, this->ids_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(
        cached_ctx_, pq_ivf_vectors_uri(), 0, timestamp);
    for (size_t i = 0; i < this->get_num_subspaces(); ++i) {
      std::string this_table_uri =
          distance_tables_uri() + "_" + std::to_string(i);
      tiledb::Array::delete_fragments(
          cached_ctx_, this_table_uri, 0, timestamp);
    }
  }

  /*****************************************************************************
   * Partitioning / repartitioning history information
   ****************************************************************************/
  auto get_previous_num_partitions() const {
    return metadata_.partition_history_.back();
  }
  auto get_num_partitions() const {
    return metadata_.partition_history_[this->history_index_];
  }
  auto append_num_partitions(size_t size) {
    metadata_.partition_history_.push_back(size);
  }
  auto get_all_num_partitions() const {
    return metadata_.partition_history_;
  }
  auto set_num_partitions(size_t size) {
    metadata_.partition_history_[this->history_index_] = size;
  }
  auto set_last_num_partitions(size_t size) {
    metadata_.partition_history_.back() = size;
  }

  /*****************************************************************************
   * Inverted index information: centroids, index, pq_parts, ids
   ****************************************************************************/
  [[nodiscard]] auto cluster_centroids_uri() const {
    return this->array_key_to_uri("cluster_centroids_array_name");
  }
  [[nodiscard]] auto flat_ivf_centroids_uri() const {
    return this->array_key_to_uri("flat_ivf_centroids_array_name");
  }
  [[nodiscard]] auto pq_ivf_centroids_uri() const {
    return this->array_key_to_uri("pq_ivf_centroids_array_name");
  }
  [[nodiscard]] auto pq_ivf_indices_uri() const {
    return this->array_key_to_uri("pq_ivf_indices_array_name");
  }
  [[nodiscard]] auto pq_ivf_vectors_uri() const {
    return this->array_key_to_uri("pq_ivf_vectors_array_name");
  }
  [[nodiscard]] auto distance_tables_uri() const {
    return this->array_key_to_uri("distance_tables_array_name");
  }

  [[nodiscard]] auto cluster_centroids_array_name() const {
    return this->array_key_to_array_name("cluster_centroids_array_name");
  }
  [[nodiscard]] auto flat_ivf_centroids_array_name() const {
    return this->array_key_to_array_name("flat_ivf_centroids_array_name");
  }
  [[nodiscard]] auto pq_ivf_centroids_array_name() const {
    return this->array_key_to_array_name("pq_ivf_centroids_array_name");
  }
  [[nodiscard]] auto pq_ivf_indices_array_name() const {
    return this->array_key_to_array_name("pq_ivf_indices_array_name");
  }
  [[nodiscard]] auto pq_ivf_vectors_array_name() const {
    return this->array_key_to_array_name("pq_ivf_vectors_array_name");
  }
  [[nodiscard]] auto distance_tables_array_name() const {
    return this->array_key_to_array_name("distance_tables_array_name");
  }

  /*****************************************************************************
   * Getters and setters for PQ related metadata: num_subspaces, sub_dimension,
   * bits_per_subspace, num_clusters
   ****************************************************************************/
  auto get_num_subspaces() const {
    return metadata_.num_subspaces_;
  }
  auto set_num_subspaces(size_t num_subspaces) {
    metadata_.num_subspaces_ = num_subspaces;
  }

  auto get_sub_dimensions() const {
    return metadata_.sub_dimensions_;
  }
  auto set_sub_dimensions(size_t sub_dimensions) {
    metadata_.sub_dimensions_ = sub_dimensions;
  }

  auto get_bits_per_subspace() const {
    return metadata_.bits_per_subspace_;
  }
  auto set_bits_per_subspace(size_t bits_per_subspace) {
    metadata_.bits_per_subspace_ = bits_per_subspace;
  }

  auto get_num_clusters() const {
    return metadata_.num_clusters_;
  }
  auto set_num_clusters(size_t num_clusters) {
    metadata_.num_clusters_ = num_clusters;
  }

  /*****************************************************************************
   * Create a ready-to-use group with default arrays
   ****************************************************************************/
  void create_default_impl() {
    if (empty(this->version_)) {
      this->version_ = current_storage_version;
    }
    this->init_valid_array_names();

    static const int32_t tile_size{
        (int32_t)(tile_size_bytes / sizeof(typename index_type::feature_type) /
                  this->get_dimensions())};
    static const tiledb_filter_type_t default_compression{
        string_to_filter(storage_formats[version_]["default_attr_filters"])};

    tiledb::Group::create(cached_ctx_, group_uri_);
    auto write_group = tiledb::Group(
        cached_ctx_, group_uri_, TILEDB_WRITE, cached_ctx_.config());

    this->metadata_.storage_version_ = version_;

    // Updates to base class metadata -- we shouldn't duplicate as it is now
    this->metadata_.dtype_ =
        type_to_string_v<typename index_type::feature_type>;

    metadata_.feature_datatype_ =
        type_to_tiledb_v<typename index_type::feature_type>;
    metadata_.id_datatype_ = type_to_tiledb_v<typename index_type::id_type>;
    metadata_.feature_type_str_ =
        type_to_string_v<typename index_type::feature_type>;
    metadata_.id_type_str_ = type_to_string_v<typename index_type::id_type>;

    metadata_.px_datatype_ =
        type_to_tiledb_v<typename index_type::indices_type>;

    metadata_.indices_type_str_ =
        type_to_string_v<typename index_type::indices_type>;

    // Initialize IVF related metadata
    metadata_.ingestion_timestamps_ = {};
    metadata_.base_sizes_ = {};
    metadata_.partition_history_ = {};
    metadata_.temp_size_ = 0;
    metadata_.dimensions_ = this->get_dimensions();

    // Create the arrays: cluster_centroids,
    // flat_ivf_centroids, pq_ivf_centroids, ivf_index,
    // ivf_ids, pq_ivf_vectors, cluster_centroids, distance_tables
    create_empty_for_matrix<
        typename index_type::flat_vector_feature_type,
        stdx::layout_left>(
        cached_ctx_,
        cluster_centroids_uri(),
        this->get_dimensions(),
        this->get_num_clusters(),
        this->get_dimensions(),
        this->get_num_clusters(),
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, cluster_centroids_uri(), cluster_centroids_array_name());

    create_empty_for_matrix<
        typename index_type::flat_vector_feature_type,
        stdx::layout_left>(
        cached_ctx_,
        flat_ivf_centroids_uri(),
        this->get_dimensions(),
        default_domain,
        this->get_dimensions(),
        default_tile_extent,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, flat_ivf_centroids_uri(), flat_ivf_centroids_array_name());

    create_empty_for_matrix<
        typename index_type::pq_vector_feature_type,
        stdx::layout_left>(
        cached_ctx_,
        pq_ivf_centroids_uri(),
        this->get_num_subspaces(),
        default_domain,
        this->get_num_subspaces(),
        default_tile_extent,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, pq_ivf_centroids_uri(), pq_ivf_centroids_array_name());

    create_empty_for_vector<typename index_type::indices_type>(
        cached_ctx_,
        pq_ivf_indices_uri(),
        default_domain,
        default_tile_extent,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, pq_ivf_indices_uri(), pq_ivf_indices_array_name());

    create_empty_for_vector<typename index_type::id_type>(
        cached_ctx_,
        this->ids_uri(),
        default_domain,
        tile_size,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, this->ids_uri(), this->ids_array_name());

    create_empty_for_matrix<
        typename index_type::pq_code_type,
        stdx::layout_left>(
        cached_ctx_,
        pq_ivf_vectors_uri(),
        this->get_num_subspaces(),
        default_domain,
        this->get_num_subspaces(),
        default_tile_extent,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, pq_ivf_vectors_uri(), pq_ivf_vectors_array_name());

    for (size_t i = 0; i < this->get_num_subspaces(); ++i) {
      std::string this_table_uri =
          distance_tables_uri() + "_" + std::to_string(i);
      std::string this_table_array_name =
          distance_tables_array_name() + "_" + std::to_string(i);
      create_empty_for_matrix<
          typename index_type::score_type,
          stdx::layout_left>(
          cached_ctx_,
          this_table_uri,
          this->get_num_clusters(),
          this->get_num_clusters(),
          this->get_num_clusters(),
          this->get_num_clusters(),
          default_compression);
      tiledb_helpers::add_to_group(
          write_group, this_table_uri, this_table_array_name);
    }

    // Store the metadata if all of the arrays were created successfully
    metadata_.store_metadata(write_group);
  }
};

#endif  // TILEDB_PQ_GROUP_H
