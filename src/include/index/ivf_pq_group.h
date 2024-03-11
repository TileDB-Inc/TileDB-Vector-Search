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

// flat_ivf_centroids, pq_ivf_centroids, partitioned_pq_vectors, partitioned_pq_index, 
// cluster_centroids, distance_tables,

[[maybe_unused]] static StorageFormat ivf_pq_storage_formats = {
    {"0.3",
     {
         // @todo Should these be kept consistent with ivf_flat?
         {"centroids_array_name", "partition_centroids"},  // These have to do
         {"index_array_name", "partition_indexes"},        // with the inverted
         {"ids_array_name", "shuffled_vector_ids"},        // index

         {"pq_parts_array_name", "shuffled_pq_vectors"},
         {"pq_codes_array_name", "pq_codes"},  // "centroids" in flat_pq
         {"distance_tables_array_name", "distance_tables"},
     }}};

template <class Index>
class ivf_pq_group;

template <class Index>
struct metadata_type_selector<ivf_pq_group<Index>> {
  using type = ivf_pq_metadata;
};

template <class Index>
class ivf_pq_group : public base_index_group<ivf_pq_group<Index>> {
  using Base = base_index_group<ivf_pq_group>;

  using Base::array_name_map_;
  using Base::cached_ctx_;
  using Base::group_uri_;
  using Base::metadata_;
  using Base::valid_array_names_;
  using Base::valid_key_names_;
  using Base::version_;

  using index_type = Index;

  // @todo These should be defined in some common place and passed as parameters
  static const int32_t default_domain{std::numeric_limits<int32_t>::max() - 1};
  static const int32_t default_tile_extent{100'000};
  static const int32_t tile_size_bytes{64 * 1024 * 1024};

 public:
  using index_group_metadata_type = ivf_pq_metadata;

  ivf_pq_group(
      const index_type& index,
      const tiledb::Context& ctx,
      const std::string& uri,
      tiledb_query_type_t rw = TILEDB_READ,
      size_t timestamp = 0,
      const std::string& version = std::string{""},
      const tiledb::Config& cfg = tiledb::Config{})
      : Base(ctx, uri, index.dimension(), rw, timestamp, version, cfg) {
  }

 public:
  void append_valid_array_names_impl() {
    for (auto&& [array_key, array_name] : ivf_pq_storage_formats[version_]) {
      valid_key_names_.insert(array_key);
      valid_array_names_.insert(array_name);
      array_name_map_[array_key] = array_name;
    }
  }

  /*****************************************************************************
   * Partitioning / repartitioning history information
   ****************************************************************************/
  auto get_previous_num_partitions() const {
    return metadata_.partition_history_.back();
  }
  auto get_num_partitions() const {
    return metadata_.partition_history_[this->timetravel_index_];
  }
  auto append_num_partitions(size_t size) {
    metadata_.partition_history_.push_back(size);
  }
  auto get_all_num_partitions() {
    return metadata_.partition_history_;
  }
  auto set_num_partitions(size_t size) {
    metadata_.partition_history_[this->timetravel_index_] = size;
  }
  auto set_last_num_partitions(size_t size) {
    metadata_.partition_history_.back() = size;
  }

  /*****************************************************************************
   * Inverted index information: centroids, index, pq_parts, ids
   ****************************************************************************/
  [[nodiscard]] auto centroids_uri() const {
    return this->array_key_to_uri("centroids_array_name");
  }
  [[nodiscard]] auto pq_parts_uri() const {
    return this->array_key_to_uri("pq_parts_array_name");
  }
  [[nodiscard]] auto ids_uri() const {
    return this->array_key_to_uri("ids_array_name");
  }
  [[nodiscard]] auto indices_uri() const {
    return this->array_key_to_uri("index_array_name");
  }
  [[nodiscard]] auto centroids_array_name() const {
    return this->array_key_to_array_name("centroids_array_name");
  }
  [[nodiscard]] auto pq_parts_array_name() const {
    return this->array_key_to_array_name("pq_parts_array_name");
  }
  [[nodiscard]] auto ids_array_name() const {
    return this->array_key_to_array_name("ids_array_name");
  }
  [[nodiscard]] auto indices_array_name() const {
    return this->array_key_to_array_name("index_array_name");
  }

  /*****************************************************************************
   * PQ encoded data information: pq_codes, distance_tables
   ****************************************************************************/
  [[nodiscard]] auto pq_codes_uri() const {
    return this->array_key_to_uri("pq_codes_array_name");
  }
  [[nodiscard]] auto distance_tables_uri() const {
    return this->array_key_to_uri("distance_tables_array_name");
  }
  [[nodiscard]] auto pq_codes_array_name() const {
    return this->array_key_to_array_name("pq_codes_array_name");
  }
  [[nodiscard]] auto distance_tables_array_name() const {
    return this->array_key_to_array_name("distance_tables_array_name");
  }

  /*****************************************************************************
   * Getters and setters for PQ related metadata: num_subspaces, sub_dimension,
   * bits_per_subspace, num_clusters
   ****************************************************************************/
  // num_subspaces
  auto get_num_subspaces() const {
    return metadata_.num_subspaces_;
  }
  auto set_num_subspaces(size_t num_subspaces) {
    metadata_.num_subspaces_ = num_subspaces;
  }

  // sub_dimension
  auto get_sub_dimension() const {
    return metadata_.sub_dimension_;
  }
  auto set_sub_dimension(size_t sub_dimension) {
    metadata_.sub_dimension_ = sub_dimension;
  }

  // bits_per_subspace
  auto get_bits_per_subspace() const {
    return metadata_.bits_per_subspace_;
  }
  auto set_bits_per_subspace(size_t bits_per_subspace) {
    metadata_.bits_per_subspace_ = bits_per_subspace;
  }

  // num_clusters
  auto get_num_clusters() const {
    return metadata_.num_clusters_;
  }
  auto set_num_clusters(size_t num_clusters) {
    metadata_.num_clusters_ = num_clusters;
  }

  /*****************************************************************************
   * Create a ready-to-use group with default arrays
   ****************************************************************************/
  void create_default_impl(const tiledb::Config& cfg) {
    if (empty(this->version_)) {
      this->version_ = current_storage_version;
    }
    this->init_valid_array_names();

    static const int32_t tile_size{
        (int32_t)(tile_size_bytes / sizeof(typename index_type::feature_type) /
                  this->get_dimension())};
    static const tiledb_filter_type_t default_compression{
        string_to_filter(storage_formats[version_]["default_attr_filters"])};

    tiledb::Group::create(cached_ctx_, group_uri_);
    auto write_group =
        tiledb::Group(cached_ctx_, group_uri_, TILEDB_WRITE, cfg);

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

    // Set the IVF related metadata
    metadata_.ingestion_timestamps_ = {0};
    metadata_.base_sizes_ = {0};
    metadata_.partition_history_ = {0};
    metadata_.temp_size_ = 0;
    metadata_.dimension_ = this->get_dimension();

    // Set the PQ related metadata: num_subspaces, sub_dimension,
    // bits_per_subspace, num_clusters
    metadata_.num_subspaces_ = this->get_num_subspaces();          // m
    metadata_.sub_dimension_ = this->get_sub_dimension();          // D* == D/m
    metadata_.bits_per_subspace_ = this->get_bits_per_subspace();  // 8
    metadata_.num_clusters_ = this->get_num_clusters();            // 2**nbits

    // Create the arrays: centroids, index, ids, pq_parts, pq_codes,
    // distance_tables
    create_empty_for_matrix<
        typename index_type::centroid_feature_type,
        stdx::layout_left>(
        cached_ctx_,
        centroids_uri(),
        this->get_dimension(),
        default_domain,
        this->get_dimension(),
        default_tile_extent,
        default_compression);
    write_group.add_member(
        centroids_array_name(), true, centroids_array_name());

    create_empty_for_vector<typename index_type::indices_type>(
        cached_ctx_,
        indices_uri(),
        default_domain,
        default_tile_extent,
        default_compression);
    // write_group.add_member(indices_uri(), true, indices_array_name());
    write_group.add_member(indices_array_name(), true, indices_array_name());

    create_empty_for_vector<typename index_type::id_type>(
        cached_ctx_, ids_uri(), default_domain, tile_size, default_compression);
    write_group.add_member(ids_array_name(), true, ids_array_name());

    create_empty_for_matrix<typename index_type::pq_type, stdx::layout_left>(
        cached_ctx_,
        pq_parts_uri(),
        this->get_num_subspaces(),
        default_domain,
        this->get_num_subspaces(),
        default_tile_extent,  // This should be much smaller for pq vectors
        default_compression);
    write_group.add_member(pq_parts_array_name(), true, pq_parts_array_name());

    create_empty_for_matrix<
        typename index_type::centroid_feature_type,
        stdx::layout_left>(
        cached_ctx_,
        pq_codes_uri(),
        this->get_num_subspaces(),
        this->get_num_clusters(),
        this->get_num_subspaces(),
        this->get_num_clusters(),
        default_compression);
    write_group.add_member(pq_codes_array_name(), true, pq_codes_array_name());

    create_empty_for_matrix<typename index_type::score_type, stdx::layout_left>(
        cached_ctx_,
        distance_tables_uri(),
        this->get_num_clusters(),
        this->get_num_clusters() * this->get_num_subspaces(),
        this->get_num_clusters(),
        this->get_num_clusters() * this->get_num_subspaces(),
        default_compression);
    write_group.add_member(
        distance_tables_array_name(), true, distance_tables_array_name());

    // Store the metadata if all of the arrays were created successfully
    metadata_.store_metadata(write_group);
  }
};

#endif  // TILEDB_PQ_GROUP_H
