/**
 * @file   ivf_flat_index_group.h
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
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * @note We use the terms "array key" and "array name".  The former is the
 string
 * that is used to look up the latter in a translation table that depends on the
 * version of the index.
 */

#ifndef TILEDB_IVF_FLAT_INDEX_GROUP_H
#define TILEDB_IVF_FLAT_INDEX_GROUP_H

#include "index/index_defs.h"
#include "index/index_group.h"
#include "index/ivf_flat_metadata.h"

[[maybe_unused]] static StorageFormat ivf_flat_storage_formats = {
    {"0.3",
     {
         {"centroids_array_name", "partition_centroids"},
         {"index_array_name", "partition_indexes"},
         {"ids_array_name", "shuffled_vector_ids"},
         {"parts_array_name", "shuffled_vectors"},
     }}};

template <class Index>
class ivf_flat_index_group;

template <class Index>
struct metadata_type_selector<ivf_flat_index_group<Index>> {
  using type = ivf_flat_index_metadata;
};

template <class Index>
class ivf_flat_index_group
    : public base_index_group<ivf_flat_index_group<Index>> {
  using Base = base_index_group<ivf_flat_index_group>;
  // using Base::Base;

  using Base::array_name_map_;
  using Base::cached_ctx_;
  using Base::group_uri_;
  using Base::metadata_;
  using Base::valid_array_names_;
  using Base::valid_key_names_;
  using Base::version_;

  using index_type = Index;
  // std::reference_wrapper<const index_type> index_;
  // index_type index_;

  static const int32_t default_domain{std::numeric_limits<int32_t>::max() - 1};
  static const int32_t default_tile_extent{100'000};
  static const int32_t tile_size_bytes{64 * 1024 * 1024};

 public:
  using index_group_metadata_type = ivf_flat_index_metadata;

  ivf_flat_index_group(
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
    for (auto&& [array_key, array_name] : ivf_flat_storage_formats[version_]) {
      valid_key_names_.insert(array_key);
      valid_array_names_.insert(array_name);
      array_name_map_[array_key] = array_name;
    }
  }

  /*
   * Partition information
   */
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

  [[nodiscard]] auto centroids_uri() const {
    return this->array_key_to_uri("centroids_array_name");
  }
  [[nodiscard]] auto parts_uri() const {
    return this->array_key_to_uri("parts_array_name");
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
  [[nodiscard]] auto parts_array_name() const {
    return this->array_key_to_array_name("parts_array_name");
  }
  [[nodiscard]] auto ids_array_name() const {
    return this->array_key_to_array_name("ids_array_name");
  }
  [[nodiscard]] auto indices_array_name() const {
    return this->array_key_to_array_name("index_array_name");
  }

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

    this->metadata_.dtype_ =
        type_to_string_v<typename index_type::feature_type>;

    metadata_.feature_datatype_ =
        type_to_tiledb_v<typename index_type::feature_type>;
    metadata_.id_datatype_ = type_to_tiledb_v<typename index_type::id_type>;
    metadata_.px_datatype_ =
        type_to_tiledb_v<typename index_type::indices_type>;

    metadata_.feature_type_str_ =
        type_to_string_v<typename index_type::feature_type>;
    metadata_.id_type_str_ = type_to_string_v<typename index_type::id_type>;
    metadata_.indices_type_str_ =
        type_to_string_v<typename index_type::indices_type>;

    metadata_.ingestion_timestamps_ = {0};
    metadata_.base_sizes_ = {0};
    metadata_.partition_history_ = {0};
    metadata_.temp_size_ = 0;
    metadata_.dimension_ = this->get_dimension();

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
    // write_group.add_member(centroids_uri(), true, centroids_array_name());
    write_group.add_member(
        centroids_array_name(), true, centroids_array_name());

    create_empty_for_matrix<
        typename index_type::feature_type,
        stdx::layout_left>(
        cached_ctx_,
        parts_uri(),
        this->get_dimension(),
        default_domain,
        this->get_dimension(),
        default_tile_extent,
        default_compression);
    // write_group.add_member(parts_uri(), true, parts_array_name());
    write_group.add_member(parts_array_name(), true, parts_array_name());

    create_empty_for_vector<typename index_type::id_type>(
        cached_ctx_, ids_uri(), default_domain, tile_size, default_compression);
    // write_group.add_member(ids_uri(), true, ids_array_name());
    write_group.add_member(ids_array_name(), true, ids_array_name());

    create_empty_for_vector<typename index_type::indices_type>(
        cached_ctx_,
        indices_uri(),
        default_domain,
        default_tile_extent,
        default_compression);
    // write_group.add_member(indices_uri(), true, indices_array_name());
    write_group.add_member(indices_array_name(), true, indices_array_name());

    // Store the metadata if all of the arrays were created successfully
    metadata_.store_metadata(write_group);
  }
};

#endif  // TILEDB_FLAT_INDEX_GROUP_H
