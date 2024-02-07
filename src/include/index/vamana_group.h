/**
 * @file   vamana_group.h
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
 */

#ifndef TILEDB_VAMANA_GROUP_H
#define TILEDB_VAMANA_GROUP_H

#include "index/index_defs.h"
#include "index/index_group.h"
#include "index/vamana_metadata.h"

/**
 * The vamana index group needs to store
 *   * vectors
 *   * graph (basically CSR)
 *     * neighbor lists
 *     * neighbor scores (distances)
 *     * "row" index
 *   * centroids (for the case of partitioned vamana)
 */
[[maybe_unused]] static StorageFormat vamana_storage_formats = {
    {"0.3",
     {
         {"feature_vectors_array_name", "feature_vectors"},
         {"adjacency_scores_array_name", "adjacency_scores"},
         {"adjacency_ids_array_name", "adjacency_ids"},
         {"adjacency_row_index_array_name", "adjacency_row_index"},

         // @todo for ivf_vamana we would also want medoids
         // {"medoids_array_name", "medoids"},
     }}};

template <class Index>
class vamana_index_group;

template <class Index>
struct metadata_type_selector<vamana_index_group<Index>> {
  using type = vamana_index_metadata;
};

template <class Index>
class vamana_index_group : public base_index_group<vamana_index_group<Index>> {
  using Base = base_index_group<vamana_index_group>;
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

  // @todo Make this controllable
  static const int32_t default_domain{std::numeric_limits<int32_t>::max() - 1};
  static const int32_t default_tile_extent{100'000};
  static const int32_t tile_size_bytes{64 * 1024 * 1024};

 public:
  using index_group_metadata_type = vamana_index_metadata;

  vamana_index_group(
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
    for (auto&& [array_key, array_name] : vamana_storage_formats[version_]) {
      valid_key_names_.insert(array_key);
      valid_array_names_.insert(array_name);
      array_name_map_[array_key] = array_name;
    }
  }

  [[nodiscard]] auto feature_vectors_uri() const {
    return this->array_key_to_uri("feature_vectors_array_name");
  }
  [[nodiscard]] auto adjacency_scores_uri() const {
    return this->array_key_to_uri("adjacency_scores_array_name");
  }
  [[nodiscard]] auto adjacency_ids_uri() const {
    return this->array_key_to_uri("adjacency_ids_array_name");
  }
  [[nodiscard]] auto adjacency_row_index_uri() const {
    return this->array_key_to_uri("adjacency_row_index_array_name");
  }
  [[nodiscard]] auto feature_vectors_array_name() const {
    return this->array_key_to_array_name("feature_vectors_array_name");
  }
  [[nodiscard]] auto adjacency_scores_array_name() const {
    return this->array_key_to_array_name("adjacency_scores_array_name");
  }
  [[nodiscard]] auto adjacency_ids_array_name() const {
    return this->array_key_to_array_name("adjacency_ids_array_name");
  }
  [[nodiscard]] auto adjacency_row_index_array_name() const {
    return this->array_key_to_array_name("adjacency_row_index_array_name");
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


    /**************************************************************************
     * Base group metadata setup
     * @todo Do this in base group
     *************************************************************************/
    this->metadata_.storage_version_ = version_;

    this->metadata_.dtype_ =
        type_to_string_v<typename index_type::feature_type>;

    metadata_.feature_datatype_ =
        type_to_tiledb_v<typename index_type::feature_type>;
    metadata_.id_datatype_ = type_to_tiledb_v<typename index_type::id_type>;

    metadata_.feature_type_str_ =
        type_to_string_v<typename index_type::feature_type>;
    metadata_.id_type_str_ = type_to_string_v<typename index_type::id_type>;

    /**************************************************************************
     * IVF group metadata setup
     *************************************************************************/
    metadata_.adjacency_scores_datatype_ =
        type_to_tiledb_v<typename index_type::score_type>;
    metadata_.adjacency_row_index_datatype =
        type_to_tiledb_v<typename index_type::score_type>;
    metadata_.adjacency_scores_type_str_ =
        type_to_string_v<typename index_type::score_type>;
    metadata_.adjacency_row_index_type_str_ =
        type_to_string_v<typename index_type::score_type>;

    metadata_.adjacency_scores_type_str_ =
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

#endif  // TILEDB_VAMANA_GROUP_H
