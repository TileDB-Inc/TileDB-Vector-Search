/**
 * @file   vamana_index.h
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

#include "detail/linalg/tdb_helpers.h"
#include "index/index_defs.h"
#include "index/index_group.h"
#include "index/vamana_metadata.h"

/**
 * The vamana index group stores:
 * - feature_vectors: the original set of vectors which we copy.
 *   - Example: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
 * - feature_vectors_ids: the IDs of the vectors in feature_vectors_array_name.
 *   - Example: [99, 100, 101]
 * - The graph (basically a CSR)
 *   - adjacency_ids: These are indexes into feature_vectors. Vertices go from 0
 * -> n-1 and each of those vertices indexes into feature_vectors. Then those
 * IDs correspond to the indexes. You can also think of it as holding the R
 * nearest neighbhors in the graph for each vertex.
 *      - Example: Here we have 100 and 101 connected, 99 and 101 connected, and
 * 99 and 10 connected. Logically you can think of it like: [[1 2], [0, 2], [0,
 * 1]], but it's stored as [1, 2, 0, 2, 0, 1]
 *   - adjacency_scores: This holds the neighbor scores (i.e. the distances)
 *      - Example: [[distance between 0 and 1, distance between 0 and 2], etc.]
 *   -  adjacency_row_index: Each entry in the row index indicates where the
 * neighbhors for that index start. 0 because that's where neighbors for vertex
 * 0 start, then 2 b/c that's where neighbors for vertex 1 start, then 4 b/c
 * that's whre neighbors for vertex 2 start, then 6 b/c that's the end.
 *      - Example: [0, 2, 4, 6]
 */
[[maybe_unused]] static StorageFormat vamana_storage_formats = {
    {"0.3",
     {
         {"adjacency_scores_array_name", "adjacency_scores"},
         {"adjacency_ids_array_name", "adjacency_ids"},
         {"adjacency_row_index_array_name", "adjacency_row_index"},

         // @todo for ivf_vamana we would also want medoids
         // {"medoids_array_name", "medoids"},
     }}};

template <class index_type>
class vamana_index_group : public base_index_group<index_type> {
  using Base = base_index_group<index_type>;

  using Base::array_key_to_array_name_;
  using Base::array_name_to_uri_;
  using Base::cached_ctx_;
  using Base::group_uri_;
  using Base::metadata_;
  using Base::valid_array_keys_;
  using Base::valid_array_names_;
  using Base::version_;

  // @todo Make this controllable
  static const int32_t default_domain{std::numeric_limits<int32_t>::max() - 1};
  static const int32_t default_tile_extent{100'000};
  static const int32_t tile_size_bytes{64 * 1024 * 1024};

 public:
  vamana_index_group(
      const tiledb::Context& ctx,
      const std::string& uri,
      tiledb_query_type_t rw = TILEDB_READ,
      TemporalPolicy temporal_policy = TemporalPolicy{TimeTravel, 0},
      const std::string& version = std::string{""},
      uint64_t dimensions = 0)
      : Base(ctx, uri, rw, temporal_policy, version, dimensions) {
    Base::load();
  }

  void append_valid_array_names_impl() {
    for (auto&& [array_key, array_name] : vamana_storage_formats[version_]) {
      valid_array_keys_.insert(array_key);
      valid_array_names_.insert(array_name);
      array_key_to_array_name_[array_key] = array_name;
      array_name_to_uri_[array_name] =
          array_name_to_uri(group_uri_, array_name);
    }
  }

  void clear_history_impl(uint64_t timestamp) {
    tiledb::Array::delete_fragments(
        cached_ctx_, this->feature_vectors_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(
        cached_ctx_, adjacency_scores_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(
        cached_ctx_, adjacency_ids_uri(), 0, timestamp);
    tiledb::Array::delete_fragments(
        cached_ctx_, adjacency_row_index_uri(), 0, timestamp);
  }

  /*
   * Graph size information
   */
  auto get_previous_num_edges() const {
    return metadata_.num_edges_history_.back();
  }
  auto get_num_edges() const {
    return metadata_.num_edges_history_[this->history_index_];
  }
  auto append_num_edges(size_t size) {
    metadata_.num_edges_history_.push_back(size);
  }
  auto get_all_num_edges() const {
    return metadata_.num_edges_history_;
  }
  auto set_num_edges(size_t size) {
    metadata_.num_edges_history_[this->history_index_] = size;
  }
  auto set_last_num_edges(size_t size) {
    metadata_.num_edges_history_.back() = size;
  }
  auto get_l_build() const {
    return metadata_.l_build_;
  }
  auto set_l_build(size_t size) {
    metadata_.l_build_ = size;
  }
  auto get_r_max_degree() const {
    return metadata_.r_max_degree_;
  }
  auto set_r_max_degree(size_t size) {
    metadata_.r_max_degree_ = size;
  }
  auto get_alpha_min() const {
    return metadata_.alpha_min_;
  }
  auto set_alpha_min(float size) {
    metadata_.alpha_min_ = size;
  }
  auto get_alpha_max() const {
    return metadata_.alpha_max_;
  }
  auto set_alpha_max(float size) {
    metadata_.alpha_max_ = size;
  }
  auto get_medoid() const {
    return metadata_.medoid_;
  }
  auto set_medoid(size_t size) {
    metadata_.medoid_ = size;
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
  [[nodiscard]] auto adjacency_scores_array_name() const {
    return this->array_key_to_array_name("adjacency_scores_array_name");
  }
  [[nodiscard]] auto adjacency_ids_array_name() const {
    return this->array_key_to_array_name("adjacency_ids_array_name");
  }
  [[nodiscard]] auto adjacency_row_index_array_name() const {
    return this->array_key_to_array_name("adjacency_row_index_array_name");
  }

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

    /**************************************************************************
     * Base group metadata setup
     * @todo Do this in base group
     * @todo Make this table-driven
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
     * Vamana group metadata setup
     * @todo Make this table-driven
     *************************************************************************/
    metadata_.adjacency_scores_datatype_ =
        type_to_tiledb_v<typename index_type::adjacency_scores_type>;
    metadata_.adjacency_row_index_datatype_ =
        type_to_tiledb_v<typename index_type::adjacency_row_index_type>;

    metadata_.adjacency_scores_type_str_ =
        type_to_string_v<typename index_type::adjacency_scores_type>;
    metadata_.adjacency_row_index_type_str_ =
        type_to_string_v<typename index_type::adjacency_row_index_type>;

    metadata_.ingestion_timestamps_ = {};
    metadata_.base_sizes_ = {};
    metadata_.num_edges_history_ = {};
    metadata_.temp_size_ = 0;
    metadata_.dimensions_ = this->get_dimensions();

    /**
     * Create the arrays: feature_vectors (matrix), feature_vectors_ids
     * (vector), adjacency_scores (vector), adjacency_ids (vector),
     * adjacency_row_index (vector).
     */
    create_empty_for_matrix<
        typename index_type::feature_type,
        stdx::layout_left>(
        cached_ctx_,
        this->feature_vectors_uri(),
        this->get_dimensions(),
        default_domain,
        this->get_dimensions(),
        default_tile_extent,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group,
        this->feature_vectors_uri(),
        this->feature_vectors_array_name());

    create_empty_for_vector<typename index_type::id_type>(
        cached_ctx_,
        this->ids_uri(),
        default_domain,
        tile_size,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, this->ids_uri(), this->ids_array_name());

    create_empty_for_vector<typename index_type::adjacency_scores_type>(
        cached_ctx_,
        adjacency_scores_uri(),
        default_domain,
        tile_size,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, adjacency_scores_uri(), adjacency_scores_array_name());

    create_empty_for_vector<typename index_type::id_type>(
        cached_ctx_,
        adjacency_ids_uri(),
        default_domain,
        tile_size,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group, adjacency_ids_uri(), adjacency_ids_array_name());

    create_empty_for_vector<typename index_type::adjacency_row_index_type>(
        cached_ctx_,
        adjacency_row_index_uri(),
        default_domain,
        tile_size,
        default_compression);
    tiledb_helpers::add_to_group(
        write_group,
        adjacency_row_index_uri(),
        adjacency_row_index_array_name());

    // Store the metadata if all of the arrays were created successfully
    metadata_.store_metadata(write_group);
  }
};

#endif  // TILEDB_VAMANA_GROUP_H
