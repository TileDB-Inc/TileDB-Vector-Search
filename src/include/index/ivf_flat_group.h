/**
 * @file   ivf_flat_group.h
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
 */

#ifndef TILEDB_IVF_FLAT_GROUP_H
#define TILEDB_IVF_FLAT_GROUP_H

#include <filesystem>
#include <string>

#include <tiledb/group_experimental.h>
#include <tiledb/tiledb>

#include "index/index_defs.h"
#include "index/ivf_flat_index.h"
#include "index/ivf_flat_metadata.h"
#include "tdb_defs.h"

template <class Index>
class ivf_flat_index_group {
  Index* index_ {nullptr};

  std::reference_wrapper<const tiledb::Context> cached_ctx_;

  std::filesystem::path centroids_array_name_;
  std::filesystem::path part_array_name_;
  std::filesystem::path ids_array_name_;
  std::filesystem::path indices_array_name_;

  std::filesystem::path partial_write_array_name_;

  std::filesystem::path input_vectors_array_name_;
  std::filesystem::path external_ids_array_name_;
  std::filesystem::path updates_array_name_;

  std::filesystem::path group_uri_;

  std::filesystem::path centroids_uri_;
  std::filesystem::path part_uri_;
  std::filesystem::path ids_uri_;
  std::filesystem::path indices_uri_;

  std::filesystem::path partial_write_array_dir_;

  std::filesystem::path input_vectors_uri_;
  std::filesystem::path external_ids_uri_;
  std::filesystem::path updates_array_uri_;

  ivf_flat_index_metadata metadata_;

 public:
  ivf_flat_index_group() = default;

  // @todo Loop through `storage_formats` as an initializer list
  ivf_flat_index_group(
      const tiledb::Context ctx,
      const std::string& uri,
      const std::string& version = current_storage_version)
      : cached_ctx_(ctx)
      , group_uri_(uri)
      , centroids_array_name_{storage_formats[version]["centroids_array_name"]}
      , part_array_name_{storage_formats[version]["part_array_name"]}
      , ids_array_name_{storage_formats[version]["ids_array_name"]}
      , indices_array_name_{storage_formats[version]["index_array_name"]}
      , partial_write_array_name_{storage_formats[version]
                                                 ["partial_write_array_name"]}
      , input_vectors_array_name_{storage_formats[version]["input_vectors_array_name"]}
      , external_ids_array_name_{storage_formats[version]["external_ids_array_name"]}
      , updates_array_name_{storage_formats[version]["updates_array_name"]}

      , centroids_uri_{group_uri_ / centroids_array_name_}
      , part_uri_{group_uri_ / part_array_name_}
      , ids_uri_{group_uri_ / ids_array_name_}
      , indices_uri_{group_uri_ / indices_array_name_}
      , partial_write_array_dir_{group_uri_ / partial_write_array_name_}
      , input_vectors_uri_{group_uri_ / input_vectors_array_name_}
      , external_ids_uri_{group_uri_ / external_ids_array_name_}
      , updates_array_uri_{group_uri_ / updates_array_name_} {
  }

  bool exists(const tiledb::Context& ctx) const {
    return tiledb::Object::object(ctx, group_uri_).type() ==
           tiledb::Object::Type::Group;
  }

  auto remove(const tiledb::Context& ctx) {
    return tiledb::Object::remove(ctx, group_uri_);
  }

  /**
   * Read metadata, etc
   */
  auto open(const tiledb::Config& cfg = tiledb::Config{}) {

    /*
     * First verify that this is the right kind of index
     */
    tiledb::Group read_group(cached_ctx_, group_uri_, TILEDB_READ, cfg);

    tiledb_datatype_t index_kind_type;
    if (!read_group.has_metadata("index_kind", &index_kind_type)) {
      throw std::runtime_error("Missing index_kind metadata");
    }
    if (index_kind_type != TILEDB_UINT32) {
      throw std::runtime_error("Wrong type for index_kind");
    }
    uint32_t index_kind_value;
    void* index_kind_addr;
    read_group.get_metadata(
        "index_kind",
        &index_kind_type,
        &index_kind_value,
        (const void**)&index_kind_addr);
    *reinterpret_cast<IndexKind*>(&index_kind_value) =
        *reinterpret_cast<IndexKind*>(index_kind_addr);
    if (index_kind_value != (uint32_t) Index::index_kind_) {
      throw std::runtime_error("Wrong value for index_kind");
    }
  }

#if 0

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
      } else if (datatype == TILEDB_UINT64) {
        *reinterpret_cast<uint64_t*>(value) =
            *reinterpret_cast<uint64_t*>(addr);
      } else if (datatype == TILEDB_FLOAT32) {
        *reinterpret_cast<float*>(value) = *reinterpret_cast<float*>(addr);
      } else {
        throw std::runtime_error("Unsupported datatype");
      }
    }

    centroids_ =
        std::move(tdbPreLoadMatrix<centroid_feature_type, stdx::layout_left>(
            *cached_ctx_, group_uri_ + "/centroids", 0, temporal_policy_));
  }

#endif


  /**
   * Create an empty group.
   *
   * @param ctx
   * @param dimension
   * @param cfg
   * @return
   *
   * @todo Take some of the tile sizes (et al) as parameters
   */
  auto create_empty(
      const tiledb::Context& ctx,
      size_t dimension,
      const tiledb::Config& cfg = tiledb::Config{}) {
    if (group_uri_.empty()) {
      throw std::runtime_error("Group URI has not been set.");
    }
    if (exists(ctx)) {
      throw std::runtime_error("Group already exists.");
    }

    using centroids_type = typename Index::feature_type;
    using feature_type = typename Index::centroids_type;
    using id_type = typename Index::id_type;
    using indices_type = typename Index::indices_type;

    static const std::string version{current_storage_version};
    static const int32_t default_domain{std::numeric_limits<int32_t>::max()};
    static const int32_t default_tile_extent{100'000};
    static const int32_t tile_size_bytes{64 * 1024 * 1024};
    static const int32_t tile_size{
        tile_size_bytes / sizeof(feature_type) / dimension};
    static const tiledb_filter_type_t default_compression{
        string_to_filter(storage_formats[version]["default_attr_filters"])};

    tiledb::Group::create(ctx, group_uri_);
    auto create_group = tiledb::Group(ctx, group_uri_, TILEDB_WRITE, cfg);

    // rows
    // cols
    // row_extents
    // col_extents
    create_array_for_matrix<centroids_type, stdx::layout_left>(
        ctx,
        centroids_uri_,
        "centroids",
        dimension,
        default_domain,
        dimension,
        default_tile_extent,
        default_compression);
    // create_group.add_member(centroids_array_name_, true, centroids_array_name_);
    create_group.add_member(centroids_uri_, false, centroids_array_name_);

    create_array_for_matrix<feature_type, stdx::layout_left>(
        ctx,
        part_uri_,
        "values",
        dimension,
        default_domain,
        dimension,
        default_tile_extent,
        default_compression);
    create_group.add_member(part_uri_, false, part_array_name_);

    create_array_for_vector<id_type>(
        ctx,
        ids_uri_,
        "values",
        default_domain,
        tile_size,
        default_compression);
    create_group.add_member(ids_uri_, false, ids_array_name_);

    create_array_for_vector<indices_type>(
        ctx,
        indices_uri_,
        "values",
        default_domain,
        default_tile_extent,
        default_compression);
    create_group.add_member(indices_uri_, false, indices_array_name_);

    // -- then ivf_flat_index.py create() constructs and returns Index
    //
    // -- constructs base
    //    uri
    //    group
    //    storage_version
    //    checks support_timetravel for index version
    //    updates_array_name
    //    updates_array_uri
    //    index_version
    //    ingestion_timestamps
    //    history_index
    //    base_sizes
    //    latest_ingestion_timestamp
    //    base_array_timestamp
    //    query_base_array
    //    update_array_timestamp
    //    does a bunch of stuff based on timestamp
    //
    // -- sets
    //    index_type = INDEX_TYPE (hardwired)
    //    db_uri (parts_array_name) -- from storage_formats.py
    //    centroids_uri (centroids_array_name)
    //    index_uri (index_array_name)
    //    ids_uri (ids_array_name)
    //    self.dtype ("dtype") -- from metadata or from db_uri
    //    base_size
    //    partition_history
    // -- loads
    //    centroids
    //    index
    //      parts and ids loaded upon query

    // -- currently ivf_flat.h open_index()
    //    reads metadata
    //    reads centroids
    //    does not read index -> part of tdb_partitioned_matrix


  }



};

#endif  // TILEDB_IVF_FLAT_GROUP_H