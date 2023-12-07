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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tiledb/group_experimental.h>
#include <tiledb/tiledb>

#include "index/index_defs.h"
#include "index/ivf_flat_metadata.h"
#include "tdb_defs.h"

class ivf_flat_index_metadata;

// template <class Index>
class ivf_flat_index_group {
  // Index* index_ {nullptr};

  std::reference_wrapper<const tiledb::Context> cached_ctx_;

  std::filesystem::path group_uri_;
  std::string version_;
  std::optional<tiledb::Group> write_group_;

  std::string centroids_array_name_;
  std::string parts_array_name_;
  std::string ids_array_name_;
  std::string indices_array_name_;

  std::string partial_write_array_name_;

  std::string input_vectors_array_name_;
  std::string external_ids_array_name_;
  std::string updates_array_name_;

  std::filesystem::path centroids_uri_;
  std::filesystem::path parts_uri_;
  std::filesystem::path ids_uri_;
  std::filesystem::path indices_uri_;

  std::filesystem::path partial_write_array_dir_;

  std::filesystem::path input_vectors_uri_;
  std::filesystem::path external_ids_uri_;
  std::filesystem::path updates_array_uri_;

  // std::unique_ptr<ivf_flat_index_metadata> metadata_;
  ivf_flat_index_metadata metadata_;

 public:
  ivf_flat_index_group() = delete;

  /**
   * Constructor
   *
   * If opening for read, if version is not set, the group will be opened
   * with whichever version is found in the metadata.  If version is set,
   * and does not match the version in the metadata, an exception will be
   * thrown.
   *
   * If opening for write, if version is not set, the group will be opened
   * with the current version.  If version is set, it will be opened with
   * the specified version.
   *
   *
   * @param ctx
   * @param uri
   * @param version
   */
  template <class Index = void>
  ivf_flat_index_group(
      const tiledb::Context& ctx,
      const std::string& uri,
      tiledb_query_type_t rw = TILEDB_READ,
      const std::string& version = std::string{""},
      const tiledb::Config& cfg = tiledb::Config{})
      : cached_ctx_(ctx)
      , group_uri_(uri), version_(version) {
    switch (rw) {
      case TILEDB_READ:
        open_for_read(cfg);
        break;
      case TILEDB_WRITE:
        open_for_write<Index>(cfg);
        break;
      default:
        throw std::runtime_error("Invalid query type.");
    }
  }

  using uri_init_type = std::tuple<std::string, std::filesystem::path*>;
  std::map<std::string, std::filesystem::path*> uri_inits;

  void open_for_read(const tiledb::Config& cfg) {


    auto read_group = tiledb::Group(cached_ctx_, group_uri_, TILEDB_READ, cfg);
    metadata_.load_metadata(read_group);
    if (!empty(version_) && metadata_.storage_version_ != version_) {
      throw std::runtime_error(
          "Version mismatch.  Requested " + version_ + " but found " +
          metadata_.storage_version_);
    } else if (empty(version_)) {
      version_ = metadata_.storage_version_;
    }
    init_array_names(version_);
    init_uris();
    for (size_t i = 0; i < read_group.member_count(); ++i) {
      auto member = read_group.member(i);
      auto name = member.name();
      if (!name || name.value().empty()) {
        throw std::runtime_error("Name is empty.");
      }
      if (uri_inits.find(*name) != uri_inits.end()) {
        *uri_inits[*name] = std::filesystem::path(member.uri());
      }
    }
  }

  /**
   * Open group for writing.  For now we assume the group does not exist and
   * needs to be created.
   *
   * @param ctx
   * @param uri
   * @param version
   */
  template <class Index>
  void open_for_write(const tiledb::Config& cfg) {
    if constexpr (std::is_same_v<Index, void>) {
      throw std::runtime_error(
          "Index type must be specified for open_for_write.");
    } else {
      if (empty(version_)) {
        version_ = current_storage_version;
      }
      init_array_names(version_);
      // Do not init_uris -- uris are set when an array is added
      // init_uris();
      write_group_ = std::make_optional<tiledb::Group>(
          cached_ctx_, group_uri_, TILEDB_WRITE, cfg);

      ivf_flat_index_metadata metadata;
      metadata.storage_version_ = version_;
      metadata.dtype_ = type_to_string_v<typename Index::feature_type>;

      metadata.feature_datatype_ =
          type_to_tiledb_v<typename Index::feature_type>;
      metadata.id_datatype_ = type_to_tiledb_v<typename Index::id_type>;
      metadata.px_datatype_ = type_to_tiledb_v<typename Index::indices_type>;

      metadata.feature_type_str_ =
          type_to_string_v<typename Index::feature_type>;
      metadata.id_type_str_ = type_to_string_v<typename Index::id_type>;
      metadata.indices_type_str_ =
          type_to_string_v<typename Index::indices_type>;

      metadata_.store_metadata(*write_group_);
    }
  }

  std::string basename_to_uri(const std::string& basename) {
    return group_uri_ / basename;
  }

  void add_array(const std::string& basename) {
    if (!write_group_ || write_group_->is_open()) {
      throw std::runtime_error("Group " + std::string(group_uri_) + " is not open for writing.");
    }
    if (uri_inits.find(basename) == uri_inits.end()) {
      throw std::runtime_error("Invalid array name " + basename);
    }
    std::filesystem::path uri = basename_to_uri(basename);
    *uri_inits[basename] = uri;
    write_group_->add_member(uri, false, basename);
  }

  void init_array_names(const std::string& version) {
    centroids_array_name_ = storage_formats[version]["centroids_array_name"];
    parts_array_name_ = storage_formats[version]["parts_array_name"];
    ids_array_name_ = storage_formats[version]["ids_array_name"];
    indices_array_name_ = storage_formats[version]["index_array_name"];
    partial_write_array_name_ =
        storage_formats[version]["partial_write_array_name"];
    input_vectors_array_name_ =
        storage_formats[version]["input_vectors_array_name"];
    external_ids_array_name_ =
        storage_formats[version]["external_ids_array_name"];
    updates_array_name_ = storage_formats[version]["updates_array_name"];
  }

  void init_uris() {
    uri_inits = {
        {centroids_array_name_, &centroids_uri_},
        {parts_array_name_, &parts_uri_},
        {ids_array_name_, &ids_uri_},
        {indices_array_name_, &indices_uri_},
        {partial_write_array_name_, &partial_write_array_dir_},
        {input_vectors_array_name_, &input_vectors_uri_},
        {external_ids_array_name_, &external_ids_uri_},
        {updates_array_name_, &updates_array_uri_},
    };
  }



  bool exists(const tiledb::Context& ctx) const {
    return tiledb::Object::object(ctx, group_uri_).type() ==
           tiledb::Object::Type::Group;
  }

  auto remove(const tiledb::Context& ctx) {
    return tiledb::Object::remove(ctx, group_uri_);
  }


#if 0
  /**
   * Read metadata, set up uris, etc.
   */
  auto open(const tiledb::Config& cfg = tiledb::Config{}) {
    auto read_group = tiledb::Group(cached_ctx_, group_uri_, TILEDB_READ, cfg);

    metadata_.load_metadata(read_group);

    for (size_t i = 0; i < read_group.member_count(); ++i) {
      auto member = read_group.member(i);
      auto name = member.name();
      if (!name || name.value().empty()) {
        throw std::runtime_error("Name is empty.");
      }
      if (uri_inits.find(*name) != uri_inits.end()) {
        *uri_inits[*name] = std::filesystem::path(member.uri());
      }
    }
    // Should do this in the appropriate place (which is tbd)
#if 0
    // Should not have group members for which arrays do not exist
    for (auto& [name, uri] : uri_inits) {
      if (uri->empty()) {
        *(uri) = group_uri_ / name;
      }
    }
#endif
  }
#endif

  /**************************************************************************
   * Getters
   **************************************************************************/

  [[nodiscard]] auto centroids_uri() const {
    return centroids_uri_;
  }
  [[nodiscard]] auto parts_uri() const {
    return parts_uri_;
  }
  [[nodiscard]] auto ids_uri() const {
    return ids_uri_;
  }
  [[nodiscard]] auto indices_uri() const {
    return indices_uri_;
  }
  [[nodiscard]] auto partial_write_array_dir() const {
    return partial_write_array_dir_;
  }
  [[nodiscard]] auto input_vectors_uri() const {
    return input_vectors_uri_;
  }
  [[nodiscard]] auto external_ids_uri() const {
    return external_ids_uri_;
  }
  [[nodiscard]] auto updates_array_uri() const {
    return updates_array_uri_;
  }
  [[nodiscard]] auto centroids_name() const {
    return centroids_array_name_;
  }
  [[nodiscard]] auto parts_name() const {
    return parts_array_name_;
  }
  [[nodiscard]] auto ids_name() const {
    return ids_array_name_;
  }
  [[nodiscard]] auto indices_name() const {
    return indices_array_name_;
  }
  [[nodiscard]] auto partial_write_array_name() const {
    return partial_write_array_name_;
  }
  [[nodiscard]] auto input_vectors_name() const {
    return input_vectors_array_name_;
  }
  [[nodiscard]] auto external_ids_name() const {
    return external_ids_array_name_;
  }
  [[nodiscard]] auto updates_array_name() const {
    return updates_array_name_;
  }

  [[nodiscard]] const std::reference_wrapper<const tiledb::Context> cached_ctx()
      const {
    return cached_ctx_;
  }
  [[nodiscard]] std::reference_wrapper<const tiledb::Context> cached_ctx() {
    return cached_ctx_;
  }

  auto dump(const std::string& msg= "") {
    if (!empty(msg)) {
      std::cout << "-------------------------------------------------------\n";
      std::cout << "# " + msg << std::endl;
    }
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Stored in ivf_flat_group:" << std::endl;
    for (auto& [name, uri] : uri_inits) {
      std::cout << name << " " << uri->string() << std::endl;
    }
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Stored in " + group_uri_.string() + ":" << std::endl;
    auto cfg = tiledb::Config();
    auto read_group = tiledb::Group(cached_ctx_, group_uri_, TILEDB_READ, cfg);
    for (size_t i = 0; i < read_group.member_count(); ++i) {
      auto member = read_group.member(i);
      auto name = member.name();
      if (!name || empty(*name)) {
        throw std::runtime_error("Name is empty.");
      }
      std::cout << *name << " " << member.uri() << std::endl;
    }
    std::cout << "-------------------------------------------------------\n";
  }

  // Defer implementation for now
#if 0
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
#endif
};

#endif  // TILEDB_IVF_FLAT_GROUP_H