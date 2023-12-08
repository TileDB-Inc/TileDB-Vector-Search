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


template <class Index>
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
  ivf_flat_index_metadata<ivf_flat_index_group> metadata_;

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
  ivf_flat_index_group(
      const tiledb::Context& ctx,
      const std::string& uri,
      tiledb_query_type_t rw = TILEDB_READ,
      const std::string& version = std::string{""},
      const tiledb::Config& cfg = tiledb::Config{})
      : cached_ctx_(ctx)
      , group_uri_(uri)
      , version_(version) {
    switch (rw) {
      case TILEDB_READ:
        open_for_read(cfg);
        break;
      case TILEDB_WRITE:
        open_for_write(cfg);
        break;
      default:
        throw std::runtime_error("Invalid query type.");
    }
  }

  using uri_init_type = std::tuple<std::string, std::filesystem::path*>;
  std::map<std::string, std::filesystem::path*> uri_inits;

  /**
   * Open group for reading.  If version_ is not set, the group will be opened
   * with whichever version is found in the metadata.  If version_ is set,
   * and does not match the version in the metadata, an exception will be
   * thrown.
   *
   * @param ctx
   */
  void open_for_read(const tiledb::Config& cfg) {
    tiledb::VFS vfs(cached_ctx_);
    if (!vfs.is_dir(group_uri_)) {
      throw std::runtime_error(
          "Group uri " + std::string(group_uri_) + " does not exist.");
    }
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
   * needs to be created.  If version_ has not been set, the group is opened
   * with the current default storage version.  Otherwise it is opened with the
   * specified version.
   *
   * @todo This is really a create.  We also need an open for update.
   *
   * @param ctx
   * @param uri
   * @param version
   */
  void open_for_write(const tiledb::Config& cfg) {
    tiledb::VFS vfs(cached_ctx_);
    if (vfs.is_dir(group_uri_)) {
      throw std::runtime_error(
          "Group uri " + std::string(group_uri_) + " exists.");
    }


    if (empty(version_)) {
      version_ = current_storage_version;
    }
    init_array_names(version_);
    // Do not init_uris -- uris are set when an array is added
    // init_uris();
    write_group_ = std::make_optional<tiledb::Group>(
        cached_ctx_, group_uri_, TILEDB_WRITE, cfg);

    ivf_flat_index_metadata<ivf_flat_index_group> metadata;
    metadata.storage_version_ = version_;

    metadata.dtype_ = type_to_string_v<typename Index::feature_type>;

    metadata.feature_datatype_ = type_to_tiledb_v<typename Index::feature_type>;
    metadata.id_datatype_ = type_to_tiledb_v<typename Index::id_type>;
    metadata.px_datatype_ = type_to_tiledb_v<typename Index::indices_type>;

    metadata.feature_type_str_ = type_to_string_v<typename Index::feature_type>;
    metadata.id_type_str_ = type_to_string_v<typename Index::id_type>;
    metadata.indices_type_str_ = type_to_string_v<typename Index::indices_type>;

    metadata_.store_metadata(*write_group_);
  }

  /**
   * @brief Convert the base name (array name) to a URI.
   *
   * @param basename
   */
  std::string basename_to_uri(const std::string& basename) {
    return group_uri_ / basename;
  }

  /**
   * @brief Add an array to the group.
   *
   * @param basename
   */
  void add_array(const std::string& basename, tiledb_datatype_t datatype = TILEDB_ANY) {
    if (!write_group_ || write_group_->is_open()) {
      throw std::runtime_error(
          "Group " + std::string(group_uri_) + " is not open for writing.");
    }
    if (uri_inits.find(basename) == uri_inits.end()) {
      throw std::runtime_error("Invalid array name " + basename);
    }
    std::filesystem::path uri = basename_to_uri(basename);
    *uri_inits[basename] = uri;

    // Make this relative so it can be moved? Or just make it absolute?
    // Our absolute formula seems to give the same URI as relative
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

  /**
   * Set up dictionary of array basenames to URIs. Have to this in a
   * function rather than a static intitializer because the names need
   * to be set first, which only happens after the group is opened
   * because we need the group version to get the correct names.
   */
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

  /**
   * @brief Test whether the group exists or not.
   * @param ctx
   */
  bool exists(const tiledb::Context& ctx) const {
    return tiledb::Object::object(ctx, group_uri_).type() ==
           tiledb::Object::Type::Group;
  }

  /**
   * @brief Delete the group.  Use with caution.
   *
   * @param ctx
   * @return void
   */
  auto remove(const tiledb::Context& ctx) {
    return tiledb::Object::remove(ctx, group_uri_);
  }

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

  /**
   * Dump the contents of the group to stdout.  Useful for "printf" debugging.
   * @todo Add test to compare the real group with the group structure.
   *
   * @param msg Optional message to print before the dump.
   */
  auto dump(const std::string& msg = "") {
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
};

#endif  // TILEDB_IVF_FLAT_GROUP_H