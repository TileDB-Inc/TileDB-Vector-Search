/**
 * @file   base_group.h
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

#ifndef TILEDB_INDEX_GROUP_H
#define TILEDB_INDEX_GROUP_H

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tiledb/group_experimental.h>
#include <tiledb/tiledb>

#include "detail/linalg/tdb_io.h"
#include "index/index_defs.h"
#include "index/index_metadata.h"
#include "mdspan/mdspan.hpp"
#include "tdb_defs.h"

/** Lookup an array name given an array key */
inline std::string array_key_to_array_name_from_map(
    const std::unordered_map<std::string, std::string>& map,
    const std::string& array_key) {
  if (map.find(array_key) == map.end()) {
    throw std::runtime_error("Invalid array key in map: " + array_key);
  }
  auto tmp = *map.find(array_key);
  return tmp.second;
};

/** Convert an array name to a uri. */
inline std::string array_name_to_uri(
    const std::string& group_uri, const std::string& array_name) {
  return (std::filesystem::path{group_uri} / std::filesystem::path{array_name})
      .string();
}

template <class Index>
class base_index_metadata;

template <class DerivedClass>
struct metadata_type_selector {
  using type = typename DerivedClass::index_metadata_type;
};

template <class IndexGroup>
class base_index_group {
  // using index_type = typename IndexGroup::index_type;
  using group_type = IndexGroup;

  // Can't do this ....
  // using index_group_metadata_type = typename
  // IndexGroup::index_group_metadata_type; Can do this
  using index_group_metadata_type =
      typename metadata_type_selector<IndexGroup>::type;

  friend IndexGroup;

 protected:
  std::reference_wrapper<const tiledb::Context> cached_ctx_;
  std::string group_uri_;
  size_t index_timestamp_{0};
  size_t group_timestamp_{0};
  size_t timetravel_index_{0};

  // std::reference_wrapper<const index_type> index_;
  std::string version_;
  tiledb_query_type_t opened_for_{TILEDB_READ};

  index_group_metadata_type metadata_;

  // Set of the names that are used in the group for this version
  std::unordered_set<std::string> valid_array_names_;
  std::unordered_set<std::string> valid_key_names_;

  // Set of names of arrays that are part of the group.  They either have
  // been read from the group or written to the group.
  std::unordered_set<std::string> active_array_names_;

  std::unordered_map<std::string, std::string> array_name_map_;

  /** Check validity of key name */
  constexpr bool is_valid_key_name(const std::string& key_name) const noexcept {
    return valid_key_names_.contains(key_name);
  }

  /** Check validity of array name */
  constexpr bool is_valid_array_name(
      const std::string& array_name) const noexcept {
    return valid_array_names_.contains(array_name);
  }

  /** Check whether the array has been put into this group */
  constexpr bool is_active_array_name(
      const std::string& array_name) const noexcept {
    return active_array_names_.contains(array_name);
  }

  /** Lookup an array name given an array key */
  constexpr auto array_key_to_array_name(const std::string& array_key) const {
    if (!is_valid_key_name(array_key)) {
      throw std::runtime_error("Invalid array key: " + array_key);
    }
    return array_key_to_array_name_from_map(array_name_map_, array_key);
  };

  /** Create the set of valid key names and array names */
  auto init_valid_array_names() {
    if (empty(version_)) {
      throw std::runtime_error("Version not set.");
    }
    for (auto&& [array_key, array_name] : storage_formats[version_]) {
      valid_key_names_.insert(array_key);
      valid_array_names_.insert(array_name);
      array_name_map_[array_key] = array_name;
    }
    static_cast<group_type*>(this)->append_valid_array_names_impl();
  }

  /**
   * @brief Add an array to the group.
   *
   * @param array_name
   *
   * @todo Could have type of array set here instead of by Index.  Might be
   * better to have it set in conjunction with array being set?
   */
  auto init_array_for_create(const std::string& array_name) {
    if (!is_valid_array_name(array_name)) {
      throw std::runtime_error(
          "Invalid array name in add_array: " + array_name);
    }
    active_array_names_.insert(array_name);

    std::filesystem::path uri = array_name_to_uri(array_name);

    return uri;
  }

  /**
   * Open group for reading.  If version_ is not set, the group will be opened
   * with whichever version is found in the metadata.  If version_ is set,
   * and does not match the version in the metadata, an exception will be
   * thrown.
   *
   * @param ctx
   */
  void init_for_open(const tiledb::Config& cfg) {
    tiledb::VFS vfs(cached_ctx_);
    if (!vfs.is_dir(group_uri_)) {
      throw std::runtime_error(
          "Group uri " + std::string(group_uri_) + " does not exist.");
    }
    auto read_group = tiledb::Group(cached_ctx_, group_uri_, TILEDB_READ, cfg);

    // Load the metadata and check the version.  We need to do this before
    // we can check the array names.

    // @todo FIXME This needs to be done in derived class
    metadata_.load_metadata(read_group);
    if (!empty(version_) && metadata_.storage_version_ != version_) {
      throw std::runtime_error(
          "Version mismatch.  Requested " + version_ + " but found " +
          metadata_.storage_version_);
    } else if (empty(version_)) {
      version_ = metadata_.storage_version_;
    }

    init_valid_array_names();

    // Get the active array names
    auto count = read_group.member_count();
    for (size_t i = 0; i < read_group.member_count(); ++i) {
      auto member = read_group.member(i);
      auto name = member.name();
      if (!name || name->empty()) {
        throw std::runtime_error("Name is empty.");
      }
      if (is_valid_array_name(*name)) {
        active_array_names_.insert(*name);
      } else {
        throw std::runtime_error(
            "Invalid array name in group: " + std::string(*name));
      }
    }
  }

  void open_for_read(const tiledb::Config& cfg) {
    init_for_open(cfg);

    if (size(metadata_.ingestion_timestamps_) == 0) {
      throw std::runtime_error("No ingestion timestamps found.");
    }
    if (index_timestamp_ == 0) {
      index_timestamp_ = metadata_.ingestion_timestamps_.back();
    }

    auto timestamp_bound = std::lower_bound(
        begin(metadata_.ingestion_timestamps_),
        end(metadata_.ingestion_timestamps_),
        index_timestamp_);
    if (timestamp_bound == end(metadata_.ingestion_timestamps_)) {
      throw std::runtime_error(
          "Requested read timestamp " + std::to_string(index_timestamp_) +
          " is beyond " +
          std::to_string(metadata_.ingestion_timestamps_.back()));
    }
    timetravel_index_ =
        std::distance(begin(metadata_.ingestion_timestamps_), timestamp_bound);

    // @todo Or index_timestamp_?
    group_timestamp_ = metadata_.ingestion_timestamps_[timetravel_index_];
  }

  /**
   * Open group for writing.  If the group does not exist, create a new one.
   *
   * If creating a new group, the version_ must be set.  If it is not set,
   * use the current default storage version.
   *
   * If opening a group for write, we open for read first to get the metadata
   * and record the timestamp at which we opened the group.  When we close the
   * group, we update the timestamp array, the sizes array, and the partitions
   * array.  We also update the metadata.
   *
   * @param ctx
   * @param uri
   * @param version
   */
  void open_for_write(const tiledb::Config& cfg) {
    tiledb::VFS vfs(cached_ctx_);

    if (vfs.is_dir(group_uri_)) {
      /** Load the current group metadata */
      init_for_open(cfg);
      if (index_timestamp_ < metadata_.ingestion_timestamps_.back()) {
        throw std::runtime_error(
            "Requested write timestamp " + std::to_string(index_timestamp_) +
            " is not greater than " +
            std::to_string(metadata_.ingestion_timestamps_.back()));
        group_timestamp_ = index_timestamp_;
      }
    } else {
      /** Create a new group */
      create_default(cfg);
    }
  }

  /**
   * Create a new group with the default arrays and metadata.
   *
   * @param cfg
   *
   * @todo Process the "base group" metadata here.
   */
  void create_default(const tiledb::Config& cfg) {
    static_cast<group_type*>(this)->create_default_impl(cfg);
  }

  /** Convert an array name to a uri. */
  constexpr std::string array_name_to_uri(
      const std::string& array_name) const noexcept {
    return array_name_to_uri(group_uri_, array_name);
  }

  /** Convert an array key to a uri. */
  constexpr std::string array_key_to_uri(const std::string& array_key) const {
    return (std::filesystem::path{group_uri_} /
            std::filesystem::path{array_key_to_array_name(array_key)})
        .string();
  }

 public:
  /**************************************************************************
   * Constructors
   **************************************************************************/
  base_index_group() = delete;

  /**
   * If opening for read, if version is not set, the group will be opened
   * with whichever version is found in the metadata.  If version is set,
   * and does not match the version in the metadata, an exception will be
   * thrown.
   *
   * If opening for write, if version is not set, the group will be opened
   * with the current version.  If version is set, it will be opened with
   * the specified version.
   *
   * @param ctx
   * @param uri
   * @param version
   * @param index
   * @param rw
   * @param timestamp
   *
   * @todo Chained parameters here too?
   */
  base_index_group(
      const tiledb::Context& ctx,
      const std::string& uri,
      uint64_t dimension,
      tiledb_query_type_t rw = TILEDB_READ,
      size_t timestamp = 0,
      const std::string& version = std::string{""},
      const tiledb::Config& cfg = tiledb::Config{})
      : cached_ctx_(ctx)
      , group_uri_(uri)
      , index_timestamp_(timestamp)
      , version_(version)
      , opened_for_(rw) {
    switch (opened_for_) {
      case TILEDB_READ:
        open_for_read(cfg);
        break;
      case TILEDB_WRITE:
        set_dimension(dimension);
        open_for_write(cfg);
        break;
      case TILEDB_MODIFY_EXCLUSIVE:
        break;
      case TILEDB_DELETE:
        break;
      case TILEDB_UPDATE:
        break;
      default:
        throw std::runtime_error("Invalid query type.");
    }
  }

  /**
   * @brief Destructor.  If opened for write, update the metadata.
   *
   * @todo Don't use default Config
   */
  ~base_index_group() {
    if (opened_for_ == TILEDB_WRITE) {
      auto cfg = tiledb::Config();
      auto write_group =
          tiledb::Group(cached_ctx_, group_uri_, TILEDB_WRITE, cfg);
      metadata_.store_metadata(write_group);
    }
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
   * @brief Self-destruct the group.  Use with caution.
   *
   * @param ctx
   * @return void
   */
  auto remove(const tiledb::Context& ctx) {
    return tiledb::Object::remove(ctx, group_uri_);
  }

  /**************************************************************************
   * Getters for read and write timestamps and sizes
   **************************************************************************/

  /** Temporary until time traveling is implemented */
  auto get_previous_ingestion_timestamp() const {
    return metadata_.ingestion_timestamps_.back();
  }
  auto get_ingestion_timestamp() const {
    return metadata_.ingestion_timestamps_[timetravel_index_];
  }
  auto append_ingestion_timestamp(size_t timestamp) {
    metadata_.ingestion_timestamps_.push_back(timestamp);
  }
  auto get_all_ingestion_timestamps() {
    return metadata_.ingestion_timestamps_;
  }

  /*
   * Base size information
   */
  auto get_previous_base_size() const {
    return metadata_.base_sizes_.back();
  }
  auto get_base_size() const {
    return metadata_.base_sizes_[timetravel_index_];
  }
  auto append_base_size(size_t size) {
    metadata_.base_sizes_.push_back(size);
  }
  auto get_all_base_sizes() {
    return metadata_.base_sizes_;
  }

  auto get_all_active_array_names() {
    return active_array_names_;
  }

  auto get_all_active_uris() {
    std::vector<std::string> uris;
    for (auto&& array_name : active_array_names_) {
      uris.push_back(array_name_to_uri(array_name));
    }
    return uris;
  }

  auto get_temp_size() const {
    return metadata_.temp_size_;
  }
  auto set_temp_size(size_t size) {
    metadata_.temp_size_ = size;
  }

  auto get_dimension() const {
    return metadata_.dimension_;
  }
  auto set_dimension(size_t dim) {
    metadata_.dimension_ = dim;
  }

  /**************************************************************************
   * Getters for names and uris
   **************************************************************************/

  [[nodiscard]] auto partial_write_array_dir() const {
    return array_key_to_uri("partial_write_array_dir");
  }
  [[nodiscard]] auto input_vectors_uri() const {
    return array_key_to_uri("input_vectors_array_name");
  }
  [[nodiscard]] auto external_ids_uri() const {
    return array_key_to_uri("external_ids_array_name");
  }
  [[nodiscard]] auto updates_array_uri() const {
    return array_key_to_uri("updates_array_name");
  }
  [[nodiscard]] auto partial_write_array_name() const {
    return array_key_to_array_name("partial_write_array_dir");
  }
  [[nodiscard]] auto input_vectors_array_name() const {
    return array_key_to_array_name("input_vectors_array_name");
  }
  [[nodiscard]] auto external_ids_array_name() const {
    return array_key_to_array_name("external_ids_array_name");
  }
  [[nodiscard]] auto updates_array_name() const {
    return array_key_to_array_name("updates_array_name");
  }
  [[nodiscard]] const std::reference_wrapper<const tiledb::Context> cached_ctx()
      const {
    return cached_ctx_;
  }
  [[nodiscard]] std::reference_wrapper<const tiledb::Context> cached_ctx() {
    return cached_ctx_;
  }
  [[nodiscard]] auto group_timestamp() const {
    return group_timestamp_;
  }

  /**************************************************************************
   * Helpful functions for debugging, testing, etc
   **************************************************************************/

  auto set_ingestion_timestamp(size_t timestamp) {
    metadata_.ingestion_timestamps_[timetravel_index_] = timestamp;
  }
  auto set_base_size(size_t size) {
    metadata_.base_sizes_[timetravel_index_] = size;
  }

  auto set_last_ingestion_timestamp(size_t timestamp) {
    metadata_.ingestion_timestamps_.back() = timestamp;
  }
  auto set_last_base_size(size_t size) {
    metadata_.base_sizes_.back() = size;
  }

  bool compare_group(const base_index_group& rhs) const {
    if (group_uri_ != rhs.group_uri_) {
      return false;
    }
    if (group_uri_ != rhs.group_uri_) {
      return false;
    }
    if (size(valid_array_names_) != size(rhs.valid_array_names_)) {
      return false;
    }
    if (valid_array_names_ != rhs.valid_array_names_) {
      return false;
    }
    if (size(valid_key_names_) != size(rhs.valid_key_names_)) {
      return false;
    }
    if (valid_key_names_ != rhs.valid_key_names_) {
      return false;
    }
    if (!metadata_.compare_metadata(rhs.metadata_)) {
      return false;
    }

    return true;
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
    std::cout << "Stored in " + group_uri_ + ":" << std::endl;
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
    std::cout << "# Active arrays:" << std::endl;
    for (auto&& array_name : active_array_names_) {
      std::cout << array_name << std::endl;
    }
    std::cout << "-------------------------------------------------------\n";
    std::cout << "# Metadata:" << std::endl;
    std::cout << "-------------------------------------------------------\n";
    metadata_.dump();
    std::cout << "-------------------------------------------------------\n";
  }
};

#endif  // TILEDB_INDEX_GROUP_H
