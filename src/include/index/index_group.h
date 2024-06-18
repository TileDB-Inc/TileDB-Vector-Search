/**
 * @file   index_group.h
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

template <class index_type>
class base_index_group {
  using group_type = index_type::group_type;
  using metadata_type = index_type::metadata_type;

 protected:
  tiledb::Context cached_ctx_;
  std::string group_uri_;
  size_t base_array_timestamp_{0};
  size_t history_index_{0};
  bool should_skip_query_{false};

  std::string version_;
  tiledb_query_type_t opened_for_{TILEDB_READ};
  std::optional<TemporalPolicy> temporal_policy_;

  metadata_type metadata_;

  // Set of the names that are used in the group for this version
  std::unordered_set<std::string> valid_array_names_;
  std::unordered_set<std::string> valid_array_keys_;

  std::unordered_map<std::string, std::string> array_key_to_array_name_;

  // Maps from the array name (not the key) to the URI of the array. Should be
  // used to get array URI's because the group_uri_ may be of the form
  // `tiledb://foo/edc4656a-3f45-43a1-8ee5-fa692a015c53` which cannot have the
  // array name added as a suffix.
  std::unordered_map<std::string, std::string> array_name_to_uri_;

  /** Lookup an array name given an array key */
  constexpr auto array_key_to_array_name(const std::string& array_key) const {
    if (!valid_array_keys_.contains(array_key)) {
      throw std::runtime_error("Invalid array key: " + array_key);
    }
    return array_key_to_array_name_from_map(
        array_key_to_array_name_, array_key);
  };

  /** Create the set of valid key names and array names */
  auto init_valid_array_names() {
    if (empty(version_)) {
      throw std::runtime_error("Version not set.");
    }
    for (auto&& [array_key, array_name] : storage_formats[version_]) {
      valid_array_keys_.insert(array_key);
      valid_array_names_.insert(array_name);
      array_key_to_array_name_[array_key] = array_name;
      array_name_to_uri_[array_name] =
          array_name_to_uri(group_uri_, array_name);
    }
    static_cast<group_type*>(this)->append_valid_array_names_impl();
  }

  /**
   * Open group for reading.  If version_ is not set, the group will be opened
   * with whichever version is found in the metadata.  If version_ is set,
   * and does not match the version in the metadata, an exception will be
   * thrown.
   *
   * @param ctx
   */
  void init_for_open(std::optional<TemporalPolicy> temporal_policy) {
    if (!exists()) {
      throw std::runtime_error(
          "Group uri " + std::string(group_uri_) + " does not exist.");
    }
    auto read_group = tiledb::Group(
        cached_ctx_, group_uri_, TILEDB_READ, cached_ctx_.config());

    // Load the metadata and check the version.  We need to do this before
    // we can check the array names.

    // @todo FIXME This needs to be done in derived class
    metadata_.load_metadata(read_group);
    if (!empty(version_) && metadata_.storage_version_ != version_) {
      throw std::runtime_error(
          "Version mismatch. Requested " + version_ + " but found " +
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
      auto uri = member.uri();
      if (uri.empty()) {
        throw std::runtime_error("Uri is empty.");
      }

      array_name_to_uri_[*name] = uri;
    }

    // This is based off of apis/python/src/tiledb/vector_search/index.py.
    if (temporal_policy.has_value()) {
      if (temporal_policy->timestamp_start() != 0) {
        // We have a (start, end) temporal_policy.
        if (temporal_policy->timestamp_start() >
            metadata_.ingestion_timestamps_[0]) {
          should_skip_query_ = true;
        } else {
          history_index_ = 0;
          base_array_timestamp_ =
              metadata_.ingestion_timestamps_[history_index_];
        }
      } else {
        // We have a (end) temporal_policy.
        history_index_ = 0;
        for (int i = 0; i < metadata_.ingestion_timestamps_.size(); i++) {
          if (metadata_.ingestion_timestamps_[i] <=
              temporal_policy->timestamp_end()) {
            history_index_ = i;
            base_array_timestamp_ =
                metadata_.ingestion_timestamps_[history_index_];
          }
        }
      }
    } else {
      // These are the defaults if no timestamp is set.
      history_index_ = metadata_.ingestion_timestamps_.size() - 1;
      base_array_timestamp_ = metadata_.ingestion_timestamps_[history_index_];
    }
  }

  void open_for_read(std::optional<TemporalPolicy> temporal_policy) {
    init_for_open(temporal_policy);

    if (size(metadata_.ingestion_timestamps_) == 0) {
      throw std::runtime_error("No ingestion timestamps found.");
    }
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
  void open_for_write(std::optional<TemporalPolicy> temporal_policy) {
    if (exists()) {
      /** Load the current group metadata */
      init_for_open(temporal_policy);
      if (!metadata_.ingestion_timestamps_.empty() &&
          base_array_timestamp_ < metadata_.ingestion_timestamps_.back()) {
        throw std::runtime_error(
            "Requested write timestamp " +
            std::to_string(base_array_timestamp_) + " is not greater than " +
            std::to_string(metadata_.ingestion_timestamps_.back()));
      }
    } else {
      /** Create a new group */
      create_default();
    }
  }

  /**
   * Create a new group with the default arrays and metadata.
   *
   * @param cfg
   *
   * @todo Process the "base group" metadata here.
   */
  void create_default() {
    if (get_dimensions() == 0) {
      throw std::runtime_error(
          "Dimensions must be set when creating a new group.");
    }
    static_cast<group_type*>(this)->create_default_impl();
  }

  /** Convert an array key to a uri. */
  constexpr std::string array_key_to_uri(const std::string& array_key) const {
    auto name = array_key_to_array_name(array_key);
    if (array_name_to_uri_.find(name) == array_name_to_uri_.end()) {
      throw std::runtime_error(
          "Invalid key when getting the URI: " + array_key +
          ". Name does not exist: " + name);
    }

    return array_name_to_uri_.at(name);
  }

  /**
   * @brief Test whether the group exists or not.
   * @param ctx
   */
  bool exists() const {
    return tiledb::Object::object(cached_ctx_, group_uri_).type() ==
           tiledb::Object::Type::Group;
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
   * @param ctx The TileDB context.
   * @param uri The group URI.
   * @param rw Whether to open for TILEDB_READ or TILEDB_WRITE.
   * @param temporal_policy The temporal policy to use.
   * @param version The storage format version.
   * @param dimensions The dimensions of the vectors in the index. Only needs to
   * be set for TILEDB_WRITE.
   *
   * @todo Chained parameters here too?
   */
  base_index_group(
      const tiledb::Context& ctx,
      const std::string& uri,
      tiledb_query_type_t rw = TILEDB_READ,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt,
      const std::string& version = std::string{""},
      uint64_t dimensions = 0)
      : cached_ctx_(ctx)
      , group_uri_(uri)
      , version_(version)
      , opened_for_(rw)
      , temporal_policy_(temporal_policy) {
    if (opened_for_ == TILEDB_WRITE) {
      set_dimensions(dimensions);
    }
  }

  /**
   * @brief Load the group - this should be called in the constructor of derived
   * classes of index_group. Note that we don't have the index_group constructor
   * call this because the derived class may need to do some setup before
   * open_for_write() is called.
   */
  void load() {
    switch (opened_for_) {
      case TILEDB_READ:
        open_for_read(temporal_policy_);
        break;
      case TILEDB_WRITE:
        open_for_write(temporal_policy_);
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
   * @brief Clears all history that is <= timestamp.
   */
  void clear_history(uint64_t timestamp) {
    if (opened_for_ != TILEDB_WRITE) {
      throw std::runtime_error("Cannot clear history in read mode.");
    }
    if (!exists()) {
      throw std::runtime_error(
          "Cannot clear history because group does not exist.");
    }

    metadata_.clear_history(timestamp);
    tiledb::Array::delete_fragments(cached_ctx_, ids_uri(), 0, timestamp);
    static_cast<group_type*>(this)->clear_history_impl(timestamp);
  }

  /**
   * @brief Destructor.  If opened for write, update the metadata.
   *
   * @todo Don't use default Config
   */
  ~base_index_group() {
    if (opened_for_ == TILEDB_WRITE && exists()) {
      auto write_group = tiledb::Group(
          cached_ctx_, group_uri_, TILEDB_WRITE, cached_ctx_.config());
      metadata_.store_metadata(write_group);
    }
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

  /** Temporary until time traveling is implemented */
  auto get_previous_ingestion_timestamp() const {
    return metadata_.ingestion_timestamps_.back();
  }
  auto get_ingestion_timestamp() const {
    return metadata_.ingestion_timestamps_[history_index_];
  }
  auto append_ingestion_timestamp(size_t timestamp) {
    metadata_.ingestion_timestamps_.push_back(timestamp);
  }
  auto get_all_ingestion_timestamps() const {
    return metadata_.ingestion_timestamps_;
  }

  auto get_previous_base_size() const {
    return metadata_.base_sizes_.back();
  }
  auto get_base_size() const {
    return metadata_.base_sizes_[history_index_];
  }
  auto append_base_size(size_t size) {
    metadata_.base_sizes_.push_back(size);
  }
  auto get_all_base_sizes() const {
    return metadata_.base_sizes_;
  }

  auto get_temp_size() const {
    return metadata_.temp_size_;
  }
  auto set_temp_size(size_t size) {
    metadata_.temp_size_ = size;
  }

  auto get_dimensions() const {
    return metadata_.dimensions_;
  }
  auto set_dimensions(size_t dim) {
    metadata_.dimensions_ = dim;
  }

  auto get_history_index() const {
    return history_index_;
  }

  auto should_skip_query() const {
    return should_skip_query_;
  }

  [[nodiscard]] auto ids_uri() const {
    return array_key_to_uri("ids_array_name");
  }
  [[nodiscard]] auto ids_array_name() const {
    return array_key_to_array_name("ids_array_name");
  }
  [[nodiscard]] auto feature_vectors_uri() const {
    return array_key_to_uri("parts_array_name");
  }
  [[nodiscard]] auto feature_vectors_array_name() const {
    return array_key_to_array_name("parts_array_name");
  }

  [[nodiscard]] const std::reference_wrapper<const tiledb::Context> cached_ctx()
      const {
    return cached_ctx_;
  }
  [[nodiscard]] std::reference_wrapper<const tiledb::Context> cached_ctx() {
    return cached_ctx_;
  }

  /**************************************************************************
   * Helpful functions for debugging, testing, etc
   **************************************************************************/

  auto set_ingestion_timestamp(size_t timestamp) {
    metadata_.ingestion_timestamps_[history_index_] = timestamp;
  }
  auto set_base_size(size_t size) {
    metadata_.base_sizes_[history_index_] = size;
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
    if (size(valid_array_keys_) != size(rhs.valid_array_keys_)) {
      return false;
    }
    if (valid_array_keys_ != rhs.valid_array_keys_) {
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
  auto dump(const std::string& msg = "") const {
    if (!empty(msg)) {
      std::cout << "-------------------------------------------------------\n";
      std::cout << "# " + msg << std::endl;
    }
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Stored in " + group_uri_ + ":" << std::endl;
    auto read_group = tiledb::Group(
        cached_ctx_, group_uri_, TILEDB_READ, cached_ctx_.config());
    for (size_t i = 0; i < read_group.member_count(); ++i) {
      auto member = read_group.member(i);
      auto name = member.name();
      if (!name || empty(*name)) {
        throw std::runtime_error("Name is empty.");
      }
      std::cout << *name << " " << member.uri() << std::endl;
    }
    std::cout << "version_: " << version_ << std::endl;
    std::cout << "history_index_: " << history_index_ << std::endl;
    std::cout << "base_array_timestamp_: " << base_array_timestamp_
              << std::endl;
    std::cout << "should_skip_query_: " << should_skip_query_ << std::endl;
    std::cout << "-------------------------------------------------------\n";
    std::cout << "# Metadata:" << std::endl;
    std::cout << "-------------------------------------------------------\n";
    metadata_.dump();
    std::cout << "-------------------------------------------------------\n";
  }
};

#endif  // TILEDB_INDEX_GROUP_H
