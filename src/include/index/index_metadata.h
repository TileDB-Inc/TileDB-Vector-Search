/**
 * @file   index_metadata.h
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
 *
 *
 * Group metadata identifiers:
 *
 *  "base_sizes",            // (json) list
 *  "dataset_type",          // "vector_search"
 *  "dtype",                 // "float32", etc (Python dtype names)
 *  "index_type",            // "FLAT", "IVF_FLAT", "VAMANA", "IVF_PQ"
 *  "ingestion_timestamps",  // (json) list
 *  "storage_version",       // "0.3"
 *  "temp_size",             // TILEDB_INT64 or TILEDB_FLOAT64
 *
 *  "feature_datatype",      // TILEDB_UINT32
 *  "id_datatype",           // TILEDB_UINT32
 *  "px_datatype",           // TILEDB_UINT32
 *
 *  "feature_type",        // std::string
 *  "id_type",             // std::string
 *  "indices_type",        // std::string
 */

#ifndef TILEDB_INDEX_METADATA_H
#define TILEDB_INDEX_METADATA_H

#include <tiledb/tiledb>

#include "index/index_defs.h"
#include "index/index_group.h"
#include "tdb_defs.h"

#include "nlohmann/json.hpp"

/**
 * @brief Metadata for an IVF_FLAT index.
 * @tparam Group
 *
 * @note Group is a template parameter out of laziness so that the Group class
 * can directly access the metadata without a plethora of getters and setters.
 * Putting ivf_flat_group directly here caused a circular dependency, perhaps
 * fixable by CRTP?
 *
 * @todo Clean up so that we don't need the template parameter.
 */
template <class IndexMetadata>
class base_index_metadata {
 protected:
  using base_sizes_type = uint64_t;
  using ingestion_timestamps_type = uint64_t;

  /**************************************************************************
   * Members set / updated by users of the group
   ******************************************************************************/

  // Make public for now in interest of expedience
 public:
  /** Record timestamps of writes to the group */
  std::vector<ingestion_timestamps_type> ingestion_timestamps_;

  /** Record size of vector array at each write at a given timestamp */
  std::vector<base_sizes_type> base_sizes_;

  /** Record size of temp data */
  int64_t temp_size_{0};

  uint32_t dimensions_{0};

  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};

  // A non-empty value indicates an expected value / default value
  std::string base_sizes_str_{""};
  std::string dataset_type_{"vector_search"};
  std::string dtype_{""};
  std::string ingestion_timestamps_str_{""};
  std::string storage_version_{current_storage_version};

  std::string feature_type_str_{""};
  std::string id_type_str_{""};

  /**************************************************************************
   * Initializer structs for metadata
   **************************************************************************/
  using metadata_string_check_type =
      std::tuple<std::string, std::string&, bool>;
  std::vector<metadata_string_check_type> metadata_string_checks{
      // name, member_variable, required
      {"dataset_type", dataset_type_, true},
      {"storage_version", storage_version_, true},
      {"dtype", dtype_, false},

      {"feature_type", feature_type_str_, false},
      {"id_type", id_type_str_, false},

      {"base_sizes", base_sizes_str_, true},
      {"ingestion_timestamps", ingestion_timestamps_str_, true},
  };

  using metadata_arithmetic_check_type =
      std::tuple<std::string, void*, tiledb_datatype_t, bool>;
  std::vector<metadata_arithmetic_check_type> metadata_arithmetic_checks{
      // name, member_variable, type, required
      {"temp_size", &temp_size_, TILEDB_INT64, true},
      {"dimensions", &dimensions_, TILEDB_UINT32, false},
      {"feature_datatype", &feature_datatype_, TILEDB_UINT32, false},
      {"id_datatype", &id_datatype_, TILEDB_UINT32, false},
  };

  template <class T>
  auto json_to_vector(const std::string& json_str) const {
    auto json = nlohmann::json::parse(json_str);
    std::vector<T> vec;
    for (auto& item : json) {
      vec.push_back(item.get<T>());
    }
    return vec;
  }

  /**
   * @brief Given a name, value, and required flag, read in the metadata
   * associated with the name and store into value.  An exception is thrown if
   * required is set and the metadata is not found, or if the metadata is found
   * but is not of the correct type. This function deals with metadata stored as
   * strings.
   * @param read_group
   * @param check
   * @return
   */
  auto check_string_metadata(
      tiledb::Group& read_group,
      const metadata_string_check_type& check) const {
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;

    auto&& [name, value, required] = check;
    if (!read_group.has_metadata(name, &v_type)) {
      if (required) {
        throw std::runtime_error("Missing metadata: " + name);
      } else {
        return;
      }
    }
    read_group.get_metadata(name, &v_type, &v_num, &v);
    if (v_type != TILEDB_STRING_ASCII && v_type != TILEDB_STRING_UTF8) {
      throw std::runtime_error(
          name + " must be a string not " + tiledb::impl::type_to_str(v_type));
    }
    std::string tmp = std::string(static_cast<const char*>(v), v_num);

    // Check for expected value
    if (!empty(value) && tmp != value) {
      throw std::runtime_error(name + " must be '" + value + "' not " + tmp);
    }
    value = tmp;
  };

  /**
   * @brief Given a name, value, and required flag, read in the metadata
   * associated with the name and store into value.  An exception is thrown if
   * required is set and the metadata is not found, or if the metadata is found
   * but is not of the correct type. This function deals with metadata stored as
   * arithmetic types.
   * @param read_group
   * @param check
   * @return
   */
  auto check_arithmetic_metadata(
      tiledb::Group& read_group,
      const metadata_arithmetic_check_type& check) const {
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;
    auto&& [name, value, type, required] = check;
    if (!read_group.has_metadata(name, &v_type)) {
      if (required) {
        throw std::runtime_error("Missing metadata: " + name);
      } else {
        return;
      }
    }
    read_group.get_metadata(name, &v_type, &v_num, &v);

    // Handle temp_size as a special case for now
    if (name == "temp_size") {
      if (v_type == TILEDB_INT64) {
        *static_cast<int64_t*>(value) = *static_cast<const int64_t*>(v);
      } else if (v_type == TILEDB_FLOAT64) {
        *static_cast<int64_t*>(value) =
            static_cast<int64_t>(*static_cast<const double*>(v));
      } else {
        throw std::runtime_error(
            "temp_size must be a int64_t or float64 not " +
            tiledb::impl::type_to_str(v_type));
      }
      return;
    }

    if (v_type != type) {
      throw std::runtime_error(
          name + " must be a " + tiledb::impl::type_to_str(type) + " not " +
          tiledb::impl::type_to_str(v_type));
    }
    switch (type) {
      case TILEDB_FLOAT64:
        *static_cast<double*>(value) = *static_cast<const double*>(v);
        break;
      case TILEDB_FLOAT32:
        *static_cast<float*>(value) = *static_cast<const float*>(v);
        break;
      case TILEDB_INT64:
        *static_cast<int64_t*>(value) = *static_cast<const int64_t*>(v);
        break;
      case TILEDB_UINT64:
        *static_cast<uint64_t*>(value) = *static_cast<const uint64_t*>(v);
        break;
      case TILEDB_UINT32:
        *static_cast<uint32_t*>(value) = *static_cast<const uint32_t*>(v);
        break;
      default:
        throw std::runtime_error("Unhandled type");
    }
  }

 public:
  base_index_metadata() = default;
  base_index_metadata(const base_index_metadata&) = default;
  base_index_metadata(base_index_metadata&&) = default;
  base_index_metadata& operator=(const base_index_metadata&) = default;
  base_index_metadata& operator=(base_index_metadata&&) = default;

  /**
   * Read all of the metadata fields from the given group.
   *
   * @param read_group
   * @return void
   *
   * @todo Dispatch on storage version
   */
  auto load_metadata(tiledb::Group& read_group) {
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;

    for (auto& check : metadata_string_checks) {
      check_string_metadata(read_group, check);
    }
    for (auto& check :
         static_cast<IndexMetadata*>(this)->metadata_string_checks_impl) {
      check_string_metadata(read_group, check);
    }
    for (auto& check : metadata_arithmetic_checks) {
      check_arithmetic_metadata(read_group, check);
    }
    for (auto& check :
         static_cast<IndexMetadata*>(this)->metadata_arithmetic_checks_impl) {
      check_arithmetic_metadata(read_group, check);
    }

    if (!read_group.has_metadata("temp_size", &v_type)) {
      throw std::runtime_error("Missing metadata: temp_size");
    }
    read_group.get_metadata("temp_size", &v_type, &v_num, &v);
    if (v_type == TILEDB_INT64) {
      temp_size_ = *static_cast<const int64_t*>(v);
    } else if (v_type == TILEDB_FLOAT64) {
      temp_size_ = static_cast<int64_t>(*static_cast<const double*>(v));
    } else {
      throw std::runtime_error(
          "temp_size must be a int64_t or float64 not " +
          tiledb::impl::type_to_str(v_type));
    }

    base_sizes_ = json_to_vector<base_sizes_type>(base_sizes_str_);
    ingestion_timestamps_ =
        json_to_vector<ingestion_timestamps_type>(ingestion_timestamps_str_);

    static_cast<IndexMetadata*>(this)->json_to_vector_impl();
  }

  /**
   * @brief Write all the metadata fields to the given group.
   *
   * @param write_group
   * @return
   */
  auto store_metadata(tiledb::Group& write_group) {
    base_sizes_str_ = to_string(nlohmann::json(base_sizes_));
    ingestion_timestamps_str_ =
        to_string(nlohmann::json(ingestion_timestamps_));

    static_cast<IndexMetadata*>(this)->vector_to_json_impl();

    for (auto&& [name, value, required] : metadata_string_checks) {
      write_group.put_metadata(
          name, TILEDB_STRING_UTF8, value.size(), value.c_str());
    }
    for (auto&& [name, value, required] :
         static_cast<IndexMetadata*>(this)->metadata_string_checks_impl) {
      write_group.put_metadata(
          name, TILEDB_STRING_UTF8, value.size(), value.c_str());
    }
    for (auto&& [name, value, type, required] : metadata_arithmetic_checks) {
      write_group.put_metadata(name, type, 1, static_cast<const void*>(value));
    }
    for (auto&& [name, value, type, required] :
         static_cast<IndexMetadata*>(this)->metadata_arithmetic_checks_impl) {
      write_group.put_metadata(name, type, 1, static_cast<const void*>(value));
    }
  }

  /**
   * @brief Clears all history that is <= timestamp.
   */
  void clear_history(uint64_t timestamp) {
    static_cast<IndexMetadata*>(this)->clear_history_impl(timestamp);

    std::vector<ingestion_timestamps_type> new_ingestion_timestamps;
    std::vector<base_sizes_type> new_base_sizes;
    for (int i = 0; i < ingestion_timestamps_.size(); i++) {
      auto ingestion_timestamp = ingestion_timestamps_[i];
      if (ingestion_timestamp > timestamp) {
        new_ingestion_timestamps.push_back(ingestion_timestamp);
        new_base_sizes.push_back(base_sizes_[i]);
      }
    }
    if (new_ingestion_timestamps.empty()) {
      new_ingestion_timestamps = {0};
      new_base_sizes = {0};
    }
    ingestion_timestamps_ = new_ingestion_timestamps;
    ingestion_timestamps_str_ =
        to_string(nlohmann::json(ingestion_timestamps_));

    base_sizes_ = new_base_sizes;
    base_sizes_str_ = to_string(nlohmann::json(base_sizes_));
  }

  /**************************************************************************
   * Helpful functions for debugging, testing, etc
   **************************************************************************/

  /**
   * @brief Compare two metadata objects for equality.
   * @param rhs The metadata object to compare *this against.
   * @return bool Whether the metadata objects are equal.
   */
  bool compare_arithmetic_metadata(
      const std::vector<metadata_arithmetic_check_type>& arithmetic_checks,
      const std::vector<metadata_arithmetic_check_type>& rhs_arithmetic_checks)
      const {
    for (size_t i = 0; i < size(arithmetic_checks); i++) {
      auto&& [name, value, type, required] = arithmetic_checks[i];
      auto&& [rhs_name, rhs_value, rhs_type, rhs_required] =
          rhs_arithmetic_checks[i];

      if (name != rhs_name) {
        return false;
      }
      if (type != rhs_type) {
        return false;
      }
      switch (type) {
        case TILEDB_FLOAT64:
          if (*static_cast<double*>(value) !=
              *static_cast<double*>(rhs_value)) {
            return false;
          }
        case TILEDB_FLOAT32:
          if (*static_cast<float*>(value) != *static_cast<float*>(rhs_value)) {
            return false;
          }
        case TILEDB_INT64:
          if (*static_cast<int64_t*>(value) !=
              *static_cast<int64_t*>(rhs_value)) {
            return false;
          }
        case TILEDB_UINT64:
          if (*static_cast<uint64_t*>(value) !=
              *static_cast<uint64_t*>(rhs_value)) {
            return false;
          }
        case TILEDB_UINT32:
          if (*static_cast<uint32_t*>(value) !=
              *static_cast<uint32_t*>(rhs_value)) {
            return false;
          }
          break;
        default:
          throw std::runtime_error("Unhandled type in compare_metadata");
      }
    }
    return true;
  }

  bool compare_string_metadata(
      const std::vector<metadata_string_check_type>& string_checks,
      const std::vector<metadata_string_check_type>& rhs_string_checks) const {
    for (size_t i = 0; i < size(string_checks); i++) {
      auto&& [name, value, required] = string_checks[i];
      auto&& [rhs_name, rhs_value, rhs_required] = rhs_string_checks[i];
      if (name != rhs_name) {
        return false;
      }
      if (value != rhs_value) {
        return false;
      }
    }
    return true;
  }

  bool compare_metadata(const base_index_metadata& rhs) const {
    // If the dataset type is different, don't bother comparing rest
    if (dataset_type_ != rhs.dataset_type_) {
      return false;
    }

    // If storage version is different, don't bother comparing rest
    if (storage_version_ != rhs.storage_version_) {
      return false;
    }
    if (base_sizes_str_ != rhs.base_sizes_str_) {
      return false;
    }
    if (ingestion_timestamps_str_ != rhs.ingestion_timestamps_str_) {
      return false;
    }
    if (compare_arithmetic_metadata(
            metadata_arithmetic_checks, rhs.metadata_arithmetic_checks) ==
        false) {
      return false;
    }
    if (compare_arithmetic_metadata(
            static_cast<const IndexMetadata*>(this)
                ->metadata_arithmetic_checks_impl,
            static_cast<const IndexMetadata&>(rhs)
                .metadata_arithmetic_checks_impl) == false) {
      return false;
    }
    if (compare_string_metadata(
            metadata_string_checks, rhs.metadata_string_checks) == false) {
      return false;
    }
    if (compare_string_metadata(
            static_cast<const IndexMetadata*>(this)
                ->metadata_string_checks_impl,
            static_cast<const IndexMetadata&>(rhs)
                .metadata_string_checks_impl) == false) {
      return false;
    }
    return true;
  }

  /**
   * @brief Dump metadata to stdout.  Useful for debugging.
   * @param write_group
   * @return void
   *
   * @todo Dispatch on storage version
   */
  auto dump_strings(
      const std::vector<metadata_string_check_type>& string_checks) const {
    for (auto&& [name, value, required] : string_checks) {
      std::cout << name << ": " << value << std::endl;
    }
  }

  auto dump_arithmetic(const std::vector<metadata_arithmetic_check_type>&
                           arithmetic_checks) const {
    for (auto&& [name, value, type, required] : arithmetic_checks) {
      switch (type) {
        case TILEDB_FLOAT64:
          std::cout << name << ": " << *static_cast<double*>(value)
                    << std::endl;
          break;
        case TILEDB_FLOAT32:
          std::cout << name << ": " << *static_cast<float*>(value) << std::endl;
          break;
        case TILEDB_INT64:
          std::cout << name << ": " << *static_cast<int64_t*>(value)
                    << std::endl;
          break;
        case TILEDB_UINT64:
          std::cout << name << ": " << *static_cast<uint64_t*>(value)
                    << std::endl;
          break;
        case TILEDB_UINT32:
          if (name == "feature_datatype" || name == "id_datatype" ||
              name == "px_datatype" || name == "adjacency_scores_datatype" ||
              name == "adjacency_row_index_datatype") {
            std::cout << name << ": " << *static_cast<uint32_t*>(value) << " ("
                      << tiledb::impl::type_to_str(
                             (tiledb_datatype_t) *
                             static_cast<uint32_t*>(value))
                      << ")" << std::endl;
          } else {
            std::cout << name << ": " << *static_cast<uint32_t*>(value)
                      << std::endl;
          }
          break;
        default:
          throw std::runtime_error(
              "Unhandled type: " + tiledb::impl::type_to_str(type));
      }
    }
  }

  auto dump() const {
    dump_strings(metadata_string_checks);
    dump_strings(
        static_cast<const IndexMetadata*>(this)->metadata_string_checks_impl);

    dump_arithmetic(metadata_arithmetic_checks);
    dump_arithmetic(static_cast<const IndexMetadata*>(this)
                        ->metadata_arithmetic_checks_impl);

    if (!empty(feature_type_str_) &&
        feature_datatype_ != string_to_datatype(feature_type_str_)) {
      throw std::runtime_error(
          "feature_datatype metadata disagree, must be " + feature_type_str_ +
          " not " + tiledb::impl::type_to_str(feature_datatype_));
    }
    if (!empty(id_type_str_) &&
        id_datatype_ != string_to_datatype(id_type_str_)) {
      throw std::runtime_error(
          "id_datatype metadata disagree, must be " + id_type_str_ + " not " +
          tiledb::impl::type_to_str(id_datatype_));
    }
    static_cast<const IndexMetadata*>(this)->dump_json_impl();
  }
};

#endif  // TILEDB_INDEX_METADATA_H
