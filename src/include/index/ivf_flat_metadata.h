/**
 * @file   ivf_flat_metadata.h
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

#ifndef TILEDB_IVF_FLAT_METADATA_H
#define TILEDB_IVF_FLAT_METADATA_H

#include <tiledb/group_experimental.h>
#include <tiledb/tiledb>

#include "index/index_defs.h"

#include "nlohmann/json.hpp"

class ivf_flat_index_metadata {
 private:
  using base_sizes_type = uint32_t;
  using ingestion_timestamps_type = uint32_t;
  using partition_history_type = uint32_t;

  // @todo Is this where we actually want to store these?
  std::vector<base_sizes_type> base_sizes_;
  std::vector<ingestion_timestamps_type> ingestion_timestamps_;
  std::vector<partition_history_type> partition_history_;
  uint64_t temp_size_{0};  // @todo ???

  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};
  tiledb_datatype_t px_datatype_{TILEDB_ANY};

  std::string base_sizes_str_{"[0]"};
  std::string dataset_type_{"vector_search"};
  std::string dtype_{""};
  std::string index_type_{"IVF_FLAT"};
  std::string ingestion_timestamps_str_{"[0]"};
  std::string partition_history_str_{"[0]"};
  std::string storage_version_{current_storage_version};

  std::string feature_type_str_{""};
  std::string id_type_str_{""};
  std::string indices_type_str_{""};

  /*
   * Group Metadata:
   *
   "base_sizes",            // (json) list
   "dataset_type",          // "vector_search"
   "dtype",                 // "float32", etc (Python dtype names)
   "index_type",            // "FLAT", "IVF_FLAT"
   "ingestion_timestamps",  // (json) list
   "partition_history",     // (json) list
   "storage_version",       // "0.3"
   "temp_size",
   *
   "feature_datatype",      // TILEDB_UINT32
   "id_datatype",           // TILEDB_UINT32
   "px_datatype",           // TILEDB_UINT32
   *
   * "feature_type",        // std::string
   * "id_type",             // std::string
   * indices_type",         // std::string
   */

 public:
  ivf_flat_index_metadata() = default;

  template <class T>
  auto json_to_vector(const std::string& json_str) {
    auto json = nlohmann::json::parse(json_str);
    std::vector<T> vec;
    for (auto& item : json) {
      vec.push_back(item.get<uint32_t>());
    }
    return vec;
  }

  using metadata_string_check_type =
      std::tuple<std::string, std::string&, bool>;
  std::vector<metadata_string_check_type> metadata_string_checks{
      {"dataset_type", dataset_type_, true},
      {"index_type", index_type_, true},
      {"storage_version", storage_version_, true},
      {"dtype", dtype_, false},
      {"base_sizes", base_sizes_str_, true},
      {"ingestion_timestamps", ingestion_timestamps_str_, true},
      {"partition_history", partition_history_str_, true},
      {"feature_type", feature_type_str_, false},
      {"id_type", id_type_str_, false},
      {"indices_type", indices_type_str_, false},
  };

  using metadata_arithmetic_check_type =
      std::tuple<std::string, void*, tiledb_datatype_t, bool>;
  std::vector<metadata_arithmetic_check_type> metadata_arithmetic_checks{
      {"temp_size", &temp_size_, TILEDB_UINT64, true},
      {"feature_datatype", &feature_datatype_, TILEDB_UINT32, false},
      {"id_datatype", &id_datatype_, TILEDB_UINT32, false},
      {"px_datatype", &px_datatype_, TILEDB_UINT32, false},
  };

  auto check_string_metadata(
      tiledb::Group& read_group, const metadata_string_check_type& check) {
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;
    auto&& [name, value, required] = check;  // copilot filled in "required"
    if (!read_group.has_metadata(std::get<0>(check), &v_type)) {
      if (required) {
        throw std::runtime_error("Missing metadata: " + std::get<0>(check));
      } else {
        return;
      }
    }
    read_group.get_metadata(std::get<0>(check), &v_type, &v_num, &v);
    if (v_type != TILEDB_STRING_ASCII && v_type != TILEDB_STRING_UTF8) {
      throw std::runtime_error(
          std::get<0>(check) + " must be a string not " +
          tiledb::impl::type_to_str(v_type));
    }
    std::string tmp = std::string(static_cast<const char*>(v), v_num);
    if (std::get<1>(check) != "" && tmp != std::get<1>(check)) {
      throw std::runtime_error(
          std::get<0>(check) + " must be '" + std::get<1>(check) + "' not " +
          tmp);
    }
    std::get<1>(check) = tmp;
  };

  auto check_arithmetic_metadata(
      tiledb::Group& read_group, const metadata_arithmetic_check_type& check) {
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;
    auto&& [name, value, type, required] =
        check;  // copilot filled in "required"
    if (!read_group.has_metadata(std::get<0>(check), &v_type)) {
      if (required) {
        throw std::runtime_error("Missing metadata: " + std::get<0>(check));
      } else {
        return;
      }
    }
    read_group.get_metadata(name, &v_type, &v_num, &v);

    // Handle temp_size as a special case for now
    if (name == "temp_size") {
      if (v_type == TILEDB_UINT64) {
        *static_cast<uint64_t*>(value) = *static_cast<const uint64_t*>(v);
      } else if (v_type == TILEDB_FLOAT64) {
        *static_cast<uint64_t*>(value) =
            static_cast<uint64_t>(*static_cast<const double*>(v));
      } else {
        throw std::runtime_error(
            "temp_size must be a uint64_t or float64 not " +
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

  /**
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
    for (auto& check : metadata_arithmetic_checks) {
      check_arithmetic_metadata(read_group, check);
    }

    if (!read_group.has_metadata("temp_size", &v_type)) {
      throw std::runtime_error("Missing metadata: temp_size");
    }
    read_group.get_metadata("temp_size", &v_type, &v_num, &v);
    if (v_type == TILEDB_UINT64) {
      temp_size_ = *static_cast<const uint64_t*>(v);
    } else if (v_type == TILEDB_FLOAT64) {
      temp_size_ = static_cast<uint64_t>(*static_cast<const double*>(v));
    } else {
      throw std::runtime_error(
          "temp_size must be a uint64_t or float64 not " +
          tiledb::impl::type_to_str(v_type));
    }

    base_sizes_ = json_to_vector<base_sizes_type>(base_sizes_str_);
    ingestion_timestamps_ =
        json_to_vector<ingestion_timestamps_type>(ingestion_timestamps_str_);
    partition_history_ =
        json_to_vector<partition_history_type>(partition_history_str_);
  }

  auto dump() {
    for (auto&& [name, value, required] : metadata_string_checks) {
      std::cout << name << ": " << value << std::endl;
    }
    for (auto&& [name, value, type, required] : metadata_arithmetic_checks) {
      switch (type) {
        case TILEDB_FLOAT64:
          std::cout << name << ": " << *static_cast<double*>(value)
                    << std::endl;
          break;
        case TILEDB_FLOAT32:
          std::cout << name << ": " << *static_cast<float*>(value) << std::endl;
          break;
        case TILEDB_UINT64:
          std::cout << name << ": " << *static_cast<uint64_t*>(value)
                    << std::endl;
          break;
        case TILEDB_UINT32:
          if (name == "feature_datatype" || name == "id_datatype" ||
              name == "px_datatype") {
            std::cout << name << ": "
                      << tiledb::impl::type_to_str(
                             (tiledb_datatype_t)*static_cast<uint32_t*>(value))
                      << std::endl;
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
};

#endif  // TILEDB_IVF_FLAT_METADATA_H