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
  uint64_t temp_size_{0};   // @todo ???

  std::string base_sizes_str_{"[0]"};
  std::string dataset_type_{"vector_search"};
  std::string dtype_{""};
  std::string index_type_{"IVF_FLAT"};
  std::string ingestion_timestamps_str_{"[0]"};
  std::string partition_history_str_{"[0]"};
  std::string storage_version_{current_storage_version};
  std::string temp_size_str_{""};

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

  /**
   *
   * @param read_group
   * @return
   *
   * @todo Use initializer list
   */
  auto load_metadata (tiledb::Group& read_group) {
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;

    using metadata_check_type = std::tuple<std::string, std::string&>;
    std::vector<metadata_check_type> metadata_checks{
        {"dataset_type", dataset_type_},
        {"index_type", index_type_},
        {"storage_version", storage_version_},
        {"dtype", dtype_},
        {"base_sizes", base_sizes_str_},
        {"ingestion_timestamps", ingestion_timestamps_str_},
        {"partition_history", partition_history_str_},
    };

    auto check_metadata = [&read_group](const auto& check) {
      tiledb_datatype_t v_type;
      uint32_t v_num;
      const void* v;
      if (!read_group.has_metadata(std::get<0>(check), &v_type)) {
        throw std::runtime_error("Missing metadata: " + std::get<0>(check));
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
            std::get<0>(check) + " must be '" + std::get<1>(check) +
            "' not " + tmp);
      }
      std::get<1>(check) = tmp;
    };

    for (auto& check : metadata_checks) {
      check_metadata(check);
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
      throw std::runtime_error("temp_size must be a uint64_t or float64 not " + tiledb::impl::type_to_str(v_type));
    }

    base_sizes_ = json_to_vector<base_sizes_type>(base_sizes_str_);
    ingestion_timestamps_ = json_to_vector<ingestion_timestamps_type>(ingestion_timestamps_str_);
    partition_history_ = json_to_vector<partition_history_type>(partition_history_str_);

#if 0
    if (!read_group.has_metadata("dataset_type", &v_type)) {
       throw std::runtime_error("Missing metadata: dataset_type");
    }
    read_group.get_metadata("dataset_type", &v_type, &v_num, &v);
    if (v_type != TILEDB_STRING_ASCII && v_type != TILEDB_STRING_UTF8) {
      throw std::runtime_error("dataset_type must be a string not " + tiledb::impl::type_to_str(v_type));
    }
    dataset_type_ = std::string(static_cast<const char*>(v), v_num);
    if (dataset_type_ != "vector_search") {
      throw std::runtime_error("dataset_type must be 'vector_search' not " + dataset_type_);
    }

    read_group.get_metadata("index_type", &v_type, &v_num, &v);
    if (v_type != TILEDB_STRING_ASCII && v_type != TILEDB_STRING_UTF8) {
      throw std::runtime_error("index_type must be a string not " + tiledb::impl::type_to_str(v_type));
    }
    index_type_ = std::string(static_cast<const char*>(v), v_num);
    if (index_type_ != "IVF_FLAT") {
      throw std::runtime_error("index_type must be 'IVF_FLAT' not " + index_type_);
    }

    read_group.get_metadata("storage_version", &v_type, &v_num, &v);
    if (v_type != TILEDB_STRING_ASCII && v_type != TILEDB_STRING_UTF8) {
      throw std::runtime_error("storage_version must be a string not " + tiledb::impl::type_to_str(v_type));
    }
    storage_version_ = std::string(static_cast<const char*>(v), v_num);
    if (storage_version_ != "0.3") {
      throw std::runtime_error("storage_version must be '0.3' not " + storage_version_);
    }

    read_group.get_metadata("dtype", &v_type, &v_num, &v);
    dtype_ = std::string(static_cast<const char*>(v), v_num);

    read_group.get_metadata("base_sizes", &v_type, &v_num, &v);
    base_sizes_str_ = std::string(static_cast<const char*>(v), v_num);

    read_group.get_metadata("ingestion_timestamps", &v_type, &v_num, &v);
    ingestion_timestamps_str_ = std::string(static_cast<const char*>(v), v_num);

    read_group.get_metadata("partition_history", &v_type, &v_num, &v);
    partition_history_str_ = std::string(static_cast<const char*>(v), v_num);

    read_group.get_metadata("temp_size", &v_type, &v_num, &v);
    if (v_type == TILEDB_UINT64) {
      temp_size_ = *static_cast<const uint64_t*>(v);
    } else if (v_type == TILEDB_FLOAT64) {
      temp_size_ = static_cast<uint64_t>(*static_cast<const double*>(v));
    } else {
      throw std::runtime_error("temp_size must be a uint64_t or float64 not " + tiledb::impl::type_to_str(v_type));
    }

    base_sizes_ = json_to_vector<base_sizes_type>(base_sizes_str_);
    ingestion_timestamps_ = json_to_vector<ingestion_timestamps_type>(ingestion_timestamps_str_);
    partition_history_ = json_to_vector<partition_history_type>(partition_history_str_);
#endif
  }
};



#endif // TILEDB_IVF_FLAT_METADATA_H