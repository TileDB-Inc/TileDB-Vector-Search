/**
* @file   ivf_flat_index_metadata.h
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
*/

#ifndef TILEDB_IVF_FLAT_INDEX_METADATA_H
#define TILEDB_IVF_FLAT_INDEX_METADATA_H

#include "index/index_metadata.h"

class ivf_flat_index_metadata
    : public base_index_metadata<ivf_flat_index_metadata> {
  using Base = base_index_metadata<ivf_flat_index_metadata>;
  friend Base;

  using Base::metadata_arithmetic_check_type;
  using Base::metadata_string_check_type;

  using partition_history_type = uint64_t;

  // public for now in interest of time
 public:
  /** Record number of partitions at each write at a given timestamp */
  std::vector<partition_history_type> partition_history_;

  tiledb_datatype_t px_datatype_{TILEDB_ANY};
  std::string index_type_{"IVF_FLAT"};
  std::string partition_history_str_{""};
  std::string indices_type_str_{""};

 protected:
  IndexKind index_kind_{IndexKind::IVFFlat};

  std::vector<metadata_string_check_type> metadata_string_checks_impl{
      // name, member_variable, required
      {"index_type", index_type_, true},
      {"indices_type", indices_type_str_, false},
      {"partition_history", partition_history_str_, true},
  };

  std::vector<metadata_arithmetic_check_type> metadata_arithmetic_checks_impl{
      {"px_datatype", &px_datatype_, TILEDB_UINT32, false},
  };

  auto json_to_vector_impl() {
    partition_history_ =
        json_to_vector<partition_history_type>(partition_history_str_);
  }

  auto vector_to_json_impl() {
    partition_history_str_ = to_string(nlohmann::json(partition_history_));
  }

  auto dump_json_impl() {
    if (!empty(indices_type_str_)) {
      if (px_datatype_ == TILEDB_ANY) {
        px_datatype_ = string_to_datatype(indices_type_str_);
      } else if (px_datatype_ != string_to_datatype(indices_type_str_)) {
        throw std::runtime_error(
            "px_datatype metadata disagree, must be " + indices_type_str_ +
            " not " + tiledb::impl::type_to_str(px_datatype_));
      }
    }
  }
};

#endif  // TILEDB_IVF_FLAT_INDEX_METADATA_H
