/**
 * @file   ivf_pq_metadata.h
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

#ifndef TILEDB_IVF_PQ_METADATA_H
#define TILEDB_IVF_PQ_METADATA_H

/******************************************************************************
 * Metadata
 *   From Base
 *   - dimensions
 *
 *   From IVF flat
 *   - index_type
 *   - indices_type
 *   - partition_history
 *   - px_datatype
 *
 *   From PQ pq
 *   - num_subspaces
 *   - sub_dimensions
 *   - bits_per_subspace
 *   - num_clusters
 *   - max_iterations
 *   - convergence_tolerance
 *   - reassign_ratio
 *
 *   Execution specific
 *   - num_threads
 *
 *
 ******************************************************************************/

#include "index/index_metadata.h"

class ivf_pq_metadata : public base_index_metadata<ivf_pq_metadata> {
  using Base = base_index_metadata<ivf_pq_metadata>;
  friend Base;

  using Base::metadata_arithmetic_check_type;
  using Base::metadata_string_check_type;

  using partition_history_type = uint64_t;

  // public for now in interest of time
 public:
  /** Record number of partitions at each write at a given timestamp */
  std::vector<partition_history_type> partition_history_;

  tiledb_datatype_t px_datatype_{TILEDB_ANY};
  std::string index_type_{"IVF_PQ"};
  std::string partition_history_str_{""};
  std::string indices_type_str_{""};

  uint64_t num_subspaces_{0};
  uint64_t sub_dimensions_{0};
  uint32_t bits_per_subspace_{0};
  uint32_t num_clusters_{0};
  uint64_t max_iterations_{0};
  float convergence_tolerance_{0.f};
  float reassign_ratio_{0.f};
  DistanceMetric distance_metric_{DistanceMetric::L2};

 protected:
  IndexKind index_kind_{IndexKind::IVFPQ};

  std::vector<metadata_string_check_type> metadata_string_checks_impl{
      // name, member_variable, required
      {"index_type", index_type_, true},
      {"indices_type", indices_type_str_, false},
      {"partition_history", partition_history_str_, true},
  };

  std::vector<metadata_arithmetic_check_type> metadata_arithmetic_checks_impl{
      {"px_datatype", &px_datatype_, TILEDB_UINT32, false},
      {"num_subspaces", &num_subspaces_, TILEDB_UINT64, true},
      {"sub_dimensions", &sub_dimensions_, TILEDB_UINT64, true},
      {"bits_per_subspace", &bits_per_subspace_, TILEDB_UINT32, true},
      {"num_clusters", &num_clusters_, TILEDB_UINT32, true},
      {"max_iterations", &max_iterations_, TILEDB_UINT64, true},
      {"convergence_tolerance", &convergence_tolerance_, TILEDB_FLOAT32, true},
      {"reassign_ratio", &reassign_ratio_, TILEDB_FLOAT32, true},
      {"distance_metric", &distance_metric_, TILEDB_UINT32, true},
  };

  void clear_history_impl(uint64_t timestamp) {
    std::vector<partition_history_type> new_partition_history;
    for (int i = 0; i < ingestion_timestamps_.size(); i++) {
      auto ingestion_timestamp = ingestion_timestamps_[i];
      if (ingestion_timestamp > timestamp) {
        new_partition_history.push_back(partition_history_[i]);
      }
    }
    if (new_partition_history.empty()) {
      new_partition_history = {0};
    }

    partition_history_ = new_partition_history;
    partition_history_str_ = to_string(nlohmann::json(partition_history_));
  }

  auto json_to_vector_impl() {
    partition_history_ =
        json_to_vector<partition_history_type>(partition_history_str_);
  }

  auto vector_to_json_impl() {
    partition_history_str_ = to_string(nlohmann::json(partition_history_));
  }

  auto dump_json_impl() const {
    if (!empty(indices_type_str_) &&
        px_datatype_ != string_to_datatype(indices_type_str_)) {
      throw std::runtime_error(
          "px_datatype metadata disagree, must be " + indices_type_str_ +
          " not " + tiledb::impl::type_to_str(px_datatype_));
    }
  }
};

#endif  // TILEDB_PQ_METADATA_H
