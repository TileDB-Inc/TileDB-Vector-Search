/**
 * @file   vamana_metadata.h
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

#ifndef TILEDB_VAMANA_METADATA_H
#define TILEDB_VAMANA_METADATA_H

#include "index/index_metadata.h"

/**
 * Vamana index metadata
 *
 * A vamana index has the original data, indexed via a graph.  The graph is
 * essentially a CSR graph, requiring an array for neighbor ids (vector
 * locations), an array for the distances to the neighbors, and an array
 * indexing the begin and end of each neighbor adjacency list.
 *
 * Since the index is dynamic, the size of the array will change over time, so
 * the extent will be int max.
 *
 * Thus we need to store the datatype and size of each of those arrays as
 * metadata
 *   - feature_vectors_type -- this is also dtype and base_sizes in the base
 *   - adjacency_scores_type -- the size is "nnz"
 *   - adjacency_ids_type -- size is "nnz", type is same as base id_datatype
 *   - adjacency_row_index_type -- the size should be the base_size plus one
 *
 * For partitioned vamana (TBD) we will also need to store partition history
 * (as with the ivf index)
 *
 *
 */
class vamana_index_metadata
    : public base_index_metadata<vamana_index_metadata> {
  using Base = base_index_metadata<vamana_index_metadata>;
  friend Base;

  using Base::metadata_arithmetic_check_type;
  using Base::metadata_string_check_type;

  using num_edges_history_type = uint64_t;

  // public for now in interest of time
 public:
  std::string index_type_{"VAMANA"};

  /** Record number of partitions at each write at a given timestamp */
  std::vector<num_edges_history_type> num_edges_history_;
  std::string num_edges_history_str_{""};

  /*
   * The type of the feature vectors and ids is "inherited"
   */
  tiledb_datatype_t adjacency_scores_datatype_{TILEDB_ANY};
  tiledb_datatype_t adjacency_row_index_datatype_{TILEDB_ANY};

  std::string adjacency_scores_type_str_{""};
  std::string adjacency_row_index_type_str_{""};

  uint64_t l_build_{0};
  uint64_t r_max_degree_{0};
  float alpha_min_{1.0};
  float alpha_max_{1.2};
  uint64_t medoid_{0};

 protected:
  IndexKind index_kind_{IndexKind::Vamana};

  std::vector<metadata_string_check_type> metadata_string_checks_impl{
      // name, member_variable, required
      {"index_type", index_type_, true},
      {"adjacency_scores_type", adjacency_scores_type_str_, false},
      {"adjacency_row_index_type", adjacency_row_index_type_str_, false},
      {"num_edges_history", num_edges_history_str_, true},
  };

  std::vector<metadata_arithmetic_check_type> metadata_arithmetic_checks_impl{
      {"adjacency_scores_datatype",
       &adjacency_scores_datatype_,
       TILEDB_UINT32,
       false},
      {"adjacency_row_index_datatype",
       &adjacency_row_index_datatype_,
       TILEDB_UINT32,
       false},
      {"l_build", &l_build_, TILEDB_UINT64, false},
      {"r_max_degree", &r_max_degree_, TILEDB_UINT64, false},
      {"alpha_min", &alpha_min_, TILEDB_FLOAT32, false},
      {"alpha_max", &alpha_max_, TILEDB_FLOAT32, false},
      {"medoid", &medoid_, TILEDB_UINT64, false},
  };

  void clear_history_impl(uint64_t timestamp) {
    std::vector<num_edges_history_type> new_num_edges_history;
    for (int i = 0; i < ingestion_timestamps_.size(); i++) {
      auto ingestion_timestamp = ingestion_timestamps_[i];
      if (ingestion_timestamp > timestamp) {
        new_num_edges_history.push_back(num_edges_history_[i]);
      }
    }
    if (new_num_edges_history.empty()) {
      new_num_edges_history = {0};
    }

    num_edges_history_ = new_num_edges_history;
    num_edges_history_str_ = to_string(nlohmann::json(num_edges_history_));
  }

  auto json_to_vector_impl() {
    num_edges_history_ =
        json_to_vector<num_edges_history_type>(num_edges_history_str_);
  }

  auto vector_to_json_impl() {
    num_edges_history_str_ = to_string(nlohmann::json(num_edges_history_));
  }

  auto dump_json_impl() const {
    if (!empty(adjacency_scores_type_str_) &&
        adjacency_scores_datatype_ !=
            string_to_datatype(adjacency_scores_type_str_)) {
      throw std::runtime_error(
          "adjacency_scores_datatype metadata disagree, must be " +
          adjacency_scores_type_str_ + " not " +
          tiledb::impl::type_to_str(adjacency_scores_datatype_));
    }
    if (!empty(adjacency_row_index_type_str_) &&
        adjacency_row_index_datatype_ !=
            string_to_datatype(adjacency_row_index_type_str_)) {
      throw std::runtime_error(
          "adjacency_row_index_datatype metadata disagree, must be " +
          adjacency_row_index_type_str_ + " not " +
          tiledb::impl::type_to_str(adjacency_row_index_datatype_));
    }
  }
};

#endif  // TILEDB_VAMANA_METADATA_H
