/**
 * @file   tdb_partitioned_matrix.h
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
 * Class for maintaining a partitioned matrix (ala a CSR matrix), with
 * TileDB arrays as the source of the data.
 *
 * The actual partitioned data is stored in a PartitionedMatirix base class
 * that contains the partitioned vectors, partitioned ids, and indexing
 * information. The data in the tdbPartitionedMatrix may represent:
 *   1) All of the partitions in the TileDB array
 *   2) A subset of the partitions in the TileDB array, but all the
 *      partitions that will be necessary for subsequent queries
 *   3) Out of core case.  A subset of the partitions in the TileDB array,
 *      which are a subset of the partitions that will be necessary for
 *      subsequent queries.  Additional partitions are loaded by calling
 *      the load() method.
 *
 * Further note that
 *   1) "Partitions" consist of a set of vectors and a set of corresponding ids
 *   2) The partitions are loaded in order, so that the first partition
 *      in the resident set is the first relevant partition in the partitioned
 *      TileDB array
 *   3) Only complete partitions are resident
 *   4) The complete ("master") indexing information for the entire partitioned
 *      matrix is stored in the tdbPartitionedMatrix base class.
 *   5) The indexing information for the resident partitions is stored in the
 *      PartitionedMatrix base class and is self-consistent with the partitions
 *      that are resident there.
 */

#ifndef TILEDB_PARTITIONED_MATRIX_H
#define TILEDB_PARTITIONED_MATRIX_H

#include <algorithm>
#include <cstddef>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include <version>

#include <tiledb/tiledb>
#include "mdspan/mdspan.hpp"

#include "detail/linalg/partitioned_matrix.h"
#include "tdb_defs.h"

#include "tdb_helpers.h"
#include "utils/timer.h"

namespace stdx {
using namespace Kokkos;
// using namespace Kokkos::Experimental;
}  // namespace stdx

/**
 * @brief Class for maintaining a partitioned matrix (ala a CSR matrix), with
 * TileDB arrays as the source of the data.
 * @tparam T
 * @tparam IdType
 * @tparam IndicesType
 * @tparam LayoutPolicy
 * @tparam I
 */
template <
    class T,
    class IdType,
    class IndicesType,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
class tdbPartitionedMatrix
    : public PartitionedMatrix<T, IdType, IndicesType, LayoutPolicy, I> {
  /****************************************************************************
   *
   * Duplication from tdbMatrix
   *
   * @todo Unify duplicated code
   *
   ****************************************************************************/
  using Base = PartitionedMatrix<T, IdType, IndicesType, LayoutPolicy, I>;
  // using Base::Base;

 public:
  using value_type = typename Base::value_type;  // should be same as T
  using typename Base::index_type;
  using typename Base::reference;
  using typename Base::size_type;

  using id_type = IdType;
  using indices_type = IndicesType;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  /*****************************************************************************
   * Information for reading from TileDB arrays
   ****************************************************************************/
  using row_domain_type = int32_t;
  using col_domain_type = int32_t;

  // For now, we assume this is always valid so we don't need to add constructor
  // arguments to limit it
  size_t num_array_rows_{0};

  // We don't actually use this
  // size_t num_array_cols_{0};

  std::reference_wrapper<const tiledb::Context> ctx_;
  // tiledb::Context ctx_;

  std::string partitioned_vectors_uri_;
  std::unique_ptr<tiledb::Array> partitioned_vectors_array_;
  tiledb::ArraySchema partitioned_vectors_schema_;

  std::string partitioned_ids_uri_;
  std::unique_ptr<tiledb::Array> partitioned_ids_array_;
  tiledb::ArraySchema ids_schema_;

  /*****************************************************************************
   * Partitioning information
   ****************************************************************************/

  // Vector of indices for all of the partitions.  We need to maintain this
  // so that we can specify the partitions (which use the master indices) from
  // to read from the partitioned TileDB array.
  std::vector<indices_type> master_indices_;  // @todo pointer and span?

  // Vector of the partition numbers of all the partitions that we want to load
  // This could be much smaller than the master indices, but is the same size
  // as the squashed indices.  We use indices_type as its type.
  std::vector<indices_type> relevant_parts_;  // @todo pointer and span?

  // Vector of indices for all of the partitions that will be loaded from the
  // TileDB array into contiguous memory.
  std::vector<indices_type> squashed_indices_;  // @todo pointer and span?

  // The total number of partitions in the partitioned array (resident plus
  // non-resident)
  size_t total_num_parts_{0};

  // The number of resident partitions
  size_t num_resident_parts_{0};

  // The offset of the first partitions in the resident vectors
  // Should be equal to first element of part_view_
  index_type resident_part_offset_{0};

  // The initial and final partition number of the resident partitions
  index_type first_resident_part_{0};
  index_type last_resident_part_{0};

  /*****************************************************************************
   * Column information
   ****************************************************************************/

  // The max number of columns that can fit in allocated memory
  size_t column_capacity{0};

  // The number of columns that are currently loaded into memory
  size_t num_resident_cols_{0};

  // The offset of the first column in the resident vector
  // Should be equal to first element of col_view_
  index_type resident_col_offset_{0};

  // The initial and final index numbers of the resident columns
  index_type first_resident_col_{0};
  index_type last_resident_col_{0};

  /*****************************************************************************
   * Accounting information
   ****************************************************************************/
  size_t max_part_size_{0};
  size_t max_resident_parts_{0};
  size_t num_loads_{0};

 public:
  tdbPartitionedMatrix(const tdbPartitionedMatrix&) = delete;
  tdbPartitionedMatrix(tdbPartitionedMatrix&&) = default;
  tdbPartitionedMatrix& operator=(tdbPartitionedMatrix&&) = default;
  tdbPartitionedMatrix& operator=(const tdbPartitionedMatrix&) = delete;

  /**
   * @brief Primary constructor.  Reads in vectors from a partitioned array,
   * as indicated by partition numbers in the relevant_partitions vector.
   * The data that is read in is also a set of partitioned vectors, but
   * with new demarcations between partitions.  The indicices_ vector in
   * the class is used for the partitions that have actually been read in.
   * This constructor does not load any data (that happense on invocations
   * of load()).
   *
   * @tparam P
   * @param ctx
   * @param partitioned_vectors_uri
   * @param indices_uri
   * @param ids_uri
   * @param relevant_parts
   * @param upper_bound
   */
  template <std::ranges::contiguous_range P>
  tdbPartitionedMatrix(
      const tiledb::Context& ctx,
      const std::string& partitioned_vectors_uri,
      const std::string& indices_uri,
      const std::string& ids_uri,
      const P& relevant_parts,
      size_t upper_bound,
      size_t timestamp = 0)
      : tdbPartitionedMatrix(
            ctx,
            partitioned_vectors_uri,
            read_vector<indices_type>(ctx, indices_uri, timestamp),
            ids_uri,
            relevant_parts,
            upper_bound,
            timestamp) {
  }

  template <std::ranges::contiguous_range P>
  tdbPartitionedMatrix(
      const tiledb::Context& ctx,
      const std::string& partitioned_vectors_uri,
      const std::string& indices_uri,
      size_t num_indices,
      const std::string& ids_uri,
      const P& relevant_parts,
      size_t upper_bound,
      size_t timestamp = 0)
      : tdbPartitionedMatrix(
            ctx,
            partitioned_vectors_uri,
            read_vector<indices_type>(
                ctx, indices_uri, 0, num_indices, timestamp),
            ids_uri,
            relevant_parts,
            upper_bound,
            timestamp) {
  }

  template <std::ranges::contiguous_range P>
  tdbPartitionedMatrix(
      const tiledb::Context& ctx,
      const std::string& partitioned_vectors_uri,
      const std::vector<indices_type>& indices,
      const std::string& ids_uri,
      const P& relevant_parts,
      size_t upper_bound,
      size_t timestamp = 0)
      : tdbPartitionedMatrix(
            ctx,
            partitioned_vectors_uri,
            indices,
            ids_uri,
            relevant_parts,
            upper_bound,
            (timestamp == 0) ?
                tiledb::TemporalPolicy() :
                tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp)) {
  }

  template <std::ranges::contiguous_range P>
  tdbPartitionedMatrix(
      const tiledb::Context& ctx,
      const std::string& partitioned_vectors_uri,
      const std::vector<indices_type>& indices,
      const std::string& ids_uri,
      const P& relevant_parts,
      size_t upper_bound,
      const tiledb::TemporalPolicy temporal_policy = {})
      : ctx_{ctx}
      , partitioned_vectors_uri_{partitioned_vectors_uri}
      , partitioned_vectors_array_(tiledb_helpers::open_array(
            tdb_func__,
            ctx_,
            partitioned_vectors_uri_,
            TILEDB_READ,
            temporal_policy))
      , partitioned_vectors_schema_{partitioned_vectors_array_->schema()}
      , partitioned_ids_uri_{ids_uri}
      , partitioned_ids_array_(tiledb_helpers::open_array(
            tdb_func__, ctx_, partitioned_ids_uri_, TILEDB_READ))
      , ids_schema_{partitioned_ids_array_->schema()}
      //      , master_indices_{std::move(indices)}
      //      , relevant_parts_(std::move(relevant_parts))
      , master_indices_{indices}
      , relevant_parts_(relevant_parts)
      , squashed_indices_(size(relevant_parts_) + 1)
      , first_resident_part_{0}
      , last_resident_part_{0} {
    total_num_parts_ = size(relevant_parts_);

    scoped_timer _{tdb_func__ + " " + partitioned_vectors_uri_};

    auto cell_order = partitioned_vectors_schema_.cell_order();
    auto tile_order = partitioned_vectors_schema_.tile_order();

    if (cell_order != tile_order) {
      throw std::runtime_error("Cell order and tile order must match");
    }

    auto domain_{partitioned_vectors_schema_.domain()};

    auto array_rows_{domain_.dimension(0)};
    auto array_cols_{domain_.dimension(1)};

    num_array_rows_ =
        (array_rows_.template domain<row_domain_type>().second -
         array_rows_.template domain<row_domain_type>().first + 1);

// We don't use this.  The active partitions naturally limits the number of
// columns that we will read in.
// Comment out for now
#if 0
    num_array_cols_ =
        (array_cols_.template domain<col_domain_type>().second -
         array_cols_.template domain<col_domain_type>().first + 1);
#endif

    if ((matrix_order_ == TILEDB_ROW_MAJOR && cell_order == TILEDB_COL_MAJOR) ||
        (matrix_order_ == TILEDB_COL_MAJOR && cell_order == TILEDB_ROW_MAJOR)) {
      throw std::runtime_error("Cell order and matrix order must match");
    }

    size_t dimension = num_array_rows_;

    // indices might not be contiguous, so we need to explicitly add the deltas
    auto total_max_cols = 0UL;
    for (size_t i = 0; i < total_num_parts_; ++i) {
      auto part_size = master_indices_[relevant_parts_[i] + 1] -
                       master_indices_[relevant_parts_[i]];
      total_max_cols += part_size;
      max_part_size_ = std::max<size_t>(max_part_size_, part_size);
    }

    if (upper_bound != 0 && upper_bound < max_part_size_) {
      throw std::runtime_error(
          "Upper bound is less than max partition size: " +
          std::to_string(upper_bound) + " < " + std::to_string(max_part_size_));
    }

    if (upper_bound == 0 || upper_bound > total_max_cols) {
      column_capacity = total_max_cols;
    } else {
      column_capacity = upper_bound;
    }

    // auto max_resident_parts = 0UL;
    size_t running_resident_parts = 0UL;
    size_t running_resident_size = 0UL;
    for (size_t i = 0; i < total_num_parts_; ++i) {
      auto part_size = master_indices_[relevant_parts_[i] + 1] -
                       master_indices_[relevant_parts_[i]];

      if (running_resident_size + part_size > column_capacity) {
        max_resident_parts_ =
            std::max(running_resident_parts, max_resident_parts_);
        running_resident_parts = 0;
        running_resident_size = 0;
      }

      running_resident_parts += 1;
      running_resident_size += part_size;
    }
    max_resident_parts_ = std::max(running_resident_parts, max_resident_parts_);

    if (max_resident_parts_ == 0) {
      max_resident_parts_ = total_num_parts_;
    }

    /*
     * Create new indices that are relative to all of the partitions that will
     * be resident.
     *
     * @todo Do this load() by load() and maintain in the PartitionedMatrix
     * base class
     */
    squashed_indices_[0] = 0;
    for (size_t i = 0; i < size(relevant_parts_); ++i) {
      squashed_indices_[i + 1] = squashed_indices_[i] +
                                 master_indices_[relevant_parts_[i] + 1] -
                                 master_indices_[relevant_parts_[i]];
    }

    /*
     * Now that we have computed the parameters for storing the partitioned
     * data, we prep the storage for subsequent loads.
     *
     * column_capacity is the max number of vectors we will ever make resident.
     * We use this to size the "nnz" of the partitioned_matrix base class.
     *
     * max_resident_parts is the max number of partitions that will ever be
     * resident at any one time.  We use this to size the index of the
     * partitioned_matrix base class.
     */
    Base::operator=(
        std::move(Base{dimension, column_capacity, max_resident_parts_}));
    // this->num_vectors() = 0;
    // this->num_partitions() = 0;
    this->num_vectors_ = 0;
    this->num_parts_ = 0;

    if (this->part_index_.size() != max_resident_parts_ + 1) {
      throw std::runtime_error(
          "Invalid partitioning, part_index_ size " +
          std::to_string(this->part_index_.size()) +
          " != " + std::to_string(max_resident_parts_ + 1));
    }
  }

  /**
   * Read in the next partitions
   * todo Allow to specify how many columns to read in
   */
  bool load() override {
    scoped_timer _{tdb_func__ + " " + partitioned_vectors_uri_};

    if (this->part_index_.size() != max_resident_parts_ + 1) {
      throw std::runtime_error(
          "Invalid partitioning, part_index_ size " +
          std::to_string(this->part_index_.size()) +
          " != " + std::to_string(max_resident_parts_ + 1));
    }

    // @todo -- col oriented only for now -- generalize!!
    {
      const size_t attr_idx = 0;
      auto attr = partitioned_vectors_schema_.attribute(attr_idx);

      std::string attr_name = attr.name();
      tiledb_datatype_t attr_type = attr.type();
      if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
        throw std::runtime_error(
            "Attribute type mismatch: " + std::to_string(attr_type) + " != " +
            std::to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
      }

      auto dimension = num_array_rows_;

      /*
       * Fit as many partitions as we can into column_capacity
       */

      first_resident_col_ = last_resident_col_;    // # columns
      first_resident_part_ = last_resident_part_;  // # partitions

      last_resident_part_ = first_resident_part_;
      for (size_t i = first_resident_part_; i < total_num_parts_; ++i) {
        auto next_part_size = squashed_indices_[i + 1] - squashed_indices_[i];

        // Continue if this partition is empty
        if (next_part_size == 0) {
          continue;
        }

        if ((last_resident_col_ + next_part_size) >
            first_resident_col_ + column_capacity) {
          break;
        }
        last_resident_col_ += next_part_size;  // FIXME ??
        last_resident_part_ = i + 1;
      }
      num_resident_cols_ = last_resident_col_ - first_resident_col_;
      resident_col_offset_ = first_resident_col_;

      assert(num_resident_cols_ <= column_capacity);
      if (num_resident_cols_ > column_capacity) {
        throw std::runtime_error(
            "Invalid partitioning, num_resident_cols_ " +
            std::to_string(num_resident_cols_) + " > " +
            std::to_string(column_capacity));
      }

      num_resident_parts_ = last_resident_part_ - first_resident_part_;
      resident_part_offset_ = first_resident_part_;

      assert(num_resident_parts_ <= max_resident_parts_);
      if (num_resident_parts_ > max_resident_parts_) {
        throw std::runtime_error(
            "Invalid partitioning, num_resident_parts_ " +
            std::to_string(num_resident_parts_) + " > " +
            std::to_string(max_resident_parts_));
      }

      if ((num_resident_cols_ == 0 && num_resident_parts_ != 0) ||
          (num_resident_cols_ != 0 && num_resident_parts_ == 0)) {
        throw std::runtime_error("Invalid partitioning");
      }
      if (num_resident_cols_ == 0) {
        return false;
      }

      if (this->part_index_.size() != max_resident_parts_ + 1) {
        throw std::runtime_error(
            "Invalid partitioning, part_index_ size " +
            std::to_string(this->part_index_.size()) +
            " != " + std::to_string(max_resident_parts_ + 1));
      }
      if (num_resident_parts_ > max_resident_parts_) {
        throw std::runtime_error(
            "Invalid partitioning, num_resident_parts_ " +
            std::to_string(num_resident_parts_) + " > " +
            std::to_string(max_resident_parts_));
      }

      /*
       * Set up the subarray to read the partitions
       */
      tiledb::Subarray subarray(ctx_, *(this->partitioned_vectors_array_));

      // Dimension 0 goes from 0 to 127
      subarray.add_range(0, 0, (int)dimension - 1);

      /**
       * Read in the next batch of partitions
       */
      size_t col_count = 0;
      for (size_t j = first_resident_part_; j < last_resident_part_; ++j) {
        size_t start = master_indices_[relevant_parts_[j]];
        size_t stop = master_indices_[relevant_parts_[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        col_count += len;
        subarray.add_range(1, (int)start, (int)stop - 1);
      }
      if (col_count != last_resident_col_ - first_resident_col_) {
        throw std::runtime_error("Column count mismatch");
      }

      auto cell_order = partitioned_vectors_schema_.cell_order();
      auto layout_order = cell_order;

      tiledb::Query query(ctx_, *(this->partitioned_vectors_array_));

      auto ptr = this->data();
      query.set_subarray(subarray)
          .set_layout(layout_order)
          .set_data_buffer(attr_name, ptr, col_count * dimension);
      // tiledb_helpers::submit_query(tdb_func__, partitioned_vectors_uri_,
      // query);
      query.submit();
      _memory_data.insert_entry(tdb_func__, col_count * dimension * sizeof(T));

      // assert(tiledb::Query::Status::COMPLETE == query.query_dstatus());
      auto qs = query.query_status();
      // @todo Handle incomplete queries.
      if (tiledb::Query::Status::COMPLETE != query.query_status()) {
        throw std::runtime_error("Query status is not complete -- fix me");
      }
    }

    /**
     * Lather, rinse, repeat for ids -- use separate scopes for partitions
     * and ids to keep from cross pollinating identifiers
     *
     * @todo -- combine these two blocks
     */
    {
      auto ids_attr_idx = 0;

      auto ids_attr = ids_schema_.attribute(ids_attr_idx);
      std::string ids_attr_name = ids_attr.name();

      tiledb::Subarray ids_subarray(ctx_, *partitioned_ids_array_);

      size_t ids_col_count = 0;
      for (size_t j = first_resident_part_; j < last_resident_part_; ++j) {
        size_t start = master_indices_[relevant_parts_[j]];
        size_t stop = master_indices_[relevant_parts_[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        ids_col_count += len;
        ids_subarray.add_range(0, (int)start, (int)stop - 1);
      }
      if (ids_col_count != last_resident_col_ - first_resident_col_) {
        throw std::runtime_error("Column count mismatch");
      }

      tiledb::Query ids_query(ctx_, *partitioned_ids_array_);

      auto ids_ptr = this->ids_.data();
      ids_query.set_subarray(ids_subarray)
          .set_data_buffer(ids_attr_name, ids_ptr, ids_col_count);
      tiledb_helpers::submit_query(tdb_func__, partitioned_ids_uri_, ids_query);
      _memory_data.insert_entry(tdb_func__, ids_col_count * sizeof(T));

      // assert(tiledb::Query::Status::COMPLETE == query.query_status());
      if (tiledb::Query::Status::COMPLETE != ids_query.query_status()) {
        throw std::runtime_error("Query status is not complete -- fix me");
      }
    }

    /*
     * Copy indices for resident partitions into Base::part_index_
     * resident_part_offset_ will be the first index into squashed
     * Also [first_resident_part_, last_resident_part_)
     */
    auto sub = squashed_indices_[resident_part_offset_];
    for (size_t i = 0; i < num_resident_parts_ + 1; ++i) {
      this->part_index_[i] = squashed_indices_[i + resident_part_offset_] - sub;
    }

    // this->num_vectors() =
    // this->num_partitions() =
    this->num_vectors_ = num_resident_cols_;
    ;
    this->num_parts_ = num_resident_parts_;
    ;

    num_loads_++;
    return true;
  }
#if 0
  auto& vectors() const {
    return *this;
  }

  index_type num_resident_parts() const {
    return last_resident_part_ - first_resident_part_;
  }

  index_type resident_part_offset() const {
    return resident_part_offset_;
  }

  index_type col_offset() const {
    return resident_col_offset_;
  }

  size_t num_loads() const {
    return num_loads_;
  }
#endif
  /**
   * Destructor.  Closes arrays if they are open.
   */
  ~tdbPartitionedMatrix() {
    // Don't really need these since tiledb::Array will close on destruction
    if (partitioned_vectors_array_->is_open()) {
      partitioned_vectors_array_->close();
    }
    if (partitioned_ids_array_->is_open()) {
      partitioned_ids_array_->close();
    }
  }
};

/**
 * Convenience class for row-major matrices.
 */
template <
    class T,
    class partitioned_ids_type,
    class indices_type,
    class I = size_t>
using tdbRowMajorPartitionedMatrix = tdbPartitionedMatrix<
    T,
    partitioned_ids_type,
    indices_type,
    stdx::layout_right,
    I>;

/**
 * Convenience class for column-major matrices.
 */
template <
    class T,
    class partitioned_ids_type,
    class indices_type,
    class I = size_t>
using tdbColMajorPartitionedMatrix = tdbPartitionedMatrix<
    T,
    partitioned_ids_type,
    indices_type,
    stdx::layout_left,
    I>;

#endif  // TILEDB_PARTITIONED_MATRIX_H
