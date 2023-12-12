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
 * Class the provides a matrix view to a partitioned TileDB array (as
 * partitioned by IVF indexing).
 *
 * The class requires the URI of a partitioned TileDB array and partioned set of
 * vector identifiers.  The class will provide a view of the requested
 * partitions and the corresponding vector identifiers.
 *
 * Also provides support for out-of-core operation.
 *
 */

#ifndef TILEDB_PARTITIONED_MATRIX_H
#define TILEDB_PARTITIONED_MATRIX_H

#include <cstddef>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <tiledb/tiledb>
#include "mdspan/mdspan.hpp"

#include "detail/linalg/tdb_defs.h"

#include "utils/timer.h"

namespace stdx {
using namespace Kokkos;
}  // namespace stdx

/**
 *
 * @note The template parameters indices_type and parts_type are deduced using
 * CTAD.  However, with the uri-based constructor, the type of the indices and
 * the shuffled_db array cannot be deduced.  Therefore, the user must specify
 * the type of the indices and the shuffled_ids array.
 */
template <
    class T,
    class shuffled_ids_type,
    class indices_type,
    class parts_type,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
class tdbPartitionedMatrix : public Matrix<T, LayoutPolicy, I> {
  /****************************************************************************
   *
   * Duplication from tdbMatrix
   *
   * @todo Unify duplicated code
   *
   ****************************************************************************/
  using Base = Matrix<T, LayoutPolicy, I>;
  using Base::Base;

 public:
  using value_type = typename Base::value_type;
  using typename Base::index_type;
  using typename Base::reference;
  using typename Base::size_type;

  using view_type = Base;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  using row_domain_type = int32_t;
  using col_domain_type = int32_t;

  log_timer constructor_timer{"tdbPartitionedMatrix constructor"};

  std::string uri_;
  std::reference_wrapper<const tiledb::Context> ctx_;
  tiledb::Array array_;
  tiledb::ArraySchema schema_;
  size_t num_array_rows_{0};
  size_t num_array_cols_{0};

  // std::tuple<index_type, index_type> row_view_;
  std::tuple<index_type, index_type> col_view_;
  // index_type row_offset_{0};
  index_type col_offset_{0};

  // For future asynchronous loads
  // std::unique_ptr<T[]> backing_data_;
  // std::future<bool> fut_;
  // size_t pending_row_offset{0};
  // size_t pending_col_offset{0};

  /****************************************************************************
   *
   * Stuff for partitioned (reshuffled) matrix
   *
   * @todo This needs to go into its own class
   *
   ****************************************************************************/
  tiledb::Array ids_array_;
  tiledb::ArraySchema ids_schema_;
  std::vector<indices_type> indices_;   // @todo pointer and span?
  std::vector<parts_type> parts_;       // @todo pointer and span?
  std::vector<shuffled_ids_type> ids_;  // @todo pointer and span?

  // The total number of p in the partitioned array
  size_t total_num_parts_{0};

  // std::tuple<index_type, index_type> row_part_view_;
  std::tuple<index_type, index_type> col_part_view_;

  // index_type row_part_offset_{0};
  index_type col_part_offset_{0};

  // The max number of columns that can fit in allocated memory
  size_t max_cols_{0};

  // The number of columns in the portion of array loaded into memory
  size_t num_cols_{0};

  // The total number of partitions in the partitioned array
  size_t max_col_parts_{0};

  // The number of partitions in the portion of array loaded into memory
  size_t num_col_parts_{0};

  size_t num_loads_{0};
  size_t max_part_size_{0};

 public:
  tdbPartitionedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      std::vector<indices_type>& indices,
      const std::vector<parts_type>& parts,
      const std::string& id_uri)
      : tdbPartitionedMatrix(
            ctx, uri, indices, parts, id_uri, /*shuffled_ids,*/ 0) {
  }

  /**
   * Gather pieces of a partitioned array into a single array (along with the
   * vector ids into a corresponding 1D array)
   *
   * @todo Column major is kind of baked in here.  Need to generalize.
   */
  tdbPartitionedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      std::vector<indices_type>& in_indices,
      const std::vector<parts_type>& in_parts,
      const std::string& ids_uri,
      // std::vector<shuffled_ids_type>& shuffled_ids,
      size_t upper_bound,
      const tiledb::TemporalPolicy temporal_policy = {})
      : constructor_timer{tdb_func__ + std::string{" constructor"}}
      , uri_{uri}
      , ctx_{ctx}
      , array_{tiledb_helpers::open_array(tdb_func__, ctx_, uri, TILEDB_READ, temporal_policy)}
      , schema_{array_.schema()}
      , ids_array_{tiledb_helpers::open_array(
            tdb_func__, ctx_, ids_uri, TILEDB_READ)}
      , ids_schema_{ids_array_.schema()}
      , indices_{in_indices}
      , parts_{in_parts}
      , col_part_view_{0, 0} {
    constructor_timer.stop();

    total_num_parts_ = size(parts_);

    scoped_timer _{tdb_func__ + " " + uri_};

    auto cell_order = schema_.cell_order();
    auto tile_order = schema_.tile_order();

    // @todo Maybe throw an exception here?  Have to properly handle since
    // this is a constructor.
    assert(cell_order == tile_order);

    auto domain_{schema_.domain()};

    auto array_rows_{domain_.dimension(0)};
    auto array_cols_{domain_.dimension(1)};

    num_array_rows_ =
        (array_rows_.template domain<row_domain_type>().second -
         array_rows_.template domain<row_domain_type>().first + 1);
    num_array_cols_ =
        (array_cols_.template domain<col_domain_type>().second -
         array_cols_.template domain<col_domain_type>().first + 1);

    if ((matrix_order_ == TILEDB_ROW_MAJOR && cell_order == TILEDB_COL_MAJOR) ||
        (matrix_order_ == TILEDB_COL_MAJOR && cell_order == TILEDB_ROW_MAJOR)) {
      throw std::runtime_error("Cell order and matrix order must match");
    }

    size_t dimension = num_array_rows_;

    // indices might not be contiguous, so we need to explicitly add the deltas
    auto total_max_cols = 0UL;
    auto max_part_size_ = 0UL;
    for (size_t i = 0; i < total_num_parts_; ++i) {
      total_max_cols += indices_[parts_[i] + 1] - indices_[parts_[i]];
      max_part_size_ = std::max<size_t>(
          max_part_size_, indices_[parts_[i] + 1] - indices_[parts_[i]]);
    }

    if (upper_bound != 0 && upper_bound < max_part_size_) {
      throw std::runtime_error(
          "Upper bound is less than max partition size: " +
          std::to_string(upper_bound) + " < " + std::to_string(max_part_size_));
    }

    if (upper_bound == 0 || upper_bound > total_max_cols) {
      max_cols_ = total_max_cols;
    } else {
      max_cols_ = upper_bound;
    }

    // @todo be more sensible -- dont use a vector and don't use inout
    if (size(ids_) < max_cols_) {
      ids_.resize(max_cols_);
    }

#ifdef __cpp_lib_smart_ptr_for_overwrite
    auto data_ = std::make_unique_for_overwrite<T[]>(dimension * max_cols_);
#else
    auto data_ = std::unique_ptr<T[]>(new T[dimension * max_cols_]);
#endif

    Base::operator=(Base{std::move(data_), dimension, max_cols_});
  }

  /**
   * Read in the next partitions
   * todo Allow to specify how many columns to read in
   */
  bool load() {
    scoped_timer _{tdb_func__ + " " + uri_};

    // @todo -- col oriented only for now -- generalize!!
    {
      const size_t attr_idx = 0;
      auto attr = schema_.attribute(attr_idx);

      std::string attr_name = attr.name();
      tiledb_datatype_t attr_type = attr.type();
      if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
        throw std::runtime_error(
            "Attribute type mismatch: " + std::to_string(attr_type) + " != " +
            std::to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
      }

      auto dimension = num_array_rows_;

      /*
       * Fit as many partitions as we can into max_cols_
       */
      std::get<0>(col_view_) = std::get<1>(col_view_);  // # columns
      std::get<0>(col_part_view_) =
          std::get<1>(col_part_view_);  // # partitions

      std::get<1>(col_part_view_) = std::get<0>(col_part_view_);
      for (size_t i = std::get<0>(col_part_view_); i < total_num_parts_; ++i) {
        auto next_part_size = indices_[parts_[i] + 1] - indices_[parts_[i]];

        // Continue if this partition is empty
        if (next_part_size == 0) {
          continue;
        }

        if ((std::get<1>(col_view_) + next_part_size) >
            std::get<0>(col_view_) + max_cols_) {
          break;
        }
        std::get<1>(col_view_) += next_part_size;  // FIXME ??
        std::get<1>(col_part_view_) = i + 1;
      }
      num_cols_ = std::get<1>(col_view_) - std::get<0>(col_view_);
      col_offset_ = std::get<0>(col_view_);

      assert(num_cols_ <= max_cols_);

      num_col_parts_ =
          std::get<1>(col_part_view_) - std::get<0>(col_part_view_);
      col_part_offset_ = std::get<0>(col_part_view_);

      if ((num_cols_ == 0 && num_col_parts_ != 0) ||
          (num_cols_ != 0 && num_col_parts_ == 0)) {
        throw std::runtime_error("Invalid partitioning");
      }
      if (num_cols_ == 0) {
        return false;
      }

      /*
       * Set up the subarray to read the partitions
       */
      tiledb::Subarray subarray(ctx_, this->array_);

      // Dimension 0 goes from 0 to 127
      subarray.add_range(0, 0, (int)dimension - 1);

      /**
       * Read in the next batch of partitions
       */
      size_t col_count = 0;
      for (size_t j = std::get<0>(col_part_view_);
           j < std::get<1>(col_part_view_);
           ++j) {
        size_t start = indices_[parts_[j]];
        size_t stop = indices_[parts_[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        col_count += len;
        subarray.add_range(1, (int)start, (int)stop - 1);
      }
      if (col_count != std::get<1>(col_view_) - std::get<0>(col_view_)) {
        throw std::runtime_error("Column count mismatch");
      }

      auto cell_order = schema_.cell_order();
      auto layout_order = cell_order;

      tiledb::Query query(ctx_, this->array_);

      // auto ptr = data_.get();

      auto ptr = this->data();
      query.set_subarray(subarray)
          .set_layout(layout_order)
          .set_data_buffer(attr_name, ptr, col_count * dimension);
      tiledb_helpers::submit_query(tdb_func__, uri_, query);
      _memory_data.insert_entry(tdb_func__, col_count * dimension * sizeof(T));

      // assert(tiledb::Query::Status::COMPLETE == query.query_status());
      if (tiledb::Query::Status::COMPLETE != query.query_status()) {
        throw std::runtime_error("Query status is not complete -- fix me");
      }
    }

    /**
     * Lather, rinse, repeat for ids -- use separate scopes for partitions
     * and ids to keep from cross pollinating identifiers
     */
    {
      auto ids_attr_idx = 0;

      auto ids_attr = ids_schema_.attribute(ids_attr_idx);
      std::string ids_attr_name = ids_attr.name();

      tiledb::Subarray ids_subarray(ctx_, ids_array_);

      size_t ids_col_count = 0;
      for (size_t j = std::get<0>(col_part_view_);
           j < std::get<1>(col_part_view_);
           ++j) {
        size_t start = indices_[parts_[j]];
        size_t stop = indices_[parts_[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        ids_col_count += len;
        ids_subarray.add_range(0, (int)start, (int)stop - 1);
      }
      if (ids_col_count != std::get<1>(col_view_) - std::get<0>(col_view_)) {
        throw std::runtime_error("Column count mismatch");
      }

      tiledb::Query ids_query(ctx_, ids_array_);

      auto ids_ptr = ids_.data();
      ids_query.set_subarray(ids_subarray)
          .set_data_buffer(ids_attr_name, ids_ptr, ids_col_count);
      ids_query.submit();
      _memory_data.insert_entry(tdb_func__, ids_col_count * sizeof(T));

      // assert(tiledb::Query::Status::COMPLETE == query.query_status());
      if (tiledb::Query::Status::COMPLETE != ids_query.query_status()) {
        throw std::runtime_error("Query status is not complete -- fix me");
      }
    }

    num_loads_++;
    return true;
  }

  auto& ids() const {
    return ids_;
  }

  index_type num_col_parts() const {
    return std::get<1>(col_part_view_) - std::get<0>(col_part_view_);
  }

  index_type col_part_offset() const {
    return col_part_offset_;
  }

  index_type col_offset() const {
    return col_offset_;
  }

  size_t num_loads() const {
    return num_loads_;
  }

  /**
   * Destructor.  Closes arrays if they are open.
   */
  ~tdbPartitionedMatrix() {
    if (array_.is_open()) {
      array_.close();
    }
    if (ids_array_.is_open()) {
      ids_array_.close();
    }
  }
};

/**
 * Convenience class for row-major matrices.
 */
template <
    class T,
    class shuffled_ids_type,
    class indices_type,
    class parts_type,
    class I = size_t>
using tdbRowMajorPartitionedMatrix = tdbPartitionedMatrix<
    T,
    shuffled_ids_type,
    indices_type,
    parts_type,
    stdx::layout_right,
    I>;

/**
 * Convenience class for column-major matrices.
 */
template <
    class T,
    class shuffled_ids_type,
    class indices_type,
    class parts_type,
    class I = size_t>
using tdbColMajorPartitionedMatrix = tdbPartitionedMatrix<
    T,
    shuffled_ids_type,
    indices_type,
    parts_type,
    stdx::layout_left,
    I>;

#endif  // TILEDB_PARTITIONED_MATRIX_H
