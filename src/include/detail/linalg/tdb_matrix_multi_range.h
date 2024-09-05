/**
 * @file   tdb_matrix_multi_range.h
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
 * Class that provides a matrix interface to a TileDB array containing
 * vectors. We load in a multi-range query using indices into the columns.
 *
 */

#ifndef TDB_MATRIX_MULTI_RANGE_H
#define TDB_MATRIX_MULTI_RANGE_H

#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "tdb_defs.h"

/**
 * Derived from `Matrix`.
 * Initialized in construction by filling from the TileDB vectors array.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class tdbBlockedMatrixMultiRange : public Matrix<T, LayoutPolicy, I> {
  using Base = Matrix<T, LayoutPolicy, I>;
  using Base::Base;

 public:
  using index_type = typename Base::index_type;
  using size_type = typename Base::size_type;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  log_timer constructor_timer{"tdbBlockedMatrixMultiRange constructor"};

  tiledb::Context ctx_;

  uint64_t dimensions_{0};

  std::string uri_;
  std::unique_ptr<tiledb::Array> array_;
  tiledb::ArraySchema schema_;

  // The indices of all the columns to load. The size of this is the total
  // number of columns.
  std::vector<I> column_indices_;

  // The max number of columns that can fit in allocated memory
  size_t column_capacity_{0};

  // The number of columns that are currently loaded into memory
  size_t num_resident_cols_{0};

  // The final index numbers of the resident columns
  size_t last_resident_col_{0};

  size_t get_elements_to_load() const {
    // Note that here we try to load column_indices_.size() vectors. If we are
    // time travelling, these vectors may not exist in the array, but we still
    // need to load them to know that they don't exist.
    return std::min(
        column_capacity_, column_indices_.size() - last_resident_col_);
  }

 public:
  tdbBlockedMatrixMultiRange(tdbBlockedMatrixMultiRange&& rhs) = default;

  /** Default destructor. array will be closed when it goes out of scope */
  virtual ~tdbBlockedMatrixMultiRange() = default;

  /**
   * @brief Construct a new tdbBlockedMatrixMultiRange object, limited to
   * `upper_bound` vectors. In this case, the `Matrix` is column-major, so the
   * number of vectors is the number of columns.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param indices The indices of the columns to read.
   * @param dimensions The number of dimensions in each vector.
   * @param upper_bound The maximum number of vectors to read.
   * @param temporal_policy The TemporalPolicy to use for reading the array
   * data.
   */
  tdbBlockedMatrixMultiRange(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::vector<I>& column_indices,
      size_type dimensions,
      size_t upper_bound,
      TemporalPolicy temporal_policy = TemporalPolicy{})
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Base(dimensions, column_indices.size())
      , dimensions_{dimensions}
      , uri_{uri}
      , array_(std::make_unique<tiledb::Array>(
            ctx, uri, TILEDB_READ, temporal_policy.to_tiledb_temporal_policy()))
      , schema_{array_->schema()}
      , column_indices_{column_indices} {
    constructor_timer.stop();

    // The default is to load all the vectors.
    if (upper_bound == 0 || upper_bound > column_indices_.size()) {
      column_capacity_ = column_indices_.size();
    } else {
      column_capacity_ = upper_bound;
    }

    // Check the cell and tile order.
    auto cell_order = schema_.cell_order();
    auto tile_order = schema_.tile_order();
    if ((matrix_order_ == TILEDB_ROW_MAJOR && cell_order == TILEDB_COL_MAJOR) ||
        (matrix_order_ == TILEDB_COL_MAJOR && cell_order == TILEDB_ROW_MAJOR)) {
      throw std::runtime_error("Cell order and matrix order must match");
    }
    if (cell_order != tile_order) {
      throw std::runtime_error("Cell order and tile order must match");
    }

#ifdef __cpp_lib_smart_ptr_for_overwrite
    auto data =
        std::make_unique_for_overwrite<T[]>(dimensions * column_capacity_);
#else
    auto data = std::unique_ptr<T[]>(new T[dimensions * column_capacity_]);
#endif
    Base::operator=(Base{std::move(data), dimensions, column_capacity_});
  }

  bool load() {
    scoped_timer _{"tdb_matrix_multi_range@load"};

    auto elements_to_load = get_elements_to_load();

    // Return early if we're at the end already.
    if (elements_to_load == 0 || dimensions_ == 0) {
      array_->close();
      return false;
    }

    const auto first_resident_col = last_resident_col_;
    last_resident_col_ += elements_to_load;
    num_resident_cols_ = last_resident_col_ - first_resident_col;

    // Set up the subarray.
    auto attr = schema_.attribute(0);
    std::string attr_name = attr.name();

    tiledb_datatype_t attr_type = attr.type();
    if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch: " + datatype_to_string(attr_type) + " != " +
          datatype_to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
    }
    tiledb::Subarray subarray(ctx_, *array_);
    // For a 128 dimension vector, Dimension 0 will go from 0 to 127.
    subarray.add_range(0, 0, static_cast<int>(dimensions_) - 1);

    // Setup the query ranges.
    for (size_t i = first_resident_col; i < last_resident_col_; ++i) {
      const auto index = static_cast<int>(column_indices_[i]);
      subarray.add_range(1, index, index);
    }

    // Execute the query.
    tiledb::Query query(ctx_, *array_);
    query.set_subarray(subarray)
        .set_layout(schema_.cell_order())
        .set_data_buffer(
            attr_name, this->data(), num_resident_cols_ * dimensions_);
    tiledb_helpers::submit_query(tdb_func__, uri_, query);
    // @todo Handle incomplete queries.
    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error(
          "[tdb_matrix_multi_range@load] Query status is not complete");
    }

    // Close the arrays if we're done.
    if (get_elements_to_load() == 0) {
      array_->close();
    }

    return true;
  }
};

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t>
using tdbColMajorMatrixMultiRange =
    tdbBlockedMatrixMultiRange<T, stdx::layout_left, I>;

#endif  // TDB_MATRIX_MULTI_RANGE_H
