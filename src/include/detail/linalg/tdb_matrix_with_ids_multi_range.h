/**
 * @file   tdb_matrix_with_ids_multi_range.h
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
 * Class that provides a matrix interface to two TileDB arrays, one containing
 * vectors and one IDs. We load in a multi-range query using indices into the columns.
 *
 */

#ifndef TDB_MATRIX_WITH_IDS_MULTI_RANGE_H
#define TDB_MATRIX_WITH_IDS_MULTI_RANGE_H

#include "detail/linalg/matrix_with_ids.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "tdb_defs.h"

/**
 * Derived from `tdbBlockedMatrix`, which we have derive from `MatrixWithIds`.
 * Initialized in construction by filling from the TileDB vectors array and the
 * the TileDB IDs array.
 */
template <
    class T,
    class IdsType = uint64_t,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
class tdbBlockedMatrixWithIdsMultiRange : public MatrixWithIds<T, IdsType, LayoutPolicy, I> {
  using Base = MatrixWithIds<T, IdsType, LayoutPolicy, I>;
  using Base::Base;

 public:
  using index_type = typename Base::index_type;
  using ids_type = typename Base::ids_type;
  using size_type = typename Base::size_type;

 private:
  log_timer constructor_timer{"tdbBlockedMatrixWithIdsMultiRange constructor"};

  tiledb::Context ctx_;

  uint64_t dimensions_{0};

  std::string uri_;
  std::unique_ptr<tiledb::Array> array_;
  tiledb::ArraySchema schema_;

  std::string ids_uri_;
  std::unique_ptr<tiledb::Array> ids_array_;
  tiledb::ArraySchema ids_schema_;

  // The indices of all the columns to load. The size of this is the total number of columns.
  std::unordered_set<I> column_indices_;

  // The max number of columns that can fit in allocated memory
  size_t column_capacity_{0};

  // The number of columns that are currently loaded into memory
  size_t num_resident_cols_{0};

  // The final index numbers of the resident columns
  index_type last_resident_col_{0};

  size_t get_elements_to_load() const {
    return std::min(column_capacity_, column_indices_.size() - last_resident_col_);
  }

 public:
  tdbBlockedMatrixWithIdsMultiRange(tdbBlockedMatrixWithIdsMultiRange&& rhs) = default;

  /** Default destructor. array will be closed when it goes out of scope */
  virtual ~tdbBlockedMatrixWithIdsMultiRange() = default;

  /**
   * @brief Construct a new tdbBlockedMatrixWithIdsMultiRange object, limited to
   * `upper_bound` vectors. In this case, the `Matrix` is column-major, so the
   * number of vectors is the number of columns.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param ids_uri URI of the TileDB array of IDs to read.
   * @param indices The indices of the columns to read.
   * @param dimensions The number of dimensions in each vector.
   * @param upper_bound The maximum number of vectors to read.
   * @param temporal_policy The TemporalPolicy to use for reading the array
   * data.
   */
  tdbBlockedMatrixWithIdsMultiRange(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri,
      const std::unordered_set<I>& column_indices,
      size_type& dimensions,
      size_t upper_bound,
      TemporalPolicy temporal_policy)
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Base(dimensions, column_indices_.size())
      , dimensions_{dimensions}
      , uri_{uri}
      , array_(std::make_unique<tiledb::Array>(ctx, uri, TILEDB_READ, temporal_policy.to_tiledb_temporal_policy()))
      , schema_{array_->schema()}
      , ids_uri_(ids_uri)
      , ids_array_(std::make_unique<tiledb::Array>(ctx, ids_uri, TILEDB_READ, temporal_policy.to_tiledb_temporal_policy()))
      , ids_schema_{ids_array_->schema()}
      , column_indices_{column_indices} {
    constructor_timer.stop();

    // The default is to load all the vectors.
    if (upper_bound == 0 || upper_bound > column_indices_.size()) {
      column_capacity_ = column_indices_.size();
    } else {
      column_capacity_ = upper_bound;
    }

    #ifdef __cpp_lib_smart_ptr_for_overwrite
      auto data = std::make_unique_for_overwrite<T[]>(dimensions * column_capacity_);
      auto ids = std::make_unique_for_overwrite<typename MatrixBase::ids_type[]>(column_capacity_);
    #else
      auto data = std::unique_ptr<T[]>(new T[dimensions * column_capacity_]);
      auto ids = std::unique_ptr<typename Base::ids_type[]>(new typename Base::ids_type[column_capacity_]);
    #endif
      Base::operator=(Base{std::move(data), std::move(ids), dimensions, column_capacity_});
  }

  // @todo Allow specification of how many columns to advance by
  bool load() {
    scoped_timer _{"tdb_matrix_with_ids_multi_range@load"};

    auto elements_to_load = get_elements_to_load();

    // Return if we're at the end.
    if (elements_to_load == 0 || dimensions_ == 0) {
      array_->close();
      ids_array_->close();
      return false;
    }

    const auto first_resident_col = last_resident_col_;
    last_resident_col_ += elements_to_load;
    
    num_resident_cols_ = last_resident_col_ - first_resident_col;

    // a. Set up the vectors subarray.
    auto attr = schema_.attribute(0);
    std::string attr_name = attr.name();
    tiledb::Subarray subarray(ctx_, *array_);
    // For a 128 dimension vector, Dimension 0 will go from 0 to 127.
    subarray.add_range(0, 0, static_cast<int>(dimensions_) - 1);

    // b. Set up the IDs subarray.
    auto ids_attr = ids_schema_.attribute(0);
    std::string ids_attr_name = ids_attr.name();
    tiledb::Subarray ids_subarray(ctx_, *ids_array_);

    // c. Setup the query ranges.
    for (size_t i = first_resident_col; i < last_resident_col_; ++i) {
      subarray.add_range(1, i, i);
      ids_subarray.add_range(0, i, i);
    }

    // d. Execute the vectors query.
    tiledb::Query query(ctx_, *array_);
    query.set_subarray(subarray)
        .set_layout(schema_.cell_order())
        .set_data_buffer(attr_name, this->data(), num_resident_cols_ * dimensions_);
    tiledb_helpers::submit_query(tdb_func__, uri_, query);
    // @todo Handle incomplete queries.
    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error(
          "[tdb_matrix_with_ids_multi_range@load] Query status is not complete");
    }

    // e. Execute the IDs query.
    tiledb::Query ids_query(ctx_, *ids_array_);
    ids_query.set_subarray(ids_subarray)
        .set_data_buffer(ids_attr_name, this->ids(), num_resident_cols_);
    tiledb_helpers::submit_query(tdb_func__, ids_uri_, ids_query);
    // @todo Handle incomplete queries.
    if (tiledb::Query::Status::COMPLETE != ids_query.query_status()) {
      throw std::runtime_error(
          "[tdb_matrix_with_ids_multi_range@load] IDs query status is not complete");
    }

    // f. Close the arrays if we're done.
    if (get_elements_to_load() == 0) {
      array_->close();
      ids_array_->close();
    }

    return true;
  }
};

/**
 * Convenience class for column-major matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using tdbColMajorMatrixWithIdsMultiRange = tdbBlockedMatrixWithIdsMultiRange<T, IdsType, stdx::layout_left, I>;

#endif  // TDB_MATRIX_WITH_IDS_MULTI_RANGE_H
