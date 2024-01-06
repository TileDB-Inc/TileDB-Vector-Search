/**
 * @file   tdb_matrix.h
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
 * Class that provides a matrix interface to a TileDB array.
 *
 * @todo Include the right headers for BLAS support.
 * @todo Refactor ala tdb_partitioned_matrix.h
 *
 */

#ifndef TDB_MATRIX_H
#define TDB_MATRIX_H

#include <future>

#include <tiledb/tiledb>

#include "detail/linalg/linalg_defs.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_helpers.h"
#include "tdb_defs.h"

/**
 * Derived from `Matrix`.  Initialized in construction by filling from a given
 * TileDB array.
 *
 * @todo Evaluate whether or not we really need to do things this way or if
 * it is sufficient to simply have one Matrix class and have a factory that
 * creates them by reading from TileDB.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class tdbBlockedMatrix : public Matrix<T, LayoutPolicy, I> {
  using Base = Matrix<T, LayoutPolicy, I>;
  using Base::Base;

 public:
  using value_type = typename Base::value_type;
  using index_type = typename Base::index_type;
  using size_type = typename Base::size_type;
  using reference = typename Base::reference;

  using view_type = Base;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  using row_domain_type = int32_t;
  using col_domain_type = int32_t;

  log_timer constructor_timer{"tdbBlockedMatrix constructor"};

  std::reference_wrapper<const tiledb::Context> ctx_;
  std::string uri_;
  std::unique_ptr<tiledb::Array> array_;
  tiledb::ArraySchema schema_;

  /** The domain for each dimension (rows and columns) */
  // size_t row_capacity_{0};
  // size_t col_capacity_{0};

  // std::tuple<index_type, index_type> row_view_;
  // std::tuple<index_type, index_type> col_view_;
  index_type first_row_;
  index_type last_row_;
  index_type first_col_;
  index_type last_col_;
  index_type first_resident_col_;
  index_type last_resident_col_;

  // The number of columns loaded into memory.  Except for the last (remainder)
  // block, this will be equal to `blocksize_`.
  index_type num_resident_cols_{0};

  // How many columns to load at a time
  index_type load_blocksize_{0};

  // How many blocks we have loaded
  size_t num_loads_{0};

  // For future asynchronous loads
  // std::unique_ptr<T[]> backing_data_;
  // std::future<bool> fut_;
  // size_t pending_row_offset{0};
  // size_t pending_col_offset{0};

 public:
  tdbBlockedMatrix(tdbBlockedMatrix&& rhs) = default;

  /** Default destructor. array will be closed when it goes out of scope */
  virtual ~tdbBlockedMatrix() = default;

  /**
   * @brief Construct a new tdbBlockedMatrix object, limited to `upper_bound`
   * vectors. In this case, the `Matrix` is row-major, so the number of vectors
   * is the number of rows.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   */
  tdbBlockedMatrix(const tiledb::Context& ctx, const std::string& uri) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrix(ctx, uri, 0, 0, 0, 0, 0, 0) {
  }

#if 0
  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t upper_bound,
      uint64_t timestamp = 0)
      : tdbBlockedMatrix(
            ctx,
            uri,
            upper_bound,
            (timestamp == 0) ?
                tiledb::TemporalPolicy() :
                tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp)) {
  }
#endif

  /**
   * @brief Construct a new tdbBlockedMatrix object, limited to `upper_bound`
   * vectors. In this case, the `Matrix` is column-major, so the number of
   * vectors is the number of columns.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param upper_bound The maximum number of vectors to read.
   * @param temporal_policy The TemporalPolicy to use for reading the array
   * data.
   */
  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t upper_bound,
      size_t timestamp = 0)
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrix(ctx, uri, 0, 0, 0, 0, upper_bound, timestamp) {
  }
#if 0
  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_array_rows,
      size_t num_array_cols,
      size_t upper_bound,
      size_t timestamp =
          0) requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrix(
            ctx,
            uri,
            num_array_rows,
            num_array_cols,
            upper_bound,
            (timestamp == 0) ?
                tiledb::TemporalPolicy() :
                tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp)) {
  }

  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t upper_bound,
      const tiledb::TemporalPolicy& temporal_policy)  // noexcept
      requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrix(ctx, uri, 0, 0, upper_bound, temporal_policy) {
  }
#endif

#if 0

  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_array_rows,
      size_t num_array_cols,
      size_t upper_bound,
      size_t timestamp = 0)  // noexcept
      requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : ctx_{ctx}
      , uri_{uri}
      , array_(std::make_unique<tiledb::Array>(
            ctx, uri, TILEDB_READ, (timestamp == 0 ?
                tiledb::TemporalPolicy() :
                tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp))))
      , schema_{array_->schema()}
      , num_array_rows_{num_array_rows}
      , num_array_cols_{num_array_cols} {
    constructor_timer.stop();
    scoped_timer _{tdb_func__ + " " + uri};

    auto cell_order = schema_.cell_order();
    auto tile_order = schema_.tile_order();

    // @todo Maybe throw an exception here instead of just an assert?
    // Have to properly handle an exception since this is a constructor.
    assert(cell_order == tile_order);

    const size_t attr_idx = 0;

    auto domain_{schema_.domain()};

    auto array_rows_{domain_.dimension(0)};
    auto array_cols_{domain_.dimension(1)};

    /* The size of the array may not be the size of domain.  Use non-zero value
     * if set in constructor */
    if (num_array_rows_ == 0) {
      num_array_rows_ =
          (array_rows_.template domain<row_domain_type>().second -
           array_rows_.template domain<row_domain_type>().first + 1);
    }
    if (num_array_cols_ == 0) {
      num_array_cols_ =
          (array_cols_.template domain<col_domain_type>().second -
           array_cols_.template domain<col_domain_type>().first + 1);
    }

    if ((matrix_order_ == TILEDB_ROW_MAJOR && cell_order == TILEDB_COL_MAJOR) ||
        (matrix_order_ == TILEDB_COL_MAJOR && cell_order == TILEDB_ROW_MAJOR)) {
      throw std::runtime_error("Cell order and matrix order must match");
    }

    size_t dimension = num_array_rows_;

    if (upper_bound == 0 || upper_bound > num_array_cols_) {
      blocksize_ = num_array_cols_;
    } else {
      blocksize_ = upper_bound;
    }

#ifdef __cpp_lib_smart_ptr_for_overwrite
    auto data_ = std::make_unique_for_overwrite<T[]>(dimension * blocksize_);
#else
    // auto data_ = std::make_unique<T[]>(new T[mat_rows_ * mat_cols_]);
    auto data_ = std::unique_ptr<T[]>(new T[dimension * blocksize_]);
#endif

    Base::operator=(Base{std::move(data_), dimension, blocksize_});
  }

  // @todo Allow specification of how many columns to advance by
  bool load() {
    scoped_timer _{tdb_func__ + " " + uri_};

    const size_t attr_idx{0};
    auto attr = schema_.attribute(attr_idx);

    std::string attr_name = attr.name();
    tiledb_datatype_t attr_type = attr.type();
    if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch: " + std::to_string(attr_type) + " != " +
          std::to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
    }

    auto dimension = num_array_rows_;
    auto num_end_elts =
        std::min(blocksize_, num_array_cols_ - std::get<1>(col_view_));

    // Return if we're at the end
    if (num_end_elts == 0) {
      return false;
    }

    // These calls change the current view
    std::get<0>(col_view_) = std::get<1>(col_view_);
    std::get<1>(col_view_) += num_end_elts;
    col_offset_ = std::get<0>(col_view_);

    num_cols_ = std::get<1>(col_view_) - std::get<0>(col_view_);
    if (num_cols_ == 0) {
      return false;
    }

    assert(std::get<1>(col_view_) <= num_array_cols_);

    // Create a subarray for the next block of columns
    tiledb::Subarray subarray(ctx_, *array_);
    subarray.add_range(0, 0, (int)dimension - 1);
    subarray.add_range(
        1, (int)std::get<0>(col_view_), (int)std::get<1>(col_view_) - 1);

    auto layout_order = schema_.cell_order();

    // Create a query
    tiledb::Query query(ctx_, *array_);
    query.set_subarray(subarray)
        .set_layout(layout_order)
        .set_data_buffer(attr_name, this->data(), num_cols_ * dimension);
    tiledb_helpers::submit_query(tdb_func__, uri_, query);
    _memory_data.insert_entry(tdb_func__, num_cols_ * dimension * sizeof(T));

    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error("Query status is not complete -- fix me");
    }

    num_loads_++;
    return true;
  }

#else
  /** General constructor */
  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t first_row,
      size_t last_row,
      size_t first_col,
      size_t last_col,
      size_t upper_bound,
      size_t timestamp = 0)  // noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : ctx_{ctx}
      , uri_{uri}
      , array_(std::make_unique<tiledb::Array>(
            ctx,
            uri,
            TILEDB_READ,
            (timestamp == 0 ?
                 tiledb::TemporalPolicy() :
                 tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp))))
      , schema_{array_->schema()}
      , first_row_{first_row}
      , last_row_{last_row}
      , first_col_{first_col}
      , last_col_{last_col} {
    constructor_timer.stop();
    scoped_timer _{tdb_func__ + " " + uri};

    if (last_row_ < first_row_) {
      throw std::runtime_error("last_row < first_row");
    }
    if (last_col_ < first_col_) {
      throw std::runtime_error("last_col < first_col");
    }

    auto cell_order = schema_.cell_order();
    auto tile_order = schema_.tile_order();

    if ((matrix_order_ == TILEDB_ROW_MAJOR && cell_order == TILEDB_COL_MAJOR) ||
        (matrix_order_ == TILEDB_COL_MAJOR && cell_order == TILEDB_ROW_MAJOR)) {
      throw std::runtime_error("Cell order and matrix order must match");
    }

    // @todo Maybe throw an exception here instead of just an assert?
    // Have to properly handle an exception since this is a constructor.
    assert(cell_order == tile_order);

    const size_t attr_idx = 0;

    auto domain_{schema_.domain()};

    auto row_domain{domain_.dimension(0)};
    auto col_domain{domain_.dimension(1)};

    /* The size of the array may not be the size of domain.  Use non-zero value
     * if set in constructor */
    if (last_row_ == 0) {
      last_row_ =
          (row_domain.template domain<row_domain_type>().second -
           row_domain.template domain<row_domain_type>().first + 1);
    }
    if (last_col_ == 0) {
      last_col_ =
          (col_domain.template domain<col_domain_type>().second -
           col_domain.template domain<col_domain_type>().first + 1);
    }

    size_t dimension = last_row_ - first_row_;
    size_t num_vectors = last_col_ - first_col_;

    // The default is to load all of the vectors
    if (upper_bound == 0 || upper_bound > num_vectors) {
      load_blocksize_ = num_vectors;
    } else {
      load_blocksize_ = upper_bound;
    }

    first_resident_col_ = first_col_;
    last_resident_col_ = first_col_;

#ifdef __cpp_lib_smart_ptr_for_overwrite
    auto data_ = std::make_unique_for_overwrite<T[]>(dimension * load_blocksize_);
#else
    auto data_ = std::unique_ptr<T[]>(new T[dimension * load_blocksize_]);
#endif

    Base::operator=(Base{std::move(data_), dimension, load_blocksize_});
  }

  // @todo Allow specification of how many columns to advance by
  bool load() {
    scoped_timer _{tdb_func__ + " " + uri_};

    const size_t attr_idx{0};
    auto attr = schema_.attribute(attr_idx);

    std::string attr_name = attr.name();
    tiledb_datatype_t attr_type = attr.type();
    if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch: " + std::to_string(attr_type) + " != " +
          std::to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
    }

    size_t dimension = last_row_ - first_row_;
    auto elements_to_load =
        std::min(load_blocksize_, last_col_  - last_resident_col_);

    // Return if we're at the end
    if (elements_to_load == 0) {
      return false;
    }

    // These calls change the current view
    first_resident_col_ = last_resident_col_;
    last_resident_col_ += elements_to_load;

    assert(last_resident_col_ != first_resident_col_);

    // Create a subarray for the next block of columns
    tiledb::Subarray subarray(ctx_, *array_);
    subarray.add_range(0, 0, (int)dimension - 1);
    subarray.add_range(
        1, (int)first_resident_col_, (int)last_resident_col_ - 1);

    auto layout_order = schema_.cell_order();

    // Create a query
    tiledb::Query query(ctx_, *array_);
    query.set_subarray(subarray)
        .set_layout(layout_order)
        .set_data_buffer(attr_name, this->data(), elements_to_load * dimension);
    tiledb_helpers::submit_query(tdb_func__, uri_, query);
    _memory_data.insert_entry(tdb_func__, elements_to_load * dimension * sizeof(T));

    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error("Query status is not complete -- fix me");
    }

    num_loads_++;
    return true;
  }
#endif

  index_type col_offset() const {
    return first_resident_col_;
  }

  index_type num_loads() const {
    return num_loads_;
  }

// We don't seem to need this?  It wouldn't work with load() afaict.
#if 0
  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t row_begin,
      size_t row_end,
      size_t col_begin,
      size_t col_end,
      uint64_t timestamp = 0)
      : tdbBlockedMatrix(
            ctx,
            uri,
            row_begin,
            row_end,
            col_begin,
            col_end,
            (timestamp == 0) ?
                tiledb::TemporalPolicy() :
                tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp)) {
  }

  /**
   * @brief General constructor.  Read a view of the array, delimited by the
   * given row and column indices.
   *
   * @param uri
   * @param row_begin
   * @param row_end
   * @param col_begin
   * @param col_end
   *
   * @todo Make this compatible with various schemas we are using
   */
  tdbBlockedMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t row_begin,
      size_t row_end,
      size_t col_begin,
      size_t col_end,
      const tiledb::TemporalPolicy temporal_policy = {})  // noexcept
      : ctx_{ctx}
      , uri_{uri}
      //      ,
      //      array_(std::make_unique<tiledb::Array>(tiledb_helpers::open_array(
      //            tdb_func__, ctx, uri, TILEDB_READ, temporal_policy)))
      , array_(std::make_unique<tiledb::Array>(
            ctx, uri, TILEDB_READ, temporal_policy))
      , schema_{array_->schema()} {
    constructor_timer.stop();
    scoped_timer _{tdb_func__ + uri};

    auto cell_order = schema_.cell_order();
    auto tile_order = schema_.tile_order();

    // @todo Maybe throw an exception here?  Have to properly handle since this
    // is a constructor.
    assert(cell_order == tile_order);

    const size_t attr_idx = 0;

    auto domain_{schema_.domain()};

    auto array_rows_{domain_.dimension(0)};
    auto array_cols_{domain_.dimension(1)};

    if (num_array_rows_ == 0) {
      num_array_rows_ =
          (array_rows_.template domain<row_domain_type>().second -
           array_rows_.template domain<row_domain_type>().first + 1);
    }
    if (num_array_cols_ == 0) {
      num_array_cols_ =
          (array_cols_.template domain<col_domain_type>().second -
           array_cols_.template domain<col_domain_type>().first + 1);
    }

    if ((matrix_order_ == TILEDB_ROW_MAJOR && cell_order == TILEDB_COL_MAJOR) ||
        (matrix_order_ == TILEDB_COL_MAJOR && cell_order == TILEDB_ROW_MAJOR)) {
      std::swap(row_begin, col_begin);
      std::swap(row_end, col_end);
    }

    if (row_begin == 0 && row_end == 0) {
      row_end = num_array_rows_;
    }
    if (col_begin == 0 && col_end == 0) {
      col_end = num_array_cols_;
    }

    std::get<0>(row_view_) = row_begin;
    std::get<1>(row_view_) = row_end;
    std::get<0>(col_view_) = col_begin;
    std::get<1>(col_view_) = col_end;
    row_offset_ = row_begin;
    col_offset_ = col_begin;

    auto num_rows = row_end - row_begin;
    auto num_cols = col_end - col_begin;

#ifdef __cpp_lib_smart_ptr_for_overwrite
    auto data_ = std::make_unique_for_overwrite<T[]>(num_rows * num_cols);
#else
    // auto data_ = std::make_unique<T[]>(new T[mat_rows_ * mat_cols_]);
    auto data_ = std::unique_ptr<T[]>(new T[num_rows * num_cols]);
#endif

    auto attr_num{schema_.attribute_num()};
    auto attr = schema_.attribute(attr_idx);

    std::string attr_name = attr.name();
    tiledb_datatype_t attr_type = attr.type();
    if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch: " + std::to_string(attr_type) + " != " +
          std::to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
    }

    // Create a subarray that reads the array up to the specified subset.
    std::vector<int32_t> subarray_vals = {
        (int32_t)row_begin,
        (int32_t)row_end - 1,
        (int32_t)col_begin,
        (int32_t)col_end - 1};
    tiledb::Subarray subarray(ctx_, *array_);
    subarray.set_subarray(subarray_vals);

    auto layout_order = cell_order;

    tiledb::Query query(ctx_, *array_);

    query.set_subarray(subarray)
        .set_layout(layout_order)
        .set_data_buffer(attr_name, data_.get(), num_rows * num_cols);
    tiledb_helpers::submit_query(tdb_func__, uri, query);
    _memory_data.insert_entry(tdb_func__, num_rows * num_cols * sizeof(T));

    // assert(tiledb::Query::Status::COMPLETE == query.query_status());
    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error("Query status is not complete -- fix me");
    }

    if ((matrix_order_ == TILEDB_ROW_MAJOR && cell_order == TILEDB_COL_MAJOR) ||
        (matrix_order_ == TILEDB_COL_MAJOR && cell_order == TILEDB_ROW_MAJOR)) {
      std::swap(num_rows, num_cols);
    }

    Base::operator=(Base{std::move(data_), num_rows, num_cols});
  }
#endif
};  // tdbBlockedMatrix

template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class tdbPreLoadMatrix : public tdbBlockedMatrix<T, LayoutPolicy, I> {
  using Base = tdbBlockedMatrix<T, LayoutPolicy, I>;
  // This just about did me in.
  // using Base::Base;

 public:
  /**
   * @brief Construct a new tdbBlockedMatrix object, limited to `upper_bound`
   * vectors. In this case, the `Matrix` is column-major, so the number of
   * vectors is the number of columns.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param upper_bound The maximum number of vectors to read.
   */
  tdbPreLoadMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t upper_bound = 0,
      uint64_t timestamp = 0)
      : tdbPreLoadMatrix(ctx, uri, 0, 0, upper_bound, timestamp) {
  }

  tdbPreLoadMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_array_rows,
      size_t num_array_cols,
      size_t upper_bound = 0,
      uint64_t timestamp = 0)
      : Base(ctx, uri, 0, num_array_rows, 0, num_array_cols, upper_bound, timestamp) {
    Base::load();
  }

  tdbPreLoadMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t upper_bound,
      const tiledb::TemporalPolicy& temporal_policy)
      : tdbPreLoadMatrix(ctx, uri, 0, 0, upper_bound, temporal_policy) {
  }

  tdbPreLoadMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_array_rows,
      size_t num_array_cols,
      size_t upper_bound,
      const tiledb::TemporalPolicy& temporal_policy)
      : Base(
            ctx,
            uri,
            num_array_rows,
            num_array_cols,
            upper_bound,
            temporal_policy) {
    Base::load();
  }
};

/**
 * Convenience class for row-major blocked matrices.
 */
template <class T, class I = size_t>
using tdbRowMajorBlockedMatrix = tdbBlockedMatrix<T, stdx::layout_right, I>;

/**
 * Convenience class for column-major blockef matrices.
 */
template <class T, class I = size_t>
using tdbColMajorBlockedMatrix = tdbBlockedMatrix<T, stdx::layout_left, I>;

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t>
using tdbRowMajorMatrix = tdbBlockedMatrix<T, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t>
using tdbColMajorMatrix = tdbBlockedMatrix<T, stdx::layout_left, I>;

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t>
using tdbRowMajorPreLoadMatrix = tdbPreLoadMatrix<T, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t>
using tdbColMajorPreLoadMatrix = tdbPreLoadMatrix<T, stdx::layout_left, I>;

/**
 * Convenience class for row-major matrices.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
using tdbMatrix = tdbBlockedMatrix<T, LayoutPolicy, I>;

#endif  // TDB_MATRIX_H
