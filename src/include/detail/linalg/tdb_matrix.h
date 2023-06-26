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
#include "detail/linalg/tdb_defs.h"

/**
 * Derived from `Matrix`.  Initialized in construction by filling from a given
 * TileDB array.
 *
 * @todo Evaluate whether or not we really need to do things this way or if
 * it is sufficient to simply have one Matrix class and have a factory that
 * creates them by reading from TileDB.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class tdbMatrix : public Matrix<T, LayoutPolicy, I> {
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
  log_timer constructor_timer{"tdbMatrix constructor"};

  std::reference_wrapper<const tiledb::Context> ctx_;
  tiledb::Array array_;
  tiledb::ArraySchema schema_;
  std::unique_ptr<T[]> backing_data_;
  size_t num_array_rows_{0};
  size_t num_array_cols_{0};

  std::tuple<index_type, index_type> row_view_;
  std::tuple<index_type, index_type> col_view_;
  index_type row_offset_{0};
  index_type col_offset_{0};

  std::future<bool> fut_;
  size_t pending_row_offset{0};
  size_t pending_col_offset{0};

 public:
  /**
   * @brief Construct a new tdbMatrix object, limited to `num_elts` vectors.
   * In this case, the `Matrix` is row-major, so the number of vectors is
   * the number of rows.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param num_elts Number of vectors to read from the array.
   */
  tdbMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_elts) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
      : tdbMatrix(ctx, uri, num_elts, 0) {
  }

  /**
   * @brief Construct a new tdbMatrix object, limited to `num_elts` vectors.
   * In this case, the `Matrix` is column-major, so the number of vectors is
   * the number of columns.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param num_elts Number of vectors to read from the array.
   */
  tdbMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_elts) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbMatrix(ctx, uri, 0, num_elts) {
  }

  /**
   * @brief Construct a new tdbMatrix object, reading all of the vectors in
   * the array.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   */
  explicit tdbMatrix(
      const tiledb::Context& ctx, const std::string& uri) noexcept
      // requires (std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbMatrix(ctx, uri, 0, 0) {
    if (global_debug) {
      std::cerr << "# tdbMatrix constructor: " << uri << std::endl;
    }
  }

  /**
   * @brief Construct a new tdbMatrix object, reading a subset of the vectors
   * and a subset of the elements in each vector.
   *
   * @param ctx
   * @param uri
   * @param num_rows
   * @param num_cols
   */
  tdbMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_rows,
      size_t num_cols) noexcept
      : tdbMatrix(ctx, uri, 0, num_rows, 0, num_cols) {
  }

  /**
   * @brief "Slice" interface.
   * @param ctx The TileDB context to use.
   * @param uri
   * @param rows pair of row indices indicating begin and end of view
   * @param cols pair of column indices indicating begin and end of view
   */
  tdbMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      std::tuple<size_t, size_t> rows,
      std::tuple<size_t, size_t> cols) noexcept
      : tdbMatrix(
            uri,
            std::get<0>(rows),
            std::get<1>(rows),
            std::get<0>(cols),
            std::get<1>(cols)) {
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
  tdbMatrix(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t row_begin,
      size_t row_end,
      size_t col_begin,
      size_t col_end)  // noexcept
      : ctx_{ctx}
      , array_{ctx, uri, TILEDB_READ}
      , schema_{array_.schema()} {
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

    num_array_rows_ =
        (array_rows_.template domain<row_domain_type>().second -
         array_rows_.template domain<row_domain_type>().first + 1);
    num_array_cols_ =
        (array_cols_.template domain<col_domain_type>().second -
         array_cols_.template domain<col_domain_type>().first + 1);

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

#ifndef __APPLE__
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
    tiledb::Subarray subarray(ctx_, array_);
    subarray.set_subarray(subarray_vals);

    auto layout_order = cell_order;

    tiledb::Query query(ctx_, array_);

    query.set_subarray(subarray)
        .set_layout(layout_order)
        .set_data_buffer(attr_name, data_.get(), num_rows * num_cols);
    query.submit();
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

 public:
  /**
   * Gather pieces of a partitioned array into a single array (along with the
   * vector ids into a corresponding 1D array)
   */
  tdbMatrix(
      const std::string& uri,
      std::vector<uint64_t>& indices,
      const std::vector<size_t>& top_top_k,
      const std::string& id_uri,
      std::vector<uint64_t>& shuffled_ids,
      size_t nthreads)
      : array_{ctx_, uri, TILEDB_READ}
      , schema_{array_.schema()} {
    size_t nprobe = size(top_top_k);
    size_t num_cols = 0;
    for (size_t i = 0; i < nprobe; ++i) {
      num_cols += indices[top_top_k[i] + 1] - indices[top_top_k[i]];
    }

    {
      scoped_timer _{"read tdb matrix " + uri};

      auto cell_order = schema_.cell_order();
      auto tile_order = schema_.tile_order();

      // @todo Maybe throw an exception here?  Have to properly handle since
      // this is a constructor.
      assert(cell_order == tile_order);

      const size_t attr_idx = 0;

      auto attr_num{schema_.attribute_num()};
      auto attr = schema_.attribute(attr_idx);

      std::string attr_name = attr.name();
      tiledb_datatype_t attr_type = attr.type();
      if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
        throw std::runtime_error(
            "Attribute type mismatch: " + std::to_string(attr_type) + " != " +
            std::to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
      }

      auto domain_{schema_.domain()};

      auto array_rows_{domain_.dimension(0)};
      auto array_cols_{domain_.dimension(1)};

      num_array_rows_ =
          (array_rows_.template domain<row_domain_type>().second -
           array_rows_.template domain<row_domain_type>().first + 1);
      num_array_cols_ =
          (array_cols_.template domain<col_domain_type>().second -
           array_cols_.template domain<col_domain_type>().first + 1);

      if ((matrix_order_ == TILEDB_ROW_MAJOR &&
           cell_order == TILEDB_COL_MAJOR) ||
          (matrix_order_ == TILEDB_COL_MAJOR &&
           cell_order == TILEDB_ROW_MAJOR)) {
        throw std::runtime_error("Cell order and matrix order must match");
      }

      size_t dimension = num_array_rows_;

#ifndef __APPLE__
      auto data_ = std::make_unique_for_overwrite<T[]>(dimension * num_cols);
#else
      auto data_ = std::unique_ptr<T[]>(new T[dimension * num_cols]);
#endif

      /**
       * Read in the partitions
       */
      size_t offset = 0;
      for (size_t j = 0; j < nprobe; ++j) {
        size_t start = indices[top_top_k[j]];
        size_t stop = indices[top_top_k[j] + 1];
        size_t len = stop - start;
        size_t num_elements = len * dimension;

        // Create a subarray that reads the array up to the specified subset.
        std::vector<int32_t> subarray_vals = {
            (int32_t)0,
            (int32_t)dimension - 1,
            (int32_t)start,
            (int32_t)stop - 1};
        tiledb::Subarray subarray(ctx_, array_);
        subarray.set_subarray(subarray_vals);

        auto layout_order = cell_order;

        tiledb::Query query(ctx_, array_);

        auto ptr = data_.get() + offset;
        query.set_subarray(subarray)
            .set_layout(layout_order)
            .set_data_buffer(attr_name, ptr, num_elements);
        query.submit();
        _memory_data.insert_entry(tdb_func__, num_elements * sizeof(T));

        // assert(tiledb::Query::Status::COMPLETE == query.query_status());
        if (tiledb::Query::Status::COMPLETE != query.query_status()) {
          throw std::runtime_error("Query status is not complete -- fix me");
        }
        offset += len;
      }

      Base::operator=(Base{std::move(data_), dimension, num_cols});
    }

    auto part_ids = std::vector<uint64_t>(num_cols);

    {
      scoped_timer _{"read partitioned vector" + id_uri};
      /**
       * Now deal with ids
       */
      auto attr_idx = 0;

      auto ids_array_ = tiledb::Array{ctx_, id_uri, TILEDB_READ};
      auto ids_schema_ = ids_array_.schema();

      auto attr_num{ids_schema_.attribute_num()};
      auto attr = ids_schema_.attribute(attr_idx);

      std::string attr_name = attr.name();

      auto domain_{ids_schema_.domain()};
      auto array_rows_{domain_.dimension(0)};

      auto total_vec_rows_{
          (array_rows_.template domain<row_domain_type>().second -
           array_rows_.template domain<row_domain_type>().first + 1)};

      size_t offset = 0;
      for (size_t j = 0; j < nprobe; ++j) {
        size_t start = indices[top_top_k[j]];
        size_t stop = indices[top_top_k[j] + 1];
        size_t len = stop - start;
        size_t num_elements = len;

        // Create a subarray that reads the array up to the specified subset.
        std::vector<int32_t> subarray_vals = {
            (int32_t)start, (int32_t)stop - 1};
        tiledb::Subarray subarray(ctx_, ids_array_);
        subarray.set_subarray(subarray_vals);

        tiledb::Query query(ctx_, ids_array_);
        auto ptr = part_ids.data() + offset;
        query.set_subarray(subarray).set_data_buffer(
            attr_name, ptr, num_elements);
        query.submit();
        _memory_data.insert_entry(tdb_func__, num_elements * sizeof(T));

        if (tiledb::Query::Status::COMPLETE != query.query_status()) {
          throw std::runtime_error("Query status is not complete -- fix me");
        }
        offset += len;
      }
      ids_array_.close();
    }
    shuffled_ids = std::move(part_ids);
  }

 private:
  using row_domain_type = int32_t;
  using col_domain_type = int32_t;

 public:
  size_t offset() const
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
  {
    return row_offset_;
  }

  size_t offset() const
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
  {
    return col_offset_;
  }

  ~tdbMatrix() noexcept {
    array_.close();
  }

  bool advance(size_t num_elts = 0)
  // requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
  {
    // std::cout << "tdbMatrix advance" << std::endl;
    // @todo attr_idx, attr_name, and cell_order / layout_order should be
    // members of the class
    size_t attr_idx = 0;
    auto attr = schema_.attribute(attr_idx);
    std::string attr_name = attr.name();
    tiledb_datatype_t attr_type = attr.type();
    auto cell_order = schema_.cell_order();
    auto layout_order = cell_order;

    if (layout_order == TILEDB_ROW_MAJOR) {
      if (num_elts == 0) {
        num_elts = std::get<1>(row_view_) - std::get<0>(row_view_);
      }
      auto num_end_elts = std::min(num_elts, num_array_rows_ - row_offset_);

      if (this->is_async()) {
        pending_row_offset = row_offset_ + num_end_elts;
      } else {
        row_offset_ += num_end_elts;
      }

      std::get<0>(row_view_) += num_elts;
      std::get<1>(row_view_) += num_end_elts;

      if (std::get<0>(row_view_) >= num_array_rows_) {
        return false;
      }
    } else if (layout_order == TILEDB_COL_MAJOR) {
      if (num_elts == 0) {
        num_elts = std::get<1>(col_view_) - std::get<0>(col_view_);
      }
      auto num_end_elts = std::min(num_elts, num_array_cols_ - col_offset_);

      if (this->is_async()) {
        pending_col_offset = col_offset_ + num_end_elts;
      } else {
        col_offset_ += num_end_elts;
      }

      std::get<0>(col_view_) += num_elts;
      std::get<1>(col_view_) += num_end_elts;

      if (std::get<0>(col_view_) >= num_array_cols_) {
        return false;
      }
    } else {
      throw std::runtime_error("Unknown cell order");
    }

    // Create a subarray that reads the array with the specified view
    std::vector<int32_t> subarray_vals = {
        (int32_t)std::get<0>(row_view_),
        (int32_t)std::get<1>(row_view_) - 1,
        (int32_t)std::get<0>(col_view_),
        (int32_t)std::get<1>(col_view_) - 1};
    tiledb::Subarray subarray(ctx_, array_);
    subarray.set_subarray(subarray_vals);

    tiledb::Query query(ctx_, array_);

    auto this_data =
        this->is_async() ? this->backing_data_.get() : this->data();

    size_t read_size = (std::get<1>(row_view_) - std::get<0>(row_view_)) *
                       (std::get<1>(col_view_) - std::get<0>(col_view_));

    query.set_subarray(subarray)
        .set_layout(layout_order)
        .set_data_buffer(attr_name, this_data, read_size);
    query.submit();
    _memory_data.insert_entry(tdb_func__, read_size * sizeof(T));

    // assert(tiledb::Query::Status::COMPLETE == query.query_status());
    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error("Query status is not complete -- fix me");
    }

    return true;
  }

  void advance_async(size_t num_elts = 0) {
    if (!backing_data_) {
#ifndef __APPLE__
      backing_data_ = std::make_unique_for_overwrite<T[]>(
          this->num_rows() * this->num_cols());
#else
      backing_data_ =
          std::unique_ptr<T[]>(new T[this->num_rows() * this->num_cols()]);
#endif
    }
    // this->data_.swap(backing_data_);
    fut_ = std::async(std::launch::async, [this, num_elts]() {
      return this->advance(num_elts);
    });
  }

  void swap_offsets() {
    std::swap(row_offset_, pending_row_offset);
    std::swap(col_offset_, pending_col_offset);
  }

  bool advance_wait(size_t num_elts = 0) {
    bool more{true};
    if (fut_.valid()) {
      more = fut_.get();
    } else {
      throw std::runtime_error("advance_wait: future is not valid");
    }
    if (!more) {
      return false;
    }
    this->storage_.swap(this->backing_data_);
    this->swap_offsets();

    return true;
  }
};

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t>
using tdbRowMajorMatrix = tdbMatrix<T, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t>
using tdbColMajorMatrix = tdbMatrix<T, stdx::layout_left, I>;

#endif  // TDB_MATRIX_H
