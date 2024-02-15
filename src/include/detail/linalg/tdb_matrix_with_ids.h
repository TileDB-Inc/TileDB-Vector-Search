/**
 * @file   tdb_matrix_with_ids.h
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

#ifndef TDB_MATRIX_WITH_IDS_H
#define TDB_MATRIX_WITH_IDS_H

// #include <future>
//
// #include <tiledb/tiledb>
//
// #include "detail/linalg/linalg_defs.h"
#include "detail/linalg/matrix_with_ids.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "tdb_defs.h"

// template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
// class tdbBlockedMatrixBase : public Matrix<T, LayoutPolicy, I> {};

// template <class T, class LayoutPolicy, class I>
// class tdbBlockedMatrixBase<T, LayoutPolicy, I, true> : public
// MatrixWithIds<T, LayoutPolicy, I> {};

/**
 * Derived from `tdbBlockedMatrix`, which we have derive from `MatrixWithIds`.
 * Initialized in construction by filling from a given TileDB array.
 */
template <
    class T,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t,
    class IdsType = size_t>
class tdbBlockedMatrixWithIds
    : public tdbBlockedMatrix<
          T,
          LayoutPolicy,
          I,
          MatrixWithIds<T, LayoutPolicy, I, IdsType>> {
  using Base = tdbBlockedMatrix<
      T,
      LayoutPolicy,
      I,
      MatrixWithIds<T, LayoutPolicy, I, IdsType>>;
  using Base::Base;

 public:
  // using value_type = typename Base::value_type;
  using index_type = typename Base::index_type;
  // using size_type = typename Base::size_type;
  // using reference = typename Base::reference;

  // using view_type = Base;

  // constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  // using row_domain_type = int32_t;
  // using col_domain_type = int32_t;

  log_timer constructor_timer{"tdbBlockedMatrixWithIds constructor"};

  // std::reference_wrapper<const tiledb::Context> ctx_;
  // std::string uri_;
  // std::unique_ptr<tiledb::Array> array_;
  // tiledb::ArraySchema schema_;

  std::string ids_uri_;
  std::unique_ptr<tiledb::Array> ids_array_;
  tiledb::ArraySchema ids_schema_;

  //  index_type first_ids_index_;
  //  index_type last_ids_index_;
  // index_type first_row_;
  // index_type last_row_;
  // index_type first_col_;
  // index_type last_col_;
  // index_type first_resident_col_;
  // index_type last_resident_col_;

  // The number of columns loaded into memory.  Except for the last (remainder)
  // block, this will be equal to `blocksize_`.
  // index_type num_resident_cols_{0};

  // How many columns to load at a time
  // index_type load_blocksize_{0};

  // How many blocks we have loaded
  // size_t num_loads_{0};

 public:
  tdbBlockedMatrixWithIds(tdbBlockedMatrixWithIds&& rhs) = default;

  /** Default destructor. array will be closed when it goes out of scope */
  virtual ~tdbBlockedMatrixWithIds() = default;

  /**
   * @brief Construct a new tdbBlockedMatrixWithIds object, limited to
   * `upper_bound` vectors. In this case, the `Matrix` is row-major, so the
   * number of vectors is the number of rows.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param ids_uri URI of the TileDB array of IDs to read.
   */
  tdbBlockedMatrixWithIds(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrixWithIds(ctx, uri, ids_uri, 0, 0, 0, 0, 0, 0) {
  }

  /**
   * @brief Construct a new tdbBlockedMatrixWithIds object, limited to
   * `upper_bound` vectors. In this case, the `Matrix` is column-major, so the
   * number of vectors is the number of columns.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param ids_uri URI of the TileDB array of IDs to read.
   * @param upper_bound The maximum number of vectors to read.
   * @param temporal_policy The TemporalPolicy to use for reading the array
   * data.
   */
  tdbBlockedMatrixWithIds(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri,
      size_t upper_bound,
      size_t timestamp = 0)
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrixWithIds(
            ctx, uri, ids_uri, 0, 0, 0, 0, upper_bound, timestamp) {
  }

  /** General constructor */
  tdbBlockedMatrixWithIds(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri,
      size_t first_row,
      size_t last_row,
      size_t first_col,
      size_t last_col,
      size_t upper_bound,
      size_t timestamp)
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrixWithIds(
            ctx,
            uri,
            ids_uri,
            first_row,
            last_row,
            first_col,
            last_col,
            upper_bound,
            (timestamp == 0 ?
                 tiledb::TemporalPolicy() :
                 tiledb::TemporalPolicy(tiledb::TimeTravel, timestamp))) {
  }

  /** General constructor */
  tdbBlockedMatrixWithIds(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri,
      size_t first_row,
      size_t last_row,
      size_t first_col,
      size_t last_col,
      size_t upper_bound,
      tiledb::TemporalPolicy temporal_policy)  // noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Base(
            ctx,
            uri,
            first_row,
            last_row,
            first_col,
            last_col,
            upper_bound,
            temporal_policy)
      , ids_uri_(ids_uri)
      , ids_array_(std::make_unique<tiledb::Array>(
            ctx, ids_uri, TILEDB_READ, temporal_policy))
      , ids_schema_{ids_array_->schema()} {
    constructor_timer.stop();
    //     scoped_timer _{tdb_func__ + " " + uri};

    //     size_t dimension = last_row_ - first_row_;
    // #ifdef __cpp_lib_smart_ptr_for_overwrite
    //       auto data_ =
    //           std::make_unique_for_overwrite<T[]>(dimension *
    //           load_blocksize_);
    //       auto ids_data_ =
    //           std::make_unique_for_overwrite<T[]>(load_blocksize_);
    // #else
    //       auto data_ = std::unique_ptr<T[]>(new T[dimension *
    //       load_blocksize_]); auto ids_data_ = std::unique_ptr<T[]>(new
    //       T[load_blocksize_]);
    // #endif
    //       Base::operator=(Base{std::move(data_), std::move(ids_data_),
    //       dimension, load_blocksize_});
  }

  // @todo Allow specification of how many columns to advance by
  bool load() {
    scoped_timer _{tdb_func__ + " " + this->ids_uri_};
    if (!Base::load()) {
      return false;
    }
    std::cout << "[tdb_matrix_with_ids] finished with base load()" << std::endl;

    const size_t attr_idx{0};
    auto attr = ids_schema_.attribute(attr_idx);

    std::string attr_name = attr.name();
    tiledb_datatype_t attr_type = attr.type();
    if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch with IDs: " + datatype_to_string(attr_type) +
          " != " +
          datatype_to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
    }

    // std::string ids_uri_;
    // std::unique_ptr<tiledb::Array> ids_array_;
    // tiledb::ArraySchema ids_schema_;

    // size_t dimension = 1;
    // auto elements_to_load =
    //     std::min(load_blocksize_, last_col_ - last_resident_col_);

    size_t dimension = 1;
    // In the Base::load() we will have already computed the number of elements
    // to load, and because returned true from there we should have a positive
    // number of elements to load.
    auto elements_to_load =
        this->last_resident_col_ - this->first_resident_col_;
    if (elements_to_load <= 0) {
      throw std::runtime_error(
          "Error computing IDs to load: " + std::to_string(elements_to_load));
    }

    // // Return if we're at the end
    // if (elements_to_load == 0) {
    //   return false;
    // }

    // // These calls change the current view
    // first_resident_col_ = last_resident_col_;
    // last_resident_col_ += elements_to_load;

    // assert(last_resident_col_ != first_resident_col_);

    // Create a subarray for the next block of columns
    tiledb::Subarray subarray(this->ctx_, *ids_array_);
    subarray.add_range(0, 0, (int)dimension - 1);
    subarray.add_range(
        1, (int)this->first_resident_col_, (int)this->last_resident_col_ - 1);

    auto layout_order = ids_schema_.cell_order();

    // Create a query
    tiledb::Query query(this->ctx_, *ids_array_);
    query.set_subarray(subarray)
        .set_layout(layout_order)
        .set_data_buffer(attr_name, this->ids(), elements_to_load * dimension);
    tiledb_helpers::submit_query(tdb_func__, ids_uri_, query);
    _memory_data.insert_entry(
        tdb_func__, elements_to_load * dimension * sizeof(T));

    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error("Query status for IDs is not complete");
    }

    // num_loads_++;
    return true;
  }

  // index_type col_offset() const {
  //   return first_resident_col_;
  // }

  // index_type num_loads() const {
  //   return num_loads_;
  // }

};  // tdbBlockedMatrix

/**
 * Convenience class for row-major blocked matrices.
 */
template <class T, class I = size_t, class IdsType = size_t>
using tdbRowMajorBlockedMatrixWithIds =
    tdbBlockedMatrixWithIds<T, stdx::layout_right, I, IdsType>;

/**
 * Convenience class for column-major blockef matrices.
 */
template <class T, class I = size_t, class IdsType = size_t>
using tdbColMajorBlockedMatrixWithIds =
    tdbBlockedMatrixWithIds<T, stdx::layout_left, I, IdsType>;

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t, class IdsType = size_t>
using tdbRowMajorMatrixWithIds =
    tdbBlockedMatrixWithIds<T, stdx::layout_right, I, IdsType>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t, class IdsType = size_t>
using tdbColMajorMatrixWithIds =
    tdbBlockedMatrixWithIds<T, stdx::layout_left, I, IdsType>;

/**
 * Convenience class for row-major matrices.
 */
template <
    class T,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t,
    class IdsType = size_t>
using tdbMatrixWithIds = tdbBlockedMatrixWithIds<T, LayoutPolicy, I, IdsType>;

#endif  // TDB_MATRIX_WITH_IDS_H
