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
 * Class that provides a matrix interface to two TileDB arrays, one containing
 * vectors and one IDs.
 *
 * @todo Include the right headers for BLAS support.
 * @todo Refactor ala tdb_partitioned_matrix.h
 *
 */

#ifndef TDB_MATRIX_WITH_IDS_H
#define TDB_MATRIX_WITH_IDS_H

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
class tdbBlockedMatrixWithIds
    : public tdbBlockedMatrix<
          T,
          LayoutPolicy,
          I,
          MatrixWithIds<T, IdsType, LayoutPolicy, I>> {
  using Base = tdbBlockedMatrix<
      T,
      LayoutPolicy,
      I,
      MatrixWithIds<T, IdsType, LayoutPolicy, I>>;
  using Base::Base;

 public:
  using index_type = typename Base::index_type;
  using ids_type = typename Base::ids_type;

 private:
  log_timer constructor_timer{"tdbBlockedMatrixWithIds constructor"};

  std::string ids_uri_;
  std::unique_ptr<tiledb::Array> ids_array_;
  tiledb::ArraySchema ids_schema_;

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
      : tdbBlockedMatrixWithIds(
            ctx, uri, ids_uri, 0, std::nullopt, 0, std::nullopt, 0, {}) {
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
      TemporalPolicy temporal_policy = {})
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : tdbBlockedMatrixWithIds(
            ctx,
            uri,
            ids_uri,
            0,
            std::nullopt,
            0,
            std::nullopt,
            upper_bound,
            temporal_policy) {
  }

  /** General constructor */
  tdbBlockedMatrixWithIds(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri,
      size_t first_row,
      std::optional<size_t> last_row,
      size_t first_col,
      std::optional<size_t> last_col,
      size_t upper_bound,
      TemporalPolicy temporal_policy)  // noexcept
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
            ctx,
            ids_uri,
            TILEDB_READ,
            temporal_policy.to_tiledb_temporal_policy()))
      , ids_schema_{ids_array_->schema()} {
    constructor_timer.stop();
  }

  // @todo Allow specification of how many columns to advance by
  bool load() {
    scoped_timer _{tdb_func__ + " " + this->ids_uri_};
    if (!Base::load()) {
      return false;
    }

    const size_t attr_idx{0};
    auto attr = ids_schema_.attribute(attr_idx);

    std::string attr_name = attr.name();
    tiledb_datatype_t attr_type = attr.type();
    if (attr_type != tiledb::impl::type_to_tiledb<ids_type>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch with IDs: " + datatype_to_string(attr_type) +
          " != " +
          datatype_to_string(
              tiledb::impl::type_to_tiledb<ids_type>::tiledb_type));
    }

    static const size_t dimension = 1;
    // In the Base::load() we will have already computed the number of elements
    // to load, and because returned true from there we should have a positive
    // number of elements to load.
    auto elements_to_load =
        this->last_resident_col_ - this->first_resident_col_;
    if (elements_to_load <= 0) {
      throw std::runtime_error(
          "Error computing IDs to load: " + std::to_string(elements_to_load));
    }

    // Create a subarray for the next block of columns
    tiledb::Subarray subarray(this->ctx_, *ids_array_);
    subarray.add_range(
        0, (int)this->first_resident_col_, (int)this->last_resident_col_ - 1);

    auto layout_order = ids_schema_.cell_order();

    // Create a query
    tiledb::Query query(this->ctx_, *ids_array_);
    query.set_subarray(subarray)
        .set_layout(layout_order)
        .set_data_buffer(attr_name, this->ids(), elements_to_load * dimension);
    tiledb_helpers::submit_query(tdb_func__, ids_uri_, query);
    _memory_data.insert_entry(
        tdb_func__, elements_to_load * dimension * sizeof(T));
    // @todo Handle incomplete queries.
    if (tiledb::Query::Status::COMPLETE != query.query_status()) {
      throw std::runtime_error("Query status for IDs is not complete");
    }

    return true;
  }
};  // tdbBlockedMatrixWithIds

template <
    class T,
    class IdsType = uint64_t,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
class tdbPreLoadMatrixWithIds
    : public tdbBlockedMatrixWithIds<T, IdsType, LayoutPolicy, I> {
  using Base = tdbBlockedMatrixWithIds<T, IdsType, LayoutPolicy, I>;

 public:
  /**
   * @brief Construct a new tdbBlockedMatrixWithIds object, limited to
   * `upper_bound` vectors. In this case, the `Matrix` is column-major, so the
   * number of vectors is the number of columns.
   *
   * @param ctx The TileDB context to use.
   * @param uri URI of the TileDB array to read.
   * @param upper_bound The maximum number of vectors to read.
   */
  tdbPreLoadMatrixWithIds(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri,
      size_t upper_bound = 0,
      TemporalPolicy temporal_policy = {})
      : tdbPreLoadMatrixWithIds(
            ctx,
            uri,
            ids_uri,
            std::nullopt,
            std::nullopt,
            upper_bound,
            temporal_policy) {
  }

  tdbPreLoadMatrixWithIds(
      const tiledb::Context& ctx,
      const std::string& uri,
      const std::string& ids_uri,
      std::optional<size_t> num_array_rows,
      std::optional<size_t> num_array_cols,
      size_t upper_bound = 0,
      TemporalPolicy temporal_policy = {})
      : Base(
            ctx,
            uri,
            ids_uri,
            0,
            num_array_rows,
            0,
            num_array_cols,
            upper_bound,
            temporal_policy) {
    Base::load();
  }
};

/**
 * Convenience class for row-major blocked matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using tdbRowMajorBlockedMatrixWithIds =
    tdbBlockedMatrixWithIds<T, IdsType, stdx::layout_right, I>;

/**
 * Convenience class for column-major blocked matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using tdbColMajorBlockedMatrixWithIds =
    tdbBlockedMatrixWithIds<T, IdsType, stdx::layout_left, I>;

/**
 * Convenience class for row-major matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using tdbRowMajorMatrixWithIds =
    tdbBlockedMatrixWithIds<T, IdsType, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using tdbColMajorMatrixWithIds =
    tdbBlockedMatrixWithIds<T, IdsType, stdx::layout_left, I>;

/**
 * Convenience class for row-major matrices.
 */
template <
    class T,
    class IdsType = uint64_t,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
using tdbMatrixWithIds = tdbBlockedMatrixWithIds<T, IdsType, LayoutPolicy, I>;

/**
 * Convenience class for row-major matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using tdbRowMajorPreLoadMatrixWithIds =
    tdbPreLoadMatrixWithIds<T, IdsType, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using tdbColMajorPreLoadMatrixWithIds =
    tdbPreLoadMatrixWithIds<T, IdsType, stdx::layout_left, I>;

#endif  // TDB_MATRIX_WITH_IDS_H
