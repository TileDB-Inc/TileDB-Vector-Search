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
#include <version>

#include <tiledb/tiledb>
#include "mdspan/mdspan.hpp"

#include "tdb_defs.h"

#include "utils/timer.h"

namespace stdx {
using namespace Kokkos;
// using namespace Kokkos::Experimental;
}  // namespace stdx

/**
 *
 * @note The template parameters indices_type and parts_type can be deduced
 * usingCTAD.  However, with the uri-based constructor, the type of the indices
 * and the partitioned_vectors array cannot be deduced.  Therefore, the user
 * must specify the type of the indices and the partitioned_ids array.  And
 * since CTAD is all or nothing, we have to pass in all of the types.
 */
template <
    class T,
    class IdType,
    class IndicesType,
    class PartsType,
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
  using value_type = typename Base::value_type;  // should be same as T
  using typename Base::index_type;
  using typename Base::reference;
  using typename Base::size_type;

  using id_type = IdType;
  using indices_type = IndicesType;
  using parts_type = PartsType;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  using row_domain_type = int32_t;
  using col_domain_type = int32_t;

    // std::reference_wrapper<const tiledb::Context> ctx_;
  tiledb::Context ctx_;

  std::string partitioned_vectors_uri_;
  tiledb::Array partitioned_vectors_array_;
  tiledb::ArraySchema partitioned_vectors_schema_;

  size_t num_array_rows_{0};
  size_t num_array_cols_{0};

  std::string partitioned_ids_uri_;
  tiledb::Array ids_array_;
  tiledb::ArraySchema ids_schema_;

  // Index for the resident partition
  std::vector<indices_type> indices_;       // @todo pointer and span?

  // ids for the vectors in the resident partition
  std::vector<id_type> ids_;                // @todo pointer and span?

  /*****************************************************************************
   * Partition information
   ****************************************************************************/

  // Vector of all the partitions that we want to load
  std::vector<parts_type> relevant_parts_;  // @todo pointer and span?

  // The total number of partitions in the partitioned array (resident plus
  // non-resident)
  size_t total_num_parts_{0};

  // The number of resident partitions
  size_t num_resident_parts_{0};

  // The offset of the first partitions in the resident vectors
  // Should be equal to first element of part_view_
  index_type resident_part_offset_{0};

  // The initial and final partition number of the resident partitions
  std::tuple<index_type, index_type> resident_part_view_;

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
  std::tuple<index_type, index_type> resident_col_view_;

  /*****************************************************************************
   * Accounting information
   ****************************************************************************/
  size_t max_part_size_{0};
  size_t num_loads_{0};

 public:

  tdbPartitionedMatrix& operator=(tdbPartitionedMatrix&&) = default;
  // tdbPartitionedMatrix() = default;

  /**
   * Constructor from group uri
   * @param ctx
   * @param group_uri
   *
   * @todo WIP
   */
  tdbPartitionedMatrix(const tiledb::Context& ctx,
                       const std::string& group_uri);

  /**
   * @brief Constructor for loading the entire partitioned array.  It sets up
   * the relevant partitions vector to include all partitions (partition
   * numbers from 0 to num_parts - 1), sets upper_bound to be zero, and
   * invokes the main constructor.
   *
   * @param ctx
   * @param partitioned_vectors_uri
   * @param indices_uri
   * @param ids_uri
   * @param num_parts
   */
    tdbPartitionedMatrix (
      const tiledb::Context& ctx,
      const std::string& partitioned_vectors_uri,
      const std::string& indices_uri,
      const std::string& ids_uri,
      size_t num_parts) {
    auto relevant_parts = Vector<parts_type>(num_parts);
    std::iota(begin(relevant_parts), end(relevant_parts), 0);
    *this = tdbPartitionedMatrix(
        ctx, partitioned_vectors_uri, indices_uri, ids_uri, relevant_parts, 0);
  }


   /*
   * @todo Column major is kind of baked in here.  Need to generalize.
   */

  /**
   * @brief Main constructor.  Reads in vectors from a partitioned array, as
   * indicated by partition numbers in the relevant_partitions vector.
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
  tdbPartitionedMatrix (
      const tiledb::Context& ctx,
      const std::string& partitioned_vectors_uri,
      const std::string& indices_uri,
      const std::string& ids_uri,
      const P& relevant_parts,
      size_t upper_bound)
      : partitioned_vectors_uri_{partitioned_vectors_uri}
      , ctx_{ctx}
      , partitioned_vectors_array_{tiledb_helpers::open_array(tdb_func__, ctx_, partitioned_vectors_uri_, TILEDB_READ)}
      , partitioned_vectors_schema_{partitioned_vectors_array_.schema()}
      , ids_array_{tiledb_helpers::open_array(
            tdb_func__, ctx_, ids_uri, TILEDB_READ)}
      , ids_schema_{ids_array_.schema()}
      , indices_{read_vector<indices_type>(ctx_, indices_uri)}
      , relevant_parts_{relevant_parts}
      , resident_part_view_{0, 0} {

    total_num_parts_ = size(relevant_parts_);

    scoped_timer _{tdb_func__ + " " + partitioned_vectors_uri_};

    auto cell_order = partitioned_vectors_schema_.cell_order();
    auto tile_order = partitioned_vectors_schema_.tile_order();

    // @todo Maybe throw an exception here?  Have to properly handle since
    // this is a constructor.
    assert(cell_order == tile_order);

    auto domain_{partitioned_vectors_schema_.domain()};

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
      total_max_cols += indices_[relevant_parts_[i] + 1] - indices_[relevant_parts_[i]];
      max_part_size_ = std::max<size_t>(
          max_part_size_, indices_[relevant_parts_[i] + 1] - indices_[relevant_parts_[i]]);
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

    // @todo be more sensible -- dont use a vector and don't use inout
    if (size(ids_) < column_capacity) {
      ids_.resize(column_capacity);
    }

#ifdef __cpp_lib_smart_ptr_for_overwrite
    auto data_ = std::make_unique_for_overwrite<T[]>(dimension * column_capacity);
#else
    auto data_ = std::unique_ptr<T[]>(new T[dimension * column_capacity]);
#endif

    Base::operator=(Base{std::move(data_), dimension, column_capacity});
  }

  /**
   * Read in the next partitions
   * todo Allow to specify how many columns to read in
   */
  bool load() {
    scoped_timer _{tdb_func__ + " " + partitioned_vectors_uri_};

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
      std::get<0>(resident_col_view_) = std::get<1>(resident_col_view_);  // # columns
      std::get<0>(resident_part_view_) =
          std::get<1>(resident_part_view_);  // # partitions

      std::get<1>(resident_part_view_) = std::get<0>(resident_part_view_);
      for (size_t i = std::get<0>(resident_part_view_); i < total_num_parts_; ++i) {
        auto next_part_size = indices_[relevant_parts_[i] + 1] - indices_[relevant_parts_[i]];
        if ((std::get<1>(resident_col_view_) + next_part_size) >
            std::get<0>(resident_col_view_) + column_capacity) {
          break;
        }
        std::get<1>(resident_col_view_) += next_part_size;  // FIXME ??
        std::get<1>(resident_part_view_) = i + 1;
      }
      num_resident_cols_ = std::get<1>(resident_col_view_) - std::get<0>(resident_col_view_);
      resident_col_offset_ = std::get<0>(resident_col_view_);

      assert(num_resident_cols_ <= column_capacity);

      num_resident_parts_ =
          std::get<1>(resident_part_view_) - std::get<0>(resident_part_view_);
      resident_part_offset_ = std::get<0>(resident_part_view_);

      if ((num_resident_cols_ == 0 && num_resident_parts_ != 0) ||
          (num_resident_cols_ != 0 && num_resident_parts_ == 0)) {
        throw std::runtime_error("Invalid partitioning");
      }
      if (num_resident_cols_ == 0) {
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
      for (size_t j = std::get<0>(resident_part_view_);
           j < std::get<1>(resident_part_view_);
           ++j) {
        size_t start = indices_[relevant_parts_[j]];
        size_t stop = indices_[relevant_parts_[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        col_count += len;
        subarray.add_range(1, (int)start, (int)stop - 1);
      }
      if (col_count != std::get<1>(resident_col_view_) - std::get<0>(resident_col_view_)) {
        throw std::runtime_error("Column count mismatch");
      }

      auto cell_order = partitioned_vectors_schema_.cell_order();
      auto layout_order = cell_order;

      tiledb::Query query(ctx_, this->array_);

      // auto ptr = data_.get();

      auto ptr = this->data();
      query.set_subarray(subarray)
          .set_layout(layout_order)
          .set_data_buffer(attr_name, ptr, col_count * dimension);
      tiledb_helpers::submit_query(tdb_func__, partitioned_vectors_uri_, query);
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
      for (size_t j = std::get<0>(resident_part_view_);
           j < std::get<1>(resident_part_view_);
           ++j) {
        size_t start = indices_[relevant_parts_[j]];
        size_t stop = indices_[relevant_parts_[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        ids_col_count += len;
        ids_subarray.add_range(0, (int)start, (int)stop - 1);
      }
      if (ids_col_count != std::get<1>(resident_col_view_) - std::get<0>(resident_col_view_)) {
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

  auto& vectors() const {
    return *this;
  }

  auto& indices() const {
    return indices_;
  }

  auto& ids() const {
    return ids_;
  }

  index_type num_col_parts() const {
    return std::get<1>(resident_part_view_) - std::get<0>(resident_part_view_);
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

  /**
   * Destructor.  Closes arrays if they are open.
   */
  ~tdbPartitionedMatrix() {
    if (partitioned_vectors_array_.is_open()) {
      partitioned_vectors_array_.close();
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
    class partitioned_ids_type,
    class indices_type,
    class parts_type,
    class I = size_t>
using tdbRowMajorPartitionedMatrix = tdbPartitionedMatrix<
    T,
    partitioned_ids_type,
    indices_type,
    parts_type,
    stdx::layout_right,
    I>;

/**
 * Convenience class for column-major matrices.
 */
template <
    class T,
    class partitioned_ids_type,
    class indices_type,
    class parts_type,
    class I = size_t>
using tdbColMajorPartitionedMatrix = tdbPartitionedMatrix<
    T,
    partitioned_ids_type,
    indices_type,
    parts_type,
    stdx::layout_left,
    I>;

#endif  // TILEDB_PARTITIONED_MATRIX_H
