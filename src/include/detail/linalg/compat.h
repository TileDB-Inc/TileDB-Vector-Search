/**
 * @file   compat.h
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
 * Classes for non-owning wrappers of otherwise owned data for use with
 * ivf_flat_index based queries. Provides compatability for current Python
 * bindings (which want to own their data).
 *
 */

#ifndef TILEDB_COMPAT_H
#define TILEDB_COMPAT_H

// ColMajorPartitionedMatrix: qv_query_heap_infinite_ram,
// nuv_query_heap_infinite_ram tdbColMajorPartitionedMatrix:
// nuv_query_heap_finite_ram -- should be okay -- called with uris

#include <cstddef>
#include "detail/linalg/matrix.h"

/**
 * @brief Partitioned matrix wrapper class.
 * @tparam T
 * @tparam IdType
 * @tparam PartIndexType
 * @tparam LayoutPolicy
 * @tparam I
 *
 * Note:  The Matrix base class may have larger capacity than the vectors
 * being stored (similarly ids_ and part_index_).  The member data num_vectors_
 * and num_parts_ store the number of valid vectors and partitions being
 * stored.
 */
template <
    class T,
    class IdType,
    class PartIndexType,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
class PartitionedMatrixWrapper {
 public:
  using value_type = T;  // should be same as T
  using index_type = PartIndexType;
  using reference = T&;
  using id_type = IdType;

 private:
  using part_index_type = PartIndexType;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 protected:
  // The partitioned vectors
  std::reference_wrapper<Matrix<T, LayoutPolicy, I>>
      parts_;  // @todo pointer and span?

  // ids for the partitioned vectors
  std::reference_wrapper<std::vector<id_type>> ids_;  // @todo pointer and span?

  // Index for partitioned vectors
  std::reference_wrapper<std::vector<part_index_type>>
      part_index_;  // @todo pointer and span?

  // Stores the number of valid vectors being stored
  size_t num_vectors_{0};

  // Stores the number of valid partitions being stored
  size_t num_parts_{0};

 public:
  PartitionedMatrixWrapper() = default;

  PartitionedMatrixWrapper(const PartitionedMatrixWrapper&) = delete;
  PartitionedMatrixWrapper& operator=(const PartitionedMatrixWrapper&) = delete;
  PartitionedMatrixWrapper(PartitionedMatrixWrapper&&) = default;
  PartitionedMatrixWrapper& operator=(PartitionedMatrixWrapper&&) = default;
  virtual ~PartitionedMatrixWrapper() = default;

  // @note Use Vector instead of std::vector
  PartitionedMatrixWrapper(
      Matrix<T, LayoutPolicy, I>& parts,
      std::vector<IdType>& ids,
      std::vector<PartIndexType>& part_index)
      : parts_{parts}
      , ids_{ids}
      , part_index_{part_index}
      , num_vectors_(::num_vectors(parts))
      , num_parts_(part_index.size() - 1) {
  }

  auto operator[](index_type i) const {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&parts_(i, 0), parts_.get().num_cols());
    } else {
      return std::span(&parts_(0, i), parts_.get().num_rows());
    }
  }

  auto operator[](index_type i) {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&parts_(i, 0), parts_.get().num_cols());
    } else {
      return std::span(&parts_(0, i), parts_.get().num_rows());
    }
  }

  auto dimensions() const {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return parts_.get().num_cols();
    } else {
      return parts_.get().num_rows();
    }
  }

  auto dimensions() {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return parts_.get().num_cols();
    } else {
      return parts_.get().num_rows();
    }
  }

  constexpr auto& num_vectors() {
    return num_vectors_;
  }

  constexpr auto num_vectors() const {
    return num_vectors_;
  }

  constexpr auto& num_partitions() const {
    return num_parts_;
  }

  constexpr auto num_partitions() {
    return num_parts_;
  }

  auto& parts() const {
    return parts_.get();
  }

  auto& parts() {
    return parts_.get();
  }

  auto& ids() const {
    return ids_.get();
  }

  auto& ids() {
    return ids_.get();
  }

  auto& indices() const {
    return part_index_.get();
  }

  auto& indices() {
    return part_index_.get();
  }

  virtual bool load() {
    return false;
  }
};

/**
 * Convenience class for row-major matrices.
 */
template <
    class T,
    class partitioned_ids_type,
    class part_index_type,
    class I = size_t>
using RowMajorPartitionedMatrixWrapper = PartitionedMatrixWrapper<
    T,
    partitioned_ids_type,
    part_index_type,
    stdx::layout_right,
    I>;

/**
 * Convenience class for column-major matrices.
 */
template <
    class T,
    class partitioned_ids_type,
    class part_index_type,
    class I = size_t>
using ColMajorPartitionedMatrixWrapper = PartitionedMatrixWrapper<
    T,
    partitioned_ids_type,
    part_index_type,
    stdx::layout_left,
    I>;

#endif  // TILEDB_COMPAT_H
