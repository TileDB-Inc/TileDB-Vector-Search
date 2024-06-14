/**
 * @file   partitioned_matrix.h
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
 * Partitioned Matrix class.  It comprises a matrix (an array of feature
 * vectors), an array of identifiers (ids), one for each vector, and an
 * array of partition indices (part_index_).  The partition indices indicate
 * partitions are stored contiguously and the ith index value indicates the
 * first element of the ith partition (equivalently, one past the end of the
 * (i-1)th partition.  The last index value is one past the end of the last
 * partition.
 *
 * This format is very similar to the linear algebra compressed sparse row
 * format.
 *
 */

#ifndef PARTITIONED_MATRIX_H
#define PARTITIONED_MATRIX_H

#include <cstddef>
#include "detail/linalg/matrix.h"
#include "detail/linalg/vector.h"

/**
 * @brief Partitioned matrix class.
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
class PartitionedMatrix : public Matrix<T, LayoutPolicy, I> {
  using Base = Matrix<T, LayoutPolicy, I>;
  // using Base::Base;

  // Make these private so they aren't accidentally used for sizing the
  // derived class
  using Base::num_cols;
  using Base::num_rows;

 public:
  using value_type = typename Base::value_type;  // should be same as T
  using typename Base::index_type;
  using typename Base::reference;
  using typename Base::size_type;
  using id_type = IdType;

 private:
  using part_index_type = PartIndexType;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 protected:
  // ids for the partitioned vectors
  std::vector<id_type> ids_;  // @todo pointer and span?

  // Index for partitioned vectors
  std::vector<part_index_type> part_index_;  // @todo pointer and span?

  // Stores the number of valid vectors being stored
  size_t num_vectors_{0};

  // Stores the number of valid partitions being stored
  size_t num_parts_{0};

 public:
  PartitionedMatrix() = default;

  /**
   * @brief Construct a partitioned matrix with a given dimension, maximum
   * number of vectors, and maximum number of partitions.
   *
   * @param dim The dimension of the vectors
   * @param max_num_vecs The maximum number of vectors
   * @param max_num_parts The maximum number of partitions
   */
  PartitionedMatrix(size_t dim, size_t max_num_vecs, size_t max_num_parts)
      : Base(dim, max_num_vecs)
      , ids_(max_num_vecs)
      , part_index_(max_num_parts + 1) {
  }

  PartitionedMatrix(const PartitionedMatrix&) = delete;
  PartitionedMatrix& operator=(const PartitionedMatrix&) = delete;
  PartitionedMatrix(PartitionedMatrix&&) = default;
  PartitionedMatrix& operator=(PartitionedMatrix&&) = default;
  virtual ~PartitionedMatrix() = default;

  // @note Use Vector instead of std::vector
  PartitionedMatrix(
      Matrix<T, LayoutPolicy, I>& parts,
      std::vector<IdType>& ids,
      std::vector<PartIndexType>& part_index)
      : Base(std::move(parts))
      , ids_(std::move(ids))
      , part_index_(std::move(part_index))
      , num_vectors_(::num_vectors(parts))
      , num_parts_(size(part_index_) - 1) {
  }

  /**
   * @brief Construct a partitioned matrix in place from a training set of data
   * and an array of labels that maps each vector of the training set to a
   * partition.
   *
   * @tparam F A feature vector array (of the training set)
   * @tparam V A contiguous range of size_t
   * @param training_set The training set of data
   * @param part_labels A vector of size num_vectors(training_set) that maps
   * each vector of the training set to a partition.
   * @param num_parts The number of partitions
   */
  template <feature_vector_array F, std::ranges::contiguous_range V>
  PartitionedMatrix(
      const F& training_set, const V& part_labels, size_t num_parts)
      : Base(::dimensions(training_set), ::num_vectors(training_set))
      , ids_(::num_vectors(training_set))
      , part_index_(num_parts + 1)
      , num_vectors_{::num_vectors(training_set)}
      , num_parts_{num_parts} {
    if (size(part_labels) != ::num_vectors(training_set)) {
      throw std::invalid_argument(
          "The number of part_labels must equal the number of vectors in the "
          "training_set.");
    }

    auto degrees = std::vector<size_t>(num_parts);

    for (size_t i = 0; i < ::num_vectors(training_set); ++i) {
      auto j = part_labels[i];
      ++degrees[j];
    }
    part_index_[0] = 0;
    std::inclusive_scan(begin(degrees), end(degrees), begin(part_index_) + 1);

    for (size_t i = 0; i < ::num_vectors(training_set); ++i) {
      size_t bin = part_labels[i];
      size_t ibin = part_index_[bin];

      if constexpr (feature_vector_array_with_ids<F>) {
        ids_[ibin] = training_set.id(i);
      } else {
        ids_[ibin] = i;
      }

      if (ibin >= this->num_cols()) {
        throw std::runtime_error(
            "[partitioned_matrix@PartitionedMatrix] ibin >= this->num_cols()");
      }
      for (size_t j = 0; j < dimensions(training_set); ++j) {
        this->operator()(j, ibin) = training_set(j, i);
      }
      ++part_index_[bin];
    }

    std::shift_right(begin(part_index_), end(part_index_), 1);
    part_index_[0] = 0;
  }

  constexpr auto& num_vectors() {
    return num_vectors_;
  }

  constexpr auto& num_vectors() const {
    return num_vectors_;
  }

  constexpr auto& num_partitions() const {
    return num_parts_;
  }
  constexpr auto& num_partitions() {
    return num_parts_;
  }

  auto& ids() const {
    return ids_;
  }

  auto& indices() const {
    return part_index_;
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
using RowMajorPartitionedMatrix = PartitionedMatrix<
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
using ColMajorPartitionedMatrix = PartitionedMatrix<
    T,
    partitioned_ids_type,
    part_index_type,
    stdx::layout_left,
    I>;

template <class PartitionedMatrix>
void debug_partitioned_matrix(
    const PartitionedMatrix& matrix,
    const std::string& msg = "",
    size_t max_size = 10) {
  auto rowsEnd = std::min(dimensions(matrix), static_cast<size_t>(max_size));
  auto colsEnd = std::min(num_vectors(matrix), static_cast<size_t>(max_size));

  debug_matrix(matrix, msg, max_size);

  std::cout << "# ids: [";
  auto end = std::min(matrix.ids().size(), static_cast<size_t>(max_size));
  for (size_t i = 0; i < end; ++i) {
    std::cout << matrix.ids()[i];
    if (i != matrix.ids().size() - 1) {
      std::cout << ", ";
    }
  }
  if (matrix.ids().size() > max_size) {
    std::cout << "...";
  }
  std::cout << "]" << std::endl;

  std::cout << "# indices: [";
  end = std::min(matrix.indices().size(), static_cast<size_t>(max_size));
  for (size_t i = 0; i < end; ++i) {
    std::cout << matrix.indices()[i];
    if (i != matrix.indices().size() - 1) {
      std::cout << ", ";
    }
  }
  if (matrix.indices().size() > max_size) {
    std::cout << "...";
  }
  std::cout << "]" << std::endl;
}

#endif  // PARTITIONED_MATRIX_H
