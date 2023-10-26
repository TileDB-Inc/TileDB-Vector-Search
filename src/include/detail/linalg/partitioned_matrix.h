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
* Class the provides a matrix view to a partitioned TileDB array (as
* partitioned by IVF indexing).
*
* The class requires the URI of a partitioned TileDB array and partitioned set of
* vector identifiers.  The class will provide a view of the requested
* partitions and the corresponding vector identifiers.
*
* Also provides support for out-of-core operation.
*
*/

#ifndef PARTITIONED_MATRIX_H
#define PARTITIONED_MATRIX_H

#include <cstddef>
#include "detail/linalg/matrix.h"

template <
    class T,
    class IdType,
    class IndicesType,
    class PartsType,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
class PartitionedMatrix : public Matrix<T, LayoutPolicy, I> {

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

  // Index for partitioned vectors
  std::vector<indices_type> indices_;       // @todo pointer and span?

  // ids for the partitioned vectors
  std::vector<id_type> ids_;

  /**
   * @brief Construct a partitioned matrix in place from a training set of data
   * and an array mapping each vector of the training set to a partition.  By
   * definition, this will only be suitable for infinite memory use, and
   * the data is already loaded, so we probably don't need to set up all the
   * machinery to do loading, etc.  But we set them up anyway.
   *
   * @tparam F
   * @tparam C
   * @tparam V
   * @param training_set
   * @param parts
   * @param num_parts
   * @param num_threads_
   */
  template <feature_vector_array F, feature_vector_array C, std::ranges::contiguous_range V>
  PartitionedMatrix(const F& training_set, const V& parts, size_t num_parts, size_t num_threads_)
      : Base(::dimension(training_set), ::num_vectors(training_set))
      , indices_{num_parts + 1}
      , ids_{::num_vectors(training_set)}
  {
    auto degrees = std::vector<size_t>(num_parts);

    auto partitioned_vectors =
        ColMajorMatrix<T>{dimension(training_set), num_vectors(training_set)};

    for (size_t i = 0; i < ::num_vectors(training_set); ++i) {
      auto j = parts[i];
      ++degrees[j];
    }
    indices_[0] = 0;
    std::inclusive_scan(begin(degrees), end(degrees), begin(indices_) + 1);

    for (size_t i = 0; i < ::num_vectors(training_set); ++i) {
      size_t bin = parts[i];
      size_t ibin = indices_[bin];

      ids_[ibin] = i;

      assert(ibin < partitioned_vectors.num_cols());
      for (size_t j = 0; j < dimension(training_set); ++j) {
        partitioned_vectors(j, ibin) = training_set(j, i);
      }
      ++indices_[bin];
    }

    std::shift_right(begin(indices_), end(indices_), 1);
    indices_[0] = 0;
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
using RowMajorPartitionedMatrix = PartitionedMatrix<
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
using ColMajorPartitionedMatrix = PartitionedMatrix<
    T,
    partitioned_ids_type,
    indices_type,
    parts_type,
    stdx::layout_left,
    I>;


#endif  // PARTITIONED_MATRIX_H