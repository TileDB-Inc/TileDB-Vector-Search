/**
 * @file   matrix_with_ids.h
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
 * A child class of Matrix that also holds IDs.
 *
 */

#ifndef TILEDB_MATRIX_WITH_IDS_H
#define TILEDB_MATRIX_WITH_IDS_H

#include "detail/linalg/linalg_defs.h"
#include "detail/linalg/matrix.h"

/**
 * @brief A 2-D matrix class that owns its storage and has an ID for either each
 * row (if row-major) or each column (if column-major). The interface is that of
 * mdspan.
 *
 * @tparam T
 * @tparam LayoutPolicy
 * @tparam I
 *
 */
template <
    class T,
    class IdsType = uint64_t,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t>
class MatrixWithIds : public Matrix<T, LayoutPolicy, I> {
  using Base = Matrix<T, LayoutPolicy, I>;

 protected:
  std::vector<IdsType> ids_;

 public:
  using ids_type = IdsType;
  using size_type = typename Base::size_type;

  MatrixWithIds() noexcept = default;

  MatrixWithIds(const MatrixWithIds&) = delete;

  MatrixWithIds& operator=(const MatrixWithIds&) = delete;

  MatrixWithIds(MatrixWithIds&&) = default;

  MatrixWithIds& operator=(MatrixWithIds&& rhs) = default;

  virtual ~MatrixWithIds() = default;

  MatrixWithIds(
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
      : Base(nrows, ncols, policy)
      , ids_(this->num_rows_) {
  }

  MatrixWithIds(
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Base(nrows, ncols, policy)
      , ids_(this->num_cols_) {
  }

  MatrixWithIds(
      std::unique_ptr<T[]>&& storage,
      std::vector<IdsType>&& ids,
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
      : Base(std::move(storage), nrows, ncols, policy)
      , ids_{std::move(ids)} {
  }

  /**
   * Initializer list constructor. Useful for testing and for examples.
   * The initializer list is assumed to be in row-major order.
   */
  MatrixWithIds(
      std::initializer_list<std::initializer_list<T>> matrix,
      const std::vector<IdsType>& ids) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
      : Base(matrix)
      , ids_{ids} {
  }

  /**
   * Initializer list constructor. Useful for testing and for examples.
   * The initializer list is assumed to be in column-major order.
   */
  MatrixWithIds(
      std::initializer_list<std::initializer_list<T>> matrix,
      const std::vector<IdsType>& ids) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Base(matrix)
      , ids_{ids} {
  }

  [[nodiscard]] size_t num_ids() const {
    return ids_.size();
  }

  std::vector<IdsType>& ids() {
    return ids_;
  }

  const std::vector<IdsType>& ids() const {
    return ids_;
  }

  auto swap(MatrixWithIds& rhs) noexcept {
    Base::swap(rhs);
    std::swap(ids_, rhs.ids_);
  }

  template <
      class T_,
      class IdsType_ = uint64_t,
      class LayoutPolicy_ = stdx::layout_right,
      class I_ = size_t>
  bool operator==(const MatrixWithIds<T_, IdsType_, LayoutPolicy_, I_>& rhs)
      const noexcept {
    return Matrix<T_, LayoutPolicy_, I_>::operator==(rhs) &&
           ((void*)this->ids() == (void*)rhs.ids() ||
            std::equal(ids().begin(), ids().end(), rhs.ids().begin()));
  }
};

/**
 * Convenience class for row-major matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using RowMajorMatrixWithIds = MatrixWithIds<T, IdsType, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class IdsType = uint64_t, class I = size_t>
using ColMajorMatrixWithIds = MatrixWithIds<T, IdsType, stdx::layout_left, I>;

template <class MatrixWithIds>
void debug_slice_with_ids(
    const MatrixWithIds& A,
    const std::string& msg = "",
    size_t rows = 6,
    size_t cols = 18) {
  auto max_size = 10;
  auto rowsEnd = std::min(dimension(A), static_cast<size_t>(max_size));
  auto colsEnd = std::min(num_vectors(A), static_cast<size_t>(max_size));

  std::cout << "# " << msg << std::endl;
  for (size_t i = 0; i < rowsEnd; ++i) {
    std::cout << "# ";
    for (size_t j = 0; j < colsEnd; ++j) {
      std::cout << (float)A(i, j) << " ";
    }
    if (A.num_cols() > max_size) {
      std::cout << "...";
    }
    std::cout << std::endl;
  }
  if (A.num_rows() > max_size) {
    std::cout << "# ..." << std::endl;
  }

  std::cout << "# ids: [";
  auto end = std::min(A.num_ids(), static_cast<size_t>(max_size));
  for (size_t i = 0; i < end; ++i) {
    std::cout << (float)A.ids()[i];
    if (i != A.num_ids() - 1) {
      std::cout << ", ";
    }
  }
  if (A.num_ids() > max_size) {
    std::cout << "...";
  }
  std::cout << "]" << std::endl;
}

template <class MatrixWithIds>
void debug_with_ids(const MatrixWithIds& A, const std::string& msg = "") {
  debug_slice_with_ids(A, msg, A.num_rows(), A.num_cols());
}

#endif  // TILEDB_MATRIX_WITH_IDS_H
