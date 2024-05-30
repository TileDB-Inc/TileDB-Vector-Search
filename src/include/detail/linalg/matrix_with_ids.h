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

 public:
  using ids_type = IdsType;
  using size_type = typename Base::size_type;

 protected:
  size_type num_ids_{0};
  std::unique_ptr<IdsType[]> ids_storage_;

 public:
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
      , num_ids_(this->num_rows_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , ids_storage_{std::make_unique_for_overwrite<IdsType[]>(this->num_rows_)}
#else
      , ids_storage_{new IdsType[this->num_rows_]}
#endif
  {
  }

  MatrixWithIds(
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Base(nrows, ncols, policy)
      , num_ids_(this->num_cols_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , ids_storage_{std::make_unique_for_overwrite<IdsType[]>(this->num_cols_)}
#else
      , ids_storage_{new IdsType[this->num_cols_]}
#endif
  {
  }

  MatrixWithIds(
      std::unique_ptr<T[]>&& storage,
      std::unique_ptr<IdsType[]>&& ids_storage,
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
      : Base(std::move(storage), nrows, ncols, policy)
      , num_ids_{std::is_same<LayoutPolicy, stdx::layout_right>::value ? this->num_rows_ : this->num_cols_}
      , ids_storage_{std::move(ids_storage)} {
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
      , num_ids_(this->num_rows_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , ids_storage_{std::make_unique_for_overwrite<IdsType[]>(this->num_rows_)}
#else
      , ids_storage_{new IdsType[this->num_rows_]}
#endif
  {
    std::copy(ids.begin(), ids.end(), ids_storage_.get());
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
      , num_ids_(this->num_cols_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , ids_storage_{std::make_unique_for_overwrite<IdsType[]>(this->num_cols_)}
#else
      , ids_storage_{new IdsType[this->num_cols_]}
#endif
  {
    std::copy(ids.begin(), ids.end(), ids_storage_.get());
  }

  [[nodiscard]] size_t num_ids() const {
    return num_ids_;
  }

  auto ids() {
    return ids_storage_.get();
  }

  auto ids() const {
    return ids_storage_.get();
  }

  auto raveled_ids() {
    return std::span(ids_storage_.get(), num_ids_);
  }

  auto raveled_ids() const {
    return std::span(ids_storage_.get(), num_ids_);
  }

  auto id(Matrix<T, LayoutPolicy, I>::index_type i) const {
    return ids_storage_[i];
  }

  auto swap(MatrixWithIds& rhs) noexcept {
    Base::swap(rhs);
    std::swap(ids_storage_, rhs.ids_storage_);
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
            std::equal(
                raveled_ids().begin(),
                raveled_ids().end(),
                rhs.raveled_ids().begin()));
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

// TODO(paris): This only works on col-major matrices, fix for row-major.
template <class MatrixWithIds>
void debug_matrix_with_ids(
    const MatrixWithIds& matrix,
    const std::string& msg = "",
    size_t max_size = 10) {
  auto rowsEnd = std::min(dimensions(matrix), static_cast<size_t>(max_size));
  auto colsEnd = std::min(num_vectors(matrix), static_cast<size_t>(max_size));

  debug_matrix(matrix, msg, max_size);

  std::cout << "# ids: [";
  auto end = std::min(matrix.num_ids(), static_cast<size_t>(max_size));
  for (size_t i = 0; i < end; ++i) {
    std::cout << (float)matrix.ids()[i];
    if (i != matrix.num_ids() - 1) {
      std::cout << ", ";
    }
  }
  if (matrix.num_ids() > max_size) {
    std::cout << "...";
  }
  std::cout << "]" << std::endl;
}

#endif  // TILEDB_MATRIX_WITH_IDS_H
