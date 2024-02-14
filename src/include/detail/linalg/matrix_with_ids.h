/**
 * @file   matrix.h
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
 * A basic matrix class.  It is essentially a wrapper around mdspan, but
 * owns its storage.
 *
 */

#ifndef TILEDB_MATRIX_WITH_ID_H
#define TILEDB_MATRIX_WITH_ID_H

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include "concepts.h"
#include "mdspan/mdspan.hpp"
#include "tdb_defs.h"

#include "utils/timer.h"

#include <version>
#include "detail/linalg/linalg_defs.h"
#include "detail/linalg/matrix.h"

// class TdbMatrixWithIds : public TdbMatrix<T, LayoutPolicy, I> {
//     std::string idsUri_;

//     bool load() {
//       TdbMatrix::load();
//       loadFromUri(idsUri_);
//     }
// }

/**
 * @brief A 2-D matrix class that owns its storage.  The interface is
 * that of mdspan.
 *
 * @tparam T
 * @tparam LayoutPolicy
 * @tparam I
 *
 * @todo Make MatrixWithIds into a range (?)
 */

template <
    class T,
    class LayoutPolicy = stdx::layout_right,
    class I = size_t,
    class IdsType = size_t>
class MatrixWithIds : public Matrix<T, LayoutPolicy, I> {
 protected:
  size_t num_ids_ = 0;
  std::unique_ptr<IdsType[]> idsStorage_;

 public:
  MatrixWithIds() noexcept = default;

  MatrixWithIds(const MatrixWithIds&) = delete;
  MatrixWithIds& operator=(const MatrixWithIds&) = delete;

  MatrixWithIds(MatrixWithIds&&) = default;

  MatrixWithIds& operator=(MatrixWithIds&& rhs) = default;
  virtual ~MatrixWithIds() = default;

  MatrixWithIds(
      Matrix<T, LayoutPolicy, I>::size_type nrows,
      Matrix<T, LayoutPolicy, I>::size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
      : Matrix<T, LayoutPolicy, I>(nrows, ncols, policy)
      , num_ids_(this->num_rows_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , idsStorage_{std::make_unique_for_overwrite<T[]>(this->num_rows_)}
#else
      , idsStorage_{new T[this->num_rows_]}
#endif
  {
    std::cout << "ctor 2" << std::endl;
  }

  MatrixWithIds(
      Matrix<T, LayoutPolicy, I>::size_type nrows,
      Matrix<T, LayoutPolicy, I>::size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Matrix<T, LayoutPolicy, I>(nrows, ncols, policy)
      , num_ids_(this->num_cols_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , idsStorage_{std::make_unique_for_overwrite<T[]>(this->num_cols_)}
#else
      , idsStorage_{new T[this->num_cols_]}
#endif
  {
    std::cout << "ctor 3" << std::endl;
  }

  MatrixWithIds(
      std::unique_ptr<T[]>&& storage,
      std::unique_ptr<T[]>&& ids_storage,
      Matrix<T, LayoutPolicy, I>::size_type nrows,
      Matrix<T, LayoutPolicy, I>::size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
      : Matrix<T, LayoutPolicy, I>(storage, nrows, ncols, policy)
      , idsStorage_{std::move(ids_storage)}
      , num_ids_{
            std::is_same<LayoutPolicy, stdx::layout_right>::value ?
                this->num_rows_ :
                this->num_cols_} {
    std::cout << "ctor 1" << std::endl;
  }

  /**
   * Initializer list constructor.  Useful for testing and for examples.
   * The initializer list is assumed to be in row-major order.
   */
  MatrixWithIds(
      std::initializer_list<std::initializer_list<T>> matrix,
      std::initializer_list<T> ids) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
      : Matrix<T, LayoutPolicy, I>(matrix)
      , num_ids_(this->num_rows_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , idsStorage_{std::make_unique_for_overwrite<IdsType[]>(this->num_rows_)}
#else
      , idsStorage_{new IdsType[this->num_rows_]}
#endif
  {
    std::cout << "row-major initializer list constructor" << std::endl;
    std::cout << "num_rows_: " << this->num_rows_ << std::endl;
    std::cout << "num_cols_: " << this->num_cols_ << std::endl;
    std::copy(ids.begin(), ids.end(), idsStorage_.get());
  }

  /**
   * Initializer list constructor.  Useful for testing and for examples.
   * The initializer list is assumed to be in column-major order.
   */
  MatrixWithIds(
      std::initializer_list<std::initializer_list<T>> matrix,
      std::initializer_list<T> ids) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : Matrix<T, LayoutPolicy, I>(matrix)
      , num_ids_(this->num_cols_)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , idsStorage_{std::make_unique_for_overwrite<IdsType[]>(this->num_cols_)}
#else
      , idsStorage_{new IdsType[this->num_cols_]}
#endif
  {
    std::cout << "col-major initializer list constructor" << std::endl;
    std::cout << "num_rows_: " << this->num_rows_ << std::endl;
    std::cout << "num_cols_: " << this->num_cols_ << std::endl;
    std::copy(ids.begin(), ids.end(), idsStorage_.get());
  }

  size_t num_ids() const {
    return num_ids_;
  }

  auto ids() {
    // return this->data_handle();
    return idsStorage_.get();
  }

  auto ids() const {
    // return this->data_handle();
    return idsStorage_.get();
  }

  auto raveledIds() {
    return std::span(idsStorage_.get(), num_ids_);
  }

  auto raveledIds() const {
    return std::span(idsStorage_.get(), num_ids_);
  }

  //  auto raveledIds() requires(std::is_same_v<LayoutPolicy,
  //  stdx::layout_right>) {
  //    return std::span(idsStorage_.get(), this->num_rows_);
  //  }
  //
  //  auto raveledIds() const requires(std::is_same_v<LayoutPolicy,
  //  stdx::layout_right>) {
  //    return std::span(idsStorage_.get(), this->num_rows_);
  //  }
  //
  //  auto raveledIds() requires(std::is_same_v<LayoutPolicy,
  //  stdx::layout_left>) {
  //    return std::span(idsStorage_.get(), this->num_cols_);
  //  }
  //
  //  auto raveledIds() const requires(std::is_same_v<LayoutPolicy,
  //  stdx::layout_left>) {
  //    return std::span(idsStorage_.get(), this->num_cols_);
  //  }

  auto id(Matrix<T, LayoutPolicy, I>::index_type i) const {
    return idsStorage_[i];
  }

  auto swap(MatrixWithIds& rhs) noexcept {
    Matrix<T, LayoutPolicy, I>::swap(rhs);
    std::swap(idsStorage_, rhs.idsStorage_);
  }

  template <
      class T_,
      class LayoutPolicy_ = stdx::layout_right,
      class I_ = size_t>
  bool operator==(
      const MatrixWithIds<T_, LayoutPolicy_, I_>& rhs) const noexcept {
    return Matrix<T_, LayoutPolicy_, I_>::operator==(rhs) &&
           ((void*)this->ids() == (void*)rhs.ids() ||
            std::equal(
                raveledIds().begin(),
                raveledIds().end(),
                rhs.raveledIds().begin()));
  }
};

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t, class IdsType = size_t>
using RowMajorMatrixWithIds = MatrixWithIds<T, stdx::layout_right, I, IdsType>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t, class IdsType = size_t>
using ColMajorMatrixWithIds = MatrixWithIds<T, stdx::layout_left, I, IdsType>;

/**********************************************************************
 *
 * Debugging utilities.
 *
 **********************************************************************/

template <class MatrixWithIds>
void debug_slice_with_ids(
    const MatrixWithIds& A,
    const std::string& msg = "",
    size_t rows = 6,
    size_t cols = 18) {
  //  TODO(paris): It used to be different, but is that a bug?
  // It should be A.num_rows() not dimension(A), right?
  //  rows = std::min(rows, dimension(A));
  //  cols = std::min(cols, num_vectors(A));

  std::cout << "# " << msg << std::endl;
  for (size_t i = 0; i < rows; ++i) {
    std::cout << "# ";
    for (size_t j = 0; j < cols; ++j) {
      std::cout << (float)A(i, j) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "# ids: [";
  for (size_t i = 0; i < A.num_ids(); ++i) {
    std::cout << A.id(i);
    if (i != A.num_ids() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

template <class MatrixWithIds>
void debug_with_ids(const MatrixWithIds& A, const std::string& msg = "") {
  debug_slice_with_ids(A, msg, A.num_rows(), A.num_cols());
}

#endif  // TILEDB_MATRIX_WITH_ID_H
