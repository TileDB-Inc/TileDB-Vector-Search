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

#ifndef TILEDB_MATRIX_H
#define TILEDB_MATRIX_H

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include "concepts.h"
#include "mdspan/mdspan.hpp"
#include "tdb_defs.h"

#include "utils/timer.h"

#include <version>
#include "detail/linalg/linalg_defs.h"

template <class I = size_t>
using matrix_extents = stdx::dextents<I, 2>;

template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class MatrixView : public stdx::mdspan<T, matrix_extents<I>, LayoutPolicy> {
  using Base = stdx::mdspan<T, stdx::dextents<I, 2>, LayoutPolicy>;
  using Base::Base;

 public:
  using layout_policy = LayoutPolicy;
  using index_type = typename Base::index_type;
  using size_type = typename Base::size_type;
  using reference = typename Base::reference;

  using span_type = std::span<T>;

 public:
  MatrixView(const Base& rhs)
      : Base(rhs) {
  }

  MatrixView(T* data_pointer, I rows, I cols)
      : Base{data_pointer, rows, cols} {
  }

  auto data() {
    return this->data_handle();
  }

  auto data() const {
    return this->data_handle();
  }

  // @todo Fully verify row and column major function correctly
  auto operator[](index_type i) {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&Base::operator()(i, 0), this->extents().extent(1));
    } else {
      return std::span(&Base::operator()(0, i), this->extents().extent(0));
    }
  }

  auto operator[](index_type i) const {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&Base::operator()(i, 0), this->extents().extent(1));
    } else {
      return std::span(&Base::operator()(0, i), this->extents().extent(0));
    }
  }

  size_type num_rows() const {
    return this->extent(0);
  }

  size_type num_rows() {
    return this->extent(0);
  }

  size_type num_cols() const {
    return this->extent(1);
  }

  size_type num_cols() {
    return this->extent(1);
  }

  ~MatrixView() = default;
};

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t>
using RowMajorMatrixView = MatrixView<T, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t>
using ColMajorMatrixView = MatrixView<T, stdx::layout_left, I>;

/**
 * @brief A 2-D matrix class that owns its storage.  The interface is
 * that of mdspan.
 *
 * @tparam T
 * @tparam LayoutPolicy
 * @tparam I
 *
 * @todo Make Matrix into a range (?)
 */

template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class Matrix : public stdx::mdspan<T, matrix_extents<I>, LayoutPolicy> {
  using Base = stdx::mdspan<T, matrix_extents<I>, LayoutPolicy>;

 public:
  using layout_policy = LayoutPolicy;
  using index_type = typename Base::index_type;
  using size_type = typename Base::size_type;
  using reference = typename Base::reference;

  using view_type = Matrix;
  using span_type = std::span<T>;

 protected:
  size_type num_rows_{0};
  size_type num_cols_{0};
  std::unique_ptr<T[]> storage_;

 public:
  // Needed because of deferred construction in derived classes
  Matrix() noexcept = default;

 public:
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;

  Matrix(Matrix&&) = default;

  Matrix& operator=(Matrix&& rhs) = default;
  virtual ~Matrix() = default;

  Matrix(
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
      : num_rows_(nrows)
      , num_cols_(ncols)
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , storage_{std::make_unique_for_overwrite<T[]>(num_rows_ * num_cols_)}
#else
      , storage_{new T[num_rows_ * num_cols_]}
#endif
  {
    Base::operator=(Base{storage_.get(), num_rows_, num_cols_});
  }

  Matrix(
      std::unique_ptr<T[]>&& storage,
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
      : num_rows_(nrows)
      , num_cols_(ncols)
      , storage_{std::move(storage)} {
    Base::operator=(Base{storage_.get(), num_rows_, num_cols_});
  }

  /**
   * Initializer list constructor.  Useful for testing and for examples.
   * The intializer list is assumed to be in row-major order.
   */
  Matrix(std::initializer_list<std::initializer_list<T>> list) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
      : num_rows_{list.size()}
      , num_cols_{list.begin()->size()}
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , storage_{std::make_unique_for_overwrite<T[]>(num_rows_ * num_cols_)}
#else
      , storage_{new T[num_rows_ * num_cols_]}
#endif
  {
    Base::operator=(Base{storage_.get(), num_rows_, num_cols_});
    auto it = list.begin();
    for (size_type i = 0; i < num_rows_; ++i, ++it) {
      std::copy(it->begin(), it->end(), (*this)[i].begin());
    }
  }

  /**
   * Initializer list constructor.  Useful for testing and for examples.
   * The initializer list is assumed to be in column-major order.
   */
  Matrix(std::initializer_list<std::initializer_list<T>> list) noexcept
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
      : num_rows_{list.begin()->size()}
      , num_cols_{list.size()}
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , storage_{std::make_unique_for_overwrite<T[]>(num_rows_ * num_cols_)}
#else
      , storage_{new T[num_rows_ * num_cols_]}
#endif
  {
    Base::operator=(Base{storage_.get(), num_rows_, num_cols_});
    auto it = list.begin();
    for (size_type i = 0; i < num_cols_; ++i, ++it) {
      std::copy(it->begin(), it->end(), (*this)[i].begin());
    }
  }

  auto data() {
    // return this->data_handle();
    return storage_.get();
  }

  auto data() const {
    // return this->data_handle();
    return storage_.get();
  }

  auto raveled() {
    return std::span(storage_.get(), num_rows_ * num_cols_);
  }

  auto raveled() const {
    return std::span(storage_.get(), num_rows_ * num_cols_);
  }

  auto operator[](index_type i) {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&Base::operator()(i, 0), num_cols_);
    } else {
      return std::span(&Base::operator()(0, i), num_rows_);
    }
  }

  auto operator[](index_type i) const {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&Base::operator()(i, 0), num_cols_);
    } else {
      return std::span(&Base::operator()(0, i), num_rows_);
    }
  }

  auto rank() const noexcept {
    return Base::extents().rank();
    // return 2;  //
  }

  auto extents() const noexcept {
    return std::vector<size_t>{
        Base::extents().extent(0), Base::extents().extent(1)};
  }

  // @note We do not have a `size()` method, due to ambiguity with
  // what it means for a matrix (and for feature vector array)
  auto num_rows() const noexcept {
    return num_rows_;
  }

  auto num_cols() const noexcept {
    return num_cols_;
  }

  auto swap(Matrix& rhs) noexcept {
    std::swap(num_rows_, rhs.num_rows_);
    std::swap(num_cols_, rhs.num_cols_);
    std::swap(storage_, rhs.storage_);
    std::swap(static_cast<Base&>(*this), static_cast<Base&>(rhs));
  }

  template <
      class T_,
      class LayoutPolicy_ = stdx::layout_right,
      class I_ = size_t>
  bool operator==(const Matrix<T_, LayoutPolicy_, I_>& rhs) const noexcept {
    return (void*)this->data() == (void*)rhs.data() ||
           (num_rows_ == rhs.num_rows() && num_cols_ == rhs.num_cols() &&
            std::equal(
                raveled().begin(), raveled().end(), rhs.raveled().begin()));
  }
};

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t>
using RowMajorMatrix = Matrix<T, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t>
using ColMajorMatrix = Matrix<T, stdx::layout_left, I>;

/**
 * Convenience class for turning 2D matrices into 1D vectors.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
auto raveled(Matrix<T, LayoutPolicy, I>& m) {
  return m.raveled();
}

/**
 * Is the matrix row-oriented?
 */
template <class Matrix>
constexpr bool is_row_oriented(const Matrix& A) {
  return std::is_same_v<typename Matrix::layout_policy, stdx::layout_right>;
}

/**
 * Is the matrix col-oriented?
 */
template <class Matrix>
constexpr bool is_col_oriented(const Matrix& A) {
  return std::is_same_v<typename Matrix::layout_policy, stdx::layout_left>;
}

/**********************************************************************
 *
 * Some debugging utilities.
 *
 * *********************************************************************/

/**
 * Print information about a Matrix.
 * @param A
 */
template <class Matrix>
std::string matrix_info(const Matrix& A, const std::string& msg = "") {
  std::string str = "# " + msg;
  if (!msg.empty()) {
    str += ": ";
  }
  str += "Shape: ( " + std::to_string(::dimensions(A)) + ", " +
         std::to_string(::num_vectors(A)) + " )";
  str += std::string(" Layout: ") +
         (is_row_oriented(A) ? "row major" : "column major");
  return str;
}

/**********************************************************************
 *
 * Submatrix view.
 *
 **********************************************************************/

template <
    class ElementType,
    class Extents,
    class LayoutPolicy,
    class AccessorPolicy>
class SubMatrixView
    : public stdx::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> {
  using Base = stdx::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  using index_type = typename Base::index_type;
  using size_type = typename Base::size_type;
  using reference = typename Base::reference;

  size_t num_rows_{0};
  size_t num_cols_{0};

 public:
  SubMatrixView() noexcept = delete;

  SubMatrixView(
      const stdx::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& m)
      : Base{m}
      , num_rows_{this->extent(0)}
      , num_cols_{this->extent(1)} {
  }

  auto data() {
    return this->get();
  }

  auto data() const {
    return this->get();
  }

  auto operator[](index_type i) {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&Base::operator()(i, 0), num_cols_);
    } else if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_left>) {
      return std::span(&Base::operator()(0, i), num_rows_);
    } else {
      static_assert(always_false<ElementType>, "Unknown layout policy");
    }
  }

  auto operator[](index_type i) const {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return std::span(&Base::operator()(i, 0), num_cols_);
    } else if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_left>) {
      return std::span(&Base::operator()(0, i), num_rows_);
    } else {
      static_assert(always_false<ElementType>, "Unknown layout policy");
    }
  }

  auto num_rows() const noexcept {
    return num_rows_;
  }

  auto num_cols() const noexcept {
    return num_cols_;
  }
};

template <
    class ElementType,
    class Extents,
    class LayoutPolicy,
    class AccessorPolicy,
    class R,
    class C>
constexpr auto SubMatrix(
    const stdx::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& m,
    R rows,
    C cols) {
  return SubMatrixView(stdx::submdspan(m, rows, cols));
}

/**********************************************************************
 *
 * Some debugging utilities.
 *
 **********************************************************************/

// TODO(paris): This only works on col-major matrices, fix for row-major.
template <class Matrix>
void debug_matrix(
    const Matrix& matrix, const std::string& msg = "", size_t max_size = 10) {
  auto rowsEnd = std::min(dimensions(matrix), static_cast<size_t>(max_size));
  auto colsEnd = std::min(num_vectors(matrix), static_cast<size_t>(max_size));

  std::cout << "# " << msg << " (" << dimensions(matrix) << " rows x "
            << num_vectors(matrix) << " cols) ("
            << (is_row_oriented(matrix) ? "row major" : "column major")
            << ", so " << _cpo::num_vectors(matrix) << " vectors with "
            << _cpo::dimensions(matrix) << " dimensions each)" << std::endl;

  for (size_t i = 0; i < rowsEnd; ++i) {
    std::cout << "# ";
    for (size_t j = 0; j < colsEnd; ++j) {
      std::cout << (float)matrix(i, j) << " ";
    }
    if (num_vectors(matrix) > max_size) {
      std::cout << "...";
    }
    std::cout << std::endl;
  }
  if (dimensions(matrix) > max_size) {
    std::cout << "# ..." << std::endl;
  }
}

template <class Matrix1, class Matrix2>
void debug_slices_diff(
    const Matrix1& A,
    const Matrix2& B,
    const std::string& msg = "",
    size_t rows = 5,
    size_t cols = 15) {
  rows = std::min(rows, A.num_rows());
  cols = std::min(cols, A.num_cols());

  std::cout << "# " << msg << std::endl;
  for (size_t i = 0; i < A.num_rows(); ++i) {
    for (size_t j = 0; j < A.num_cols(); ++j) {
      if (A(i, j) != B(i, j)) {
        std::cout << "A(" << i << ", " << j << ") = " << A(i, j) << " != "
                  << "B(" << i << ", " << j << ") = " << B(i, j) << std::endl;
        if (--cols == 0) {
          break;
        }
      }
    }
    if (--rows == 0) {
      break;
    }
    cols = std::min(cols, A.num_cols());
  }
}

#endif  // TILEDB_MATRIX_H
