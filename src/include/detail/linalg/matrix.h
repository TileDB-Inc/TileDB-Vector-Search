
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
#include "mdspan/mdspan.hpp"

#include "utils/timer.h"

#include "detail/linalg/linalg_defs.h"

template <class I = size_t>
using matrix_extents = stdx::dextents<I, 2>;

/**
 * @brief A 2-D matrix class that owns its storage.  The interface is
 * that of mdspan.
 *
 * @tparam T
 * @tparam LayoutPolicy
 * @tparam I
 *
 * @todo Make an alias for extents.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class Matrix : public stdx::mdspan<T, matrix_extents<I>, LayoutPolicy> {
  using Base = stdx::mdspan<T, matrix_extents<I>, LayoutPolicy>;
  // using Base::Base;

 public:
  using layout_policy = LayoutPolicy;
  using index_type = typename Base::index_type;
  using size_type = typename Base::size_type;
  using reference = typename Base::reference;

  using view_type = Matrix;

 protected:
  size_type num_rows_{0};
  size_type num_cols_{0};

  // private:
  std::unique_ptr<T[]> storage_;

 public:
  Matrix() noexcept = default;

  Matrix(
      size_type nrows,
      size_type ncols,
      LayoutPolicy policy = LayoutPolicy()) noexcept
      : num_rows_(nrows)
      , num_cols_(ncols)
      , storage_{new T[num_rows_ * num_cols_]} {
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

  Matrix(Matrix&& rhs) noexcept
      : num_rows_{rhs.num_rows_}
      , num_cols_{rhs.num_cols_}
      , storage_{std::move(rhs.storage_)} {
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
      , storage_{new T[num_rows_ * num_cols_]} {
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
      , storage_{new T[num_rows_ * num_cols_]} {
    Base::operator=(Base{storage_.get(), num_rows_, num_cols_});
    auto it = list.begin();
    for (size_type i = 0; i < num_cols_; ++i, ++it) {
      std::copy(it->begin(), it->end(), (*this)[i].begin());
    }
  }

  auto& operator=(Matrix&& rhs) noexcept {
    num_rows_ = rhs.num_rows_;
    num_cols_ = rhs.num_cols_;
    storage_ = std::move(rhs.storage_);
    Base::operator=(Base{storage_.get(), num_rows_, num_cols_});
    return *this;
  }

  auto data() {
    return storage_.get();
  }

  auto data() const {
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

  auto span() const noexcept {
    if constexpr (std::is_same_v<LayoutPolicy, stdx::layout_right>) {
      return num_cols();
    } else {
      return num_rows();
    }
  }

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

// @todo these are k
template <class T, class I>
size_t size(const Matrix<T, stdx::layout_right, I>& m) {
  return m.num_rows();
}

template <class T, class I>
size_t size(const Matrix<T, stdx::layout_left, I>& m) {
  return m.num_cols();
}

/**
 * Is the matrix row-oriented?
 */
template <class Matrix>
constexpr bool is_row_oriented(const Matrix& A) {
  return std::is_same_v<typename Matrix::layout_policy, stdx::layout_right>;
}

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
  str += "Shape: ( " + std::to_string(A.num_rows()) + ", " +
         std::to_string(A.num_cols()) + " )";
  str += std::string(" Layout: ") +
         (is_row_oriented(A) ? "row major" : "column major");
  return str;
}

/**********************************************************************
 *
 * Some debugging utilities.
 *
 * *********************************************************************/

/**
 * Print information about a std::vector -- overload.
 * @param A
 */
template <class T>
std::string matrix_info(const std::vector<T>& A, const std::string& msg = "") {
  std::string str = "# " + msg;
  if (!msg.empty()) {
    str += ": ";
  }
  str += "Shape: (" + std::to_string(A.size()) + " )";
  return str;
}

/**
 * Print information about a std::span -- overload.
 * @param A
 */
template <class T>
std::string matrix_info(const std::span<T>& A, const std::string& msg = "") {
  std::string str = "# " + msg;
  if (!msg.empty()) {
    str += ": ";
  }
  str += "Shape: (" + std::to_string(A.size()) + " )";
  return str;
}

static bool matrix_printf = false;

template <class Matrix>
void debug_matrix(const Matrix& A, const std::string& msg = "") {
  if (matrix_printf) {
    std::cout << matrix_info(A, msg) << std::endl;
  }
}

template <class Matrix>
void debug_slice(
    const Matrix& A,
    const std::string& msg = "",
    size_t rows = 5,
    size_t cols = 15) {
  if (matrix_printf) {
    rows = std::min(rows, A.num_rows());
    cols = std::min(cols, A.num_cols());

    std::cout << "# " << msg << std::endl;
    for (size_t i = 0; i < rows; ++i) {
      std::cout << "# ";
      for (size_t j = 0; j < cols; ++j) {
        std::cout << (float)A(i, j) << "\t";
      }
      std::cout << std::endl;
    }
  }
}

template <class Matrix1, class Matrix2>
void debug_slices_diff(
    const Matrix1& A,
    const Matrix2& B,
    const std::string& msg = "",
    size_t rows = 5,
    size_t cols = 15) {
  if (matrix_printf) {
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
}

#endif  // TILEDB_MATRIX_H
