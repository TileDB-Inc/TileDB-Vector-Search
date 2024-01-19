/**
 * @file   vector.h
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
 * Simple vector class, basically a span that owns it data.
 *
 * @todo Migrate from std::vector to this class.
 *
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <initializer_list>
#include <span>
#include <tiledb/tiledb>
#include <vector>

template <class T>
std::vector<T> read_vector(
    const tiledb::Context& ctx,
    const std::string&,
    size_t start_pos,
    size_t end_pos,
    uint64_t timestamp);

template <class M>
concept is_view = requires(M) { typename M::view_type; };

template <class T>
using VectorView = std::span<T>;

/**
 * @brief A 1-D vector class that owns its storage.
 * @tparam T
 *
 * @todo More verification.  Replace uses of std::vector with this class.
 */
template <class T>
class Vector : public std::span<T> {
  using Base = std::span<T>;
  using Base::Base;

 public:
  using index_type = typename Base::difference_type;
  using size_type = typename Base::size_type;
  using reference = typename Base::reference;

 private:
  size_type nrows_;
  std::unique_ptr<T[]> storage_;

 public:
  explicit Vector(index_type nrows) noexcept
      : nrows_(nrows)
      , storage_{new T[nrows_]} {
    Base::operator=(Base{storage_.get(), nrows_});
  }

  Vector(index_type nrows, std::unique_ptr<T[]> storage)
      : nrows_(nrows)
      , storage_{std::move(storage)} {
    Base::operator=(Base{storage_.get(), nrows_});
  }

  Vector(std::initializer_list<T> lst)
      : nrows_(lst.size())
#ifdef __cpp_lib_smart_ptr_for_overwrite
      , storage_{std::make_unique_for_overwrite<T[]>(nrows_)}
#else
      , storage_{new T[nrows_]}
#endif
  {
    Base::operator=(Base{storage_.get(), nrows_});
    std::copy(lst.begin(), lst.end(), storage_.get());
  }

  Vector(Vector&& rhs) noexcept
      : nrows_{rhs.nrows_}
      , storage_{std::move(rhs.storage_)} {
    Base::operator=(Base{storage_.get(), nrows_});
  }

  constexpr reference operator()(size_type idx) const noexcept {
    return Base::operator[](idx);
  }

  Vector& operator=(const VectorView<T>& rhs) {
    std::copy(rhs.begin(), rhs.end(), this->begin());
    return *this;
  }

  constexpr size_type num_rows() const noexcept {
    return nrows_;
  }
};

template <class V>
void debug_vector(const V& v, const std::string& msg = "") {
  std::cout << msg;
  for (size_t i = 0; i < dimension(v); ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << "\n";
}

template <std::ranges::forward_range V>
void debug_vector(const V& v, const std::string& msg = "") {
  std::cout << msg;
  for (auto&& i : v) {
    std::cout << i << " ";
  }
  std::cout << "\n";
}

#endif
