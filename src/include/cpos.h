/**
 * @file   unit_concepts.cc
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
 * Customization point objects for TileDB-Vector-Search types
 *
 */

#ifndef TILEDB_CPOS_H
#define TILEDB_CPOS_H

#include <concepts>

template <class T>
concept semi_integral = std::integral<T> && !std::same_as<T, bool>;

// @todo Have to fix this!!!  Linear algebra should not leak into these CPOs
#include "mdspan/mdspan.hpp"
namespace stdx {
using namespace Kokkos;
using namespace Kokkos::Experimental;
}  // namespace stdx

// ----------------------------------------------------------------------------
// Customization point objects (CPOs) -- implemented as "niebloids"
// ----------------------------------------------------------------------------
template <class T>
concept _member_num_rows = requires(T t) {
  { t.num_rows() } -> semi_integral;
};

template <class T>
concept _member_num_cols = requires(T t) {
  { t.num_cols() } -> semi_integral;
};

template <class T>
concept row_major = std::
    same_as<typename std::remove_cvref_t<T>::layout_policy, stdx::layout_right>;

template <class T>
concept col_major = std::
    same_as<typename std::remove_cvref_t<T>::layout_policy, stdx::layout_left>;

// ----------------------------------------------------------------------------
// dimension CPO
// ----------------------------------------------------------------------------
namespace _dimension {
void dimension(auto&) = delete;
void dimension(const auto&) = delete;

template <class T>
concept _member_dimension = requires(T t) {
  { t.dimension() } -> semi_integral;
};

template <class T>
concept _member_size = requires(T t) {
  { t.size() } -> semi_integral;
};

struct _fn {
  template <_member_dimension T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.dimension();
  }

  template <class V>
  requires _member_size<V> &&(!_member_num_rows<V>)&&std::is_arithmetic_v<
      std::ranges::range_value_t<V>> auto constexpr
  operator()(V&& v) const noexcept {
    return v.size();
  }

  template <class V>
  requires _member_num_rows<V> &&
      std::is_arithmetic_v<std::ranges::range_value_t<V>>
  constexpr auto operator()(V&& v) const noexcept {
    return v.num_rows();
  }

  // @todo This is a total temporary hack
  template <class M>
  requires _member_num_rows<M> && _member_num_cols<M> && row_major<M>
  auto constexpr operator()(M&& m) const noexcept {
    return m.num_cols();
  }

  template <class M>
  requires _member_num_rows<M> && _member_num_cols<M> && col_major<M>
  auto constexpr operator()(M&& m) const noexcept {
    return m.num_rows();
  }

  // @todo Leaking abstraction?
  template <class T, class I>
  auto constexpr operator()(const stdx::mdspan<T, I, stdx::layout_left>& m) const noexcept {
    return m.extent(0);
  }

  template <class T, class I>
  auto constexpr operator()(const stdx::mdspan<T, I, stdx::layout_right>& m) const noexcept {
    return m.extent(1);
  }
};
}  // namespace _dimension

inline namespace _cpo {
inline constexpr auto dimension = _dimension::_fn{};
}  // namespace _cpo

// ----------------------------------------------------------------------------
// num_vectors CPO
// ----------------------------------------------------------------------------
namespace _num_vectors {
void num_vectors(auto&) = delete;
void num_vectors(const auto&) = delete;

template <class T>
concept _member_num_vectors = requires(T t) {
  {t.num_vectors()};
};

struct _fn {
  template <_member_num_vectors T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.num_vectors();
  }

  // @todo This is a total temporary hack -- abstraction violation
  template <class M>
  requires _member_num_rows<M> && _member_num_cols<M> && row_major<M>
  auto constexpr operator()(const M& m) const noexcept {
    return m.num_rows();
  }

  template <class M>
  requires _member_num_rows<M> && _member_num_cols<M> && col_major<M>
  auto constexpr operator()(const M& m) const noexcept {
    return m.num_cols();
  }

  // @todo Leaking abstraction?
  template <class T, class I>
  auto constexpr operator()(const stdx::mdspan<T, I, stdx::layout_left>& m) const noexcept {
    return m.extent(1);
  }

  template <class T, class I>
  auto constexpr operator()(const stdx::mdspan<T, I, stdx::layout_right>& m) const noexcept {
    return m.extent(0);
  }

};
}  // namespace _num_vectors

inline namespace _cpo {
inline constexpr auto num_vectors = _num_vectors::_fn{};
}  // namespace _cpo

// ----------------------------------------------------------------------------
// data CPO
// @todo Figure out what is wrong with const
// ----------------------------------------------------------------------------
namespace _data {
void data(auto&) = delete;
void data(const auto&) = delete;

template <class T>
concept _member_data = requires(T t) {
  {t.data()};
};

template <class T>
concept _member_data_handle = requires(T t) {
  {t.data_handle()};
};

struct _fn {
  template <_member_data T>
  requires (!_member_data_handle<T>)
  constexpr auto operator()(T&& t) const noexcept {
    return t.data();
  }

  template <_member_data_handle T>
  requires (!_member_data<T>)
  constexpr auto operator()(T&& t) const noexcept {
    return t.data_handle();
  }

  template <_member_data_handle T>
  requires (_member_data<T>)
  constexpr auto operator()(T&& t) const noexcept {
    return t.data_handle();
  }

};
}  // namespace _data
inline namespace _cpo {
inline constexpr auto data = _data::_fn{};
}  // namespace _cpo

// ----------------------------------------------------------------------------
// extents CPO
// ----------------------------------------------------------------------------
namespace _extents {
void extents(auto&) = delete;
void extents(const auto&) = delete;

template <class T>
concept _member_extents = requires(T t) {
  { t.extents() };
};

template<typename T>
concept _is_mdspan =
  std::same_as<typename std::remove_cvref_t<T>,
          stdx::mdspan<typename std::remove_cvref_t<T>::value_type,
                              typename std::remove_cvref_t<T>::extents_type,
                              typename std::remove_cvref_t<T>::layout_type>>;

struct _fn {

  template <_member_extents T>
  requires (!_is_mdspan<T>)
  auto constexpr operator()(T&& t) const noexcept {
    return t.extents();
  }

  template <_is_mdspan M>
  auto constexpr operator()(M&& m) const noexcept {
    return std::vector<size_t>{m.extents().extent(0), m.extents().extent(1)};
  }

};
}  // namespace _extents
inline namespace _cpo {
inline constexpr auto extents = _extents::_fn{};
}  // namespace _cpo

// ----------------------------------------------------------------------------
// num_partitions CPO
// ----------------------------------------------------------------------------
namespace _num_partitions {
void num_partitions(auto&) = delete;
void num_partitions(const auto&) = delete;

template <class T>
concept _member_num_partitions = requires(T t) {
  {t.num_partitions()};
};

struct _fn {
  template <_member_num_partitions T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.num_partitions();
  }
};
}  // namespace _num_partitions

inline namespace _cpo {
inline constexpr auto num_partitions = _num_partitions::_fn{};
}  // namespace _cpo

#endif  // TILEDB_CPOS_H