

#ifndef TILEDB_CPOS_H
#define TILEDB_CPOS_H

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
concept row_major = std::is_same_v<typename T::layout_policy, stdx::layout_right>;

template <class T>
concept col_major = std::is_same_v<typename T::layout_policy, stdx::layout_left>;

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
    requires _member_size<V> && (!_member_num_rows<V>) &&
             std::is_arithmetic_v<std::ranges::range_value_t<V>>
  auto constexpr operator()(const V& v) const noexcept {
    return v.size();
  }

  template <class V>
    requires _member_num_rows<V> &&
             std::is_arithmetic_v<std::ranges::range_value_t<V>>
  auto constexpr operator()(const V& v) const noexcept {
    return v.num_rows();
  }

  // @todo This is a total temporary hack
  template <class M>
    requires _member_num_rows<M> && _member_num_cols<M> && row_major<M>
  auto constexpr operator()(const M& m) const noexcept {
    return m.num_cols();
  }

  template <class M>
    requires _member_num_rows<M> && _member_num_cols<M> && col_major<M>
  auto constexpr operator()(const M& m) const noexcept {
    return m.num_rows();
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
  { t.num_vectors() };
};

struct _fn {
  template <_member_num_vectors T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.num_vectors();
  }

  // @todo This is a total temporary hack
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
};
}  // namespace _num_vectors

inline namespace _cpo {
inline constexpr auto num_vectors = _num_vectors::_fn{};
}  // namespace _cpo


// ----------------------------------------------------------------------------
// num_partitions CPO
// ----------------------------------------------------------------------------
namespace _num_partitions {
void num_partitions(auto&) = delete;
void num_partitions(const auto&) = delete;

template <class T>
concept _member_num_partitions = requires(T t) {
  { t.num_partitions() };
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

#endif // TILEDB_CPOS_H