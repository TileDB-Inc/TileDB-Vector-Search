/**
 * @file   api.h
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
 * Nascent C++ API (including concepts).
 *
 */

#ifndef TDB_API_H
#define TDB_API_H

#include <concepts>
#include <ranges>

// ----------------------------------------------------------------------------
// Convenience concepts
// ----------------------------------------------------------------------------

template <typename R>
concept range_of_ranges =
    std::ranges::range<R> && std::ranges::range<std::ranges::range_value_t<R>>;

template <typename R>
  requires range_of_ranges<R>
using inner_range_t = std::ranges::range_value_t<R>;

template <typename R>
  requires range_of_ranges<R>
using inner_iterator_t = std::ranges::iterator_t<inner_range_t<R>>;

template <typename R>
using inner_const_iterator_t = std::ranges::iterator_t<const inner_range_t<R>>;

template <typename R>
using inner_value_t = std::ranges::range_value_t<inner_range_t<R>>;

template <typename R>
using inner_reference_t = std::ranges::range_reference_t<inner_range_t<R>>;

// ----------------------------------------------------------------------------
// Utility concepts
// ----------------------------------------------------------------------------

/**
 * @brief A concept for types that have `operator[]` defined.
 *
 * @tparam T The type to check.
 */
template <class T>
concept subscriptable_container =
    std::ranges::random_access_range<T> &&
    requires(
        T& x,
        const T& y,
        const std::iter_difference_t<std::ranges::iterator_t<T>> n) {
      {
        x[n]
      } -> std::same_as<std::iter_reference_t<std::ranges::iterator_t<T>>>;
      {
        y[n]
      }
      -> std::same_as<std::iter_reference_t<std::ranges::iterator_t<const T>>>;
    };

/**
 * @brief A concept for types that have `operator()` defined.
 *
 * @tparam R The return type.
 * @tparam T The type to check.
 * @tparam Args The argument types.
 */
template <class R, class T, class... Args>
concept callable = std::is_invocable_r_v<R, T, Args...>;

template <class R>
concept callable_range =
    std::ranges::range<R> &&
    callable<
        std::iter_reference_t<std::ranges::iterator_t<R>>,
        R,
        std::iter_difference_t<std::ranges::iterator_t<R>>>;

// ----------------------------------------------------------------------------
// Expected member functions for feature_vectors -- will be used by CPOs
// ----------------------------------------------------------------------------

template <class T>
concept dimensionable = requires(T t) {
  { t.dimension() };
};

/**
 * @brief Returns the dimension of a feature vector (which in vector search
 * terminology is the number of entries, aka number of rows, aka size.
 *
 * @tparam R The feature vector type.
 * @param r The feature vector.
 * @return The dimension of the feature vector.
 */
template <dimensionable R>
auto dimension(const R& r) {
  return r.dimension();
}

template <class T>
concept vectorable = requires(T t) {
  { t.num_vectors() };
};


template <class T>
concept partitionable = requires(T t) {
  { t.num_partitions() };
};


// ----------------------------------------------------------------------------
// feature_vector concept
// ----------------------------------------------------------------------------

template <typename R>
concept feature_vector =
    std::ranges::random_access_range<R> && /* std::ranges::sized_range<R> && */
    std::ranges::contiguous_range<R> &&
    (subscriptable_container<R> || callable_range<R>)&&requires(R r) {
      { r.dimension() } -> std::same_as<typename R::size_type>;
    };

template <class R>
concept query_vector = feature_vector<R>;

// ----------------------------------------------------------------------------
// vector_range concept
//
// A vector range is a range of feature vectors.
// We will want at least these kinds:
//   contiguous -- all the vectors are stored in a single contiguous array
//   partitioned -- the range is a range of partitions, each of which is a
//                  contiguous range of vectors
//   distributed?-- the range is a range of partitions, each of which is a
// ----------------------------------------------------------------------------
template <class D>
concept feature_vector_range =
    feature_vector<inner_range_t<D>> &&
    std::ranges::random_access_range<D> && /* std::ranges::sized_range<D> && */
    subscriptable_container<D> &&
    feature_vector<std::ranges::range_value_t<D>> &&
    requires(D d, const std::iter_difference_t<std::ranges::iterator_t<D>> n) {
      { d.num_vectors() } -> std::same_as<typename D::size_type>;
      { d.dimension() } -> std::same_as<inner_range_t<typename D::size_type>>;

      // Returns a feature_vector
      // { d[n] } ->
      // std::same_as<std::iter_reference_t<std::ranges::iterator_t<D>>>; { d[n]
      // } -> feature_vector;
    };

/**
 * @brief A concept for contiguous vector ranges.  The member function data()
 * returns a pointer to the underlying contiguous one-dimensional storage.
 * @tparam D
 *
 */
template <class D>
concept contiguous_feature_vector_range =
    feature_vector_range<D> && std::ranges::contiguous_range<D>;

// ----------------------------------------------------------------------------
// partitioned_feature_vector_range concept
// ----------------------------------------------------------------------------

template <class D>
concept partitioned_feature_vector_range =
    feature_vector_range<D> && partitionable<D>;

template <class D>
concept contiguous_partitioned_feature_vector_range =
    partitioned_feature_vector_range<D> && std::ranges::contiguous_range<D>;

// ----------------------------------------------------------------------------
// partition_index concept
// ----------------------------------------------------------------------------
template <class P>
concept partition_index =
    std::ranges::random_access_range<P> && std::ranges::contiguous_range<P> &&
    subscriptable_container<P>;

// ----------------------------------------------------------------------------
// vector_search_index concept
// ----------------------------------------------------------------------------
template <typename I>
concept vector_search_index = requires(I i) {
  { i.train() };
  { i.add() };
  { i.search() };
};

// ----------------------------------------------------------------------------
// Customization point objects (CPOs) -- implemented as "niebloids"
// ----------------------------------------------------------------------------

namespace _dimension {
void dimension(auto&) = delete;
void dimension(const auto&) = delete;

struct _fn {
  template <dimensionable T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.dimension();
  }
};
}  // namespace _dimension

inline namespace _cpo {
inline constexpr auto dimension = _dimension::_fn{};
}  // namespace _cpo

namespace _num_partitions {
void num_partitions(auto&) = delete;
void num_partitions(const auto&) = delete;

struct _fn {
  template <partitionable T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.num_partitions();
  }
};
}  // namespace _num_partitions

inline namespace _cpo {
inline constexpr auto num_partitions = _num_partitions::_fn{};
}  // namespace _cpo

namespace _num_vectors {
void num_vectors(auto&) = delete;
void num_vectors(const auto&) = delete;

struct _fn {
  template <vectorable T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.num_vectors();
  }
};
}  // namespace _num_vectors

inline namespace _cpo {
inline constexpr auto num_vectors = _num_vectors::_fn{};
}  // namespace _cpo


#endif  // TDB_API_H