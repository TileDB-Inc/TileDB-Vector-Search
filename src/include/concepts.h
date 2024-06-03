/**
 * @file   concepts.h
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

#ifndef TDB_CONCEPTS_H
#define TDB_CONCEPTS_H

#include <concepts>
#include <ranges>

#include "cpos.h"

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
concept subscriptable_range =
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
concept dimensionable = requires(const T& t) {
  { dimensions(t) } -> semi_integral;
};

template <class T>
concept vectorable = requires(const T& t) {
  { num_vectors(t) } -> semi_integral;
};

template <class T>
concept partitionable = requires(const T& t) {
  { t.indices() };
  { t.ids() };
};

// ----------------------------------------------------------------------------
// feature_vector concept
// ----------------------------------------------------------------------------

template <typename R>
concept feature_vector =
    std::ranges::random_access_range<R> && std::ranges::contiguous_range<R> &&
    dimensionable<R> && (subscriptable_range<R> || callable_range<R>);

template <class R>
concept query_vector = feature_vector<R>;

// ----------------------------------------------------------------------------
// feature_vector_array concept
//
// A vector range is a range of feature vectors.
// We will want at least these kinds:
//   contiguous -- all the vectors are stored in a single contiguous array
//   partitioned -- the range is a range of partitions, each of which is a
//                  contiguous range of vectors
//   distributed?-- the range is a range of partitions, each of which is a
//
// @todo operator()(size_t, size_t) ?
// ----------------------------------------------------------------------------
template <class D>
concept feature_vector_array = requires(D d, size_t n) {
  { num_vectors(d) } -> semi_integral;
  { dimensions(d) } -> semi_integral;
  { d[n] } -> feature_vector;
};

template <class D>
concept feature_vector_array_with_ids =
    feature_vector_array<D> && requires(D d, size_t i) {
      { d.ids() };
      { d.id(i) };
    };

/**
 * @brief A concept for contiguous vector ranges.  The member function data()
 * returns a pointer to the underlying contiguous one-dimensional storage.
 * @tparam D
 *
 */
// @todo -- add ranges::contiguous_range as a requirement
template <class D>
concept contiguous_feature_vector_array =
    feature_vector_array<D> && requires(D d) {
      { data(d) } -> std::same_as<std::add_pointer_t<typename D::reference>>;
    };

template <class T>
concept query_vector_array = feature_vector_array<T>;

template <class T>
concept contiguous_query_vector_array = contiguous_feature_vector_array<T>;

// ----------------------------------------------------------------------------
// partitioned_feature_vector_range concept (WIP)
// ----------------------------------------------------------------------------

template <class D>
concept partitioned_feature_vector_array =
    feature_vector_array<D> && partitionable<D>;

template <class D>
concept contiguous_partitioned_feature_vector_array =
    partitioned_feature_vector_array<D> && std::ranges::contiguous_range<D> &&
    requires(D d) {
      { d.vectors() };
      { d.indices() };
      { d.ids() };
    };

// ----------------------------------------------------------------------------
// partition_index concept (WIP)
// ----------------------------------------------------------------------------
template <class P>
concept partition_index =
    std::ranges::random_access_range<P> && std::ranges::contiguous_range<P> &&
    subscriptable_range<P>;

// ----------------------------------------------------------------------------
// vector_search_index concept (WIP)
// ----------------------------------------------------------------------------
template <typename I>
concept vector_search_index = requires(I i) {
  { i.train() };
  { i.add() };
  { i.search() };
};

// ----------------------------------------------------------------------------
// distance function concepts (for function objects)
// - A distance function takes two feature vectors and returns a distance.
// - A sub_distance function takes two feature vectors, a start and a stop, and
//   returns a distance.
// - A cached_sub_distance function takes a start and a stop in its constructor
//   and takes two feature vectors in operator() and returns a distance.
// ----------------------------------------------------------------------------

/**
 * @brief A concept for distance functions.
 *
 * @tparam F The function type.  Must be invocable on U and V.
 * @tparam U The first feature vector type.  Must be a feature_vector.
 * @tparam V The second feature vector type.  Must be a feature_vector.
 */
template <class F, class U, class V>
concept distance_function =
    feature_vector<U> && feature_vector<V> && std::regular_invocable<F, U, V> &&
    std::regular_invocable<F, V, U> && std::regular_invocable<F, U, U> &&
    std::regular_invocable<F, V, V>;

/**
 * @brief A concept for distance functions operating on subranges of a feature
 * vector.
 *
 * @tparam F The function type.  Must be invocable on U, V, size_t, size_t.
 * @tparam U The first feature vector type.  Must be a feature_vector.
 * @tparam V The second feature vector type.  Must be a feature_vector.
 */
template <class F, class U, class V>
concept sub_distance_function =
    feature_vector<U> && feature_vector<V> &&
    // perhaps could also just use requires clause that f(u, v, i, j) is valid
    std::regular_invocable<F, U, V, size_t, size_t> &&
    std::regular_invocable<F, V, U, size_t, size_t> &&
    std::regular_invocable<F, U, U, size_t, size_t> &&
    std::regular_invocable<F, V, V, size_t, size_t>;

template <class F, class U, class V>
concept uncached_sub_distance_function = sub_distance_function<F, U, V>;
/**
 * @brief A concept for distance functions operating on subranges of a feature
 * vector, where the subrange is specified at construction time.  Thereafter,
 * the object acts like a distance function.
 * @tparam F The function type.  Must be constructable with size_t, size_t and
 * invocable on U, V.
 * @tparam U The first feature vector type.  Must be a feature_vector.
 * @tparam V The second feature vector type.  Must be a feature_vector.
 */
template <class F, class U, class V>
concept cached_sub_distance_function =
    distance_function<F, U, V> && feature_vector<U> && feature_vector<V> &&
    requires(size_t i, size_t j) { F(i, j); };

#endif  // TDB_CONCEPTS_H
