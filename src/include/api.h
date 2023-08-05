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
 * Nascent C++ API (including concepts) for code organization and to help with
 * testing.
 *
 */

#ifndef TDB_API_H
#define TDB_API_H

#include <concepts>
#include <ranges>

template <class T>
concept subscriptable = requires(const T& x) {
  x[0];
};

template <class R, class T, class... Args>
concept callable = std::is_invocable_r_v<R, T, Args...>;

template <class R>
concept callable_range = callable<
    std::iter_reference_t<std::ranges::iterator_t<R>>,
    R,
    std::iter_difference_t<std::ranges::iterator_t<R>>>;

template <typename R>
concept feature_vector = std::ranges::random_access_range<R> &&
    std::ranges::sized_range<R> && std::ranges::contiguous_range<R> &&
    subscriptable<R> && callable_range<R> && requires(R t) {
  {
    t.data()
    } -> std::same_as<
        std::remove_reference_t<std::ranges::range_reference_t<R>>*>;
  { t.dimension() } -> std::convertible_to<std::size_t>;
};

template <class R>
concept query_vector = feature_vector<R>;

/**
 * @brief Returns the dimension of a feature vector.
 *
 * @tparam R The feature vector type.
 * @param r The feature vector.
 * @return The dimension of the feature vector.
 */
template <feature_vector R>
auto dimension(const R& r) {
  return r.dimension();
}

template <feature_vector R>
auto data(const R& r) {
  return r.data();
}

#endif  // TDB_API_H