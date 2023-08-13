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
 * Nascent concepts for code organization and to help with testing.
 *
 */

#ifndef TDB_CONCEPTS_H
#define TDB_CONCEPTS_H

#include <concepts>
#include <ranges>
#include <span>
#include <type_traits>

template <typename T>
concept has_load_member = requires(T&& t) {
  t.load();
};

template <class T>
constexpr bool is_loadable_v = has_load_member<T>;

template <class T>
bool load(T&& t) {
  return false;
}

template <has_load_member T>
bool load(T&& t) {
  return t.load();
}

template <typename T>
concept has_col_offset = requires(T&& t) {
  t.col_offset();
};

template <typename T>
concept has_num_col_parts = requires(T&& t) {
  t.num_col_parts();
};

template <typename T>
concept feature_vector = requires(T t) {
  typename T::value_type;
  //  typename T::index_type;
  //  typename T::size_type;
  //  typename T::reference;
  { t.size() } -> std::convertible_to<std::size_t>;
  requires(
      requires(T t) {
        { t[0] } -> std::convertible_to<typename T::value_type>;
      } ||
      requires(T t) {
        { t(0) } -> std::convertible_to<typename T::value_type>;
      });
  { t.data() } -> std::convertible_to<typename T::value_type*>;
  { num_features(t) } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept query_vector = feature_vector<T>;

template <typename T>
concept vector_database = requires(T t) {
  typename T::value_type;
  //  typename T::index_type;
  //  typename T::size_type;
  //  typename T::reference;
  { t.size() } -> std::convertible_to<std::size_t>;
  { t.rank() } -> std::convertible_to<std::size_t>;
  { t[0] } -> std::convertible_to<std::span<typename T::value_type>>;
  { t(0, 0) } -> std::convertible_to<typename T::value_type>;
  { t.data() } -> std::convertible_to<typename T::value_type*>;
  {t.rank() == 2};
  { raveled(t) } -> std::convertible_to<std::span<typename T::value_type>>;
};

template <typename T>
concept query_set = vector_database<T>;

#endif
