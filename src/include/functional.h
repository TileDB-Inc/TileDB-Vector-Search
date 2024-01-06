/**
 * @file   functional.h
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
 * Implementations of various utilities similar to those in <functional>
 *
 */

#ifndef TDB_FUNCTIONAL_H
#define TDB_FUNCTIONAL_H

#include <functional>

namespace stdx {}  // namespace stdx

template <class T = void>
struct first_less {
  constexpr bool operator()(const T& lhs, const T& rhs) const {
    return std::get<0>(lhs) < std::get<0>(rhs);
  }
};

template <>
struct first_less<void> {
  template <class T1, class T2>
  constexpr bool operator()(const T1& lhs, const T2& rhs) const {
    return std::get<0>(lhs) < std::get<0>(rhs);
  }
};

#endif
