/**
 * @file   linalg_defs.h
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
 */

#ifndef TDB_LINALG_DEFS_H
#define TDB_LINALG_DEFS_H

#include <string>
#include <tiledb/tiledb>
#include "mdspan/mdspan.hpp"

namespace stdx {
using namespace Kokkos;
// using namespace Kokkos::Experimental;
}  // namespace stdx

template <class LayoutPolicy>
struct order_traits {
  constexpr static auto order{TILEDB_ROW_MAJOR};
};

template <>
struct order_traits<stdx::layout_right> {
  constexpr static auto order{TILEDB_ROW_MAJOR};
};

template <>
struct order_traits<stdx::layout_left> {
  constexpr static auto order{TILEDB_COL_MAJOR};
};

template <class LayoutPolicy>
constexpr auto order_v = order_traits<LayoutPolicy>::order;

#endif  // TDB_LINALG_DEFS_H
