/**
 * * @file   test/randomize.h
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

#ifndef TILEDB_RANDOMIZE_H
#define TILEDB_RANDOMIZE_H

#include <random>

template <std::ranges::range R>
void randomize(R& r, std::tuple<int, int> range = {0, 128}) {
  using element_type = std::ranges::range_value_t<R>;

  std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(2514908090);

  if constexpr (std::is_floating_point_v<element_type>) {
    std::uniform_real_distribution<element_type> dist(
        std::get<0>(range), std::get<1>(range));
    for (auto& x : r) {
      x = dist(gen);
    }
  } else {
    if constexpr (sizeof(element_type) == 1u) {
      // For MSVC the std::uniform_int_distribution does not compile for char
      // types e.g. char, uint8_t, int8_t, signed char...
      std::uniform_int_distribution<uint16_t> dist(
          std::get<0>(range), std::get<1>(range));
      for (auto& x : r) {
        x = static_cast<element_type>(dist(gen));
      }
    } else {
      std::uniform_int_distribution<element_type> dist(
          std::get<0>(range), std::get<1>(range));
      for (auto& x : r) {
        x = dist(gen);
      }
    }
  }
}

template <std::ranges::range R>
void randomize(R&& r, std::tuple<int, int> range = {0, 128}) {
  randomize(r, range);
}

#endif  // TILEDB_RANDOMIZE_H
