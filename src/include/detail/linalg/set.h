/**
 * @file set.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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

#ifndef TILEDB_SET_H
#define TILEDB_SET_H

#include <unordered_set>

template <typename T>
void debug_unordered_set(
    const std::unordered_set<T>& s,
    const std::string& msg = "",
    size_t max_size = 10) {
  if (s.empty()) {
    std::cout << msg << ": (empty)\n";
    return;
  }

  size_t end = std::min(max_size, s.size());
  if (!msg.empty()) {
    std::cout << msg << ": ";
  }
  std::cout << "[";
  size_t i = 0;
  for (const auto& val : s) {
    std::cout << val;
    if (i >= end) {
      std::cout << "...";
      break;
    }
    if (i != end - 1) {
      std::cout << ", ";
    }
    i++;
  }
  std::cout << "]\n";
}

#endif  // TILEDB_SET_H
