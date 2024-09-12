/**
 * @file   unit_algorithm.cc
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

#include <catch2/catch_all.hpp>
#include "algorithm.h"
#include "api/feature_vector_array.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/vector.h"

TEST_CASE("for_each", "[algorithm]") {
  size_t length = GENERATE(
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      13,
      17,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048,
      4096);
  std::vector<int> v(length);
  std::iota(begin(v), end(v), 1);

  SECTION("sequential") {
    stdx::for_each(begin(v), end(v), [](int& x) { x *= 2; });
  }

  SECTION("parallel") {
    auto nthreads = GENERATE(1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 63);

    stdx::for_each(
        stdx::execution::parallel_policy(nthreads),
        begin(v),
        end(v),
        [](int& x) { x *= 2; });
  }

  SECTION("indexed parallel") {
    size_t nthreads = GENERATE(1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 63);
    std::vector<unsigned> indices(nthreads);

    stdx::for_each(
        stdx::execution::indexed_parallel_policy(nthreads),
        begin(v),
        end(v),
        [&indices](int& x, auto&& n, auto&& i) mutable {
          x += i + 1;
          indices[n] = x;
        });

    if (length > 0) {
      for (size_t n = 0; n < nthreads; ++n) {
        size_t block_size = (length + nthreads - 1) / nthreads;
        auto start = std::min<size_t>(n * block_size, length);
        auto stop = std::min<size_t>((n + 1) * block_size, length);
        unsigned j = stop * 2;
        if (stop != start) {
          CHECK(indices[n] == j);
        } else {
          CHECK(indices[n] == 0);
        }
      }
    }
  }

  CHECK(size(v) == length);

  switch (length) {
    case 4096:
      CHECK(v[4095] == 8192);
      [[fallthrough]];
    case 2048:
      CHECK(v[2047] == 4096);
      [[fallthrough]];
    case 1024:
      CHECK(v[1023] == 2048);
      [[fallthrough]];
    case 512:
      CHECK(v[511] == 1024);
      [[fallthrough]];
    case 256:
      CHECK(v[255] == 512);
      [[fallthrough]];
    case 128:
      CHECK(v[127] == 256);
      [[fallthrough]];
    case 64:
      CHECK(v[63] == 128);
      [[fallthrough]];
    case 32:
      CHECK(v[31] == 64);
      [[fallthrough]];
    case 17:
      CHECK(v[16] == 34);
      [[fallthrough]];
    case 13:
      CHECK(v[12] == 26);
      [[fallthrough]];
    case 11:
      CHECK(v[10] == 22);
      [[fallthrough]];
    case 10:
      CHECK(v[9] == 20);
      [[fallthrough]];
    case 9:
      CHECK(v[8] == 18);
      [[fallthrough]];
    case 8:
      CHECK(v[7] == 16);
      [[fallthrough]];
    case 7:
      CHECK(v[6] == 14);
      [[fallthrough]];
    case 6:
      CHECK(v[5] == 12);
      [[fallthrough]];
    case 5:
      CHECK(v[4] == 10);
      [[fallthrough]];
    case 4:
      CHECK(v[3] == 8);
      [[fallthrough]];
    case 3:
      CHECK(v[2] == 6);
      [[fallthrough]];
    case 2:
      CHECK(v[1] == 4);
      [[fallthrough]];
    case 1:
      CHECK(v[0] == 2);
      break;
    case 0:
      CHECK(v.empty());
      break;
    default:
      CHECK(false);
  }
}

TEST_CASE("range_for_each", "[algorithm]") {
  size_t k = 100;
  size_t num_vectors = 100;
  size_t dimensions = 128;

  auto queries = ColMajorMatrix<float>(dimensions, num_vectors);
  for (size_t i = 0; i < num_vectors; ++i) {
    for (size_t j = 0; j < dimensions; ++j) {
      queries[i][j] = i;
    }
  }

  // Test with both a vector (b/c it's simple) and matrix (b/c it's what we use
  // in practice).
  std::vector<size_t> thread_used(num_vectors, 0);
  auto thread_used_matrix = ColMajorMatrix<float>(k, num_vectors);

  for (size_t num_threads = 1;
       num_threads <= std::thread::hardware_concurrency();
       ++num_threads) {
    stdx::range_for_each(
        stdx::execution::indexed_parallel_policy{num_threads},
        queries,
        [&](auto&& vector, auto thread_index, auto index) {
          thread_used[index] = thread_index;
          for (size_t j = 0; j < k; ++j) {
            thread_used_matrix[index][j] = thread_index;
          }
        });

    size_t block_size = (num_vectors + num_threads - 1) / num_threads;

    for (size_t i = 0; i < num_vectors; ++i) {
      size_t expected_thread = i / block_size;
      CHECK(thread_used[i] == expected_thread);

      for (size_t j = 0; j < k; ++j) {
        CHECK(thread_used_matrix[i][j] == expected_thread);
      }
    }
  }
}
