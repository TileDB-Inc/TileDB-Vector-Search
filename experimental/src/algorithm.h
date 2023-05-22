/**
 * @file   algorithm.h
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
 * Header-only library of some basic linear algebra data structures and
 * operations. Uses C++23 reference implementation of mdspan from Sandia
 * National Laboratories.
 *
 */

#ifndef TDB_ALGORITHM_H
#define TDB_ALGORITHM_H

#include <algorithm>
#include <concepts>
#include <execution>
#include <functional>
#include <future>
#include <thread>
#include <utility>

namespace stdx {

namespace execution {
struct parallel_policy {
  const size_t nthreads_;
  parallel_policy(size_t nthreads = std::thread::hardware_concurrency())
      : nthreads_(nthreads) {
  }
};
class sequenced_policy {};
class unsequenced_policy {};
struct parallel_unsequenced_policy {
  const size_t nthreads_;
  parallel_unsequenced_policy(
      size_t nthreads = std::thread::hardware_concurrency())
      : nthreads_(nthreads) {
  }
};
struct indexed_parallel_policy {
  const size_t nthreads_;
  indexed_parallel_policy(size_t nthreads = std::thread::hardware_concurrency())
      : nthreads_(nthreads) {
  }
};
};  // namespace execution

template <std::random_access_iterator RandomIt, class UnaryFunction>
void for_each(RandomIt first, RandomIt last, UnaryFunction f) {
  std::for_each(first, last, f);
}

// @todo:  Use `advance()` to handle non-random access iterators
template <std::random_access_iterator RandomIt, class UnaryFunction>
void for_each(
    stdx::execution::parallel_policy&& par,
    RandomIt begin,
    RandomIt end,
    UnaryFunction f) {
  size_t container_size = end - begin;
  size_t nthreads = par.nthreads_;
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<RandomIt>(begin + n * block_size, end);
    auto stop = std::min<RandomIt>(start + block_size, end);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async,
          [start, stop, f = std::forward<UnaryFunction>(f)]() mutable {
            std::for_each(start, stop, std::forward<UnaryFunction>(f));
          }));
    }
  }
  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].wait();
  }
}

template <std::random_access_iterator RandomIt, class UnaryFunction>
void for_each(
    stdx::execution::indexed_parallel_policy&& par,
    RandomIt begin,
    RandomIt end,
    UnaryFunction f) {
  size_t container_size = end - begin;
  size_t nthreads = par.nthreads_;
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async,
          [n,
           begin,
           start,
           stop,
           f = std::forward<UnaryFunction>(f)]() mutable {
            for (size_t i = start; i < stop; ++i) {
              std::forward<UnaryFunction>(f)(begin[i], n, i);
            }
          }));
    }
  }
  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].wait();
  }
}

template <class /*std::ranges::random_access_range*/ Range, class UnaryFunction>
void range_for_each(
    stdx::execution::indexed_parallel_policy&& par,
    Range&& range,
    UnaryFunction f) {
  size_t container_size = size(range);
  size_t nthreads = par.nthreads_;
  size_t block_size = (container_size + nthreads - 1) / nthreads;

  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  for (size_t n = 0; n < nthreads; ++n) {
    auto start = std::min<size_t>(n * block_size, container_size);
    auto stop = std::min<size_t>((n + 1) * block_size, container_size);

    if (start != stop) {
      futs.emplace_back(std::async(
          std::launch::async,
          [n,
           &range,
           start,
           stop,
           f = std::forward<UnaryFunction>(f)]() mutable {
            for (size_t i = start; i < stop; ++i) {
              std::forward<UnaryFunction>(f)(range[i], n, i);
            }
          }));
    }
  }
  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].wait();
  }
}

}  // namespace stdx

#endif  // TDB_ALGORITHM_H