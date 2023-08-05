/**
 * @file   unit_api.cc
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

#include <catch2/catch_all.hpp>
#include "api.h"
#include "detail/linalg/vector.h"

TEST_CASE("api: test test", "[api]") {
  REQUIRE(true);
}

TEST_CASE("api: random_access_range", "[api]") {
  CHECK(!std::ranges::random_access_range<int>);

  CHECK(std::ranges::random_access_range<std::vector<int>>);
  CHECK(std::ranges::random_access_range<std::vector<double>>);
  CHECK(std::ranges::random_access_range<std::vector<std::vector<int>>>);
  CHECK(std::ranges::random_access_range<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!std::ranges::random_access_range<decltype(d)>);
  CHECK(std::ranges::random_access_range<std::span<int>>);

  CHECK(std::ranges::random_access_range<Vector<int>>);
}

TEST_CASE("api: sized_range", "[api]") {
  CHECK(!std::ranges::sized_range<int>);

  CHECK(std::ranges::sized_range<std::vector<int>>);
  CHECK(std::ranges::sized_range<std::vector<double>>);
  CHECK(std::ranges::sized_range<std::vector<std::vector<int>>>);
  CHECK(std::ranges::sized_range<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!std::ranges::sized_range<decltype(d)>);
  CHECK(std::ranges::sized_range<std::span<int>>);

  CHECK(std::ranges::sized_range<Vector<int>>);
}

TEST_CASE("api: subscriptable", "[api]") {
  CHECK(!subscriptable<int>);
  CHECK(subscriptable<std::vector<int>>);
  CHECK(subscriptable<std::vector<double>>);
  CHECK(subscriptable<std::vector<std::vector<int>>>);
  CHECK(subscriptable<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(subscriptable<decltype(d)>);
  CHECK(subscriptable<std::span<int>>);

  CHECK(subscriptable<Vector<int>>);
}

template <class T>
requires callable<int, T, int>
void foo(const T&) {
}

struct bar {
  int operator()(int) {
    return 0;
  }
};

TEST_CASE("api: invocable", "[api]") {
  foo(bar{});
  std::invoke(bar{}, 1);
  std::invoke(Vector<int>{}, 1);

  CHECK(std::is_invocable_r_v<int, bar, int>);
  CHECK(std::is_invocable_r_v<int, bar, int>);
  CHECK(std::is_invocable_r_v<
        std::iter_reference_t<Vector<int>::iterator>,
        Vector<int>,
        int>);
  CHECK(std::is_invocable_r_v<int, Vector<int>, int>);
}

TEST_CASE("api: callable_range", "[api]") {
  CHECK(!callable_range<int>);
  CHECK(!callable_range<std::vector<int>>);
  CHECK(!callable_range<std::vector<double>>);
  CHECK(!callable_range<std::vector<std::vector<int>>>);
  CHECK(!callable_range<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!callable_range<decltype(d)>);
  CHECK(!callable_range<std::span<int>>);

  CHECK(callable_range<Vector<int>>);

  CHECK(callable<int, bar, int>);
}
