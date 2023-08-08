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
#include "utils/print_types.h"

#include <list>
#include <string>
#include <vector>

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

TEST_CASE("api: contiguous range", "[api]") {
  CHECK(!std::ranges::contiguous_range<int>);
  CHECK(std::ranges::contiguous_range<std::vector<int>>);
  CHECK(std::ranges::contiguous_range<std::vector<double>>);
  CHECK(std::ranges::contiguous_range<std::vector<std::vector<int>>>);
  CHECK(std::ranges::contiguous_range<std::array<int, 3>>);
  CHECK(!std::ranges::contiguous_range<std::list<int>>);
  CHECK(!std::ranges::contiguous_range<std::list<std::vector<int>>>);
  CHECK(std::ranges::contiguous_range<std::span<int>>);
  CHECK(std::ranges::contiguous_range<Vector<int>>);
}

TEST_CASE("api: subscriptable_container", "[api]") {
  using sv = std::vector<int>;
  using svi = std::ranges::iterator_t<sv>;
  using svri = std::iter_reference_t<svi>;

  // print_types(sv{}, sv{}.begin(), sv{}.cbegin(), svi{}, svi{}[0],
  // std::iter_value_t<svi>{});

  CHECK(!subscriptable_container<int>);
  CHECK(subscriptable_container<std::vector<int>>);
  CHECK(subscriptable_container<std::vector<double>>);
  CHECK(subscriptable_container<const std::vector<double>>);
  CHECK(subscriptable_container<std::vector<std::vector<int>>>);
  CHECK(subscriptable_container<std::array<int, 3>>);

  int* d = new int[3];  // random access iterator but not random access range
  CHECK(!subscriptable_container<decltype(d)>);
  CHECK(subscriptable_container<std::span<int>>);

  CHECK(subscriptable_container<Vector<int>>);
}

TEST_CASE("api: range_of_ranges", "[api]") {
  CHECK(!range_of_ranges<int>);
  CHECK(!range_of_ranges<std::vector<int>>);
  CHECK(!range_of_ranges<std::vector<double>>);
  CHECK(range_of_ranges<std::vector<std::vector<int>>>);
  CHECK(range_of_ranges<std::vector<std::list<double>>>);
  CHECK(range_of_ranges<std::vector<std::list<std::vector<std::string>>>>);
  CHECK(!range_of_ranges<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!range_of_ranges<decltype(d)>);

  int** e = nullptr;
  CHECK(!range_of_ranges<decltype(e)>);

  CHECK(!range_of_ranges<std::span<int>>);

  CHECK(!range_of_ranges<Vector<int>>);
}

TEST_CASE("api: inner_range", "[api]") {
  CHECK(std::is_same_v<
        inner_range_t<std::vector<std::vector<int>>>,
        std::vector<int>>);
  CHECK(std::is_same_v<
        inner_range_t<std::vector<std::vector<int>>>,
        typename std::vector<std::vector<int>>::value_type>);
}

TEST_CASE("api: inner_iterator_t", "[api]") {
  CHECK(std::is_same_v<
        inner_iterator_t<std::vector<std::vector<int>>>,
        std::vector<int>::iterator>);
}

TEST_CASE("api: inner_const_iterator_t", "[api]") {
  // print_types(inner_const_iterator_t<std::vector<std::vector<int>>>{},
  // std::vector<int>::const_iterator{ });
  CHECK(std::is_same_v<
        inner_const_iterator_t<std::vector<std::vector<int>>>,
        std::vector<int>::const_iterator>);
}

TEST_CASE("api: inner_value_t", "[api]") {
  CHECK(std::is_same_v<
        inner_value_t<std::vector<std::vector<int>>>,
        std::vector<int>::value_type>);
}

TEST_CASE("api: inner_reference_t", "[api]") {
  CHECK(std::is_same_v<
        inner_reference_t<std::vector<std::vector<int>>>,
        std::vector<int>::reference>);
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

TEST_CASE("api: Vector", "[api]") {
  CHECK(!range_of_ranges<Vector<int>>);
  CHECK(std::ranges::random_access_range<Vector<int>>);
  CHECK(std::ranges::sized_range<Vector<int>>);
  CHECK(std::ranges::contiguous_range<Vector<int>>);
  CHECK(subscriptable_container<Vector<int>>);
  CHECK(requires { Vector<int>{}.size(); });
  CHECK(std::ranges::sized_range<Vector<int>>);
}

template <class T>
class dummy_cpos {
 public:
  std::vector<T> vec;
  std::vector<std::vector<T>> vec_of_vec;
  std::vector<std::vector<std::vector<T>>> vec_of_vec_of_vec;

  auto dimension() const {return vec.size();}
  auto num_vectors() const {return vec_of_vec.size();}
  auto num_partitions() const {return vec_of_vec.size();}
};

// dimension
// num_vectors
// num_partitions




template <class T>
class dummy_feature_vector : public std::vector<T> {
 public:
  using std::vector<T>::vector;

  auto dimension() const { return this->size(); }
};




TEST_CASE("api: feature_vector", "[api]") {
  CHECK(!feature_vector<int>);
  CHECK(!feature_vector<std::vector<int>>);
  CHECK(!feature_vector<std::vector<double>>);
  CHECK(!feature_vector<std::vector<std::vector<int>>>);
  CHECK(!feature_vector<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!feature_vector<decltype(d)>);
  CHECK(!feature_vector<std::span<int>>);

  CHECK(feature_vector<dummy_feature_vector<int>>);
}

TEST_CASE("api: query_vector", "[api]") {
  CHECK(!query_vector<int>);
  CHECK(!query_vector<std::vector<int>>);
  CHECK(!query_vector<std::vector<double>>);
  CHECK(!query_vector<std::vector<std::vector<int>>>);
  CHECK(!query_vector<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!query_vector<decltype(d)>);
  CHECK(!query_vector<std::span<int>>);

  CHECK(!query_vector<Vector<int>>);
  CHECK(query_vector<dummy_feature_vector<int>>);
}

TEST_CASE("api: feature_vector_range", "[api]") {
}

TEST_CASE("api: contiguous_feature_vector_range", "[api]") {
}

TEST_CASE("api: partitioned_feature_vector_range", "[api]") {
}

TEST_CASE("api: contiguous_partitioned_feature_vector_range", "[api]") {
}
