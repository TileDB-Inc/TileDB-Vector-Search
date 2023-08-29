/**
 * @file   unit_concepts.cc
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
#include "concepts.h"
#include "detail/linalg/vector.h"
#include "detail/linalg/matrix.h"
#include "utils/print_types.h"

#include <list>
#include <string>
#include <vector>

TEST_CASE("concepts: test test", "[concepts]") {
  REQUIRE(true);
}

TEST_CASE("concepts: random_access_range", "[concepts]") {
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

TEST_CASE("concepts: sized_range", "[concepts]") {
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

TEST_CASE("concepts: contiguous range", "[concepts]") {
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

TEST_CASE("concepts: range_of_ranges", "[concepts]") {
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

TEST_CASE("concepts: inner_range", "[concepts]") {
  CHECK(std::is_same_v<
        inner_range_t<std::vector<std::vector<int>>>,
        std::vector<int>>);
  CHECK(std::is_same_v<
        inner_range_t<std::vector<std::vector<int>>>,
        typename std::vector<std::vector<int>>::value_type>);
}

TEST_CASE("concepts: inner_iterator_t", "[concepts]") {
  CHECK(std::is_same_v<
        inner_iterator_t<std::vector<std::vector<int>>>,
        std::vector<int>::iterator>);
}

TEST_CASE("concepts: inner_const_iterator_t", "[concepts]") {
  // print_types(inner_const_iterator_t<std::vector<std::vector<int>>>{},
  // std::vector<int>::const_iterator{ });
  CHECK(std::is_same_v<
        inner_const_iterator_t<std::vector<std::vector<int>>>,
        std::vector<int>::const_iterator>);
}

TEST_CASE("concepts: inner_value_t", "[concepts]") {
  CHECK(std::is_same_v<
        inner_value_t<std::vector<std::vector<int>>>,
        std::vector<int>::value_type>);
}

TEST_CASE("concepts: inner_reference_t", "[concepts]") {
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


TEST_CASE("concepts: invocable", "[concepts]") {
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

TEST_CASE("concepts: subscriptable_container", "[concepts]") {
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

  CHECK(subscriptable_container<sv>);
  CHECK(!subscriptable_container<svi>);
  CHECK(!subscriptable_container<svri>);

  int* d = new int[3];  // random access iterator but not random access range
  CHECK(!subscriptable_container<decltype(d)>);
  CHECK(subscriptable_container<std::span<int>>);

  CHECK(subscriptable_container<Vector<int>>);
}


TEST_CASE("concepts: callable_range", "[concepts]") {
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

TEST_CASE("concepts: Vector", "[concepts]") {
  CHECK(!range_of_ranges<Vector<int>>);
  CHECK(std::ranges::random_access_range<Vector<int>>);
  CHECK(std::ranges::sized_range<Vector<int>>);
  CHECK(std::ranges::contiguous_range<Vector<int>>);
  CHECK(subscriptable_container<Vector<int>>);
  CHECK(requires { Vector<int>{}.size(); });
  CHECK(std::ranges::sized_range<Vector<int>>);
}

TEST_CASE("concepts: dimensionable", "[concepts]") {
  CHECK(!dimensionable<int>);
  CHECK(dimensionable<std::vector<int>>);
  CHECK(dimensionable<std::vector<double>>);
  CHECK(!dimensionable<std::vector<std::vector<int>>>);
  CHECK(dimensionable<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!dimensionable<decltype(d)>);
  CHECK(dimensionable<std::span<int>>);

  CHECK(dimensionable<Vector<int>>);
}

struct dummy_vectorable {
  auto num_vectors() const { return 0; }
};

TEST_CASE("concepts: vectorable", "[concepts]") {
  CHECK(!vectorable<int>);
  CHECK(!vectorable<std::vector<int>>);
  CHECK(!vectorable<std::vector<double>>);
  CHECK(!vectorable<std::vector<std::vector<int>>>);
  CHECK(!vectorable<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!vectorable<decltype(d)>);
  CHECK(!vectorable<std::span<int>>);

  CHECK(!vectorable<Vector<int>>);

  CHECK(vectorable<dummy_vectorable>);
}

TEST_CASE("concepts: partitionable", "[concepts]") {
  CHECK(!partitionable<int>);
  CHECK(!partitionable<std::vector<int>>);
  CHECK(!partitionable<std::vector<double>>);
  CHECK(!partitionable<std::vector<std::vector<int>>>);
  CHECK(!partitionable<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!partitionable<decltype(d)>);
  CHECK(!partitionable<std::span<int>>);

  CHECK(!partitionable<Vector<int>>);
}

template <class T>
class dummy_feature_vector : public std::vector<T> {
 public:
  using std::vector<T>::vector;

  auto dimension() const { return this->size(); }
};


TEST_CASE("concepts: feature_vector", "[concepts]") {
  CHECK(!feature_vector<int>);
  CHECK(feature_vector<std::vector<int>>);
  CHECK(feature_vector<std::vector<double>>);
  CHECK(!feature_vector<std::vector<std::vector<int>>>);
  CHECK(feature_vector<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!feature_vector<decltype(d)>);
  CHECK(feature_vector<std::span<int>>);

  CHECK(feature_vector<dummy_feature_vector<int>>);
}


TEST_CASE("concepts: query_vector", "[concepts]") {
  CHECK(!query_vector<int>);
  CHECK(query_vector<std::vector<int>>);
  CHECK(query_vector<std::vector<double>>);
  CHECK(!query_vector<std::vector<std::vector<int>>>);
  CHECK(query_vector<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!query_vector<decltype(d)>);
  CHECK(query_vector<std::span<int>>);

  CHECK(query_vector<Vector<int>>);
  CHECK(query_vector<dummy_feature_vector<int>>);
}

TEST_CASE("concepts: feature_vector_range", "[concepts]") {

  CHECK(!feature_vector_range<int>);
  CHECK(!feature_vector_range<std::vector<int>>);
  CHECK(!feature_vector_range<std::vector<double>>);
  CHECK(!feature_vector_range<std::vector<std::vector<int>>>);
}

TEST_CASE("concepts: contiguous_feature_vector_range", "[concepts]") {
}

TEST_CASE("concepts: partitioned_feature_vector_range", "[concepts]") {
}

TEST_CASE("concepts: contiguous_partitioned_feature_vector_range", "[concepts]") {
}
