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
#include "cpos.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/matrix_with_ids.h"
#include "detail/linalg/partitioned_matrix.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_matrix_with_ids.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "detail/linalg/vector.h"
#include "scoring.h"
#include "utils/print_types.h"

#include <list>
#include <string>
#include <vector>

template <class V, class U>
inline auto sum_of_squares(V const& a, U const& b) {
  float sum{0.0};
  size_t size_a = size(a);

  if constexpr (
      std::unsigned_integral<std::remove_reference_t<decltype(a[0])>> ||
      std::unsigned_integral<std::remove_reference_t<decltype(b[0])>>) {
    for (size_t i = 0; i < size_a; ++i) {
      float diff = (float)a[i] - (float)b[i];
      sum += diff * diff;
    }
  } else {
    for (size_t i = 0; i < size_a; ++i) {
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
  }
  return sum;
}

TEST_CASE("std::unsigned_integral", "[concepts]") {
  std::vector<unsigned> v(1);
  CHECK(std::unsigned_integral<unsigned>);
  CHECK(std::same_as<decltype(v[0]), unsigned&>);
  CHECK(std::same_as<std::remove_reference_t<decltype(v[0])>, unsigned>);
  CHECK(std::same_as<std::remove_cvref_t<decltype(v[0])>, unsigned>);
}

TEMPLATE_TEST_CASE(
    "std::unsigned_integral sum",
    "[concepts]",
    int,
    unsigned,
    long,
    unsigned long,
    size_t,
    float,
    uint8_t) {
  std::vector<TestType> a{1, 2, 3};
  std::vector<TestType> b{4, 5, 6};
  std::span<TestType> c{a};
  std::span<TestType> d{b};

  auto sum = sum_of_squares(a, b);
  CHECK(sum == 27.0);
  auto sum2 = sum_of_squares(c, d);
  CHECK(sum2 == 27.0);
}

TEST_CASE("random_access_range", "[concepts]") {
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

TEST_CASE("sized_range", "[concepts]") {
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

TEST_CASE("contiguous range", "[concepts]") {
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

TEST_CASE("range_of_ranges", "[concepts]") {
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

TEST_CASE("inner_range", "[concepts]") {
  CHECK(std::is_same_v<
        inner_range_t<std::vector<std::vector<int>>>,
        std::vector<int>>);
  CHECK(std::is_same_v<
        inner_range_t<std::vector<std::vector<int>>>,
        typename std::vector<std::vector<int>>::value_type>);
}

TEST_CASE("inner_iterator_t", "[concepts]") {
  CHECK(std::is_same_v<
        inner_iterator_t<std::vector<std::vector<int>>>,
        std::vector<int>::iterator>);
}

TEST_CASE("inner_const_iterator_t", "[concepts]") {
  CHECK(std::is_same_v<
        inner_const_iterator_t<std::vector<std::vector<int>>>,
        std::vector<int>::const_iterator>);
}

TEST_CASE("inner_value_t", "[concepts]") {
  CHECK(std::is_same_v<
        inner_value_t<std::vector<std::vector<int>>>,
        std::vector<int>::value_type>);
}

TEST_CASE("inner_reference_t", "[concepts]") {
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

TEST_CASE("invocable", "[concepts]") {
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

TEST_CASE("subscriptable_range", "[concepts]") {
  using sv = std::vector<int>;
  using svi = std::ranges::iterator_t<sv>;
  using svri = std::iter_reference_t<svi>;

  CHECK(!subscriptable_range<int>);
  CHECK(subscriptable_range<std::vector<int>>);
  CHECK(subscriptable_range<std::vector<double>>);
  CHECK(subscriptable_range<const std::vector<double>>);
  CHECK(subscriptable_range<std::vector<std::vector<int>>>);
  CHECK(subscriptable_range<std::array<int, 3>>);

  CHECK(subscriptable_range<sv>);
  CHECK(!subscriptable_range<svi>);
  CHECK(!subscriptable_range<svri>);

  int* d = new int[3];  // random access iterator but not random access range
  CHECK(!subscriptable_range<decltype(d)>);
  CHECK(subscriptable_range<std::span<int>>);

  CHECK(subscriptable_range<Vector<int>>);
}

TEST_CASE("callable_range", "[concepts]") {
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

TEST_CASE("Vector", "[concepts]") {
  CHECK(!range_of_ranges<Vector<int>>);
  CHECK(std::ranges::random_access_range<Vector<int>>);
  CHECK(std::ranges::sized_range<Vector<int>>);
  CHECK(std::ranges::contiguous_range<Vector<int>>);
  CHECK(subscriptable_range<Vector<int>>);
  CHECK(requires { Vector<int>{}.size(); });
  CHECK(std::ranges::sized_range<Vector<int>>);
}

TEST_CASE("dimensionable", "[concepts]") {
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
  auto num_vectors() const {
    return 0;
  }
};

TEST_CASE("vectorable", "[concepts]") {
  CHECK(!vectorable<int>);
  CHECK(vectorable<std::vector<int>>);
  CHECK(vectorable<std::vector<double>>);
  CHECK(!vectorable<std::vector<std::vector<int>>>);
  CHECK(vectorable<std::array<int, 3>>);

  int* d = nullptr;
  CHECK(!vectorable<decltype(d)>);
  CHECK(vectorable<std::span<int>>);

  CHECK(vectorable<Vector<int>>);

  CHECK(vectorable<dummy_vectorable>);
}

TEST_CASE("partitionable", "[concepts]") {
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
  using base = typename std::vector<T>;

  // If both size and dimension are available, dimensions() cpo is ambiguous
 private:
  using base::size;

 public:
  auto dimensions() const {
    return this->size();
  }
};

template <class T>
class dummy_feature_vector_with_size : public std::vector<T> {
 public:
  using base = typename std::vector<T>;
};

template <feature_vector R>
void foo(const R& r) {
}

void bar() {
  foo(dummy_feature_vector<int>{});
}

TEST_CASE("feature_vector", "[concepts]") {
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

TEST_CASE("query_vector", "[concepts]") {
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

TEST_CASE("feature_vector_array", "[concepts]") {
  CHECK(!feature_vector_array<int>);
  CHECK(!feature_vector_array<std::vector<int>>);
  CHECK(!feature_vector_array<std::vector<double>>);
  CHECK(!feature_vector_array<std::vector<std::vector<int>>>);
}

// Placeholders
TEST_CASE("contiguous_feature_vector_range", "[concepts]") {
}

TEST_CASE("partitioned_feature_vector_range", "[concepts]") {
}

TEST_CASE("contiguous_partitioned_feature_vector_range", "[concepts]") {
}

TEST_CASE("distance_function", "[concepts]") {
  CHECK(!distance_function<
        int,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!distance_function<
        std::vector<int>,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(distance_function<
        sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(distance_function<
        logging_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(distance_function<
        counting_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(distance_function<
        inner_product_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(distance_function<
        sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  // cached_sub_sum_of_squares_distance is a distance_function because it takes
  // the sub start & top in the constructor, not the operator. The naming makes
  // it seem like it should be a sub_distance_function, but it is not.
  CHECK(distance_function<
        cached_sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  // But uncached_sub_sum_of_squares_distance takes the sub start & top in the
  // operator so is not a distance_function.
  CHECK(!distance_function<
        uncached_sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
}

TEST_CASE("sub_distance_function", "[concepts]") {
  CHECK(!sub_distance_function<
        int,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!sub_distance_function<
        std::vector<int>,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(!sub_distance_function<
        sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!sub_distance_function<
        logging_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!sub_distance_function<
        counting_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(!sub_distance_function<
        inner_product_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  // See distance_function for a discussion on these two.
  CHECK(!sub_distance_function<
        sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!sub_distance_function<
        cached_sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(sub_distance_function<
        uncached_sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
}

TEST_CASE("cached_sub_distance_function", "[concepts]") {
  CHECK(!cached_sub_distance_function<
        int,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!cached_sub_distance_function<
        std::vector<int>,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(!cached_sub_distance_function<
        sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!cached_sub_distance_function<
        logging_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!cached_sub_distance_function<
        counting_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(!cached_sub_distance_function<
        inner_product_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);

  CHECK(cached_sub_distance_function<
        sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(cached_sub_distance_function<
        cached_sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
  CHECK(!cached_sub_distance_function<
        uncached_sub_sum_of_squares_distance,
        dummy_feature_vector_with_size<float>,
        dummy_feature_vector_with_size<float>>);
}
