/**
 * @file   time_l2.cc
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
 * Program to test performance of different query algorithms.
 *
 */

#include <catch2/catch_all.hpp>
#include <type_traits>

#include <iostream>
#include "utils/timer.h"

template <class V>
float l2_float(const V& u, const V& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = u[i] - v[i];
    sum += d * d;
  }
  return sum;
}

template <class U, class V>
float l2_float_2(const U& u, const V& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = u[i] - v[i];
    sum += d * d;
  }
  return sum;
}

template <class U, class V>
float l2_float_2_constexpr(const U& u, const V& v) {
  float sum = 0;
  if constexpr (std::is_same_v<decltype(u[0]), decltype(v[0])>) {
    for (size_t i = 0; i < u.size(); ++i) {
      float d = u[i] - v[i];
      sum += d * d;
    }
  } else {
    for (size_t i = 0; i < u.size(); ++i) {
      float d = (float)u[i] - (float)v[i];
      sum += d * d;
    }
  }
  return sum;
}

template <class V>
float l21(const V& u, const V& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = (float)u[i] - (float)v[i];
    sum += d * d;
  }
  return sum;
}

template <class V>
float l21i(const V& u, const V& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = int{static_cast<int>(u[i])} - int{static_cast<int>(v[i])};
    sum += d * d;
  }
  return sum;
}

template <class V>
float l21i_cast(const V& u, const V& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = (int)u[i] - (int)v[i];
    sum += d * d;
  }
  return sum;
}

template <class V, class U>
float l22(const V& u, const U& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = ((float)u[i]) - ((float)v[i]);
    sum += d * d;
  }
  return sum;
}

template <class V, class U>
float l33(const U& u, const V& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = ((float)u[i]) - ((float)v[i]);
    sum += d * d;
  }
  return sum;
}

template <class V, class U>
float l22_x(const V& u, const U& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    float d = ((float)v[i]) - ((float)u[i]);
    sum += d * d;
  }
  return sum;
}

template <class V, class U>
float l22_yack(const V& u, const U& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    if constexpr (
        std::is_same_v<typename V::value_type, float> &&
        std::is_same_v<typename U::value_type, uint8_t>) {
      float d = u[i] - ((float)v[i]);
      sum += d * d;
    } else if constexpr (
        std::is_same_v<typename V::value_type, uint8_t> &&
        std::is_same_v<typename U::value_type, float>) {
      float d = ((float)u[i]) - v[i];
      sum += d * d;
    } else {
      sum += 0;
    }
  }
  return sum;
}

template <class V, class U>
float l22_yack_2(const V& u, const U& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    if constexpr (
        std::is_same_v<typename V::value_type, float> &&
        std::is_same_v<typename U::value_type, uint8_t>) {
      float d = u[i] - ((float)v[i]);
      sum += d * d;
    } else if constexpr (
        std::is_same_v<typename V::value_type, uint8_t> &&
        std::is_same_v<typename U::value_type, float>) {
      float d = v[i] - ((float)u[i]);
      sum += d * d;
    } else {
      sum += 0;
    }
  }
  return sum;
}

template <class V, class U>
float l22_yack_swap(const V& u, const U& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    if constexpr (
        std::is_same_v<typename V::value_type, float> &&
        std::is_same_v<typename U::value_type, uint8_t>) {
      return l22(v, u);
    } else if constexpr (
        std::is_same_v<typename V::value_type, uint8_t> &&
        std::is_same_v<typename U::value_type, float>) {
      float d = v[i] - ((float)u[i]);
      sum += d * d;
    } else {
      sum += 0;
    }
  }
  return sum;
}

template <class V, class U>
float l22_yack_swap_2(const V& u, const U& v) {
  float sum = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    if constexpr (
        std::is_same_v<typename V::value_type, float> &&
        std::is_same_v<typename U::value_type, uint8_t>) {
      return l22(v, u);
    } else if constexpr (
        std::is_same_v<typename V::value_type, uint8_t> &&
        std::is_same_v<typename U::value_type, float>) {
      return l22(u, v);
    } else {
      sum += 0;
    }
  }
  return sum;
}

template <class V, class U>
float l22_just_swap(const V& u, const U& v) {
  return l22(v, u);
}

using typelist = std::tuple<
    std::tuple<float, float>,
    std::tuple<uint8_t, float>,
    std::tuple<uint8_t, float>,
    std::tuple<uint8_t, uint8_t>>;

TEMPLATE_LIST_TEST_CASE("time queries", "[queries]", typelist) {
  size_t dimension = 128;
  size_t repeats = 100'000'000;

  using left_type = std::tuple_element_t<0, TestType>;
  using right_type = std::tuple_element_t<1, TestType>;

  std::vector<left_type> u(dimension), v(dimension);
  std::vector<right_type> x(dimension), y(dimension);
  std::iota(begin(u), end(u), 0);
  std::iota(begin(v), end(v), 3);
  std::iota(begin(x), end(x), 5);
  std::iota(begin(y), end(y), 7);

  {
    scoped_timer _{"float float"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l2_float(u, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float float_2"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l2_float_2(u, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float float_2_constexpr"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l2_float_2_constexpr(u, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float float l21"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l21(u, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char char l21"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l21(x, y);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char char l21i"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l21i(x, y);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char char l21i_cast"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l21i_cast(x, y);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float float l22"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22(u, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char char l22"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22(x, y);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float float l33"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l33(u, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char char l33"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l33(x, y);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22(x, u);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22(u, x);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22 you choose"};
    float sum = 0;
    if constexpr (
        std::is_same_v<typename decltype(x)::value_type, uint8_t> &&
        std::is_same_v<typename decltype(u)::value_type, float>) {
      for (size_t i = 0; i < repeats; ++i) {
        sum += l22(x, u);
      }
    } else {
      for (size_t i = 0; i < repeats; ++i) {
        sum += l22(u, x);
      }
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22 you choose"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22(u, x);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22_x"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_x(x, u);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22_x"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_x(u, x);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22_yack"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack(u, x);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22_yack"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack(y, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22_yack_2"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack_2(u, x);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22_just_swap"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_just_swap(y, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22_just_swap"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_just_swap(u, x);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22_yack_swap"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack_swap(y, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22_yack_swap_2"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack_swap_2(u, x);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22_yack_swap_2"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack_swap_2(y, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22_yack_swap"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack_swap(u, x);
    }
    CHECK(sum == 1.0);
  }

  {
    scoped_timer _{"char float l22_yack_2"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack_2(y, v);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"float char l22_yack_2"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22_yack_2(v, y);
    }
    CHECK(sum == 1.0);
  }

  {
    scoped_timer _{"float char l22"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22(v, y);
    }
    CHECK(sum == 1.0);
  }
  {
    scoped_timer _{"char float l22"};
    float sum = 0;
    for (size_t i = 0; i < repeats; ++i) {
      sum += l22(y, v);
    }
    CHECK(sum == 1.0);
  }
}
