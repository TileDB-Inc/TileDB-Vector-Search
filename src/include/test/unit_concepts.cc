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
 */

#include <catch2/catch_all.hpp>
#include <mdspan/mdspan.hpp>
#include <span>
#include "../concepts.h"

template <class T>
struct loadable_class {
  bool load() const { return true; }
};

template <class T>
struct not_loadable_class {
  bool not_load() const { return true; }
};

template <class T>
void maybe_load (const T& t) {
  if constexpr (is_loadable_v<T>) {
    CHECK(t.load());
  } else if constexpr (!is_loadable_v<T>){
    CHECK(t.not_load());
  }
}

TEST_CASE("loadable", "[concepts]") {
  CHECK(is_loadable_v<loadable_class<int>>);
  CHECK(!is_loadable_v<not_loadable_class<int>>);
  loadable_class<int> lc;
  not_loadable_class<int> nlc;
  CHECK(is_loadable_v<decltype(lc)>);
  CHECK(!is_loadable_v<decltype(nlc)>);
  if (is_loadable_v<decltype(lc)>) {
    CHECK(lc.load());
  }
  if (is_loadable_v<decltype(nlc)>) {
    CHECK(nlc.not_load());
  }
  maybe_load(lc);
  maybe_load(nlc);
}

template <class T>
struct feature_vector_0 {
  using value_type = T;
  std::size_t size() const {
    return 0;
  }
  std::size_t dim() const {
    return 0;
  }
  T operator[](std::size_t i) const {
    return 0;
  }
  T* data() const {
    return nullptr;
  }
};

template <class FV>
auto num_features(const FV& fv) {
  return fv.size();
}

template <class FV>
requires feature_vector<FV>
auto a(const FV& fv) {
  return true;
}

TEST_CASE("feature_vector_0", "[concepts]") {
  feature_vector_0<float> fv;
  CHECK(a(fv));
  // REQUIRE(feature_vector<feature_vector_0<float>>);
}

template <class T>
struct vector_database_0 {
  using value_type = T;
  std::size_t size() const {
    return 0;
  }
  std::size_t dim() const {
    return 0;
  }
  std::span<T> operator[](std::size_t i) const {
    return std::span<T>(nullptr, 0);
  }
  T operator()(std::size_t i, std::size_t j) const {
    return 0;
  }
  T* data() const {
    return nullptr;
  }
  constexpr auto rank() const noexcept {
    return 2;
  }
  std::span<T> raveled() const {
    static T x;
    return std::span<T>(&x, 1);
  }
};

template <class DB>
auto num_vectors(const DB& db) {
  return db.span();
}

template <class DB>
auto raveled(const DB& db) {
  return db.raveled();
}

template <class DB>
requires vector_database<DB>
auto b(const DB& db) {
  return true;
}

TEST_CASE("vector_database_0", "[concepts]") {
  vector_database_0<float> db;
  CHECK(b(db));
  // REQUIRE(feature_vector<feature_vector_0<float>>);
}