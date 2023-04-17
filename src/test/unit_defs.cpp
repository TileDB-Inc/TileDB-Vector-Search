//
// Created by Andrew Lumsdaine on 4/14/23.
//

#include <catch2/catch.hpp>
#include <set>
#include <vector>
#include "../defs.h"

TEST_CASE("defs: test test", "[defs]") {
  REQUIRE(true);
}

TEST_CASE("defs: vector test", "[defs]") {
  std::vector<std::vector<float>> a {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
  std::vector<float> b {0, 0, 0, 0};

  std::vector<float> c { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  std::vector<std::span<float>> d;
  for (size_t i = 0; i < 4; ++i) {
    d.push_back(std::span<float>(c.data() + i * 3, 3));
  }

  SECTION("column sum") {
    col_sum(a, b, [](auto x) { return x; });
    CHECK(b[0] == (1 + 2 + 3));
    CHECK(b[1] == (4 + 5 + 6));
    CHECK(b[2] == (7 + 8 + 9));
    CHECK(b[3] == (10 + 11 + 12));
  }

  SECTION("column sum of squares") {
    col_sum(a, b, [](auto x) { return x * x; });
    CHECK(b[0] == (1 + 4 + 9));
    CHECK(b[1] == (16 + 25 + 36));
    CHECK(b[2] == (49 + 64 + 81));
    CHECK(b[3] == (100 + 121 + 144));
  }

  SECTION("column sum of squares with span") {
    col_sum(d, b, [](auto x) { return x * x; });
    CHECK(b[0] == (1 + 4 + 9));
    CHECK(b[1] == (16 + 25 + 36));
    CHECK(b[2] == (49 + 64 + 81));
    CHECK(b[3] == (100 + 121 + 144));
  }
}

TEST_CASE("defs: std::set", "[defs]") {
  std::set<int> a;

  SECTION("insert in ascending order") {
    for (auto &&i: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert(i);
    }
  }
  SECTION("insert in descending order") {
    for (auto &&i: {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert(i);
    }
  }
  CHECK(a.size() == 10);
  CHECK(a.count(0) == 1);
  CHECK(*begin(a) == 0);
  CHECK(*rbegin(a) == 9);
}

TEST_CASE("defs: fixed_min_set", "[defs]") {
  fixed_min_set<int> a(5);

  SECTION("insert in ascending order") {
    for (auto &&i: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert(i);
    }
  }

  SECTION("insert in descending order") {
    for (auto &&i: {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert(i);
    }
  }
  CHECK(a.size() == 5);
  CHECK(a.count(0) == 1);
  CHECK(*begin(a) == 0);
  CHECK(*rbegin(a) == 4);

}