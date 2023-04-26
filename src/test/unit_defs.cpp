//
// Created by Andrew Lumsdaine on 4/14/23.
//

#include <catch2/catch_all.hpp>
#include <algorithm>
#include <iterator>
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


TEST_CASE("defs: std::set with pairs", "[defs]") {
  using element = std::pair<float, int>;
  std::set<element> a;

  SECTION("insert in ascending order") {
    for (auto &&i: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert({10-i, i});
    }
    CHECK(begin(a)->first == 1);
    CHECK(begin(a)->second == 9);
    CHECK(rbegin(a)->first == 10.0);
    CHECK(rbegin(a)->second == 0);
  }
  SECTION("insert in descending order") {
    for (auto &&i: {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert({10+i, i});
    }
    CHECK(begin(a)->first == 10.0);
    CHECK(begin(a)->second == 0);
    CHECK(rbegin(a)->first == 19.0);
    CHECK(rbegin(a)->second == 9);
  }
  CHECK(a.size() == 10);
  //CHECK(*begin(a) == element{10, 0});
  //CHECK(*rbegin(a) == element{9, 1});
  }


TEST_CASE("defs: std::priority_queue", "[defs]") {

  SECTION("construct with trivial vector") {
    std::vector<int> v;
    SECTION("pq"){
      std::priority_queue pq(begin(v), end(v));
      CHECK(pq.size() == 0);
    }
    SECTION("fms"){
      int k = 3;
      fixed_min_set pq(3, begin(v), end(v));
      CHECK(pq.size() == 0);
    }

    CHECK(v.size() == 0);
  }

  SECTION("construct with init vector iterators (ascending)") {
    std::vector<int> v {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::priority_queue pq(begin(v), end(v));
    CHECK(v.size() == 10);
    CHECK(pq.size() == 10);
  }

  SECTION("construct with init vector iterators (descending)") {
    std::vector<int> v {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::priority_queue pq(begin(v), end(v));
    CHECK(v.size() == 10);
    CHECK(pq.size() == 10);

    SECTION("add element to pq") {
      pq.push(10);
      CHECK(v.size() == 10);
      CHECK(pq.size() == 11);      
    }
  }

  SECTION("construct with init vector container (descending)") {
    std::vector<int> v {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::priority_queue pq(begin(v), end(v), std::less<>{}, v);

    SECTION("add element to pq") {
      pq.push(10);
      CHECK(v.size() == 10);
      CHECK(pq.size() == 21);
    }
  }

  SECTION("large pq"){
    std::priority_queue<int> pq;
    for (size_t i = 0; i < 1000; ++i) {
      pq.push(i);
    }
    CHECK(size(pq) == 1000);
  }
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

  // CHECK(*begin(a.c()) == 0);
 //  CHECK(*rbegin(a.c()) == 4);
}

TEST_CASE("defs: large fixed_min_set", "[defs]") {

  SECTION("insert in descending order") {
    fixed_min_set<int> a(500);
    
    for (size_t i = 0 ; i < 1000; ++i) {
      a.insert(i);
    }
    std::sort(begin(a.c()), end(a.c()));
    for (size_t i = 0 ; i < 500; ++i) {
      CHECK((a.c())[i] == (int) i);
    }
  }
}


TEST_CASE("defs: fixed_min_set with pairs", "[defs]") {
  fixed_min_set_pair<float, int> a(5);

  SECTION("insert in ascending order") {
    for (auto &&i: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      a.insert({10-i, i});
    }
    SECTION("push pop") {
    CHECK(a.size() == 5);
    CHECK(a.top().first == 5.0);
    CHECK(a.top().second == 5);
    a.pop();
    CHECK(a.top().first == 4.0);
    CHECK(a.top().second == 6);
    a.pop();
    CHECK(a.top().first == 3.0);
    CHECK(a.top().second == 7);
    a.pop();
    CHECK(a.top().first == 2.0);
    CHECK(a.top().second == 8);
    a.pop();
    CHECK(a.top().first == 1.0);
    CHECK(a.top().second == 9);
    a.pop();
    CHECK(a.empty());
    }
  }
  SECTION("insert in descending order") {
    for (auto &&i: {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}) {
      a.insert({10+i, i});
    }
    CHECK(a.size() == 5);
    CHECK(a.top().first == 14.0);
    CHECK(a.top().second == 4);
    a.pop();
    CHECK(a.size() == 4);
    CHECK(a.top().first == 13.0);
    CHECK(a.top().second == 3);
    a.pop();
    CHECK(a.size() == 3);
    CHECK(a.top().first == 12.0);
    CHECK(a.top().second == 2);
    a.pop();
    CHECK(a.size() == 2);
    CHECK(a.top().first == 11.0);
    CHECK(a.top().second == 1);
    a.pop();
    CHECK(a.size() == 1);
    CHECK(a.top().first == 10.0);
    CHECK(a.top().second == 0);
    a.pop();
    CHECK(a.empty());
  }
  CHECK(a.size() == 0);
}


TEST_CASE("defs: get_top_k", "[defs]") {
  std::vector<int> v(1000);
  std::iota(begin(v), end(v), 0);
  std::random_shuffle(begin(v), end(v));

  std::vector<int> i(1000);
  std::iota(begin(i), end(i), 0);

  int k = 20;
  std::vector<int> top_k(k);

  get_top_k(v, top_k, i, k);

  //  template <class V, class L, class I>
  //auto get_top_k(V const& scores, L & top_k, I & index, int k) {

  for (int i = 0; i < k; ++i) {
    CHECK(v[top_k[i]] == i);
  }
}
