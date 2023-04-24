//
// Created by Andrew Lumsdaine on 4/20/23.
//
#include <catch2/catch_all.hpp>
#include "../sift_array.h"

TEST_CASE("sift_array: test test", "[sift_db]") {
  REQUIRE(true);
}

TEST_CASE("sift_array: test exceptions", "[sift_array]") {
  SECTION("file does not exist") {
    REQUIRE_THROWS_WITH(sift_array<float>("no_such_file"), "[TileDB::Array] Error: Cannot open array; Array does not exist.");
  }
}

TEST_CASE("sift_array: open files", "[sift_array]") {
  auto base = sift_array<float>("arrays/siftsmall_base");
  std::cout << "a" << std::endl;
  CHECK(base.size() == 10'000);

  auto query = sift_array<float>("arrays/siftsmall_query");
  std::cout << "b" << std::endl;
  CHECK(query.size() == 100);

  auto truth = sift_array<int>("arrays/siftsmall_groundtruth");
  std::cout << "c" << std::endl;
  CHECK(truth.size() == 100);
}
