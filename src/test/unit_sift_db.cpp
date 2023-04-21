//
// Created by Andrew Lumsdaine on 4/12/23.
//

#include <catch2/catch_all.hpp>
#include "../sift_db.h"

TEST_CASE("sift_db: test test", "[sift_db]") {
  REQUIRE(true);
}

TEST_CASE("sift_db: test exceptions", "[sift_db]") {
  SECTION("file does not exist") {
    REQUIRE_THROWS_WITH(sift_db<float>("no_such_file", 128), "file no_such_file does not exist");
  }
  SECTION("wrong dimension") {
    REQUIRE_THROWS_WITH(sift_db<float>("siftsmall/siftsmall_base.fvecs", 17), "dimension mismatch: 128 != 17");
  }
}

TEST_CASE("sift_db: open files", "[sift_db]") {
  auto base = sift_db<float>("siftsmall/siftsmall_base.fvecs", 128);
  REQUIRE(base.size() == 10'000);

  auto query = sift_db<float>("siftsmall/siftsmall_query.fvecs", 128);
  REQUIRE(query.size() == 100);

  auto truth = sift_db<float>("siftsmall/siftsmall_groundtruth.ivecs", 100);
  REQUIRE(truth.size() == 100);
}
