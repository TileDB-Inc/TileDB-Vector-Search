/**
 * @file   unit_flatpq_index.cc
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

#include "detail/flat/qv.h"
#include "flatpq_index.h"
#include "gen_graphs.h"
#include "query_common.h"
#include "scoring.h"

TEST_CASE("flatpq_index: test test", "[flatpq_index]") {
  REQUIRE(true);
}

TEST_CASE(
    "flatpq_index: flat qv_partition with sub distance", "[flatpq_index]") {
  {
    auto sub_sift_base = ColMajorMatrix<float>(2, num_vectors(sift_base));
    for (size_t i = 0; i < num_vectors(sift_base); i++) {
      for (size_t j = 0; j < 2; j++) {
        sub_sift_base(j, i) = sift_base(j, i);
      }
    }
    auto sub_sift_query = ColMajorMatrix<float>(2, num_vectors(sift_query));
    for (size_t i = 0; i < num_vectors(sift_query); i++) {
      for (size_t j = 0; j < 2; j++) {
        sub_sift_query(j, i) = sift_query(j, i);
      }
    }

    auto aa = detail::flat::qv_partition(sub_sift_base, sub_sift_query, 1);
    auto bb = detail::flat::qv_partition(
        sub_sift_base, sub_sift_query, 1, sum_of_squares_distance{});
    auto cc = detail::flat::qv_partition(
        sift_base, sift_query, 1, sub_sum_of_squares_distance{0, 2});
    CHECK(std::equal(aa.begin(), aa.end(), bb.begin()));
    CHECK(std::equal(aa.begin(), aa.end(), cc.begin()));
  }

  {
    auto sub_sift_base = ColMajorMatrix<float>(2, num_vectors(sift_base));
    for (size_t i = 0; i < num_vectors(sift_base); i++) {
      for (size_t j = 1; j < 3; j++) {
        sub_sift_base(j - 1, i) = sift_base(j, i);
      }
    }
    auto sub_sift_query = ColMajorMatrix<float>(2, num_vectors(sift_query));
    for (size_t i = 0; i < num_vectors(sift_query); i++) {
      for (size_t j = 1; j < 3; j++) {
        sub_sift_query(j - 1, i) = sift_query(j, i);
      }
    }

    auto aa = detail::flat::qv_partition(sub_sift_base, sub_sift_query, 1);
    auto bb = detail::flat::qv_partition(
        sub_sift_base, sub_sift_query, 1, sum_of_squares_distance{});
    auto cc = detail::flat::qv_partition(
        sift_base, sift_query, 1, sub_sum_of_squares_distance{1, 3});
    CHECK(std::equal(aa.begin(), aa.end(), bb.begin()));
    CHECK(std::equal(aa.begin(), aa.end(), cc.begin()));
  }
}

TEST_CASE("flatpq_index: flat qv_partition hypercude", "[flatpq_index]") {
  bool debug = false;

  size_t k_near = 8;
  size_t k_far = 8;

  auto hypercube0 = build_hypercube(k_near, k_far, 0xdeadbeef);
  auto hypercube1 = build_hypercube(k_near, k_far, 0xdeadbeef);

  auto init_centroids = ColMajorMatrix<float>(3, 8);
  auto pre_centroids = ColMajorMatrix<float>(3, 8);
  for (size_t i = 0; i < 8; i++) {
    for (size_t j = 0; j < 3; j++) {
      pre_centroids(j, i) = hypercube0(j, i) + 0.025 * (((i + j) % 2) - 0.5);
      init_centroids(j, i) = pre_centroids(j, i);
    }
  }

  auto hypercube2 = ColMajorMatrix<float>(6, num_vectors(hypercube0));
  auto centroids2 = ColMajorMatrix<float>(6, 8);

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < num_vectors(hypercube0); ++i) {
      hypercube2(j, i) = hypercube0(j, i);
      hypercube2(j+3, i) = hypercube1(j, i);
    }
  }

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 8; ++i) {
      centroids2(j, i) = init_centroids(j, i);
      centroids2(j+3, i) = init_centroids(j, i);
    }
  }

  auto expected_centroids0 = ColMajorMatrix<float>(3, 8);
  auto expected_centroids1 = ColMajorMatrix<float>(3, 8);
  for (size_t i = 0; i < 8; i++) {
    for (size_t j = 0; j < 3; j++) {
      expected_centroids0(j, i) = init_centroids(j, i);
      expected_centroids1(j, i) = init_centroids(j, i);
    }
  }

  sub_kmeans(hypercube0, expected_centroids0, 0, 3, 8, 0.00, 10, 1);
  sub_kmeans(hypercube1, expected_centroids1, 0, 3, 8, 0.00, 10, 1);

  auto compare_centroids = [](auto&& a, auto&&b, size_t ba, size_t bb) {
    for (size_t i = 0; i < 8; ++i){
      for (size_t j = 0; j < 3; ++j) {
        CHECK(a(j + ba, i) == b(j + bb, i));
      }
    }
  };


  SECTION("print centroids2") {
    if (debug) {
      std::cout << "\ncentroids2" << std::endl;
      for (size_t j = 0; j < 6; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << centroids2(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  SECTION("subspace = 3 (0)")  {

    sub_kmeans(hypercube0, init_centroids, 0, 3, 8, 0.00, 10, 1);
    compare_centroids(expected_centroids0, init_centroids, 0, 0);

    if (debug) {
      std::cout << "\nsubspace = 3 (0)" << std::endl;

      for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << init_centroids(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  SECTION("subspace = 3 (0) again")  {
    sub_kmeans(hypercube0, init_centroids, 0, 3, 8, 0.00, 10, 1);
    compare_centroids(expected_centroids0, init_centroids, 0, 0);

    if(debug) {
      std::cout << "\nsubspace = 3 (0)" << std::endl;
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << init_centroids(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  SECTION("subspace = 3 (1)")  {
        sub_kmeans(hypercube1, init_centroids, 0, 3, 8, 0.00, 10, 1);
        compare_centroids(expected_centroids1, init_centroids, 0, 0);

    if(debug) {
      std::cout << "\nsubspace = 3 (1)" << std::endl;

      for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << init_centroids(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  SECTION("subspace = 3,0") {
    sub_kmeans(hypercube2, centroids2, 0, 3, 8, 0.000, 10, 1);
    compare_centroids(expected_centroids0, centroids2, 0, 0);
    compare_centroids(pre_centroids, centroids2, 0, 3);

    if (debug) {
      std::cout << "\nsubspace = 3,0" << std::endl;

      for (size_t j = 0; j < 6; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << centroids2(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  SECTION("subspace = 0,3") {
    sub_kmeans(hypercube2, centroids2, 3, 6, 8, 0.000, 10, 1);
    compare_centroids(expected_centroids1, centroids2, 0, 3);
    compare_centroids(pre_centroids, centroids2, 0, 0);

    if (debug) {
      std::cout << "\nsubspace = 0,3" << std::endl;
      for (size_t j = 0; j < 6; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << centroids2(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  SECTION("subspace = 3,3") {
    sub_kmeans(hypercube2, centroids2, 0, 3, 8, 0.000, 10, 1);
    compare_centroids(pre_centroids, centroids2, 0, 3);
    sub_kmeans(hypercube2, centroids2, 3, 6, 8, 0.000, 10, 1);
    compare_centroids(expected_centroids0, centroids2, 0, 0);
    compare_centroids(expected_centroids1, centroids2, 0, 3);


    if (debug) {
      std::cout << "\nsubspace = 3,3" << std::endl;

      for (size_t j = 0; j < 6; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << centroids2(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  SECTION("subspace = 3,3 alt") {
    sub_kmeans(hypercube2, centroids2, 3, 6, 8, 0.000, 10, 1);
    compare_centroids(pre_centroids, centroids2, 0, 0);
    sub_kmeans(hypercube2, centroids2, 0, 3, 8, 0.000, 10, 1);
    compare_centroids(expected_centroids0, centroids2, 0, 0);
    compare_centroids(expected_centroids1, centroids2, 0, 3);


    if (debug) {
      std::cout << "\nsubspace = 3,3" << std::endl;

      for (size_t j = 0; j < 6; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << centroids2(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }


  SECTION("subspace = 6") {
    sub_kmeans(hypercube2, centroids2, 0, 6, 8, 0.00, 10, 1);
    compare_centroids(expected_centroids0, centroids2, 0, 0);
    compare_centroids(expected_centroids1, centroids2, 0, 3);

    if (debug) {
      std::cout << "\nsubspace = 6" << std::endl;

      for (size_t j = 0; j < 6; ++j) {
        for (size_t i = 0; i < 8; ++i) {
          std::cout << centroids2(j, i) << " ";
        }
        std::cout << std::endl;
      }
    }
  }
}

TEST_CASE("flatpq_index: create flatpq_index", "[flatpq_index]") {
  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(128, 16, 8);
  REQUIRE(true);
}


TEST_CASE("flatpq_index: train flatpq_index", "[flatpq_index]") {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);

  REQUIRE(true);
}

TEST_CASE("flatpq_index: add flatpq_index", "[flatpq_index]") {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);
  pq_idx.add(hypercube);

  REQUIRE(true);
}

TEST_CASE("flatpq_index: verify flatpq_index with hypercube", "[flatpq_index]") {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);
  pq_idx.add(hypercube);
  auto avg_error = pq_idx.verify_pq(hypercube);

  CHECK(avg_error < 0.02);
}


TEST_CASE("flatpq_index: verify flatpq_index with stacked hypercube", "[flatpq_index]") {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube0 = build_hypercube(k_near, k_far, 0xdeadbeef);
  auto hypercube1 = build_hypercube(k_near, k_far, 0xbeefdead);

  auto hypercube4 = ColMajorMatrix<float>(12, num_vectors(hypercube0));
  auto centroids2 = ColMajorMatrix<float>(12, 8);

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < num_vectors(hypercube4); ++i) {
      hypercube4(j, i) = hypercube0(j, i);
      hypercube4(j+3, i) = hypercube1(j, i);
      hypercube4(j+6, i) = hypercube0(j, i);
      hypercube4(j+9, i) = hypercube1(j, i);
    }
  }

  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(12, 4, 8, 32);
  pq_idx.train(hypercube4);
  pq_idx.add(hypercube4);
  auto avg_error = pq_idx.verify_pq(hypercube4);

  CHECK(avg_error < 0.02);
}

TEST_CASE("flatpq_index: verify flatpq_index with siftsmall", "[flatpq_index]") {
  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_base_uri, 0);
  training_set.load();

  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(128, 16, 8, 256);
  pq_idx.train(training_set);
  pq_idx.add(training_set);
  auto avg_error = pq_idx.verify_pq(training_set);

  CHECK(avg_error < 0.05);
}