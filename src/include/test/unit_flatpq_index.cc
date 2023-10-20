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

TEST_CASE("normalize matrix", "[flatpq_index]") {
  auto hypercube = build_hypercube<float>(2, 2, 0xdeadbeef);
  auto normalized = normalize_matrix(hypercube);

  debug_slice(hypercube);
  debug_slice(normalized);
}

TEMPLATE_TEST_CASE("flatpq_index: create flatpq_index", "[flatpq_index]", float, uint8_t) {
  auto pq_idx = flatpq_index<TestType, uint32_t, uint32_t>(128, 16, 8);
  REQUIRE(true);
}


TEMPLATE_TEST_CASE("flatpq_index: train flatpq_index", "[flatpq_index]", float, uint8_t) {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flatpq_index<TestType, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);

  REQUIRE(true);
}

TEMPLATE_TEST_CASE("flatpq_index: add flatpq_index", "[flatpq_index]", float, uint8_t) {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flatpq_index<TestType, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);
  pq_idx.add(hypercube);

  REQUIRE(true);
}

TEMPLATE_TEST_CASE("flatpq_index: verify flatpq_index encoding with hypercube", "[flatpq_index]", float, uint8_t) {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flatpq_index<TestType, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);
  pq_idx.add(hypercube);
  auto avg_error = pq_idx.verify_pq_encoding(hypercube);

  CHECK(avg_error < 0.02);
}


TEMPLATE_TEST_CASE("flatpq_index: verify flatpq_index encoding with stacked hypercube", "[flatpq_index]", float, uint8_t) {
  size_t k_near = 0;
  size_t k_far = 0;

  auto hypercube0 = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);
  auto hypercube1 = build_hypercube<TestType>(k_near, k_far, 0xbeefdead);

  auto hypercube2 = ColMajorMatrix<TestType>(6, num_vectors(hypercube0));
  auto hypercube4 = ColMajorMatrix<TestType>(12, num_vectors(hypercube0));

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < num_vectors(hypercube4); ++i) {
      hypercube2(j, i) = hypercube0(j, i);
      hypercube2(j+3, i) = hypercube1(j, i);
      hypercube4(j, i) = hypercube0(j, i);
      hypercube4(j+3, i) = hypercube1(j, i);
      hypercube4(j+6, i) = hypercube0(j, i);
      hypercube4(j+9, i) = hypercube1(j, i);
    }
  }

  auto pq_idx = flatpq_index<TestType, uint32_t, uint32_t>(6, 2, 8, 8);
  pq_idx.train(hypercube2);
  pq_idx.add(hypercube2);
  auto avg_error = pq_idx.verify_pq_encoding(hypercube2);
  CHECK(avg_error < 0.02);

  auto avg_dist = pq_idx.verify_pq_distances(hypercube2);
  CHECK(avg_dist < 0.02);
}


TEST_CASE("flatpq_index: verify pq_encoding and pq_distances with siftsmall", "[flatpq_index]") {
  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_base_uri, 0);
  training_set.load();

  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(128, 16, 8, 256);
  pq_idx.train(training_set);
  pq_idx.add(training_set);

  SECTION("pq_encoding") {
    auto avg_error = pq_idx.verify_pq_encoding(training_set);
    CHECK(avg_error < 0.05);
  }
  SECTION("pq_distances") {
    auto avg_error = pq_idx.verify_pq_distances(training_set);
    CHECK(avg_error < 0.1);
  }
}

TEMPLATE_TEST_CASE("flatpq_index: verify pq_distances with stacked hypercube", "[flatpq_index]", float, uint8_t) {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube0 = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);
  auto hypercube1 = build_hypercube<TestType>(k_near, k_far, 0xbeefdead);

  auto hypercube4 = ColMajorMatrix<TestType>(12, num_vectors(hypercube0));
  auto centroids2 = ColMajorMatrix<TestType>(12, 8);

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < num_vectors(hypercube4); ++i) {
      hypercube4(j, i) = hypercube0(j, i);
      hypercube4(j+3, i) = hypercube1(j, i);
      hypercube4(j+6, i) = hypercube0(j, i);
      hypercube4(j+9, i) = hypercube1(j, i);
    }
  }

  auto pq_idx = flatpq_index<TestType, uint32_t, uint32_t>(12, 4, 8, 32);
  pq_idx.train(hypercube4);
  pq_idx.add(hypercube4);
  auto avg_error = pq_idx.verify_pq_distances(hypercube4);

  CHECK(avg_error < 0.02);
}

TEST_CASE("flatpq_index: query stacked hypercube", "[flatpq_index]") {
  size_t k_dist = GENERATE(0,  32);

  size_t k_near = k_dist;
  size_t k_far = k_dist;

  auto hypercube0 = build_hypercube(k_near, k_far, 0xdeadbeef);
  auto hypercube1 = build_hypercube(k_near, k_far, 0xbeefdead);

  auto hypercube2 = ColMajorMatrix<float>(6, num_vectors(hypercube0));
  auto centroids2 = ColMajorMatrix<float>(6, 8);

  auto hypercube4 = ColMajorMatrix<float>(12, num_vectors(hypercube0));
  auto centroids4 = ColMajorMatrix<float>(12, 8);

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < num_vectors(hypercube4); ++i) {
      hypercube2(j, i) = hypercube0(j, i);
      hypercube2(j+3, i) = hypercube1(j, i);

      hypercube4(j, i) = hypercube0(j, i);
      hypercube4(j+3, i) = hypercube1(j, i);
      hypercube4(j+6, i) = hypercube0(j, i);
      hypercube4(j+9, i) = hypercube1(j, i);
    }
  }

  auto pq_idx2 = flatpq_index<float, uint32_t, uint32_t>(6, 2, 8, k_dist == 0 ? 8: 16);
  pq_idx2.train(hypercube2);
  pq_idx2.add(hypercube2);

  auto pq_idx4 = flatpq_index<float, uint32_t, uint32_t>(12, 4, 8, k_dist == 0 ? 8 : 16);
  pq_idx4.train(hypercube4);
  pq_idx4.add(hypercube4);

// centroids
//  1	-1	-1	1	1	-1	0.5	-1
//  1	-1	-1	-1	1	1	-0.5	1
//  1	-1	1	-1	-1	1	0.5	-1
//  1	-1	-1	1	1	-1	0.5	-1
//  1	-1	-1	-1	1	1	-0.5	1
//  1	-1	1	-1	-1	1	0.5	-1
//
// feature vectors
//  0	-1	-1	-1	-1	1	1	1	1
//  0	-1	-1	1	1	-1	-1	1	1
//  0	-1	1	-1	1	-1	1	-1	1
//  0	-1	-1	-1	-1	1	1	1	1
//  0	-1	-1	1	1	-1	-1	1	1
//  0	-1	1	-1	1	-1	1	-1	1
//
// reconstructed
//  0.5	-1	-1	-1	-1	1	0.5	1	1
//  -0.5	-1	-1	1	1	-1	-0.5	1	1
//  0.5	-1	1	-1	1	-1	0.5	-1	1
//  0.5	-1	-1	-1	-1	1	0.5	1	1
//  -0.5	-1	-1	1	1	-1	-0.5	1	1
//  0.5	-1	1	-1	1	-1	0.5	-1	1

  using enc_type = std::tuple<std::vector<float>, std::vector<float>>;
  auto expected = std::vector<enc_type> {
      {{ 1,  1,  1,  1,  1,  1}, {0, 0}},
      {{ 1,  1,  1, -1, -1, -1}, {0, 1}},
      {{ 1,  1, -1, -1, -1,  1}, {4, 2}},
      {{-1,  1, -1,  1,  1,  1}, {7, 0}},
      {{ 1,  1,  1, -1,  1, -1}, {0, 7}},
      {{ 1, -1, -1, .5, -.5, .5}, {3, 6}},
      {{.5, -.5, .5, -1,  1,  1}, {6, 5}},
  };

  SECTION("Test encode + decode") {

    if (k_dist == 0) {
      for (auto&& [v, pq] : expected) {
        auto pq0 = pq_idx2.encode(v);
        CHECK(std::equal(pq0.begin(), pq0.end(), pq.begin()));

        auto v0 = pq_idx2.decode<float>(pq0);
        CHECK(std::equal(v0.begin(), v0.end(), v.begin()));
      }
    } else {
      auto counter = 0;
      for (auto&& [v, pq] : expected) {
        auto pq0 = pq_idx2.encode(v);
        auto v0 = pq_idx2.decode<float>(pq0);
        auto diff = sum_of_squares(v, v0);
        if (counter++ < 5) {
          CHECK(diff < 0.02);
        }
      }
    }
  }

  SECTION("Test encode + decode, uint8_t") {
    using enc_type_8 = std::tuple<std::vector<uint8_t>, std::vector<uint8_t>>;
    auto expected_8 = std::vector<enc_type> {
        {{ 127,  127,  127,  127,  127,  127}, {0, 0}},
        {{ 127,  127,  127, 0, 0, 0}, {0, 1}},
        {{ 127,  127, 0, 0, 0,  127}, {4, 2}},
        {{0,  127, 0,  127,  127,  127}, {7, 0}},
        {{ 127,  127,  127, 0,  127, 0}, {0, 7}},
        {{ 127, 0, 0, 95, 31, 95}, {3, 6}},
        {{95, 31, 95, 0,  127,  127}, {6, 5}},
    };

    auto pq_idx2_8 = flatpq_index<uint8_t, uint32_t, uint32_t>(6, 2, 8, k_dist == 0 ? 8: 16);
    auto hypercube2_8 = normalize_matrix(hypercube2);

    pq_idx2_8.train(hypercube2_8);
    pq_idx2_8.add(hypercube2_8);

    debug_slice(pq_idx2_8.centroids_, "\npq_idx2_8 centroids");

    if (k_dist == 0) {
      for (auto&& [v, pq] : expected_8) {
        auto pq0 = pq_idx2_8.encode(v);
        CHECK(std::equal(pq0.begin(), pq0.end(), pq.begin()));

        auto v0 = pq_idx2_8.decode<uint8_t>(pq0);
        CHECK(std::equal(v0.begin(), v0.end(), v.begin()));
      }
    } else {
      auto counter = 0;
      for (auto&& [v, _] : expected_8) {
        auto pq0 = pq_idx2_8.encode(v);
        auto v0 = pq_idx2_8.decode<uint8_t>(pq0);
        auto diff = sum_of_squares(v, v0);
        auto normalize = 0.5*(sum_of_squares(v) + sum_of_squares(v0));
        auto rel_diff = diff/normalize;
        if (counter++ < 5) {
          CHECK(rel_diff < 0.05);
        }
      }
    }
  }

  SECTION("Test sub_distance_asymmetric") {
    // debug_slice(pq_idx2.centroids_, "\npq_idx2 centroids");

    // k == 32 centroids[0] and centroids[10]
    //    1.00693       0.966238
    //   -1.03368       1.02702
    //    0.961354      1.00195
    //    1.03518       1.01545
    //   -0.998946      0.998451
    //    0.996525      1.00204

    auto counter = 0;
    for (auto&& [vx, _] : expected) {
      auto pq0 = pq_idx2.encode(vx);
      auto v1 = pq_idx2.decode<float>(pq0);

      auto diffx1 = sum_of_squares(vx, v1);

      auto sub_diffx0 = pq_idx2.sub_distance_asymmetric(vx, pq0);
      auto sub_diff10 = pq_idx2.sub_distance_asymmetric(v1, pq0);

      auto rel_diff = [](auto&& diff1, auto&& diff2, auto&& v1, auto&& v2) {
        auto d = 0.5 * (sum_of_squares(v1) + sum_of_squares(v2));
        if (d == 0.0) {
          d = 1.0;
        }
        return (std::abs(diff1 - diff2) / d);
      };

      if (k_dist == 0 || counter++ < 5) {
        CHECK(rel_diff(diffx1, sub_diffx0, vx, v1) < 0.005);
        CHECK(rel_diff(diffx1, sub_diff10, vx, v1) < 0.005);
      }
    }
  }

#if 0
  SECTION("Test sub_distance_symmetric") {
    for (auto&& [v, pq] : expected) {
      auto pq0 = pq_idx2.encode(v);
      auto v0 = pq_idx2.decode<float>(pq0);
      auto diff = sum_of_squares(v, v0);
      auto sub_diff = pq_idx2.sub_distance_symmetric(pq, pq0);
      auto d = 0.5 * (diff + sub_diff);
      if (d == 0.0) {
        d = 1.0;
      }
      CHECK((std::abs(diff - sub_diff) / d) < 0.002);
    }
  }
#endif

  SECTION("Test query 2") {
    auto query =
        ColMajorMatrix<float>{{-1, -1, -1, -1, -1, -1 }};
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx2.query(query, 4);

    auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
        hypercube2, query, 4, 8, sum_of_squares_distance{});

    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k[i]), end(top_k[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k(j, i) << " ";
      }
      std::cout << std::endl;
    }
    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k_pq[i]), end(top_k_pq[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k_pq(j, i) << " ";
      }
      std::cout << std::endl;
    }
    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k_scores[i]), end(top_k_scores[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k_scores(j, i) << " ";
      }
      std::cout << std::endl;
    }
    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k_pq_scores[i]), end(top_k_pq_scores[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k_pq_scores(j, i) << " ";
      }
      std::cout << std::endl;
    }
  }

#if 0
  SECTION("Test query 4") {
    auto query =
        ColMajorMatrix<float>{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx4.query(query, 8);

    auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
        hypercube4, query, 8, 8, sum_of_squares_distance{});

    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k[i]), end(top_k[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k(j, i) << " ";
      }
      std::cout << std::endl;
    }
    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k_pq[i]), end(top_k_pq[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k_pq(j, i) << " ";
      }
      std::cout << std::endl;
    }
    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k_scores[i]), end(top_k_scores[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k_scores(j, i) << " ";
      }
      std::cout << std::endl;
    }
    for (size_t i = 0; i < num_vectors(top_k); ++i) {
      std::sort(begin(top_k_pq_scores[i]), end(top_k_pq_scores[i]));
      for (size_t j = 0; j < top_k.num_rows(); ++j) {
        std::cout << top_k_pq_scores(j, i) << " ";
      }
      std::cout << std::endl;
    }
  }
#endif

  REQUIRE(true);

}

TEST_CASE("flatpq_index: query siftsmall", "[flatpq_index]") {
  auto k_nn = 10;

  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_base_uri, 0);
  training_set.load();

  auto query_set = tdbColMajorMatrix<float>(ctx, siftsmall_query_uri, 0);
  query_set.load();

  auto groundtruth_set = tdbColMajorMatrix<int32_t>(ctx, siftsmall_groundtruth_uri, 0);
  groundtruth_set.load();

  auto pq_idx = flatpq_index<float, uint32_t, uint32_t>(128, 16, 8, 256);
  pq_idx.train(training_set);
  pq_idx.add(training_set);
  auto&& [top_k_pq_scores, top_k_pq] = pq_idx.query(query_set, 10);

  auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
      training_set, query_set, k_nn, 1, sum_of_squares_distance{});

  auto intersections0 = (long)count_intersections(top_k_pq, top_k, k_nn);
  double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
  CHECK(recall0 > 0.7);

  auto intersections1 = (long)count_intersections(top_k_pq, groundtruth_set, k_nn);
  double recall1 = intersections1 / ((double)top_k_pq.num_cols() * k_nn);
  CHECK(recall1 > 0.7);

  std::cout << "Recall: " << recall0 << " " << recall1 << std::endl;
}

TEST_CASE("flatpq_index: query 1M", "[flatpq_index]") {
  auto num_vectors = 10'000;
  auto num_queries = 100;
  auto k_nn = 10;

  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<uint8_t>(ctx, bigann1M_base_uri, num_vectors);
  training_set.load();

  auto query_set = tdbColMajorMatrix<uint8_t>(ctx, bigann1M_query_uri, num_queries);
  query_set.load();

  auto groundtruth_set = tdbColMajorMatrix<int32_t>(ctx, siftsmall_groundtruth_uri, num_queries);
  groundtruth_set.load();

  auto pq_idx = flatpq_index<uint8_t, uint32_t, uint32_t>(128, 16, 8, 256);
  pq_idx.train(training_set);
  pq_idx.add(training_set);
  auto&& [top_k_pq_scores, top_k_pq] = pq_idx.query(query_set, 10);

  auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
      training_set, query_set, k_nn, 8, sum_of_squares_distance{});

  auto intersections0 = (long)count_intersections(top_k_pq, top_k, k_nn);
  double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
  std::cout << "Recall: " << recall0 << std::endl;
  CHECK(recall0 > 0.7);

//  auto intersections1 = (long)count_intersections(top_k_pq, groundtruth_set, k_nn);
//  double recall1 = intersections1 / ((double)top_k_pq.num_cols() * k_nn);
//  CHECK(recall1 > 0.7);

  // std::cout << "Recall: " << recall0 << " " << recall1 << std::endl;
  // std::cout << "Recall: " << recall1 << std::endl;

}

TEST_CASE("flatpq_index: flatpq_index write and read", "[flatpq_index]") {
  size_t dimension_{128};
  size_t num_subspaces_{16};
  size_t bits_per_subspace_{8};
  size_t num_clusters_{256};

  tiledb::Context ctx;
  std::string flatpq_index_uri = "/tmp/tmp_flatpq_index";
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_base_uri, 0);
  load(training_set);

  auto idx = flatpq_index<float, uint32_t, uint32_t>(
      dimension_, num_subspaces_, bits_per_subspace_, num_clusters_);
  idx.train(training_set);
  idx.add(training_set);

  idx.write_index(flatpq_index_uri, true);
  auto idx2 = flatpq_index<float, uint32_t, uint32_t>(ctx, flatpq_index_uri);

  CHECK(idx.compare_metadata(idx2));

  CHECK(idx.compare_pq_vectors(idx2));
  CHECK(idx.compare_centroids(idx2));
  CHECK(idx.compare_distance_tables(idx2));
  auto foo = 0;

}