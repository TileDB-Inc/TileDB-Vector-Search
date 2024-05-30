/**
 * @file   unit_flat_pq_index.cc
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
#include <cmath>

#include "detail/flat/qv.h"
#include "index/flat_pq_index.h"
#include "scoring.h"
#include "test/utils/array_defs.h"
#include "test/utils/gen_graphs.h"
#include "test/utils/query_common.h"

TEST_CASE("flat qv_partition with sub distance", "[flat_pq_index]") {
  {
    auto sub_sift_base =
        ColMajorMatrix<sift_feature_type>(2, num_vectors(sift_base));
    for (size_t i = 0; i < num_vectors(sift_base); i++) {
      for (size_t j = 0; j < 2; j++) {
        sub_sift_base(j, i) = sift_base(j, i);
      }
    }
    auto sub_sift_query =
        ColMajorMatrix<sift_feature_type>(2, num_vectors(sift_query));
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
    auto sub_sift_base =
        ColMajorMatrix<sift_feature_type>(2, num_vectors(sift_base));
    for (size_t i = 0; i < num_vectors(sift_base); i++) {
      for (size_t j = 1; j < 3; j++) {
        sub_sift_base(j - 1, i) = sift_base(j, i);
      }
    }
    auto sub_sift_query =
        ColMajorMatrix<sift_feature_type>(2, num_vectors(sift_query));
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

TEST_CASE("flat qv_partition hypercube", "[flat_pq_index]") {
  const bool debug = false;

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
      hypercube2(j + 3, i) = hypercube1(j, i);
    }
  }

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < 8; ++i) {
      centroids2(j, i) = init_centroids(j, i);
      centroids2(j + 3, i) = init_centroids(j, i);
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

  auto compare_centroids = [](auto&& a, auto&& b, size_t ba, size_t bb) {
    for (size_t i = 0; i < 8; ++i) {
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

  SECTION("subspace = 3 (0)") {
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

  SECTION("subspace = 3 (0) again") {
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

  SECTION("subspace = 3 (1)") {
    sub_kmeans(hypercube1, init_centroids, 0, 3, 8, 0.00, 10, 1);
    compare_centroids(expected_centroids1, init_centroids, 0, 0);

    if (debug) {
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

TEST_CASE("normalize matrix", "[flat_pq_index]") {
  const bool debug = false;

  auto hypercube = build_hypercube<float>(2, 2, 0xdeadbeef);
  auto normalized = normalize_matrix(hypercube);

  if (debug) {
    debug_matrix(hypercube);
    debug_matrix(normalized);
  }
}

TEMPLATE_TEST_CASE("create flat_pq_index", "[flat_pq_index]", float, uint8_t) {
  auto pq_idx = flat_pq_index<TestType, uint32_t, uint32_t>(128, 16, 8);
  REQUIRE(true);
}

TEMPLATE_TEST_CASE("train flat_pq_index", "[flat_pq_index]", float, uint8_t) {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flat_pq_index<TestType, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);

  REQUIRE(true);
}

TEMPLATE_TEST_CASE("add flat_pq_index", "[flat_pq_index]", float, uint8_t) {
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flat_pq_index<TestType, uint32_t, uint32_t>(3, 1, 8);
  pq_idx.train(hypercube);
  pq_idx.add(hypercube);

  REQUIRE(true);
}

TEMPLATE_TEST_CASE(
    "verify flat_pq_index encoding with hypercube",
    "[flat_pq_index]",
    float,
    uint8_t) {
  size_t k_near = 0;
  size_t k_far = 0;

  auto hypercube = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);

  auto pq_idx = flat_pq_index<TestType, uint32_t, uint32_t>(3, 1, 8, 8);
  pq_idx.train(hypercube);
  pq_idx.add(hypercube);
  auto avg_error = pq_idx.verify_pq_encoding(hypercube);

  CHECK(avg_error < 0.05);
}

TEMPLATE_TEST_CASE(
    "verify flat_pq_index encoding with stacked hypercube",
    "[flat_pq_index]",
    float,
    uint8_t) {
  /*
   * Note -- Reassignment breaks this small example if k_near and k_far
   * are equal to zero because some of the centroids are duplicates of one
   * another, and so the span of the centroid space is much less than the
   * number of clusters
   */
  size_t k_near = 32;
  size_t k_far = 32;

  auto hypercube0 = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);
  auto hypercube1 = build_hypercube<TestType>(k_near, k_far, 0xbeefdead);

  auto hypercube2 = ColMajorMatrix<TestType>(6, num_vectors(hypercube0));
  auto hypercube4 = ColMajorMatrix<TestType>(12, num_vectors(hypercube0));

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < num_vectors(hypercube4); ++i) {
      hypercube2(j, i) = hypercube0(j, i);
      hypercube2(j + 3, i) = hypercube1(j, i);
      hypercube4(j, i) = hypercube0(j, i);
      hypercube4(j + 3, i) = hypercube1(j, i);
      hypercube4(j + 6, i) = hypercube0(j, i);
      hypercube4(j + 9, i) = hypercube1(j, i);
    }
  }

  auto pq_idx2 = flat_pq_index<TestType, uint32_t, uint32_t>(6, 2, 8, 32);
  pq_idx2.train(hypercube2);
  pq_idx2.add(hypercube2);
  auto pq_idx4 = flat_pq_index<TestType, uint32_t, uint32_t>(12, 4, 8, 32);
  pq_idx4.train(hypercube4);
  pq_idx4.add(hypercube4);

  SECTION("pq_encoding") {
    auto avg_error2 = pq_idx2.verify_pq_encoding(hypercube2);
    CHECK(avg_error2 < 0.01);
    auto avg_error4 = pq_idx4.verify_pq_encoding(hypercube4);
    CHECK(avg_error4 < 0.01);
  }

  SECTION("pq_distances") {
    auto avg_error2 = pq_idx2.verify_pq_distances(hypercube2);
    CHECK(avg_error2 < 0.1);
    auto avg_error4 = pq_idx4.verify_pq_distances(hypercube4);
    CHECK(avg_error4 < 0.1);
  }

  SECTION("asymmetric_pq_distances") {
    auto [max_error2, avg_error2] =
        pq_idx2.verify_asymmetric_pq_distances(hypercube2);
    CHECK(avg_error2 < 0.075);
    auto [max_erro4, avg_error4] =
        pq_idx4.verify_asymmetric_pq_distances(hypercube4);
    CHECK(avg_error4 < 0.075);
  }

  SECTION("symmetric_pq_distances") {
    auto [max_error2, avg_error2] =
        pq_idx2.verify_symmetric_pq_distances(hypercube2);
    CHECK(avg_error2 < 0.1);
    auto [max_erro4, avg_error4] =
        pq_idx4.verify_symmetric_pq_distances(hypercube4);
    CHECK(avg_error4 < 0.1);
  }
}

TEST_CASE(
    "verify pq_encoding and pq_distances with siftsmall", "[flat_pq_index]") {
  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<siftsmall_feature_type>(
      ctx, siftsmall_inputs_uri, 2500);
  training_set.load();

  auto pq_idx = flat_pq_index<
      siftsmall_feature_type,
      siftsmall_ids_type,
      siftsmall_indices_type>(128, 16, 8, 50);
  pq_idx.train(training_set);
  pq_idx.add(training_set);

  SECTION("pq_encoding") {
    auto avg_error = pq_idx.verify_pq_encoding(training_set);
    CHECK(avg_error < 0.08);
  }
  SECTION("pq_distances") {
    auto avg_error = pq_idx.verify_pq_distances(training_set);
    CHECK(avg_error < 0.15);
  }
  SECTION("asymmetric_pq_distances") {
    auto [max_error, avg_error] =
        pq_idx.verify_asymmetric_pq_distances(training_set);
    CHECK(avg_error < 0.08);
  }
  SECTION("symmetric_pq_distances") {
    auto [max_error, avg_error] =
        pq_idx.verify_symmetric_pq_distances(training_set);
    CHECK(avg_error < 0.15);
  }
}

TEMPLATE_TEST_CASE(
    "query stacked hypercube", "[flat_pq_index]", float, uint8_t) {
  size_t k_dist = GENERATE(/*0,*/ 32);
  size_t k_near = k_dist;
  size_t k_far = k_dist;

  auto hypercube0 = build_hypercube<TestType>(k_near, k_far, 0xdeadbeef);
  auto hypercube1 = build_hypercube<TestType>(k_near, k_far, 0xbeefdead);

  auto hypercube2 = ColMajorMatrix<TestType>(6, num_vectors(hypercube0));
  auto hypercube4 = ColMajorMatrix<TestType>(12, num_vectors(hypercube0));

  for (size_t j = 0; j < 3; ++j) {
    for (size_t i = 0; i < num_vectors(hypercube4); ++i) {
      hypercube2(j, i) = hypercube0(j, i);
      hypercube2(j + 3, i) = hypercube1(j, i);

      hypercube4(j, i) = hypercube0(j, i);
      hypercube4(j + 3, i) = hypercube1(j, i);
      hypercube4(j + 6, i) = hypercube0(j, i);
      hypercube4(j + 9, i) = hypercube1(j, i);
    }
  }

  auto pq_idx2 = flat_pq_index<TestType, uint32_t, uint32_t>(
      6, 2, 8, k_dist == 0 ? 8 : 16);
  pq_idx2.train(hypercube2);
  pq_idx2.add(hypercube2);

  auto pq_idx4 = flat_pq_index<TestType, uint32_t, uint32_t>(
      12, 4, 8, k_dist == 0 ? 8 : 16);
  pq_idx4.train(hypercube4);
  pq_idx4.add(hypercube4);

  using enc_type = std::tuple<std::vector<TestType>, std::vector<uint8_t>>;
  auto expected = std::vector<enc_type>{
      {{127, 127, 127, 127, 127, 127}, {0, 0}},
      {{127, 127, 127, 0, 0, 0}, {0, 1}},
      {{127, 127, 0, 0, 0, 127}, {4, 2}},
      {{0, 127, 0, 127, 127, 127}, {7, 0}},
      {{127, 127, 127, 0, 127, 0}, {0, 7}},
      {{127, 0, 0, 95, 31, 95}, {3, 6}},
      {{95, 31, 95, 0, 127, 127}, {6, 5}},
  };

  SECTION("Test sub_distance_*symmetric, hypercube2") {
    for (auto&& [vx, pqx] : expected) {
      // asymmetric (vx, pqx)
      // asymmetric (decode(pqx), encode(vx))
      // symmetric (encode(vx), pqx)
      // sum_of_squares(vx, decode(pqx))

      // symmetric (encode(vx), decode(encode(pqx))
      // asymmetric (decode(encode(vx)), pqx)
      // sum_of_squares (decode(encode(vx)), decode(pqx))
      // sum_of_squares (decode(encode(vx)), vx);

      auto a_vx_pqx2 = pq_idx2.sub_distance_asymmetric(vx, pqx);
      auto a_dpqx_evx2 = pq_idx2.sub_distance_asymmetric(
          pq_idx2.decode(pqx), pq_idx2.encode(vx));
      auto s_evx_pqx2 = pq_idx2.sub_distance_symmetric(pq_idx2.encode(vx), pqx);
      auto ss_vx_dpqx2 = l2_distance(vx, pq_idx2.decode(pqx));
      auto s_evx_edpqx2 = pq_idx2.sub_distance_symmetric(
          pq_idx2.encode(vx), pq_idx2.encode(pq_idx2.decode(pqx)));
      auto a_evx_edpqx2 = pq_idx2.sub_distance_asymmetric(
          pq_idx2.decode(pq_idx2.encode(vx)), pqx);
      auto ss_devx_dpqx2 =
          l2_distance(pq_idx2.decode(pq_idx2.encode(vx)), pq_idx2.decode(pqx));
      auto ss_devx_vx2 = l2_distance(pq_idx2.decode(pq_idx2.encode(vx)), vx);

      auto scale = l2_distance(vx);
      if (a_vx_pqx2 > 1) {
        scale *= a_vx_pqx2;
      }

      CHECK(a_vx_pqx2 / scale < 0.00005);
      CHECK(a_dpqx_evx2 / scale < 0.00005);
      CHECK(s_evx_pqx2 / scale < 0.00005);
      CHECK(ss_vx_dpqx2 / scale < 0.00005);
      CHECK(s_evx_edpqx2 / scale < 0.00005);
      CHECK(a_evx_edpqx2 / scale < 0.00005);
      CHECK(ss_devx_dpqx2 / scale < 0.00005);
      CHECK(ss_devx_vx2 / scale < 0.000075);
    }
  }

  // Need to implement with its own expected four-element set
#if 0
  SECTION("Test sub_distance_*symmetric, hypercube4") {
    for (auto&& [vx, pqx] : expected) {
      auto a_vx_pqx4 = pq_idx4.sub_distance_asymmetric(vx, pqx);
      auto a_dpqx_evx4 = pq_idx4.sub_distance_asymmetric(
          pq_idx4.decode(pqx), pq_idx4.encode(vx));
      auto s_evx_pqx4 = pq_idx4.sub_distance_symmetric(pq_idx4.encode(vx), pqx);
      auto ss_vx_dpqx4 = l2_distance(vx, pq_idx4.decode(pqx));
      auto s_evx_edpqx4 = pq_idx4.sub_distance_symmetric(
          pq_idx4.encode(vx), pq_idx4.encode(pq_idx4.decode(pqx)));
      auto a_evx_edpqx4 = pq_idx4.sub_distance_symmetric(
          pq_idx4.decode(pq_idx4.encode(vx)), pqx);
      auto ss_devx_dpqx4 = l2_distance(
          pq_idx4.decode(pq_idx4.encode(vx)), pq_idx4.decode(pqx));
      auto ss_devx_vx4 = l2_distance(pq_idx4.decode(pq_idx4.encode(vx)), vx);

      auto scale = l2_distance(vx);
      if (a_vx_pqx4 > 1) {
        scale *= a_vx_pqx4;
      }

      REQUIRE(scale > 0);
      REQUIRE(!std::isnan(scale));

      CHECK(!std::isnan(a_vx_pqx4));
      CHECK(!std::isnan(a_dpqx_evx4));
      CHECK(!std::isnan(s_evx_pqx4));
      CHECK(!std::isnan(ss_vx_dpqx4));
      CHECK(!std::isnan(s_evx_edpqx4));
      CHECK(!std::isnan(a_evx_edpqx4));
      CHECK(!std::isnan(ss_devx_dpqx4));
      CHECK(!std::isnan(ss_devx_vx4));

      CHECK(!std::isnan(a_vx_pqx4 / scale));
      CHECK(!std::isnan(a_dpqx_evx4 / scale));
      CHECK(!std::isnan(s_evx_pqx4 / scale));
      CHECK(!std::isnan(ss_vx_dpqx4 / scale));
      CHECK(!std::isnan(s_evx_edpqx4 / scale));
      CHECK(!std::isnan(a_evx_edpqx4 / scale));
      CHECK(!std::isnan(ss_devx_dpqx4 / scale));
      CHECK(!std::isnan(ss_devx_vx4 / scale));

      CHECK(a_vx_pqx4 / scale < 0.0005);
      CHECK(a_dpqx_evx4 / scale < 0.0005);
      CHECK(s_evx_pqx4 / scale < 0.0005);
      CHECK(ss_vx_dpqx4 / scale < 0.0005);
      CHECK(s_evx_edpqx4 / scale < 0.0005);
      CHECK(a_evx_edpqx4 / scale < 0.0005);
      CHECK(ss_devx_dpqx4 / scale < 0.0005);
      CHECK(ss_devx_vx4 / scale < 0.0005);
    }
  }
#endif

// Add CHECKs / REQUIREs?
#if 0
  SECTION("Test asymmetric query 2") {
    auto query = ColMajorMatrix<TestType>{{0, 0, 0, 0, 0, 0}};
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx2.asymmetric_query(query, 4);

    auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
        hypercube2, query, 4, 8, l2_distance_distance{});

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

  SECTION("Test query 4") {
    auto query =
        ColMajorMatrix<TestType>{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx4.query(query, 8);

    auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
        hypercube4, query, 8, 8, l2_distance_distance{});

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
}

TEST_CASE("query siftsmall", "[flat_pq_index]") {
  const bool debug = false;

  auto k_nn = 10;

  tiledb::Context ctx;
  auto training_set =
      tdbColMajorMatrix<siftsmall_feature_type>(ctx, siftsmall_inputs_uri, 0);
  training_set.load();

  auto query_set =
      tdbColMajorMatrix<siftsmall_feature_type>(ctx, siftsmall_query_uri, 0);
  query_set.load();

  auto groundtruth_set = tdbColMajorMatrix<siftsmall_groundtruth_type>(
      ctx, siftsmall_groundtruth_uri, 0);
  groundtruth_set.load();

  auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
      training_set, query_set, k_nn, 1, l2_distance);

  auto pq_idx = flat_pq_index<
      siftsmall_feature_type,
      siftsmall_ids_type,
      siftsmall_indices_type>(128, 16, 8, 256);
  pq_idx.train(training_set);
  pq_idx.add(training_set);

  SECTION("asymmetric") {
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx.asymmetric_query(query_set, 10);

    auto intersections0 = (long)count_intersections(top_k_pq, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
    CHECK(recall0 > 0.7);

    auto intersections1 =
        (long)count_intersections(top_k_pq, groundtruth_set, k_nn);
    double recall1 = intersections1 / ((double)top_k_pq.num_cols() * k_nn);
    CHECK(recall1 > 0.7);

    if (debug) {
      std::cout << "Recall: " << recall0 << " " << recall1 << std::endl;
    }
  }

  SECTION("symmetric") {
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx.symmetric_query(query_set, 10);

    auto intersections0 = (long)count_intersections(top_k_pq, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
    CHECK(recall0 > 0.6);

    auto intersections1 =
        (long)count_intersections(top_k_pq, groundtruth_set, k_nn);
    double recall1 = intersections1 / ((double)top_k_pq.num_cols() * k_nn);
    CHECK(recall1 > 0.6);

    if (debug) {
      std::cout << "Recall: " << recall0 << " " << recall1 << std::endl;
    }
  }
}

TEST_CASE("query 1M", "[flat_pq_index]") {
  const bool debug = false;

  auto num_vectors = num_bigann1M_vectors;
  auto num_queries = 100;
  auto k_nn = 10;

  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<bigann1M_feature_type>(
      ctx, bigann1M_inputs_uri, num_vectors);
  training_set.load();

  auto query_set = tdbColMajorMatrix<bigann1M_feature_type>(
      ctx, bigann1M_query_uri, num_queries);
  query_set.load();

  auto groundtruth_set = tdbColMajorMatrix<bigann1M_groundtruth_type>(
      ctx, siftsmall_groundtruth_uri, num_queries);
  groundtruth_set.load();

  auto&& [top_k_scores, top_k] = detail::flat::qv_query_heap(
      training_set, query_set, k_nn, 8, l2_distance);

  auto pq_idx = flat_pq_index<
      bigann1M_feature_type,
      bigann1M_ids_type,
      bigann1M_indices_type>(128, 16, 8, 256);
  pq_idx.train(training_set);
  pq_idx.add(training_set);

  SECTION("asymmetric") {
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx.asymmetric_query(query_set, 10);

    auto intersections0 = (long)count_intersections(top_k_pq, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);

    if (debug) {
      std::cout << "Recall: " << recall0 << std::endl;
    }
    CHECK(recall0 > 0.65);
  }

  SECTION("symmetric") {
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx.symmetric_query(query_set, 10);

    auto intersections0 = (long)count_intersections(top_k_pq, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);

    if (debug) {
      std::cout << "Recall: " << recall0 << std::endl;
    }
    CHECK(recall0 > 0.55);
  }

  SECTION("asymmetric") {
    auto&& [top_k_pq_scores, top_k_pq] = pq_idx.asymmetric_query(query_set, 10);

    auto intersections0 = (long)count_intersections(top_k_pq, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);

    if (debug) {
      std::cout << "Recall: " << recall0 << std::endl;
    }
    CHECK(recall0 > 0.65);
  }
}

TEST_CASE("flat_pq_index write and read", "[flat_pq_index]") {
  const bool debug = false;

  size_t dimensions_{128};
  size_t num_subspaces_{16};
  size_t bits_per_subspace_{8};
  size_t num_clusters_{256};

  tiledb::Context ctx;
  std::string flatpq_index_uri =
      (std::filesystem::temp_directory_path() / "tmp_flatpq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(flatpq_index_uri)) {
    vfs.remove_dir(flatpq_index_uri);
  }
  auto training_set =
      tdbColMajorMatrix<siftsmall_feature_type>(ctx, siftsmall_inputs_uri, 0);
  load(training_set);

  auto idx = flat_pq_index<
      siftsmall_feature_type,
      siftsmall_ids_type,
      siftsmall_indices_type>(
      dimensions_, num_subspaces_, bits_per_subspace_, num_clusters_);
  idx.train(training_set);
  idx.add(training_set);

  idx.write_index(flatpq_index_uri);
  auto idx2 = flat_pq_index<
      siftsmall_feature_type,
      siftsmall_ids_type,
      siftsmall_indices_type>(ctx, flatpq_index_uri);

  CHECK(idx.compare_metadata(idx2));

  CHECK(idx.compare_pq_vectors(idx2));
  CHECK(idx.compare_centroids(idx2));
  CHECK(idx.compare_distance_tables(idx2));
  auto foo = 0;
}
