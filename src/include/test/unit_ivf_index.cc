/**
 * @file   unit_ivf_index.cc
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

#include <fstream>
#include <iostream>
#include <vector>

#include "../ivf_index.h"
#include "../linalg.h"
#include "gen_graphs.h"
#include "query_common.h"

TEST_CASE("ivf_index: test test", "[ivf_index]") {
  REQUIRE(true);
}

// kmeans and kmeans indexing still WIP

void debug_centroids(auto& index) {
  std::cout << "\nDebug Centroids:\n" << std::endl;
  for (size_t j = 0; j < index.get_centroids().num_rows(); ++j) {
    for (size_t i = 0; i < index.get_centroids().num_cols(); ++i) {
      std::cout << index.get_centroids()(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

TEST_CASE("ivf_index: test kmeans initializations", "[ivf_index][init]") {
  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);
  std::copy(begin(data), end(data), training_data.data());

  auto index =
      ivf_index<float, uint32_t, uint32_t>(4, 3, 10, 1e-4, 1, Catch::rngSeed());

  SECTION("random") {
    std::cout << "random" << std::endl;
    index.kmeans_random_init(training_data);
  }

  SECTION("kmeans++") {
    std::cout << "kmeans++" << std::endl;
    index.kmeans_pp(training_data);
  }

  CHECK(index.get_centroids().num_cols() == 3);
  CHECK(index.get_centroids().num_rows() == 4);

  // debug_centroids(index);

  for (size_t i = 0; i < index.get_centroids().num_cols() - 1; ++i) {
    for (size_t j = i + 1; j < index.get_centroids().num_cols(); ++j) {
      CHECK(!std::equal(
          index.get_centroids()[i].begin(),
          index.get_centroids()[i].end(),
          index.get_centroids()[j].begin()));
    }
  }

  size_t outer_counts = 0;
  for (size_t i = 0; i < index.get_centroids().num_cols(); ++i) {
    size_t inner_counts = 0;
    for (size_t j = 0; j < training_data.num_cols(); ++j) {
      inner_counts += std::equal(
          index.get_centroids()[i].begin(),
          index.get_centroids()[i].end(),
          training_data[j].begin());
    }
    CHECK(inner_counts == 1);
    outer_counts += inner_counts;
  }
  CHECK(outer_counts == index.get_centroids().num_cols());
}

TEST_CASE("ivf_index: test kmeans", "[ivf_index][kmeans]") {
  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);
  std::copy(begin(data), end(data), training_data.data());

  auto index =
      ivf_index<float, size_t, size_t>(4, 3, 10, 1e-4, 1, Catch::rngSeed());

  SECTION("random") {
    std::cout << "random" << std::endl;
    index.train(training_data, kmeans_init::random);
  }

  SECTION("kmeans++") {
    std::cout << "kmeans++" << std::endl;
    index.train(training_data, kmeans_init::kmeanspp);
  }
}

/*
 * Test with some data scraped from sklearn
 * More significant testing of kmeans (more significant comparisons against
 * sklearn) are done in python
 */
TEST_CASE("ivf_index: debug w/ sk", "[ivf_index]") {
  ColMajorMatrix<float> training_data{
      {1.0573647, 5.082087},
      {-6.229642, -1.3590931},
      {0.7446737, 6.3828287},
      {-7.698864, -3.0493321},
      {2.1362762, -4.4448104},
      {1.04019, -4.0389647},
      {0.38996044, 5.7235265},
      {1.7470839, -4.717076}};
  ColMajorMatrix<float> queries{{-7.3712273, -1.1178735}};
  ColMajorMatrix<float> sklearn_centroids{
      {-6.964253, -2.2042127}, {1.6411834, -4.400284}, {0.7306664, 5.7294807}};

  SECTION("one iteration") {
    std::cout << "one iteration" << std::endl;
    auto index = ivf_index<float, size_t, size_t>(
        sklearn_centroids.num_rows(),
        sklearn_centroids.num_cols(),
        1,
        1e-4,
        1,
        Catch::rngSeed());
    index.set_centroids(sklearn_centroids);
    index.train(training_data, kmeans_init::none);
    // debug_centroids(index);
  }

  SECTION("two iterations") {
    std::cout << "two iterations" << std::endl;
    auto index = ivf_index<float, size_t, size_t>(
        sklearn_centroids.num_rows(),
        sklearn_centroids.num_cols(),
        2,
        1e-4,
        1,
        Catch::rngSeed());
    index.set_centroids(sklearn_centroids);
    index.train(training_data, kmeans_init::none);
    // debug_centroids(index);
  }

  SECTION("five iterations") {
    std::cout << "five iterations" << std::endl;
    auto index = ivf_index<float, size_t, size_t>(
        sklearn_centroids.num_rows(),
        sklearn_centroids.num_cols(),
        5,
        1e-4,
        1,
        Catch::rngSeed());
    index.set_centroids(sklearn_centroids);
    index.train(training_data, kmeans_init::none);
    // debug_centroids(index);
  }

  SECTION("five iterations, perturbed") {
    std::cout << "five iterations, perturbed" << std::endl;
    for (size_t i = 0; i < sklearn_centroids.num_cols(); ++i) {
      for (size_t j = 0; j < sklearn_centroids.num_rows(); ++j) {
        sklearn_centroids(j, i) *= 0.8;
      }
    }

    sklearn_centroids(0, 0) += 0.25;
    auto index = ivf_index<float, size_t, size_t>(
        sklearn_centroids.num_rows(),
        sklearn_centroids.num_cols(),
        5,
        1e-4,
        1,
        Catch::rngSeed());
    index.set_centroids(sklearn_centroids);
    index.train(training_data, kmeans_init::none);
    // debug_centroids(index);
  }

  SECTION("five iterations") {
    std::cout << "five iterations" << std::endl;
    auto index = ivf_index<float, size_t, size_t>(
        sklearn_centroids.num_rows(),
        sklearn_centroids.num_cols(),
        5,
        1e-4,
        1,
        Catch::rngSeed());
    index.train(training_data, kmeans_init::random);
    // debug_centroids(index);
  }
}

TEST_CASE("ivf_index: ivf_index write and read", "[ivf_index]") {
  size_t dimension = 128;
  size_t nlist = 100;
  size_t nprobe = 10;
  size_t k_nn = 10;
  size_t nthreads = 1;

  tiledb::Context ctx;
  std::string ivf_index_uri = "/tmp/tmp_ivf_index";
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_base_uri, 0);
  load(training_set);

  auto idx = ivf_index<float, uint32_t, uint32_t>(dimension, nlist, nthreads);

  idx.train(training_set, kmeans_init::kmeanspp);
  idx.add(training_set);

  idx.write_index(ivf_index_uri, true);
  auto idx2 = ivf_index<float, uint32_t, uint32_t>(ctx, ivf_index_uri);
  idx2.read_index_infinite();

  CHECK(idx.compare_metadata(idx2));
  CHECK(idx.compare_centroids(idx2));
  CHECK(idx.compare_feature_vectors(idx2));
  CHECK(idx.compare_indices(idx2));
  CHECK(idx.compare_partitioned_ids(idx2));
}

TEMPLATE_TEST_CASE(
    "flativf_index: query stacked hypercube",
    "[flativf_index]",
    float,
    uint8_t) {
  size_t k_dist = GENERATE(0, 32);
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

  // Test with just a single partition -- should match flat index
  SECTION("nlist = 1") {
    size_t k_nn = 6;
    size_t nlist = 1;

    auto ivf_idx2 = ivf_index<TestType, uint32_t, uint32_t>(
        128, nlist, 4, 1.e-4, 1);  // dim nlist maxiter eps nthreads
    ivf_idx2.train(hypercube2);
    ivf_idx2.add(hypercube2);

    auto ivf_idx4 =
        ivf_index<TestType, uint32_t, uint32_t>(128, nlist, 4, 1.e-4, 1);
    ivf_idx4.train(hypercube4);
    ivf_idx4.add(hypercube4);

    auto top_k_ivf_scores = ColMajorMatrix<float>();
    auto top_k_ivf = ColMajorMatrix<unsigned>();
    auto top_k_scores = ColMajorMatrix<float>();
    auto top_k = ColMajorMatrix<unsigned long long>();
    auto query2 = ColMajorMatrix<TestType>();
    auto query4 = ColMajorMatrix<TestType>();

    SECTION("query2/4 = 0...") {
      query2 = ColMajorMatrix<TestType>{{0, 0, 0, 0, 0, 0}};
      query4 = ColMajorMatrix<TestType>{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
    }
    SECTION("query2/4 = 127...") {
      query2 = ColMajorMatrix<TestType>{{127, 127, 127, 127, 127, 127}};
      query4 = ColMajorMatrix<TestType>{
          {127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}};
    }
    SECTION("query2/4 = 0...") {
      query2 = ColMajorMatrix<TestType>{{0, 0, 0, 127, 127, 127}};
      query4 = ColMajorMatrix<TestType>{
          {0, 0, 0, 0, 0, 0, 127, 127, 127, 127, 127, 127}};
    }
    SECTION("query2/4 = 127...") {
      query2 = ColMajorMatrix<TestType>{{127, 127, 127, 0, 0, 0}};
      query4 = ColMajorMatrix<TestType>{
          {127, 127, 127, 127, 127, 127, 0, 0, 0, 0, 0, 0}};
    }
    SECTION("query2/4 = 127...") {
      query2 = ColMajorMatrix<TestType>{
          {127, 0, 127, 0, 127, 0}, {0, 127, 0, 127, 0, 127}};
      query4 = ColMajorMatrix<TestType>{
          {127, 0, 127, 0, 127, 0, 127, 0, 127, 0, 127, 0},
          {0, 127, 0, 127, 0, 127, 0, 127, 0, 127, 0, 127}};
    }

    std::tie(top_k_scores, top_k) = detail::flat::qv_query_heap(
        hypercube2, query2, k_nn, 1, sum_of_squares_distance{});
    std::tie(top_k_ivf_scores, top_k_ivf) =
        ivf_idx2.qv_query_heap_infinite_ram(query2, k_nn, 1);  // k, nprobe
    auto intersections0 = (long)count_intersections(top_k_ivf, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
    CHECK(intersections0 == k_nn * num_vectors(query2));
    CHECK(recall0 == 1.0);

    std::tie(top_k_scores, top_k) = detail::flat::qv_query_heap(
        hypercube4, query4, k_nn, 1, sum_of_squares_distance{});
    std::tie(top_k_ivf_scores, top_k_ivf) =
        ivf_idx4.qv_query_heap_infinite_ram(query4, k_nn, 1);  // k, nprobe

    auto intersections1 = (long)count_intersections(top_k_ivf, top_k, k_nn);
    double recall1 = intersections1 / ((double)top_k.num_cols() * k_nn);
    CHECK(intersections1 == k_nn * num_vectors(query4));
    CHECK(recall1 == 1.0);
  }
}

// base is 10k, learn is 25k
struct siftsmall_test_init {
  using feature_type = float;
  using id_type = uint32_t;
  using px_type = uint64_t;

  tiledb::Context ctx_;
  size_t nlist;
  size_t nprobe;

  size_t k_nn = 10;
  size_t nthreads = 0;
  size_t max_iter = 10;
  float tolerance = 1e-4;

  siftsmall_test_init(tiledb::Context ctx, size_t nl)
      : ctx_{ctx}
      , nlist(nl)
      , nprobe(std::min<size_t>(10, nlist))
      , training_set(tdbColMajorMatrix<feature_type>(ctx_, siftsmall_base_uri))
      , query_set(tdbColMajorMatrix<feature_type>(ctx_, siftsmall_query_uri))
      , groundtruth_set(
            tdbColMajorMatrix<int32_t>(ctx_, siftsmall_groundtruth_uri))
      , idx(128, nlist, max_iter, tolerance, nthreads, 0xdeadbeef) {
    training_set.load();
    query_set.load();
    groundtruth_set.load();
    std::tie(top_k_scores, top_k) = detail::flat::qv_query_heap(
        training_set, query_set, k_nn, 1, sum_of_squares_distance{});

    idx.train(training_set);
    idx.add(training_set);
  }

  auto get_write_read_idx() {
    std::string tmp_ivf_index_uri = "/tmp/tmp_ivf_index";
    idx.write_index(tmp_ivf_index_uri, true);
    auto idx0 = ivf_index<feature_type, id_type, px_type>(ctx_, tmp_ivf_index_uri);
    return idx0;
  }

  tdbColMajorMatrix<float> training_set;
  tdbColMajorMatrix<float> query_set;
  tdbColMajorMatrix<int32_t> groundtruth_set;
  ColMajorMatrix<float> top_k_scores;
  ColMajorMatrix<unsigned long long> top_k;
  ivf_index<feature_type, id_type, px_type> idx;

  auto verify(auto&& top_k_ivf) {
    // These are helpful for debugging
    // debug_slice(top_k_ivf, "top_k_ivf");
    // debug_slice(top_k_ivf_scores, "top_k_ivf_scores");

    auto intersectionsm1 =
        (long)count_intersections(top_k, groundtruth_set, k_nn);
    double recallm1 = intersectionsm1 / ((double)top_k.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersectionsm1 == num_vectors(top_k) * dimension(top_k));
      CHECK(recallm1 == 1.0);
    }
    CHECK(recallm1 > .99);

    // @todo There is randomness in initialization of kmeans, use a fixed seed
    auto intersections0 = (long)count_intersections(top_k_ivf, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersections0 == num_vectors(top_k) * dimension(top_k));
      CHECK(recall0 == 1.0);
    }
    CHECK(recall0 > .965);

    auto intersections1 =
        (long)count_intersections(top_k_ivf, groundtruth_set, k_nn);
    double recall1 = intersections1 / ((double)top_k_ivf.num_cols() * k_nn);
    if (nlist == 1) {
      CHECK(intersections1 == num_vectors(top_k) * dimension(top_k));
      CHECK(recall1 == 1.0);
    }
    CHECK(recall1 > 0.965);

    // std::cout << "Recall: " << recall0 << " " << recall1 << std::endl;
  }
};

// Note:  In-place only makes sense for infinite ram case
// @todo Use a fixed seed for initializing kmeans
TEST_CASE(
    "ivf_index: Build index and query in place, infinite", "[ivf_index]") {
  tiledb::Context ctx;
  size_t nlist = GENERATE(1, 100);

  auto init = siftsmall_test_init(ctx, nlist);

  auto&& [nprobe, k_nn, nthreads, max_iter, tolerance] = std::tie(
      init.nprobe, init.k_nn, init.nthreads, init.max_iter, init.tolerance);
  auto&& [idx, training_set, query_set, groundtruth_set] = std::tie(
      init.idx, init.training_set, init.query_set, init.groundtruth_set);

  auto top_k_ivf_scores = ColMajorMatrix<float>();
  auto top_k_ivf = ColMajorMatrix<unsigned>();

  SECTION("infinite") {
    INFO("infinite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.query_infinite_ram(query_set, k_nn, nprobe);
  }

  SECTION("qv_infinite") {
    INFO("qv_infinite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.qv_query_heap_infinite_ram(query_set, k_nn, nprobe);
  }

  SECTION("nuv_infinite") {
    INFO("nuv_infinite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.qv_query_heap_infinite_ram(query_set, k_nn, nprobe);
  }

  SECTION("nuv_infinite_reg_blocked") {
    INFO("nuv_infinite_reg_blocked");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.qv_query_heap_infinite_ram(query_set, k_nn, nprobe);
  }
  init.verify(top_k_ivf);
}

TEST_CASE("Build index, write, read and query, infinite", "[ivf_index]") {
  tiledb::Context ctx;
  size_t nlist = GENERATE(1, 100);

  auto init = siftsmall_test_init(ctx, nlist);

  auto&& [nprobe, k_nn, nthreads, max_iter, tolerance] = std::tie(
      init.nprobe, init.k_nn, init.nthreads, init.max_iter, init.tolerance);
  auto&& [_, training_set, query_set, groundtruth_set] = std::tie(
      init.idx, init.training_set, init.query_set, init.groundtruth_set);
  auto idx = init.get_write_read_idx();

  auto top_k_ivf_scores = ColMajorMatrix<float>();
  auto top_k_ivf = ColMajorMatrix<typename decltype(init)::id_type>();

  SECTION("infinite") {
    INFO("infinite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.query_infinite_ram(query_set, k_nn, nprobe);
  }

  SECTION("qv_infinite") {
    INFO("qv_infinite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.qv_query_heap_infinite_ram(query_set, 10, 10);
  }

  SECTION("nuv_infinite") {
    INFO("nuv_infinite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.qv_query_heap_infinite_ram(query_set, 10, 10);
  }

  SECTION("nuv_infinite_reg_blocked") {
    INFO("nuv_infinite_reg_blocked");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.qv_query_heap_infinite_ram(query_set, 10, 10);
  }
  init.verify(top_k_ivf);
}

TEST_CASE("Build index, write, read and query, finite", "ivf_index") {
  tiledb::Context ctx;
  size_t nlist = GENERATE(1, 100);

  auto init = siftsmall_test_init(ctx, nlist);

  auto&& [nprobe, k_nn, nthreads, max_iter, tolerance] = std::tie(
      init.nprobe, init.k_nn, init.nthreads, init.max_iter, init.tolerance);
  auto&& [_, training_set, query_set, groundtruth_set] = std::tie(
      init.idx, init.training_set, init.query_set, init.groundtruth_set);
  auto idx = init.get_write_read_idx();

  auto top_k_ivf_scores = ColMajorMatrix<float>();
  auto top_k_ivf = ColMajorMatrix<unsigned>();

  SECTION("finite") {
    INFO("finite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.query_finite_ram(query_set, k_nn, nprobe);
  }

  SECTION("nuv_finite") {
    INFO("nuv_finite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.nuv_query_heap_finite_ram(query_set, k_nn, nprobe);
  }

  SECTION("nuv_finite_reg_blocked") {
    INFO("nuv_finite_reg_blocked");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.nuv_query_heap_finite_ram_reg_blocked(query_set, k_nn, nprobe);
  }
  init.verify(top_k_ivf);
}

TEST_CASE(
    "Build index, write, read and query, finite, out of core", "[ivf_index]") {
  tiledb::Context ctx;
  size_t nlist = 100;
  size_t upper_bound = 5000;

  auto init = siftsmall_test_init(ctx, nlist);

  auto&& [nprobe, k_nn, nthreads, max_iter, tolerance] = std::tie(
      init.nprobe, init.k_nn, init.nthreads, init.max_iter, init.tolerance);
  auto&& [_, training_set, query_set, groundtruth_set] = std::tie(
      init.idx, init.training_set, init.query_set, init.groundtruth_set);
  auto idx = init.get_write_read_idx();

  auto top_k_ivf_scores = ColMajorMatrix<float>();
  auto top_k_ivf = ColMajorMatrix<unsigned>();

  SECTION("nuv_finite") {
    INFO("nuv_finite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.nuv_query_heap_finite_ram(query_set, k_nn, nprobe, upper_bound);
  }

  SECTION("nuv_finite_reg_blocked") {
    INFO("nuv_finite_reg_blocked");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.nuv_query_heap_finite_ram_reg_blocked(
            query_set, k_nn, nprobe, upper_bound);
  }

  SECTION("finite") {
    INFO("finite");
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.query_finite_ram(query_set, k_nn, nprobe, upper_bound);
  }

  init.verify(top_k_ivf);
}


TEST_CASE("Read from externally written index", "[ivf_index]") {
// f_type: float
// id_type: uint32
// px_type: uint64
  using feature_type = typename siftsmall_test_init::feature_type;
  using id_type = typename siftsmall_test_init::id_type;
  using px_type = typename siftsmall_test_init::px_type;

  auto k_nn = 10;
  auto nprobe = 20;
  auto nlist = 100;

  tiledb::Context ctx;
  auto query_set = tdbColMajorMatrix<float>(ctx, siftsmall_query_uri);
  query_set.load();
  auto groundtruth_set = tdbColMajorMatrix<int32_t>(ctx, siftsmall_groundtruth_uri);
  groundtruth_set.load();

  auto top_k_ivf_scores = ColMajorMatrix<float>();
  auto top_k_ivf = ColMajorMatrix<id_type>();

  auto init = siftsmall_test_init(ctx, nlist);
  std::string tmp_ivf_index_uri = "/tmp/tmp_ivf_index";
  init.idx.write_index(tmp_ivf_index_uri, true);

// Just some sanity checking and for interactive debugging
#if 0
  SECTION("compare cli and init") {
    auto idx1 = ivf_index<feature_type, id_type, px_type>(ctx, siftsmall_flatIVF_index_uri);
    idx1.open_index(ctx, siftsmall_flatIVF_index_uri);
    idx1.read_index_infinite();

    auto idx2 = ivf_index<feature_type, id_type, px_type>(ctx, tmp_ivf_index_uri);
    idx2.open_index(ctx, tmp_ivf_index_uri);
    idx2.read_index_infinite();

    CHECK(idx1.compare_metadata(idx2));
    CHECK(idx1.compare_centroids(idx2));
    CHECK(idx1.compare_feature_vectors(idx2));
    CHECK(idx1.compare_indices(idx2));
    CHECK(idx1.compare_partitioned_ids(idx2));
  }
#endif

  SECTION("read cli generated") {
    INFO("infinite cli");
    auto idx = ivf_index<feature_type, id_type, px_type>(ctx, siftsmall_flatIVF_index_uri);
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.query_infinite_ram(query_set, k_nn, nprobe);
  }

  SECTION("read init generated") {
    INFO("infinite init");
    auto idx = ivf_index<feature_type, id_type, px_type>(ctx, tmp_ivf_index_uri);
    std::tie(top_k_ivf_scores, top_k_ivf) =
        idx.query_infinite_ram(query_set, k_nn, nprobe);
  }

  auto intersections1 =
      (long)count_intersections(top_k_ivf, groundtruth_set, k_nn);
  double recall1 = intersections1 / ((double)top_k_ivf.num_cols() * k_nn);
  if (nlist == 1) {
    CHECK(intersections1 == num_vectors(top_k_ivf) * dimension(top_k_ivf));
    CHECK(recall1 == 1.0);
  }
  CHECK(recall1 > 0.965);
}