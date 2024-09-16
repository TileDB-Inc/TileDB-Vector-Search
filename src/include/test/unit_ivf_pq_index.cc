/**
 * @file   unit_ivf_pq_index.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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

#include <catch2/catch_all.hpp>
#include "index/ivf_pq_index.h"
#include "test/utils/array_defs.h"
#include "test/utils/gen_graphs.h"
#include "test/utils/query_common.h"
#include "test/utils/test_utils.h"

struct dummy_pq_index {
  using feature_type = float;
  using flat_vector_feature_type = feature_type;
  using id_type = int;
  using indices_type = int;
  using centroid_feature_type = float;
  using pq_code_type = uint8_t;
  using pq_vector_feature_type = pq_code_type;
  using score_type = float;

  uint64_t dimensions() const {
    return 128;
  }
  uint32_t num_subspaces() const {
    return 16;
  }
  auto num_clusters() const {
    return 256;
  }
  auto sub_dimensions() const {
    return 8;
  }
  auto bits_per_subspace() const {
    return 8;
  }
};

void debug_flat_ivf_centroids(const auto& index) {
  std::cout << "\nDebug Centroids:\n" << std::endl;
  for (size_t j = 0; j < index.get_flat_ivf_centroids().num_rows(); ++j) {
    for (size_t i = 0; i < index.get_flat_ivf_centroids().num_cols(); ++i) {
      std::cout << index.get_flat_ivf_centroids()(j, i) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

TEST_CASE("construct different types", "[ivf_pq_index]") {
  ivf_pq_index<int8_t, uint32_t, uint32_t> index1{};
  ivf_pq_index<uint8_t, uint32_t, uint32_t> index2{};
  ivf_pq_index<float, uint32_t, uint32_t> index3{};
  ivf_pq_index<int8_t, uint32_t, uint64_t> index4{};
  ivf_pq_index<uint8_t, uint32_t, uint64_t> index5{};
  ivf_pq_index<float, uint32_t, uint64_t> index6{};
  ivf_pq_index<int8_t, uint64_t, uint32_t> index7{};
  ivf_pq_index<uint8_t, uint64_t, uint32_t> index8{};
  ivf_pq_index<float, uint64_t, uint32_t> index9{};
  ivf_pq_index<int8_t, uint64_t, uint64_t> index10{};
  ivf_pq_index<uint8_t, uint64_t, uint64_t> index11{};
  ivf_pq_index<float, uint64_t, uint64_t> index12{};
}

TEST_CASE("default construct two", "[ivf_pq_index]") {
  ivf_pq_index<float, uint32_t, uint32_t> x;
  ivf_pq_index<float, uint32_t, uint32_t> y;
  CHECK(x.compare_cached_metadata(y));
  CHECK(y.compare_cached_metadata(x));
}

TEST_CASE("test kmeans initializations", "[ivf_pq_index][init]") {
  const bool debug = false;

  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);
  std::copy(begin(data), end(data), training_data.data());

  auto index = ivf_pq_index<float, uint32_t, uint32_t>(
      /*4,*/ 3, 2, 10, 1e-4);

  index.set_flat_ivf_centroids(ColMajorMatrix<float>(4, 3));

  SECTION("random") {
    if (debug) {
      std::cout << "random" << std::endl;
    }
    index.kmeans_random_init(training_data);
  }

  SECTION("kmeans++") {
    if (debug) {
      std::cout << "kmeans++" << std::endl;
    }
    index.kmeans_pp(training_data);
  }

  CHECK(index.get_flat_ivf_centroids().num_cols() == 3);
  CHECK(index.get_flat_ivf_centroids().num_rows() == 4);

  // debug_centroids(index);

  for (size_t i = 0; i < index.get_flat_ivf_centroids().num_cols() - 1; ++i) {
    for (size_t j = i + 1; j < index.get_flat_ivf_centroids().num_cols(); ++j) {
      CHECK(!std::equal(
          index.get_flat_ivf_centroids()[i].begin(),
          index.get_flat_ivf_centroids()[i].end(),
          index.get_flat_ivf_centroids()[j].begin()));
    }
  }

  size_t outer_counts = 0;
  for (size_t i = 0; i < index.get_flat_ivf_centroids().num_cols(); ++i) {
    size_t inner_counts = 0;
    for (size_t j = 0; j < training_data.num_cols(); ++j) {
      inner_counts += std::equal(
          index.get_flat_ivf_centroids()[i].begin(),
          index.get_flat_ivf_centroids()[i].end(),
          training_data[j].begin());
    }
    CHECK(inner_counts == 1);
    outer_counts += inner_counts;
  }
  CHECK(outer_counts == index.get_flat_ivf_centroids().num_cols());
}

TEST_CASE("test kmeans", "[ivf_pq_index][kmeans]") {
  const bool debug = false;

  std::vector<float> data = {8, 6, 7, 5, 3, 3, 7, 2, 1, 4, 1, 3, 0, 5, 1, 2,
                             9, 9, 5, 9, 2, 0, 2, 7, 7, 9, 8, 6, 7, 9, 6, 6};

  ColMajorMatrix<float> training_data(4, 8);
  std::copy(begin(data), end(data), training_data.data());

  auto index = ivf_pq_index<float, size_t, size_t>(
      /*4,*/ 3, 2, 10, 1e-4);

  SECTION("random") {
    if (debug) {
      std::cout << "random" << std::endl;
    }
    index.train_ivf(training_data, kmeans_init::random);
  }

  SECTION("kmeans++") {
    if (debug) {
      std::cout << "kmeans++" << std::endl;
    }
    index.train_ivf(training_data, kmeans_init::kmeanspp);
  }
}

TEST_CASE("debug w/ sk", "[ivf_pq_index]") {
  const bool debug = false;

  ColMajorMatrix<float> training_data{
      {{1.0573647, 5.082087},
       {-6.229642, -1.3590931},
       {0.7446737, 6.3828287},
       {-7.698864, -3.0493321},
       {2.1362762, -4.4448104},
       {1.04019, -4.0389647},
       {0.38996044, 5.7235265},
       {1.7470839, -4.717076}}};
  ColMajorMatrix<float> queries{{{-7.3712273, -1.1178735}}};
  ColMajorMatrix<float> sklearn_centroids{
      {{-6.964253, -2.2042127},
       {1.6411834, -4.400284},
       {0.7306664, 5.7294807}}};

  SECTION("one iteration") {
    if (debug) {
      std::cout << "one iteration" << std::endl;
    }
    auto index = ivf_pq_index<float, size_t, size_t>(
        /*sklearn_centroids.num_rows(),*/
        sklearn_centroids.num_cols(),
        2,
        1,
        1e-4);
    index.set_flat_ivf_centroids(sklearn_centroids);
    index.train_ivf(training_data, kmeans_init::none);
    if (debug) {
      debug_flat_ivf_centroids(index);
    }
  }

  SECTION("two iterations") {
    if (debug) {
      std::cout << "two iterations" << std::endl;
    }
    auto index = ivf_pq_index<float, size_t, size_t>(
        /*sklearn_centroids.num_rows(),*/
        sklearn_centroids.num_cols(),
        2,
        2,
        1e-4);
    index.set_flat_ivf_centroids(sklearn_centroids);
    index.train_ivf(training_data, kmeans_init::none);
    if (debug) {
      debug_flat_ivf_centroids(index);
    }
    // debug_centroids(index);
  }

  SECTION("five iterations") {
    if (debug) {
      std::cout << "five iterations" << std::endl;
    }
    auto index = ivf_pq_index<float, size_t, size_t>(
        /* sklearn_centroids.num_rows(), */
        sklearn_centroids.num_cols(),
        2,
        5,
        1e-4);
    index.set_flat_ivf_centroids(sklearn_centroids);
    index.train_ivf(training_data, kmeans_init::none);
    if (debug) {
      debug_flat_ivf_centroids(index);
    }
    // debug_centroids(index);
  }

  SECTION("five iterations, perturbed") {
    if (debug) {
      std::cout << "five iterations, perturbed" << std::endl;
    }
    for (size_t i = 0; i < sklearn_centroids.num_cols(); ++i) {
      for (size_t j = 0; j < sklearn_centroids.num_rows(); ++j) {
        sklearn_centroids(j, i) *= 0.8;
      }
    }

    sklearn_centroids(0, 0) += 0.25;
    auto index = ivf_pq_index<float, size_t, size_t>(
        /* sklearn_centroids.num_rows(), */
        sklearn_centroids.num_cols(),
        2,
        5,
        1e-4);
    index.set_flat_ivf_centroids(sklearn_centroids);
    index.train_ivf(training_data, kmeans_init::none);
    if (debug) {
      debug_flat_ivf_centroids(index);
    }
    // debug_centroids(index);
  }

  SECTION("five iterations") {
    if (debug) {
      std::cout << "five iterations" << std::endl;
    }
    auto index = ivf_pq_index<float, size_t, size_t>(
        /* sklearn_centroids.num_rows(), */
        sklearn_centroids.num_cols(),
        2,
        5,
        1e-4);
    index.train_ivf(training_data, kmeans_init::random);
    if (debug) {
      debug_flat_ivf_centroids(index);
    }
    // debug_centroids(index);
  }
}

TEST_CASE("ivf_index write and read", "[ivf_pq_index]") {
  size_t dimension = 128;
  size_t nlist = 100;
  uint32_t num_subspaces = 16;
  uint32_t max_iterations = 4;
  size_t nprobe = 10;
  size_t k_nn = 10;

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  std::string ivf_index_uri =
      (std::filesystem::temp_directory_path() / "tmp_ivf_index").string();
  if (vfs.is_dir(ivf_index_uri)) {
    vfs.remove_dir(ivf_index_uri);
  }

  // Create and write an index.
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_inputs_uri, 100);
  load(training_set);
  std::vector<siftsmall_ids_type> ids(num_vectors(training_set));
  std::iota(begin(ids), end(ids), 0);
  auto idx = ivf_pq_index<float, uint32_t, uint32_t>(
      nlist, num_subspaces, max_iterations);
  CHECK(idx.num_vectors() == 0);
  idx.train_ivf(training_set, kmeans_init::kmeanspp);
  idx.add(training_set, ids);
  CHECK(idx.num_vectors() == ::num_vectors(training_set));
  idx.write_index(ctx, ivf_index_uri);
  CHECK(idx.num_vectors() == ::num_vectors(training_set));

  // Load it from URI.
  auto idx2 = ivf_pq_index<float, uint32_t, uint32_t>(ctx, ivf_index_uri);
  CHECK(idx2.num_vectors() == ::num_vectors(training_set));

  // Check that the two indexes are the same.
  CHECK(idx.compare_cached_metadata(idx2));
  CHECK(idx.compare_cached_metadata(idx2));
  CHECK(idx.compare_cluster_centroids(idx2));
  CHECK(idx.compare_flat_ivf_centroids(idx2));
  CHECK(idx.compare_pq_ivf_vectors(idx2));
  CHECK(idx.compare_ivf_index(idx2));
  CHECK(idx.compare_ivf_ids(idx2));
  CHECK(idx.compare_pq_ivf_vectors(idx2));
}

TEST_CASE(
    "verify pq_encoding and pq_distances with siftsmall", "[ivf_pq_index]") {
  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<siftsmall_feature_type>(
      ctx, siftsmall_inputs_uri, 2500);
  training_set.load();
  std::vector<siftsmall_ids_type> ids(num_vectors(training_set));
  std::iota(begin(ids), end(ids), 0);

  auto pq_idx = ivf_pq_index<
      siftsmall_feature_type,
      siftsmall_ids_type,
      siftsmall_indices_type>(20, 16, 50);
  pq_idx.train_ivf(training_set);
  pq_idx.add(training_set, ids);
  CHECK(pq_idx.num_vectors() == ::num_vectors(training_set));

  SECTION("pq_encoding") {
    auto avg_error = pq_idx.verify_pq_encoding(training_set);
    CHECK(avg_error < 0.081);
  }
  SECTION("asymmetric_pq_distances") {
    auto [max_error, avg_error] =
        pq_idx.verify_asymmetric_pq_distances(training_set);
    CHECK(avg_error < 0.081);
  }
  SECTION("symmetric_pq_distances") {
    auto avg_error = pq_idx.verify_symmetric_pq_distances(training_set);
    CHECK(avg_error < 0.15);
  }
}

// Current code requires that the number of vectors in the training set be at
// least as large as the number of clusters.
//
#if 0
TEMPLATE_TEST_CASE(
    "query stacked hypercube",
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

  std::vector<uint32_t> ids(num_vectors(hypercube0));
  std::iota(begin(ids), end(ids), 0);

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
  SECTION("nlist = 1") {
    size_t k_nn = 6;
    size_t nlist = 1;

    auto ivf_idx2 = ivf_pq_index<TestType, uint32_t, uint32_t>(
        /*128,*/ nlist, 2, 4, 1.e-4);  // dim nlist maxiter eps nthreads
    ivf_idx2.train_ivf(hypercube2);
    ivf_idx2.add(hypercube2, ids);
    auto ivf_idx4 = ivf_pq_index<TestType, uint32_t, uint32_t>(
        /*128,*/ nlist, 2, 4, 1.e-4);
    ivf_idx4.train_ivf(hypercube4);
    ivf_idx4.add(hypercube4, ids);

    auto top_k_ivf_scores = ColMajorMatrix<float>();
    auto top_k_ivf = ColMajorMatrix<unsigned>();
    auto top_k_scores = ColMajorMatrix<float>();
    auto top_k = ColMajorMatrix<uint64_t>();
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
        ivf_idx2.query_infinite_ram(query2, k_nn, 1);  // k, nprobe
    size_t intersections0 = count_intersections(top_k_ivf, top_k, k_nn);
    double recall0 = intersections0 / ((double)top_k.num_cols() * k_nn);
    CHECK(intersections0 == k_nn * num_vectors(query2));
    CHECK(recall0 == 1.0);

    std::tie(top_k_scores, top_k) = detail::flat::qv_query_heap(
        hypercube4, query4, k_nn, 1, sum_of_squares_distance{});
    std::tie(top_k_ivf_scores, top_k_ivf) =
        ivf_idx4.query_infinite_ram(query4, k_nn, 1);  // k, nprobe

    size_t intersections1 = (long)count_intersections(top_k_ivf, top_k, k_nn);
    double recall1 = intersections1 / ((double)top_k.num_cols() * k_nn);
    CHECK(intersections1 == k_nn * num_vectors(query4));
    CHECK(recall1 == 1.0);
  }
}
#endif

TEST_CASE("build index and infinite query in place", "[ivf_pq_index]") {
  tiledb::Context ctx;
  // size_t nlist = GENERATE(1, 100);
  size_t nlist = 20;
  using s = siftsmall_test_init_defaults;
  using index = ivf_pq_index<s::feature_type, s::id_type, s::px_type>;

  auto init = siftsmall_test_init<index>(ctx, nlist, 16);

  auto&& [nprobe, k_nn, nthreads, max_iterations, convergence_tolerance] =
      std::tie(
          init.nprobe,
          init.k_nn,
          init.nthreads,
          init.max_iterations,
          init.convergence_tolerance);
  auto&& [_, training_set, query_set, groundtruth_set] = std::tie(
      init.idx, init.training_set, init.query_set, init.groundtruth_set);
  auto idx = init.get_write_read_idx();

  auto top_k_ivf_scores = ColMajorMatrix<float>();
  auto top_k_ivf = ColMajorMatrix<siftsmall_ids_type>();

  std::tie(top_k_ivf_scores, top_k_ivf) = idx.query(query_set, k_nn, nprobe);

  // NOTE: Can be used to debug the results:
  // debug_slice(top_k_ivf, "top_k_ivf");
  // debug_slice(top_k_ivf_scores, "top_k_ivf_scores");
  // debug_slice(groundtruth_set, "groundtruth_set");

  init.verify(top_k_ivf);
}

TEST_CASE("ivf_pq_index write and read", "[ivf_pq_index]") {
  tiledb::Context ctx;
  std::string ivf_pq_index_uri =
      (std::filesystem::temp_directory_path() / "tmp_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(ivf_pq_index_uri)) {
    vfs.remove_dir(ivf_pq_index_uri);
  }
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_inputs_uri, 0);
  load(training_set);
  std::vector<siftsmall_ids_type> ids(num_vectors(training_set));
  std::iota(begin(ids), end(ids), 0);

  auto idx = ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(
      10, siftsmall_dimensions / 2);
  idx.train(training_set, ids);
  idx.add(training_set, ids);
  uint64_t write_timestamp = 1000;
  idx.write_index(
      ctx, ivf_pq_index_uri, TemporalPolicy(TimeTravel, write_timestamp));
  CHECK(idx.num_vectors() == ::num_vectors(training_set));

  {
    // Load the index and check metadata.
    auto idx2 = ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(
        ctx, ivf_pq_index_uri);
    CHECK(idx2.num_vectors() == ::num_vectors(training_set));
    CHECK(idx2.group().get_dimensions() == sift_dimensions);
    CHECK(idx2.group().get_temp_size() == 0);

    CHECK(idx2.group().get_all_num_partitions().size() == 1);
    CHECK(idx2.group().get_all_base_sizes().size() == 1);
    CHECK(idx2.group().get_all_ingestion_timestamps().size() == 1);

    CHECK(idx2.group().get_all_num_partitions()[0] > 0);
    CHECK(idx2.group().get_all_base_sizes()[0] == num_sift_vectors);
    CHECK(idx2.group().get_all_ingestion_timestamps()[0] == write_timestamp);

    // Can't compare groups because a write_index does not create a group
    // @todo Should it?
    // CHECK(idx.compare_group(idx2));

    auto idx3 = ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(
        ctx, ivf_pq_index_uri);
    CHECK(idx2 == idx3);
  }

  {
    // Clear history.
    ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>::clear_history(
        ctx, ivf_pq_index_uri, write_timestamp + 10);
    auto idx2 = ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(
        ctx, ivf_pq_index_uri);

    CHECK(idx2.num_vectors() == 0);
    CHECK(idx2.group().get_dimensions() == sift_dimensions);
    CHECK(idx2.group().get_temp_size() == 0);

    CHECK(idx2.group().get_all_num_partitions().size() == 1);
    CHECK(idx2.group().get_all_base_sizes().size() == 1);
    CHECK(idx2.group().get_all_ingestion_timestamps().size() == 1);

    CHECK(idx2.group().get_all_num_partitions()[0] == 0);
    CHECK(idx2.group().get_all_base_sizes()[0] == 0);
    CHECK(idx2.group().get_all_ingestion_timestamps()[0] == 0);
  }
}

TEST_CASE("query empty index", "[ivf_pq_index]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  size_t num_vectors = 0;
  uint64_t dimensions = 10;
  size_t nlist = 1;
  auto index = ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(
      nlist, dimensions / 2);
  auto queries =
      ColMajorMatrix<siftsmall_feature_type>{{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}};

  // We can train and add to an empty index.
  {
    auto data =
        ColMajorMatrixWithIds<siftsmall_feature_type>(dimensions, num_vectors);
    index.train(data, data.raveled_ids());
    index.add(data, data.raveled_ids());
    CHECK(index.num_vectors() == num_vectors);
  }

  // We can query an empty index.
  {
    size_t k_nn = 1;
    auto&& [scores, ids] = index.query(queries, k_nn, nlist);
    CHECK(_cpo::num_vectors(scores) == _cpo::num_vectors(queries));
    CHECK(_cpo::num_vectors(ids) == _cpo::num_vectors(queries));
    CHECK(_cpo::dimensions(scores) == k_nn);
    CHECK(_cpo::dimensions(ids) == k_nn);
    CHECK(scores(0, 0) == std::numeric_limits<float>::max());
    CHECK(ids(0, 0) == std::numeric_limits<uint64_t>::max());
  }

  // We can write an empty index.
  auto ivf_index_uri =
      (std::filesystem::temp_directory_path() / "ivf_index").string();
  {
    if (vfs.is_dir(ivf_index_uri)) {
      vfs.remove_dir(ivf_index_uri);
    }
    index.write_index(ctx, ivf_index_uri);
  }

  // We can load and query an empty index.
  {
    auto index_infinite =
        ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(
            ctx, ivf_index_uri);
    CHECK(index_infinite.num_vectors() == num_vectors);
    size_t k_nn = 1;
    auto&& [scores, ids] = index_infinite.query(queries, k_nn, nlist);
    CHECK(_cpo::num_vectors(scores) == _cpo::num_vectors(queries));
    CHECK(_cpo::num_vectors(ids) == _cpo::num_vectors(queries));
    CHECK(_cpo::dimensions(scores) == k_nn);
    CHECK(_cpo::dimensions(ids) == k_nn);
    CHECK(scores(0, 0) == std::numeric_limits<float>::max());
    CHECK(ids(0, 0) == std::numeric_limits<uint64_t>::max());
  }
  {
    size_t upper_bound = 11;
    auto index_finite =
        ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(
            ctx, ivf_index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
    CHECK(index_finite.num_vectors() == num_vectors);
    size_t k_nn = 1;
    auto&& [scores, ids] = index_finite.query(queries, k_nn, nlist, 9);
    CHECK(_cpo::num_vectors(scores) == _cpo::num_vectors(queries));
    CHECK(_cpo::num_vectors(ids) == _cpo::num_vectors(queries));
    CHECK(_cpo::dimensions(scores) == k_nn);
    CHECK(_cpo::dimensions(ids) == k_nn);
    CHECK(scores(0, 0) == std::numeric_limits<float>::max());
    CHECK(ids(0, 0) == std::numeric_limits<uint64_t>::max());
  }
}

TEST_CASE("query simple", "[ivf_pq_index]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  size_t num_vectors = 4;
  uint64_t dimensions = 4;
  size_t nlist = 1;
  uint32_t num_subspaces = 2;
  uint32_t max_iterations = 1;
  float convergence_tolerance = 0.000025f;
  float reassign_ratio = 0.09f;
  std::optional<TemporalPolicy> temporal_policy = std::nullopt;
  using feature_type = float;
  using id_type = uint32_t;
  auto index = ivf_pq_index<feature_type, id_type>(
      nlist,
      num_subspaces,
      max_iterations,
      convergence_tolerance,
      reassign_ratio,
      temporal_policy,
      DistanceMetric::SUM_OF_SQUARES);
  auto ivf_index_uri =
      (std::filesystem::temp_directory_path() / "ivf_index").string();

  CHECK(index.num_vectors() == 0);
  CHECK(index.nlist() == nlist);

  // We can train, add, query, and then write the index.
  {
    auto training = ColMajorMatrixWithIds<feature_type>{
        {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}},
        {11, 22, 33, 44}};
    index.train(training, training.raveled_ids());
    index.add(training, training.raveled_ids());

    CHECK(index.num_vectors() == ::num_vectors(training));

    size_t k_nn = 1;
    size_t nprobe = nlist;
    for (int i = 1; i <= 4; ++i) {
      auto value = static_cast<feature_type>(i);
      auto queries =
          ColMajorMatrix<feature_type>{{{value, value, value, value}}};
      auto&& [scores, ids] = index.query(queries, k_nn, nprobe);
      CHECK(scores(0, 0) == 0);
      CHECK(ids(0, 0) == i * 11);
    }

    if (vfs.is_dir(ivf_index_uri)) {
      vfs.remove_dir(ivf_index_uri);
    }
    index.write_index(ctx, ivf_index_uri);
  }

  // We can load and query the index.
  {
    std::unique_ptr<ivf_pq_index<feature_type, id_type>> index2;
    SECTION("infinite") {
      index2 = std::make_unique<ivf_pq_index<feature_type, id_type>>(
          ctx, ivf_index_uri);
      CHECK(index2->upper_bound() == 0);
    }
    SECTION("finite") {
      size_t upper_bound = 97;
      index2 = std::make_unique<ivf_pq_index<feature_type, id_type>>(
          ctx, ivf_index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
      CHECK(index2->upper_bound() == upper_bound);
    }
    CHECK(index2->num_vectors() == 4);

    size_t k_nn = 1;
    size_t nprobe = nlist;
    for (int i = 1; i <= 4; ++i) {
      auto value = static_cast<feature_type>(i);
      auto queries =
          ColMajorMatrix<feature_type>{{{value, value, value, value}}};
      auto&& [scores_from_finite, ids_from_finite] =
          index2->query(queries, k_nn, nprobe, 5);
      CHECK(scores_from_finite(0, 0) == 0);
      CHECK(ids_from_finite(0, 0) == i * 11);
      auto&& [scores, ids] = index2->query(queries, k_nn, nprobe);
      CHECK(scores(0, 0) == 0);
      CHECK(ids(0, 0) == i * 11);
    }
  }
}

TEST_CASE("k_factor", "[ivf_pq_index]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  size_t num_vectors = 500;
  uint64_t dimensions = 4;
  size_t nlist = 4;
  uint32_t num_subspaces = 1;
  uint32_t max_iterations = 1;
  float convergence_tolerance = 0.000025f;
  float reassign_ratio = 0.09f;

  size_t nprobe = nlist;
  size_t k_nn = 40;
  float k_factor = 2.f;
  size_t upper_bound = 350;

  std::optional<TemporalPolicy> temporal_policy = std::nullopt;
  using feature_type = float;
  using id_type = uint32_t;
  auto index = ivf_pq_index<feature_type, id_type>(
      nlist,
      num_subspaces,
      max_iterations,
      convergence_tolerance,
      reassign_ratio,
      temporal_policy,
      DistanceMetric::L2);
  auto ivf_index_uri =
      (std::filesystem::temp_directory_path() / "ivf_index").string();
  CHECK(index.num_vectors() == 0);
  CHECK(index.nlist() == nlist);

  // We can train, add, query, and then write the index.
  std::vector<id_type> ids(num_vectors);
  size_t num_equal_no_reranking = 0;
  {
    std::vector<std::vector<feature_type>> vectors;
    for (int i = 1; i <= num_vectors; ++i) {
      std::vector<feature_type> vector(dimensions, i);
      vectors.push_back(vector);
    }
    for (int i = 1; i <= num_vectors; ++i) {
      ids[i - 1] = i;
    }

    auto training = ColMajorMatrixWithIds<feature_type, id_type>{vectors, ids};
    index.train(training, training.raveled_ids());
    index.add(training, training.raveled_ids());

    CHECK(index.num_vectors() == ::num_vectors(training));

    auto queries = ColMajorMatrix<feature_type>{{{1, 1, 1, 1}}};
    {
      auto&& [scores_reranking, ids_reranking] =
          index.query(queries, k_nn, nprobe, k_factor);
      CHECK(
          k_nn == check_single_vector_num_equal<uint32_t>(ids_reranking, ids));
      CHECK(scores_reranking(0, 0) == 0);

      auto&& [scores_no_reranking, ids_no_reranking] =
          index.query(queries, k_nn, nprobe, 1.f);
      num_equal_no_reranking =
          check_single_vector_num_equal(ids_no_reranking, ids);
      CHECK(num_equal_no_reranking != k_nn);
      CHECK(num_equal_no_reranking >= 5);
    }

    CHECK(index.num_vectors() == ::num_vectors(training));

    if (vfs.is_dir(ivf_index_uri)) {
      vfs.remove_dir(ivf_index_uri);
    }
    index.write_index(ctx, ivf_index_uri);
  }

  // We can open the index by URI and query.
  {
    auto queries = ColMajorMatrix<feature_type>{{{1, 1, 1, 1}}};

    // infinite.
    {
      auto index_infinite =
          ivf_pq_index<feature_type, id_type>(ctx, ivf_index_uri);
      CHECK(index_infinite.num_vectors() == num_vectors);
      CHECK(index_infinite.upper_bound() == 0);
      auto&& [scores_reranking, ids_reranking] =
          index_infinite.query(queries, k_nn, nprobe, k_factor);
      CHECK(
          k_nn == check_single_vector_num_equal<uint32_t>(ids_reranking, ids));
      CHECK(scores_reranking(0, 0) == 0);

      auto&& [scores_no_reranking, ids_no_reranking] =
          index_infinite.query(queries, k_nn, nprobe, 1.f);
      CHECK(
          num_equal_no_reranking ==
          check_single_vector_num_equal(ids_no_reranking, ids));
    }

    // finite.
    {
      size_t upper_bound = 300;
      auto index_finite = ivf_pq_index<feature_type, id_type>(
          ctx, ivf_index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
      CHECK(index_finite.num_vectors() == num_vectors);
      CHECK(index_finite.upper_bound() == upper_bound);

      auto&& [scores_reranking, ids_reranking] =
          index_finite.query(queries, k_nn, nprobe, k_factor);
      CHECK(
          k_nn == check_single_vector_num_equal<uint32_t>(ids_reranking, ids));
      CHECK(scores_reranking(0, 0) == 0);

      auto&& [scores_no_reranking, ids_no_reranking] =
          index_finite.query(queries, k_nn, nprobe, 1.f);
      CHECK(
          num_equal_no_reranking ==
          check_single_vector_num_equal(ids_no_reranking, ids));
    }
  }
}

TEST_CASE("ivf_pq_index query index written twice", "[ivf_pq_index]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  std::string index_uri =
      (std::filesystem::temp_directory_path() / "tmp_ivf_pq_index").string();
  if (vfs.is_dir(index_uri)) {
    vfs.remove_dir(index_uri);
  }

  using feature_type_type = uint8_t;
  using id_type_type = uint32_t;
  using partitioning_index_type_type = uint32_t;
  auto feature_type = "uint8";
  auto id_type = "uint32";
  auto partitioning_index_type = "uint32";
  uint64_t dimensions = 3;
  size_t n_list = 1;
  uint32_t num_subspaces = 1;
  uint32_t max_iterations = 3;

  // Write the empty index.
  {
    auto index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(n_list, dimensions / 2);
    auto data =
        ColMajorMatrixWithIds<feature_type_type, id_type_type>(dimensions, 0);
    index.train(data, data.raveled_ids());
    index.add(data, data.raveled_ids());
    index.write_index(ctx, index_uri, TemporalPolicy(TimeTravel, 0));
  }

  // Train the index at timestamp 99.
  {
    auto index = ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>(ctx, index_uri);
    auto data = ColMajorMatrixWithIds<feature_type_type, id_type_type>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}, {1, 2, 3, 4}};
    index.train(data, data.raveled_ids());
    index.add(data, data.raveled_ids());
    index.write_index(ctx, index_uri, TemporalPolicy(TimeTravel, 99));
  }

  // Load the index and query.
  {
    std::unique_ptr<ivf_pq_index<
        feature_type_type,
        id_type_type,
        partitioning_index_type_type>>
        index;
    SECTION("infinite") {
      index = std::make_unique<ivf_pq_index<
          feature_type_type,
          id_type_type,
          partitioning_index_type_type>>(ctx, index_uri);
      CHECK(index->upper_bound() == 0);
    }
    SECTION("finite") {
      size_t upper_bound = 5;
      index = std::make_unique<ivf_pq_index<
          feature_type_type,
          id_type_type,
          partitioning_index_type_type>>(
          ctx, index_uri, IndexLoadStrategy::PQ_OOC, upper_bound);
      CHECK(index->upper_bound() == upper_bound);
    }
    auto queries = ColMajorMatrix<feature_type_type>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}};

    auto&& [scores, ids] = index->query(queries, 1, n_list, 5);
    CHECK(std::equal(
        scores.data(),
        scores.data() + 4,
        std::vector<float>{0, 0, 0, 0}.begin()));
    CHECK(std::equal(
        ids.data(), ids.data() + 4, std::vector<uint32_t>{1, 2, 3, 4}.begin()));
  }
}

TEST_CASE("pq encoding has no error with <= 256 vectors", "[ivf_pq_index]") {
  using feature_type = float;
  using id_type = uint32_t;
  using partitioning_index_type = uint32_t;

  size_t n_list = 16;
  size_t dimensions = 300;
  uint32_t num_subspaces = dimensions;
  for (auto num_vectors : std::vector<size_t>{255, 256, 257}) {
    auto index = ivf_pq_index<feature_type, id_type, partitioning_index_type>(
        n_list, num_subspaces);
    std::vector<std::vector<feature_type>> vectors(num_vectors);
    std::vector<id_type> vector_ids(num_vectors);
    for (int i = 0; i < num_vectors; ++i) {
      for (int j = 0; j < dimensions; ++j) {
        feature_type value = (feature_type)rand() / RAND_MAX;
        vectors[i].push_back(value);
      }
    }
    auto data =
        ColMajorMatrixWithIds<feature_type, id_type>{vectors, vector_ids};
    index.train(data, data.raveled_ids());
    index.add(data, data.raveled_ids());

    // With less than 256 vectors, the PQ encoding should be perfect. This is
    // because for each subspace, we compute 256 centroids, so we have one for
    // each vector. After that, we just make sure the error is low.
    auto error = index.verify_pq_encoding(data);
    if (num_vectors <= 256) {
      CHECK(error == 0.f);
    } else {
      CHECK(error >= 0.0f);
      CHECK(error <= 1e-4f);
    }
  }
}
