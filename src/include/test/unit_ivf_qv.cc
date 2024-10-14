/**
 * @file   unit_ivf_qv.cc
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
#include "detail/ivf/dist_qv.h"  // dist_qv_finite_ram
#include "detail/ivf/qv.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"
#include "test/utils/array_defs.h"
#include "test/utils/query_common.h"

TEST_CASE("infinite all or none", "[ivf qv]") {
  // vq_query_infinite_ram
  // vq_query_infinite_ram_2

  tiledb::Context ctx;

  auto num_queries = GENERATE(101, 0);

  // auto parts = tdbColMajorMatrix<test_feature_type>(ctx, sift_parts_uri);
  // auto ids = read_vector<uint64_t>(ctx, sift_ids_uri);
  // auto index = sizes_to_indices(sizes);

  auto centroids =
      tdbColMajorMatrix<test_feature_type>(ctx, sift_centroids_uri);
  centroids.load();
  auto query =
      tdbColMajorMatrix<test_feature_type>(ctx, sift_query_uri, num_queries);
  query.load();
  auto index = read_vector<test_indices_type>(ctx, sift_index_uri);
  auto groundtruth =
      tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
  groundtruth.load();

  SECTION("all") {
    auto nprobe = GENERATE(1, 5);
    auto k_nn = GENERATE(1, 5);
    auto nthreads = GENERATE(1, 5);
    // std::cout << nprobe << " " << k_nn << " " << nthreads << std::endl;

    auto top_centroids =
        detail::ivf::ivf_top_centroids(centroids, query, nprobe, nthreads);

    auto&& [active_partitions, active_queries] =
        detail::ivf::partition_ivf_flat_index<test_ids_type>(
            centroids, query, nprobe, nthreads);

    auto infinite_parts =
        std::vector<test_indices_type>(::num_vectors(centroids));
    std::iota(begin(infinite_parts), end(infinite_parts), 0);
    auto inf_mat = tdbColMajorPartitionedMatrix<
        test_feature_type,
        test_ids_type,
        test_indices_type>(
        ctx, sift_parts_uri, sift_index_uri, sift_ids_uri, infinite_parts, 0);
    inf_mat.load();

    auto&& [D00, I00, _] = detail::ivf::query_infinite_ram(
        inf_mat, active_partitions, query, active_queries, k_nn, nthreads);

    auto check_size = [&D00 = D00, &I00 = I00](const auto& D, auto& I) {
      CHECK(D00.num_rows() == D.num_rows());
      CHECK(D00.num_cols() == D.num_cols());
      CHECK(I00.num_rows() == I.num_rows());
      CHECK(I00.num_cols() == I.num_cols());
    };
    auto intersections00 = (long)count_intersections(I00, groundtruth, k_nn);
    if (nprobe != 1 && k_nn != 1 && num_queries != 1) {
      CHECK(intersections00 != 0);
    }

    CHECK(!std::equal(
        D00.data(),
        D00.data() + D00.size(),
        std::vector<test_feature_type>(D00.size(), 0.0).data()));

    SECTION("detail::ivf::qv_query_heap_infinite_ram") {
      auto&& [D01, I01] = detail::ivf::qv_query_heap_infinite_ram(
          top_centroids, inf_mat, query, nprobe, k_nn, nthreads);

      check_size(D01, I01);
      auto intersections01 = (long)count_intersections(I01, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections01) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D01.data()));
    }
    SECTION("detail::ivf::nuv_query_heap_infinite_ram") {
      auto&& [D02, I02] = detail::ivf::nuv_query_heap_infinite_ram(
          inf_mat, active_partitions, query, active_queries, k_nn, nthreads);

      check_size(D02, I02);
      auto intersections02 = (long)count_intersections(I02, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections02) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D02.data()));
    }
    SECTION("detail::ivf::nuv_query_heap_infinite_ram_reg_blocked") {
      auto&& [D03, I03] = detail::ivf::nuv_query_heap_infinite_ram_reg_blocked(
          inf_mat, active_partitions, query, active_queries, k_nn, nthreads);

      check_size(D03, I03);
      auto intersections03 = (long)count_intersections(I03, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections03) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D03.data()));
    }
  }
}

TEST_CASE("finite all or none", "[ivf qv]") {
  // vq_query_infinite_ram
  // vq_query_infinite_ram_2

  tiledb::Context ctx;

  auto num_queries = GENERATE(1, 2253, 0);

  // auto parts = tdbColMajorMatrix<test_feature_type>(ctx, sift_parts_uri);
  // auto ids = read_vector<uint64_t>(ctx, sift_ids_uri);
  // auto index = sizes_to_indices(sizes);

  auto centroids =
      tdbColMajorMatrix<test_feature_type>(ctx, sift_centroids_uri);
  centroids.load();
  auto query =
      tdbColMajorMatrix<test_feature_type>(ctx, sift_query_uri, num_queries);
  query.load();
  auto index = read_vector<test_indices_type>(ctx, sift_index_uri);
  auto groundtruth =
      tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
  groundtruth.load();

  size_t max_part_size = 0;
  for (size_t i = 0; i < size(index) - 1; ++i) {
    max_part_size = std::max<size_t>(max_part_size, index[i + 1] - index[i]);
  }
  // std::cout << max_part_size << std::endl;

  auto upper_bound = GENERATE(/*1953, 1954, */ 10000, 0);

  SECTION("all") {
    auto nprobe = GENERATE(7 /*, 1*/);
    auto k_nn = GENERATE(9 /*, 1*/);
    auto nthreads = GENERATE(5 /*, 1*/);
    // std::cout << upper_bound << " " << nprobe << " " << num_queries << " "
    //           << k_nn << " " << nthreads << std::endl;

    auto&& [active_partitions, active_queries] =
        detail::ivf::partition_ivf_flat_index<test_ids_type>(
            centroids, query, nprobe, nthreads);

    // @todo This is kind of a hack
    auto infinite_parts =
        std::vector<test_indices_type>(::num_vectors(centroids));
    std::iota(begin(infinite_parts), end(infinite_parts), 0);
    auto inf_mat = tdbColMajorPartitionedMatrix<
        test_feature_type,
        test_ids_type,
        test_indices_type>(
        ctx, sift_parts_uri, sift_index_uri, sift_ids_uri, infinite_parts, 0);
    inf_mat.load();

    auto&& [D00, I00, _] = detail::ivf::query_infinite_ram(
        inf_mat, active_partitions, query, active_queries, k_nn, nthreads);

    auto check_size = [&D00 = D00, &I00 = I00](const auto& D, auto& I) {
      CHECK(D00.num_rows() == D.num_rows());
      CHECK(D00.num_cols() == D.num_cols());
      CHECK(I00.num_rows() == I.num_rows());
      CHECK(I00.num_cols() == I.num_cols());
    };
    auto intersections00 = (long)count_intersections(I00, groundtruth, k_nn);
    if (nprobe != 1 && k_nn != 1 && num_queries != 1) {
      CHECK(intersections00 != 0);
    }
    CHECK(!std::equal(
        D00.data(),
        D00.data() + D00.size(),
        std::vector<test_feature_type>(D00.size(), 0.0).data()));

    SECTION("detail::ivf::qv_query_heap_finite_ram") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      auto&& [D01, I01] = detail::ivf::
          qv_query_heap_finite_ram<test_feature_type, test_ids_type>(
              ctx,
              sift_parts_uri,
              centroids,
              query,
              index,
              sift_ids_uri,
              nprobe,
              k_nn,
              upper_bound,
              nthreads);
      check_size(D01, I01);
      auto intersections01 = (long)count_intersections(I01, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections01) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D01.data()));
    }
    SECTION("detail::ivf::nuv_query_heap_finite_ram") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      auto&& [D02, I02] = detail::ivf::nuv_query_heap_finite_ram(
          fin_mat, query, active_queries, k_nn, upper_bound, nthreads);

      check_size(D02, I02);
      auto intersections02 = (long)count_intersections(I02, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections02) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D02.data()));
    }

    SECTION("detail::ivf::nuv_query_heap_finite_ram_reg_blocked") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);
      auto&& [D03, I03] = detail::ivf::nuv_query_heap_finite_ram_reg_blocked(
          fin_mat, query, active_queries, k_nn, upper_bound, nthreads);

      check_size(D03, I03);
      auto intersections03 = (long)count_intersections(I03, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections03) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D03.data()));
    }

    SECTION("detail::ivf::nuv_query_finite_ram") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      auto&& [D04, I04, _] = detail::ivf::query_finite_ram(
          fin_mat, query, active_queries, k_nn, upper_bound, nthreads);

      check_size(D04, I04);
      auto intersections04 = (long)count_intersections(I04, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections04) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D04.data()));
    }

    SECTION(
        "detail::ivf::qv_query_heap_finite_ram, inner_product_distance does "
        "not crash"
        "function") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      auto&& [D01, I01] = detail::ivf::
          qv_query_heap_finite_ram<test_feature_type, test_ids_type>(
              ctx,
              sift_parts_uri,
              centroids,
              query,
              index,
              sift_ids_uri,
              nprobe,
              k_nn,
              upper_bound,
              nthreads,
              0,
              inner_product);

      check_size(D01, I01);
    }

    SECTION(
        "detail::ivf::qv_query_heap_finite_ram, counting_l2_distance "
        "function") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      counting_l2_distance.reset();
      counting_l2_distance.num_comps_ = 99;
      auto&& [D01, I01] = detail::ivf::
          qv_query_heap_finite_ram<test_feature_type, test_ids_type>(
              ctx,
              sift_parts_uri,
              centroids,
              query,
              index,
              sift_ids_uri,
              nprobe,
              k_nn,
              upper_bound,
              nthreads,
              0,
              counting_l2_distance);
      CHECK(counting_l2_distance.num_comps_ != 0);
      CHECK(counting_l2_distance.num_comps_ != 99);
      CHECK(counting_l2_distance.num_comps_ < 100 * 10'000 + 99);

      check_size(D01, I01);
      auto intersections01 = (long)count_intersections(I01, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections01) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D01.data()));
    }

    SECTION("detail::ivf::nuv_query_heap_finite_ram, counting l2 distance") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      counting_l2_distance.reset();
      counting_l2_distance.num_comps_ = 99;
      auto&& [D02, I02] = detail::ivf::nuv_query_heap_finite_ram(
          fin_mat,
          query,
          active_queries,
          k_nn,
          upper_bound,
          nthreads,
          counting_l2_distance);
      CHECK(counting_l2_distance.num_comps_ != 0);
      CHECK(counting_l2_distance.num_comps_ != 99);
      CHECK(counting_l2_distance.num_comps_ < 100 * 10'000 + 99);

      check_size(D02, I02);
      auto intersections02 = (long)count_intersections(I02, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections02) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D02.data()));
    }

    SECTION(
        "detail::ivf::nuv_query_heap_finite_ram_reg_blocked, counting l2 "
        "distance") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      counting_l2_distance.reset();
      counting_l2_distance.num_comps_ = 99;
      auto&& [D03, I03] = detail::ivf::nuv_query_heap_finite_ram_reg_blocked(
          fin_mat,
          query,
          active_queries,
          k_nn,
          upper_bound,
          nthreads,
          counting_l2_distance);
      CHECK(counting_l2_distance.num_comps_ != 0);
      CHECK(counting_l2_distance.num_comps_ != 99);
      CHECK(counting_l2_distance.num_comps_ < 100 * 10'000 + 99);

      check_size(D03, I03);
      auto intersections03 = (long)count_intersections(I03, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections03) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D03.data()));
    }

    SECTION("detail::ivf::nuv_query_finite_ram, counting l2 distance") {
      auto fin_mat = tdbColMajorPartitionedMatrix<
          test_feature_type,
          test_ids_type,
          test_indices_type>(
          ctx,
          sift_parts_uri,
          sift_index_uri,
          sift_ids_uri,
          active_partitions,
          upper_bound);

      counting_l2_distance.reset();
      counting_l2_distance.num_comps_ = 99;
      auto&& [D04, I04, _] = detail::ivf::query_finite_ram(
          fin_mat,
          query,
          active_queries,
          k_nn,
          upper_bound,
          nthreads,
          counting_l2_distance);
      CHECK(counting_l2_distance.num_comps_ != 0);
      CHECK(counting_l2_distance.num_comps_ != 99);
      CHECK(counting_l2_distance.num_comps_ < 100 * 10'000 + 99);

      check_size(D04, I04);
      auto intersections04 = (long)count_intersections(I04, groundtruth, k_nn);
      CHECK(std::labs(intersections00 - intersections04) < 12);
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D04.data()));
    }
  }
}

TEST_CASE("ivf_qv: dist_qv", "[ivf qv]") {
  tiledb::Context ctx;

  auto num_queries = GENERATE(1, 101, 0);
  auto num_nodes = GENERATE(1, 5);
  auto nprobe = GENERATE(1, 5);
  auto k_nn = GENERATE(1, 5);
  auto nthreads = GENERATE(/*1,*/ std::thread::hardware_concurrency());
  auto upper_bound = GENERATE(1953, 1954, 0);

  auto centroids =
      tdbColMajorMatrix<test_feature_type>(ctx, sift_centroids_uri);
  centroids.load();
  auto query =
      tdbColMajorMatrix<test_feature_type>(ctx, sift_query_uri, num_queries);
  query.load();
  auto index = read_vector<test_indices_type>(ctx, sift_index_uri);
  auto groundtruth =
      tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
  groundtruth.load();

  auto&& [active_partitions, active_queries] =
      detail::ivf::partition_ivf_flat_index<test_ids_type>(
          centroids, query, nprobe, nthreads);

  auto infinite_parts =
      std::vector<test_indices_type>(::num_vectors(centroids));
  std::iota(begin(infinite_parts), end(infinite_parts), 0);
  auto inf_mat = tdbColMajorPartitionedMatrix<
      test_feature_type,
      test_ids_type,
      test_indices_type>(
      ctx, sift_parts_uri, sift_index_uri, sift_ids_uri, infinite_parts, 0);
  inf_mat.load();

  auto&& [D00, I00, _] = detail::ivf::query_infinite_ram(
      inf_mat, active_partitions, query, active_queries, k_nn, nthreads);

  [[maybe_unused]] auto check_size = [&D00 = D00, &I00 = I00](
                                         auto& D, auto& I) {
    CHECK(D00.num_rows() == D.num_rows());
    CHECK(D00.num_cols() == D.num_cols());
    CHECK(I00.num_rows() == I.num_rows());
    CHECK(I00.num_cols() == I.num_cols());
  };

  auto intersections00 = (long)count_intersections(I00, groundtruth, k_nn);
  if (nprobe != 1 && k_nn != 1 && num_queries != 1) {
    CHECK(intersections00 != 0);
  }
  CHECK(!std::equal(
      D00.data(),
      D00.data() + D00.size(),
      std::vector<test_feature_type>(D00.size(), 0.0).data()));

  SECTION("dist_qv_finite_ram") {
    auto num_nodes = GENERATE(5 /*, 1 */);
    // std::cout << "num nodes " << num_nodes << std::endl;

    auto&& [D05, I05] = detail::ivf::dist_qv_finite_ram<db_type, ids_type>(
        ctx,
        sift_parts_uri,
        centroids,
        query,
        index,
        sift_ids_uri,
        nprobe,
        k_nn,
        upper_bound,
        nthreads,
        num_nodes);

    check_size(D05, I05);
    auto intersections05 = (long)count_intersections(I05, groundtruth, k_nn);
    CHECK(std::labs(intersections00 - intersections05) < 12);
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D05.data()));
  }

  SECTION("dist_qv_finite_ram couting_l2_distance") {
    auto num_nodes = GENERATE(5 /*, 1 */);

    counting_l2_distance.reset();
    counting_l2_distance.num_comps_ = 99;
    auto&& [D05, I05] = detail::ivf::dist_qv_finite_ram<db_type, ids_type>(
        ctx,
        sift_parts_uri,
        centroids,
        query,
        index,
        sift_ids_uri,
        nprobe,
        k_nn,
        upper_bound,
        nthreads,
        num_nodes,
        0,
        counting_l2_distance);
    CHECK(counting_l2_distance.num_comps_ != 0);
    CHECK(counting_l2_distance.num_comps_ != 99);
    CHECK(counting_l2_distance.num_comps_ < 100 * 10'000 + 99);

    check_size(D05, I05);
    auto intersections05 = (long)count_intersections(I05, groundtruth, k_nn);
    CHECK(std::labs(intersections00 - intersections05) < 12);

    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D05.data()));
  }
}

TEST_CASE("indices", "[ivf qv]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  using feature_type = float;
  using id_type = uint64_t;
  using part_index_type = uint64_t;

  std::string partitioned_vectors_uri =
      (std::filesystem::temp_directory_path() / "partitioned_vectors").string();
  std::string ids_uri =
      (std::filesystem::temp_directory_path() / "ids").string();

  // Create the partitioned_matrix.
  size_t num_vectors = GENERATE(37, 99, 377);
  uint64_t dimensions = 3;
  size_t num_parts = num_vectors;
  size_t num_queries = GENERATE(1, 9, 38);
  std::vector<part_index_type> indices;
  {
    auto training_set = ColMajorMatrix<feature_type>(dimensions, num_vectors);
    for (size_t i = 0; i < dimensions; ++i) {
      for (size_t j = 0; j < num_vectors; ++j) {
        training_set(i, j) = j;
      }
    }

    std::vector<id_type> part_labels(num_vectors, 0);
    for (size_t i = 0; i < num_vectors; ++i) {
      part_labels[i] = i % num_parts;
    }

    auto partitioned_matrix =
        ColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
            training_set, part_labels, num_parts);
    if (vfs.is_dir(partitioned_vectors_uri)) {
      vfs.remove_dir(partitioned_vectors_uri);
    }
    if (vfs.is_dir(ids_uri)) {
      vfs.remove_dir(ids_uri);
    }
    write_matrix(ctx, partitioned_matrix, partitioned_vectors_uri);
    write_vector(ctx, partitioned_matrix.ids(), ids_uri);

    indices = partitioned_matrix.indices();
  }

  // Load the partitioned_matrix but skip some parts.
  size_t part_to_skip = GENERATE(0, 2, 5, 10);
  std::vector<part_index_type> relevant_parts;
  for (size_t i = 0; i < num_parts; ++i) {
    if (i == part_to_skip) {
      continue;
    }
    relevant_parts.push_back(i);
  }
  auto tdb_partitioned_matrix =
      tdbColMajorPartitionedMatrix<feature_type, id_type, part_index_type>(
          ctx, partitioned_vectors_uri, indices, ids_uri, relevant_parts, 0);
  tdb_partitioned_matrix.load();

  std::vector<std::vector<feature_type>> queries_vector;
  for (size_t i = 0; i < num_queries; ++i) {
    std::vector<feature_type> query_vector;
    for (size_t j = 0; j < dimensions; ++j) {
      query_vector.push_back(i);
    }
    queries_vector.push_back(query_vector);
  }
  auto queries = ColMajorMatrix<feature_type>{queries_vector};
  CHECK(::num_vectors(queries) == num_queries);

  std::vector<std::vector<part_index_type>> active_queries;
  for (size_t i = 0; i < relevant_parts.size(); ++i) {
    std::vector<part_index_type> queries_for_part;
    for (size_t j = 0; j < ::num_vectors(queries); ++j) {
      queries_for_part.push_back(j);
    }
    active_queries.push_back(queries_for_part);
  }
  size_t k_nn = 3;
  size_t first_part = 0;
  size_t last_part = num_parts - 1;
  size_t part_offset = 0;
  size_t indices_offset = 0;

  auto min_heap_per_query = detail::ivf::apply_query(
      tdb_partitioned_matrix,
      std::optional<std::vector<int>>{},
      queries,
      active_queries,
      k_nn,
      first_part,
      last_part,
      part_offset,
      indices_offset);

  const auto& ids = tdb_partitioned_matrix.ids();
  std::set<id_type> ids_set;
  for (size_t i = 0; i < ids.size(); ++i) {
    ids_set.insert(ids[i]);
  }

  for (size_t i = 0; i < num_queries; ++i) {
    // Note this sorting doesn't get done on each insertion for performance, but
    // when testing we can explicitly call it.
    min_heap_per_query[i].self_sort();

    CHECK(min_heap_per_query[i].size() == k_nn);

    auto distance = std::get<0>(min_heap_per_query[i][0]);
    auto id = std::get<1>(min_heap_per_query[i][0]);
    auto index = std::get<2>(min_heap_per_query[i][0]);
    if (!ids_set.count(i)) {
      // In this case, we did not load the vector that is exactly the same as
      // our query vector. While we could still check the distance is not crazy,
      // we don't care about case as much. Rather, this point of this test is to
      // verify that we can get back exactly the same vector as a query vector
      // with the correct id and index.
      CHECK(distance < 5.f);
      continue;
    }

    // We are looking at the first element of the min heap, i.e the smallest
    // distance we found from this query vector to the tdb_partitioned_matrix.
    // It should be 0 because it's the same vector against itself.
    CHECK(distance == 0);
    // First let's look at the id, which should be the same as the index of the
    // query vector. This is just from how we set up the test: when we construct
    // queries, we just use the index as the value.
    CHECK(id == i);
    // Now let's look at the index. The big thing to notice here is that unlike
    // you may expect, this index is not relative to the current load. Instead
    // it is global, i.e. index = 1 means that we are pointing at the vector at
    // index 1 in partitioned_vectors_uri.
    CHECK(index == i);
  }
}
