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
#include "detail/ivf/dist_qv.h"
#include "detail/ivf/qv.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"
#include "query_common.h"

bool global_verbose = false;
bool global_debug = true;

TEST_CASE("qv: test test", "[qv]") {
  REQUIRE(true);
}

TEST_CASE("ivf qv: infinite all or none", "[ivf qv][ci-skip]") {
  // vq_query_infinite_ram
  // vq_query_infinite_ram_2

  tiledb::Context ctx;

  auto num_queries = GENERATE(101, 0);

  // auto parts = tdbColMajorMatrix<db_type>(ctx, parts_uri);
  // auto ids = read_vector<uint64_t>(ctx, ids_uri);
  // auto index = sizes_to_indices(sizes);

  auto centroids = tdbColMajorMatrix<db_type>(ctx, centroids_uri);
  centroids.load();
  auto query = tdbColMajorMatrix<db_type>(ctx, query_uri, num_queries);
  query.load();
  auto index = read_vector<indices_type>(ctx, index_uri);

  SECTION("all") {
    auto nprobe = GENERATE(1, 5);
    auto k_nn = GENERATE(1, 5);
    auto nthreads = GENERATE(1, 5);
    std::cout << nprobe << " " << k_nn << " " << nthreads << std::endl;

    auto&& [D00, I00] = detail::ivf::query_infinite_ram<db_type, ids_type>(
        ctx,
        parts_uri,
        centroids,
        query,
        index,
        ids_uri,
        nprobe,
        k_nn,
        nthreads);

    auto&& [D01, I01] =
        detail::ivf::qv_query_heap_infinite_ram<db_type, ids_type>(
            ctx,
            parts_uri,
            centroids,
            query,
            index,
            ids_uri,
            nprobe,
            k_nn,
            nthreads);

    auto&& [D02, I02] =
        detail::ivf::nuv_query_heap_infinite_ram<db_type, ids_type>(
            ctx,
            parts_uri,
            centroids,
            query,
            index,
            ids_uri,
            nprobe,
            k_nn,
            nthreads);

    auto&& [D03, I03] =
        detail::ivf::nuv_query_heap_infinite_ram_reg_blocked<db_type, ids_type>(
            ctx,
            parts_uri,
            centroids,
            query,
            index,
            ids_uri,
            nprobe,
            k_nn,
            nthreads);

    CHECK(D00.num_rows() == D01.num_rows());
    CHECK(D00.num_cols() == D01.num_cols());
    CHECK(I00.num_rows() == I01.num_rows());
    CHECK(I00.num_cols() == I01.num_cols());
    CHECK(D00.num_rows() == D02.num_rows());
    CHECK(D00.num_cols() == D02.num_cols());
    CHECK(I00.num_rows() == I02.num_rows());
    CHECK(I00.num_cols() == I02.num_cols());
    CHECK(D00.num_rows() == D03.num_rows());
    CHECK(D00.num_cols() == D03.num_cols());
    CHECK(I00.num_rows() == I03.num_rows());
    CHECK(I00.num_cols() == I03.num_cols());

    for (size_t i = 0; i < I00.num_cols(); ++i) {
      std::sort(begin(I00[i]), end(I00[i]));
      std::sort(begin(I01[i]), end(I01[i]));
      std::sort(begin(I02[i]), end(I02[i]));
      std::sort(begin(I03[i]), end(I03[i]));
    }

    CHECK(!std::equal(
        D00.data(),
        D00.data() + D00.size(),
        std::vector<db_type>(D00.size(), 0.0).data()));
    CHECK(!std::equal(
        I00.data(),
        I00.data() + I00.size(),
        std::vector<indices_type>(I00.size(), 0.0).data()));
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D01.data()));
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D02.data()));
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D03.data()));
  }
}

TEST_CASE("ivf qv: finite all or none", "[ivf qv][ci-skip]") {
  // vq_query_infinite_ram
  // vq_query_infinite_ram_2

  tiledb::Context ctx;

  auto num_queries = GENERATE(101, 0);

  // auto parts = tdbColMajorMatrix<db_type>(ctx, parts_uri);
  // auto ids = read_vector<uint64_t>(ctx, ids_uri);
  // auto index = sizes_to_indices(sizes);

  auto centroids = tdbColMajorMatrix<db_type>(ctx, centroids_uri);
  centroids.load();
  auto query = tdbColMajorMatrix<db_type>(ctx, query_uri, num_queries);
  query.load();
  auto index = read_vector<indices_type>(ctx, index_uri);
  auto groundtruth = tdbColMajorMatrix<groundtruth_type>(ctx, groundtruth_uri);
  groundtruth.load();

  size_t max_part_size = 0;
  for (size_t i = 0; i < size(index) - 1; ++i) {
    max_part_size = std::max<size_t>(max_part_size, index[i + 1] - index[i]);
  }
  // std::cout << max_part_size << std::endl;

  auto upper_bound = GENERATE(1953, 1954, 0);

  SECTION("all") {
    auto nprobe = GENERATE(7 /*, 1*/);
    auto k_nn = GENERATE(9 /*, 1*/);
    auto nthreads = GENERATE(5 /*, 1*/);
    std::cout << upper_bound << " " << nprobe << " " << num_queries << " "
              << k_nn << " " << nthreads << std::endl;

    auto&& [D00, I00] = detail::ivf::query_infinite_ram<db_type, ids_type>(
        ctx,
        parts_uri,
        centroids,
        query,
        index,
        ids_uri,
        nprobe,
        k_nn,
        nthreads);

    auto&& [D01, I01] =
        detail::ivf::qv_query_heap_finite_ram<db_type, ids_type>(
            ctx,
            parts_uri,
            centroids,
            query,
            index,
            ids_uri,
            nprobe,
            k_nn,
            upper_bound,
            nthreads);

    auto&& [D02, I02] =
        detail::ivf::nuv_query_heap_finite_ram<db_type, ids_type>(
            ctx,
            parts_uri,
            centroids,
            query,
            index,
            ids_uri,
            nprobe,
            k_nn,
            upper_bound,
            nthreads);

    auto&& [D03, I03] =
        detail::ivf::nuv_query_heap_finite_ram_reg_blocked<db_type, ids_type>(
            ctx,
            parts_uri,
            centroids,
            query,
            index,
            ids_uri,
            nprobe,
            k_nn,
            upper_bound,
            nthreads);

    auto&& [D04, I04] = detail::ivf::query_finite_ram<db_type, ids_type>(
        ctx,
        parts_uri,
        centroids,
        query,
        index,
        ids_uri,
        nprobe,
        k_nn,
        upper_bound,
        nthreads);

    auto check_size = [&D00 = D00, &I00 = I00](auto& D, auto& I) {
      CHECK(D00.num_rows() == D.num_rows());
      CHECK(D00.num_cols() == D.num_cols());
      CHECK(I00.num_rows() == I.num_rows());
      CHECK(I00.num_cols() == I.num_cols());
    };
    check_size(D00, I00);
    check_size(D01, I01);
    check_size(D02, I02);
    check_size(D03, I03);
    check_size(D04, I04);

    auto intersections00 = count_intersections(I00, groundtruth, k_nn);
    auto intersections01 = count_intersections(I01, groundtruth, k_nn);
    auto intersections02 = count_intersections(I02, groundtruth, k_nn);
    auto intersections03 = count_intersections(I03, groundtruth, k_nn);
    auto intersections04 = count_intersections(I04, groundtruth, k_nn);
    // auto intersectionsGT = count_intersections(groundtruth, groundtruth,
    // k_nn);

    // std::cout << "num intersections " << intersections00 << " / " <<
    // intersectionsGT << std::endl;
    std::cout << "num intersections " << intersections00 << std::endl;
    if (nprobe != 1 && k_nn != 1 && num_queries != 1) {
      CHECK(intersections00 != 0);
    }
    CHECK(intersections00 == intersections01);
    CHECK(intersections00 == intersections02);
    CHECK(intersections00 == intersections03);
    CHECK(intersections00 == intersections04);

    debug_slices_diff(D00, D01, "D00 vs D01");
    debug_slices_diff(D00, D02, "D00 vs D02");
    debug_slices_diff(D00, D03, "D00 vs D03");
    debug_slices_diff(D00, D04, "D00 vs D04");

    CHECK(!std::equal(
        D00.data(),
        D00.data() + D00.size(),
        std::vector<db_type>(D00.size(), 0.0).data()));
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D01.data()));
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D02.data()));
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D03.data()));
    CHECK(std::equal(D00.data(), D00.data() + D00.size(), D04.data()));

#if 1

    SECTION("dist_qv_finite_ram") {
      auto num_nodes = GENERATE(5 /*, 1,*/);
      std::cout << "num nodes " << num_nodes << std::endl;

      auto&& [D05, I05] = detail::ivf::dist_qv_finite_ram<db_type, ids_type>(
          ctx,
          parts_uri,
          centroids,
          query,
          index,
          ids_uri,
          nprobe,
          k_nn,
          upper_bound,
          nthreads,
          num_nodes);

      check_size(D05, I05);
      auto intersections05 = count_intersections(I05, groundtruth, k_nn);
      CHECK(intersections00 == intersections05);
      debug_slices_diff(D00, D05, "D00 vs D05");
      CHECK(std::equal(D00.data(), D00.data() + D00.size(), D05.data()));
    }
#endif
  }
}
