/**
 * @file   unit_timetravel.cc
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
 * Test time traveling support in TileDB-Vector-Search
 *
 */

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <limits>
#include <vector>

#include "detail/linalg/tdb_matrix.h"

#include "index/index_group.h"
#include "index/index_metadata.h"
#include "index/ivf_flat_index.h"

#include "query_common.h"

#include "utils/print_types.h"

TEST_CASE("timetravel: test test", "[timetravel]") {
  REQUIRE(true);
}

TEST_CASE("timetravel: open", "[timetravel]") {
  tiledb::Context ctx;

  auto index = ivf_flat_index<float, uint32_t, uint32_t>(100);
  auto training_set =
      tdbColMajorPreLoadMatrix<float>(ctx, siftsmall_inputs_uri);
  index.train(training_set);
  index.add(training_set);

  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();

  index.write_index(ctx, "/tmp/timetravel_test_open", true);

  // @todo: index is fake -- just used to set types of group and metadata
  auto grp = ivf_flat_index_group(ctx, "/tmp/timetravel_test_open", index);
  // grp.dump("unit_timetravel test open");

  auto x = grp.get_all_base_sizes();
  auto y = grp.get_all_num_partitions();
  auto z = grp.get_all_ingestion_timestamps();

  SECTION("check metadata") {
    CHECK(std::equal(
        x.begin(), x.end(), begin(std::vector<uint32_t>{0, 10'000})));
    CHECK(std::equal(y.begin(), y.end(), begin(std::vector<uint32_t>{0, 100})));

    CHECK(size(z) == 2);
    CHECK(z[0] == 0);
    CHECK(z[1] != 0);
    CHECK(z[1] <= now);
    CHECK(std::abs((long)z[1] - (long)now) < 2'000);
  }

  SECTION("check array timestamps") {
    auto c = grp.centroids_uri();
    auto p = grp.parts_uri();
    auto i = grp.indices_uri();
    auto j = grp.ids_uri();

    auto ac = tiledb::Array(ctx, c, TILEDB_READ);
    auto fc = tiledb::FragmentInfo(ctx, c);
    fc.load();
    auto nc = fc.fragment_num();
    auto tc = fc.timestamp_range(0);
    // std::cout << "centroid tc: " << tc.first << ", " << tc.second <<
    // std::endl;
    CHECK(tc.first == tc.second);
    CHECK((long)tc.first == z[1]);

    auto ap = tiledb::Array(ctx, p, TILEDB_READ);
    auto fp = tiledb::FragmentInfo(ctx, p);
    fp.load();
    auto np = fp.fragment_num();
    auto tp = fp.timestamp_range(0);
    // std::cout << "part tp: " << tp.first << ", " << tp.second << std::endl;
    CHECK(tp.first == tp.second);
    CHECK((long)tp.first == z[1]);

    auto ai = tiledb::Array(ctx, i, TILEDB_READ);
    auto fi = tiledb::FragmentInfo(ctx, i);
    fi.load();
    auto ni = fi.fragment_num();
    auto ti = fi.timestamp_range(0);
    // std::cout << "index ti: " << ti.first << ", " << ti.second << std::endl;
    CHECK(ti.first == ti.second);
    CHECK((long)ti.first == z[1]);

    auto aj = tiledb::Array(ctx, j, TILEDB_READ);
    auto fj = tiledb::FragmentInfo(ctx, j);
    fj.load();
    auto nj = fj.fragment_num();
    auto tj = fj.timestamp_range(0);
    // std::cout << "ids tj: " << tj.first << ", " << tj.second << std::endl;
    CHECK(tj.first == tj.second);
    CHECK((long)tj.first == z[1]);
  }
}
