/**
 * @file unit_ivf_flat_metadata.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023 TileDB
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
 */

#include <catch2/catch_all.hpp>
#include <tiledb/tiledb>

#include <filesystem>
#include <string>

#include "array_defs.h"
#include "detail/linalg/tdb_matrix.h"
#include "index/ivf_flat_index.h"
#include "index/ivf_flat_metadata.h"

TEST_CASE("ivf_flat_metadata: test test", "[ivf_flat_metadata]") {
  REQUIRE(true);
}

TEST_CASE("ivf_flat_metadata: default constructor", "[ivf_flat_metadata]") {
  auto x = ivf_flat_index_metadata();
  ivf_flat_index_metadata y;
}

TEST_CASE(
    "ivf_flat_metadata: default constructor dump", "[ivf_flat_metadata]") {
  auto x = ivf_flat_index_metadata();
  x.dump();

  ivf_flat_index_metadata y;
  y.dump();
}

TEST_CASE(
    "ivf_flat_metadata: load metadata from index", "[ivf_flat_metadata]") {
  tiledb::Context ctx;
  tiledb::Config cfg;

  std::string uri =
      (std::filesystem::temp_directory_path() / "tmp_ivf_index").string();
  auto training_vectors =
      tdbColMajorPreLoadMatrix<float>(ctx, siftsmall_inputs_uri, 0);
  auto idx = ivf_flat_index<float, uint32_t, uint32_t>(100, 1);
  idx.train(training_vectors, kmeans_init::kmeanspp);
  idx.add(training_vectors);
  idx.write_index(ctx, uri, true);

  auto read_group = tiledb::Group(ctx, uri, TILEDB_READ, cfg);

  auto x = ivf_flat_index_metadata();
  x.load_metadata(read_group);

  // Compare two constructed objects.
  ivf_flat_index_metadata y;
  y.load_metadata(read_group);
  CHECK(x.compare_metadata(y));
}
