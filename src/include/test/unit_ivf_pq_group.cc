/**
 * @file   unit_ivf_pq_group.cc
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

#include "index/ivf_pq_group.h"

TEST_CASE("ivf_pqt_group: test test", "[ivf_pq_group]") {
  REQUIRE(true);
}

TEST_CASE("ivf_pq_group: create tiledb::Group", "[ivf_pq_group]") {
  tiledb::Context ctx;
  tiledb::Config cfg;
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "ivf_pq_group_test_groups")
          .string();

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }
  tiledb::Group::create(ctx, tmp_uri);
  std::unique_ptr<tiledb::Group> write_group_;
  write_group_ =
      std::make_unique<tiledb::Group>(ctx, tmp_uri, TILEDB_WRITE, cfg);
}

struct dummy_pq_index {
  using feature_type = float;
  using flat_vector_feature_type = feature_type;
  using id_type = int;
  using indices_type = int;
  using centroid_feature_type = float;
  using pq_code_type = uint8_t;
  using pq_vector_feature_type = pq_code_type;
  using score_type = float;

  using group_type = ivf_pq_group<dummy_pq_index>;
  using metadata_type = ivf_pq_metadata;
};

TEST_CASE(
    "ivf_pq_group: write constructor - create and open", "[ivf_pq_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_pq_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_pq_group x = ivf_pq_group<dummy_pq_index>(
      ctx, tmp_uri, TILEDB_WRITE, {}, "", 10, 256, 16);
  CHECK(x.get_dimensions() == 10);
  CHECK(x.get_num_clusters() == 256);
}
