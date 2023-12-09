/**
* @file   unit_ivf_flat_group.cc
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
*/

#include <catch2/catch_all.hpp>
#include <tiledb/tiledb>

#include <filesystem>
#include <string>

#include "index/ivf_flat_group.h"
#include "query_common.h"

#include "utils/print_types.h"


TEST_CASE("ivf_flat_group: test test", "[ivf_flat_group]") {
  REQUIRE(true);
}

TEST_CASE("ivf_flat_group: create tiledb::Group", "[ivf_flat_group]") {
  tiledb::Context ctx;
  tiledb::Config cfg;
  std::string tmp_uri = "/tmp/ivf_flat_group_test_groups";

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri))
    vfs.remove_dir(tmp_uri);
  tiledb::Group::create(ctx, tmp_uri);
  std::unique_ptr<tiledb::Group> write_group_;
  write_group_ = std::make_unique<tiledb::Group>(
      ctx, tmp_uri, TILEDB_WRITE, cfg);
}

struct dummy_index {
  using feature_type = float;
  using id_type = int;
  using indices_type = int;
  using centroids_type = float;

  auto dimension() const { return 10; }
};

TEST_CASE("ivf_flat_group: constructor", "[ivf_flat_group]") {
  tiledb::Context ctx;

  auto foo = dummy_index{};
  auto n = foo.dimension();
  std::reference_wrapper<const dummy_index> bar = foo;
  auto m = bar.get().dimension();

  auto x = ivf_flat_index_group(ctx, group_uri, foo);
  auto y = ivf_flat_index_group(ctx, group_uri, dummy_index{});
}

TEST_CASE("ivf_flat_group: default constructor", "[ivf_flat_group]") {
  tiledb::Context ctx;
  auto x = ivf_flat_index_group(ctx, group_uri, dummy_index{});
  x.dump("Default constructor");
}

TEST_CASE("ivf_flat_group: read constructor", "[ivf_flat_group]") {
  tiledb::Context ctx;
  auto x = ivf_flat_index_group(ctx, group_uri, dummy_index{}, TILEDB_READ);
  x.dump("Read constructor");
}

TEST_CASE("ivf_flat_group: read constructor with version", "[ivf_flat_group]") {
  tiledb::Context ctx;
  auto x = ivf_flat_index_group(ctx, group_uri, dummy_index{}, TILEDB_READ, "0.3");
  x.dump("Read constructor with version");
}

TEST_CASE("ivf_flat_group: read constructor for non-existent group", "[ivf_flat_group]") {
  tiledb::Context ctx;

  CHECK_THROWS_WITH(ivf_flat_index_group(ctx, "I dont exist", dummy_index{}), "Group uri I dont exist does not exist.");
}

TEST_CASE("ivf_flat_group: write constructor", "[ivf_flat_group]") {
  std::string tmp_uri = "/tmp/ivf_flat_group_test_write_constructor";

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_index_group x = ivf_flat_index_group(ctx, tmp_uri, dummy_index{}, TILEDB_WRITE);
  x.dump("Write constructor");
}
