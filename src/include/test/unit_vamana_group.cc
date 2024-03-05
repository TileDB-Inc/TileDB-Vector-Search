/**
 * @file   unit_vamana_group.cc
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
#include "array_defs.h"
#include "index/vamana_group.h"

TEST_CASE("vamana_group: test test", "[vamana_group]") {
  REQUIRE(true);
}

struct dummy_index {
  using feature_type = float;
  using id_type = int;
  using adjacency_row_index_type = int;
  using score_type = float;

  constexpr static tiledb_datatype_t feature_datatype = TILEDB_FLOAT32;
  constexpr static tiledb_datatype_t id_datatype = TILEDB_UINT64;
  constexpr static tiledb_datatype_t adjacency_row_index_datatype =
      TILEDB_UINT64;
  constexpr static tiledb_datatype_t adjacency_scores_datatype = TILEDB_FLOAT32;
  constexpr static tiledb_datatype_t adjacency_ids_datatype = TILEDB_UINT64;

  auto dimension() const {
    return 10;
  }
};

TEST_CASE("vamana_group: verify member types exist", "[vamana_group") {
  tiledb::Context ctx;

  auto x = vamana_index_group(dummy_index{}, ctx, vamana_nano_group_uri);
}

TEST_CASE("vamana_group: constructor", "[vamana_group]") {
  tiledb::Context ctx;

  auto foo = dummy_index{};
  auto n = foo.dimension();
  std::reference_wrapper<const dummy_index> bar = foo;
  auto m = bar.get().dimension();

  auto x = vamana_index_group(dummy_index{}, ctx, vamana_nano_group_uri);
  auto y = vamana_index_group(foo, ctx, vamana_nano_group_uri);
}

TEST_CASE("vamana_group: default constructor", "[vamana_group]") {
  tiledb::Context ctx;
  auto x = vamana_index_group(dummy_index{}, ctx, vamana_nano_group_uri);
  x.dump("Default constructor");
}

TEST_CASE("vamana_group: read constructor", "[vamana_group]") {
  tiledb::Context ctx;
  auto x = vamana_index_group(
      dummy_index{}, ctx, vamana_nano_group_uri, TILEDB_READ);
  x.dump("Read constructor");
}

TEST_CASE("vamana_group: read constructor with version", "[vamana_group]") {
  tiledb::Context ctx;
  auto x = vamana_index_group(
      dummy_index{}, ctx, vamana_nano_group_uri, TILEDB_READ, 0, "0.3");
  x.dump("Read constructor with version");
}

// The catch2 check for exception doesn't seem to be working correctly
// @todo Fix this
#if 0
TEST_CASE(
    "vamana_group: read constructor for non-existent group",
    "[vamana_group]") {
  tiledb::Context ctx;

  CHECK_THROWS_WITH(
      vamana_index_group(dummy_index{}, ctx, "I dont exist"),
      "Group uri I dont exist does not exist.");
}
#endif

TEST_CASE("vamana_group: write constructor - create", "[vamana_group]") {
  std::string tmp_uri = "/tmp/vamana_group_test_write_constructor";

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create");
}

TEST_CASE(
    "vamana_group: write constructor - create and open", "[vamana_group]") {
  std::string tmp_uri = "/tmp/vamana_group_test_write_constructor";

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create before open");

  vamana_index_group y =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - open");
}

TEST_CASE(
    "vamana_group: write constructor - create and read", "[vamana_group]") {
  std::string tmp_uri = "/tmp/vamana_group_test_write_constructor";

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create before open");

  vamana_index_group y =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
  x.dump("Write constructor - open for read");
}

TEST_CASE(
    "vamana_group: write constructor - create, write, and read",
    "[vamana_group]") {
  std::string tmp_uri = "/tmp/vamana_group_test_write_constructor";

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create before open");

  vamana_index_group y =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - open for write");

  vamana_index_group z =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
  x.dump("Write constructor - open for read");
}

TEST_CASE(
    "vamana_group: group metadata - bases, ingestions, partitions",
    "[vamana_group]") {
  std::string tmp_uri = "/tmp/vamana_group_test_write_constructor";

  size_t expected_ingestion = 867;
  size_t expected_base = 5309;
  size_t expected_partitions = 42;
  size_t expected_temp_size = 314159;
  size_t expected_dimension = 128;

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  size_t offset = 0;

  vamana_index_group x =
      vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);

  SECTION("Just set") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);
  }

  SECTION("Just append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.append_ingestion_timestamp(expected_ingestion);
    x.append_base_size(expected_base);
    x.append_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);
  }

  SECTION("Set then append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);

    offset = 13;

    x.append_ingestion_timestamp(expected_ingestion + offset);
    x.append_base_size(expected_base + offset);
    x.append_num_edges(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimension(expected_dimension + offset);

    CHECK(size(x.get_all_ingestion_timestamps()) == 2);
    CHECK(size(x.get_all_base_sizes()) == 2);
    CHECK(size(x.get_all_num_edges()) == 2);
  }

  SECTION("Set then set") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);

    offset = 13;

    x.set_ingestion_timestamp(expected_ingestion + offset);
    x.set_base_size(expected_base + offset);
    x.set_num_edges(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimension(expected_dimension + offset);

    CHECK(size(x.get_all_ingestion_timestamps()) == 1);
    CHECK(size(x.get_all_base_sizes()) == 1);
    CHECK(size(x.get_all_num_edges()) == 1);
  }

  CHECK(x.get_previous_ingestion_timestamp() == expected_ingestion + offset);
  CHECK(x.get_previous_base_size() == expected_base + offset);
  CHECK(x.get_previous_num_edges() == expected_partitions + offset);
  CHECK(x.get_temp_size() == expected_temp_size + offset);
  CHECK(x.get_dimension() == expected_dimension + offset);
}
