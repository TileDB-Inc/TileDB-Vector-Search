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

#include <tiledb/group_experimental.h>

#include <filesystem>
#include <string>

#include "array_defs.h"
#include "index/ivf_flat_group.h"

TEST_CASE("ivf_flat_group: test test", "[ivf_flat_group]") {
  REQUIRE(true);
}

// This test is for debugging and checks whether a particular group can be
// opened
#if 0
TEST_CASE("ivf_flat_group: read a tiledb::Group", "[ivf_flat_group]") {
  tiledb::Context ctx;
  tiledb::Config cfg;
  std::string tmp_uri = siftsmall_group_uri;

  auto read_group = tiledb::Group(ctx, tmp_uri, TILEDB_READ, cfg);

  for (size_t i = 0; i < read_group.member_count(); ++i) {
    auto member = read_group.member(i);
    auto name = member.name();
    if (!name || empty(*name)) {
      throw std::runtime_error("Name is empty.");
    }
    std::cout << i <<  ": " << *name << " " << member.uri() << std::endl;
  }
}
#endif

TEST_CASE("ivf_flat_group: create tiledb::Group", "[ivf_flat_group]") {
  tiledb::Context ctx;
  tiledb::Config cfg;
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "ivf_flat_group_test_groups")
          .string();

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri))
    vfs.remove_dir(tmp_uri);
  tiledb::Group::create(ctx, tmp_uri);
  std::unique_ptr<tiledb::Group> write_group_;
  write_group_ =
      std::make_unique<tiledb::Group>(ctx, tmp_uri, TILEDB_WRITE, cfg);
}

struct dummy_index {
  using feature_type = float;
  using id_type = int;
  using indices_type = int;
  using centroid_feature_type = float;

  auto dimension() const {
    return 10;
  }
};

// The catch2 check for exception doesn't seem to be working correctly
// @todo Fix this
#if 0
TEST_CASE(
    "ivf_flat_group: read constructor for non-existent group",
    "[ivf_flat_group]") {
  tiledb::Context ctx;

  CHECK_THROWS_WITH(
      ivf_flat_index_group(dummy_index{}, ctx, "I dont exist"),
      "Group uri I dont exist does not exist.");
}
#endif

TEST_CASE("ivf_flat_group: write constructor - create", "[ivf_flat_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_index_group x =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create");
}

TEST_CASE(
    "ivf_flat_group: write constructor - create and open", "[ivf_flat_group]") {
  bool debug = false;
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_index_group x =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create before open");

  ivf_flat_index_group y =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - open");
}

TEST_CASE(
    "ivf_flat_group: write constructor - create and read", "[ivf_flat_group]") {
  bool debug = false;
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_index_group x =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create before open");

  ivf_flat_index_group y =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
  x.dump("Write constructor - open for read");
}

TEST_CASE(
    "ivf_flat_group: write constructor - create, write, and read",
    "[ivf_flat_group]") {
  bool debug = false;
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_index_group x =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - create before open");

  ivf_flat_index_group y =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
  x.dump("Write constructor - open for write");

  ivf_flat_index_group z =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
  x.dump("Write constructor - open for read");
}

TEST_CASE(
    "ivf_flat_group: group metadata - bases, ingestions, partitions",
    "[ivf_flat_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  size_t expected_ingestion = 867;
  size_t expected_base = 5309;  // OMG, copilot filled in 5309 after I typed 867
  size_t expected_partitions = 42;
  size_t expected_temp_size = 314159;
  size_t expected_dimension = 128;

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  size_t offset = 0;

  ivf_flat_index_group x =
      ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);

  SECTION("Just set") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);
  }

  SECTION("Just append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.append_ingestion_timestamp(expected_ingestion);
    x.append_base_size(expected_base);
    x.append_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);
  }

  SECTION("Set then append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);

    offset = 13;

    x.append_ingestion_timestamp(expected_ingestion + offset);
    x.append_base_size(expected_base + offset);
    x.append_num_partitions(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimension(expected_dimension + offset);

    CHECK(
        size(x.get_all_ingestion_timestamps()) ==
        2);  // OMG copilot set this to 2 here, to 1 below
    CHECK(size(x.get_all_base_sizes()) == 2);
    CHECK(size(x.get_all_num_partitions()) == 2);
  }

  SECTION("Set then set") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_index_group(dummy_index{}, ctx, tmp_uri, TILEDB_WRITE);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimension(expected_dimension);

    offset = 13;

    x.set_ingestion_timestamp(expected_ingestion + offset);
    x.set_base_size(expected_base + offset);
    x.set_num_partitions(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimension(expected_dimension + offset);

    CHECK(size(x.get_all_ingestion_timestamps()) == 1);
    CHECK(size(x.get_all_base_sizes()) == 1);
    CHECK(size(x.get_all_num_partitions()) == 1);
  }

  CHECK(x.get_previous_ingestion_timestamp() == expected_ingestion + offset);
  CHECK(x.get_previous_base_size() == expected_base + offset);
  CHECK(x.get_previous_num_partitions() == expected_partitions + offset);
  CHECK(x.get_temp_size() == expected_temp_size + offset);
  CHECK(x.get_dimension() == expected_dimension + offset);
}
