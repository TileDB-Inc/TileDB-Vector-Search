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
#include "test/utils/array_defs.h"

// This test is for debugging and checks whether a particular group can be
// opened
#if 0
TEST_CASE("read a tiledb::Group", "[ivf_flat_group]") {
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

TEST_CASE("create tiledb::Group", "[ivf_flat_group]") {
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

  using group_type = ivf_flat_group<dummy_index>;
  using metadata_type = ivf_flat_index_metadata;
};

TEST_CASE("read constructor for non-existent group", "[ivf_flat_group]") {
  tiledb::Context ctx;

  CHECK_THROWS_WITH(
      ivf_flat_group<dummy_index>(ctx, "I dont exist"),
      "Group uri I dont exist does not exist.");
}

TEST_CASE("write constructor - create", "[ivf_flat_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_group x =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);
}

TEST_CASE("write constructor - create and open", "[ivf_flat_group]") {
  bool debug = false;
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_group x =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  ivf_flat_group y =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);
}

TEST_CASE("write constructor - create and read", "[ivf_flat_group]") {
  bool debug = false;
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_group x =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  ivf_flat_group y = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
}

TEST_CASE("write constructor - create, write, and read", "[ivf_flat_group]") {
  bool debug = false;
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_group x =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  ivf_flat_group y =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  ivf_flat_group z = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
}

TEST_CASE(
    "group metadata - bases, ingestions, partitions", "[ivf_flat_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "ivf_flat_group_test_write_constructor")
                            .string();

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

  ivf_flat_group x =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  SECTION("Just set") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);
  }

  SECTION("Just append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.append_ingestion_timestamp(expected_ingestion);
    x.append_base_size(expected_base);
    x.append_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);
  }

  SECTION("Set then append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);

    offset = 13;

    x.append_ingestion_timestamp(expected_ingestion + offset);
    x.append_base_size(expected_base + offset);
    x.append_num_partitions(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimensions(expected_dimension + offset);

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
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_partitions(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);

    offset = 13;

    x.set_ingestion_timestamp(expected_ingestion + offset);
    x.set_base_size(expected_base + offset);
    x.set_num_partitions(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimensions(expected_dimension + offset);

    CHECK(size(x.get_all_ingestion_timestamps()) == 1);
    CHECK(size(x.get_all_base_sizes()) == 1);
    CHECK(size(x.get_all_num_partitions()) == 1);
  }

  CHECK(x.get_previous_ingestion_timestamp() == expected_ingestion + offset);
  CHECK(x.get_previous_base_size() == expected_base + offset);
  CHECK(x.get_previous_num_partitions() == expected_partitions + offset);
  CHECK(x.get_temp_size() == expected_temp_size + offset);
  CHECK(x.get_dimensions() == expected_dimension + offset);
}

TEST_CASE("storage version", "[ivf_flat_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "ivf_flat_group").string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  size_t expected_ingestion = 23094;
  size_t expected_base = 9234;
  size_t expected_partitions = 200;
  size_t expected_temp_size = 11;
  size_t expected_dimension = 19238;
  auto offset = 2345;

  ivf_flat_group x =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  SECTION("0.3") {
    x = ivf_flat_group<dummy_index>(
        ctx, tmp_uri, TILEDB_WRITE, TemporalPolicy{TimeTravel, 0}, "0.3", 10);
  }

  SECTION("current_storage_version") {
    x = ivf_flat_group<dummy_index>(
        ctx,
        tmp_uri,
        TILEDB_WRITE,
        TemporalPolicy{TimeTravel, 0},
        current_storage_version,
        10);
  }

  x.set_ingestion_timestamp(expected_ingestion + offset);
  x.set_base_size(expected_base + offset);
  x.set_num_partitions(expected_partitions + offset);
  x.set_temp_size(expected_temp_size + offset);
  x.set_dimensions(expected_dimension + offset);

  CHECK(size(x.get_all_ingestion_timestamps()) == 1);
  CHECK(size(x.get_all_base_sizes()) == 1);
  CHECK(size(x.get_all_num_partitions()) == 1);
  CHECK(x.get_previous_ingestion_timestamp() == expected_ingestion + offset);
  CHECK(x.get_previous_base_size() == expected_base + offset);
  CHECK(x.get_previous_num_partitions() == expected_partitions + offset);
  CHECK(x.get_temp_size() == expected_temp_size + offset);
  CHECK(x.get_dimensions() == expected_dimension + offset);
}

TEST_CASE("invalid storage version", "[ivf_flat_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "ivf_flat_group").string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }
  CHECK_THROWS(
      ivf_flat_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, TemporalPolicy{TimeTravel, 0}, "invalid"),
      10);
}

TEST_CASE("mismatched storage version", "[ivf_flat_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "ivf_flat_group").string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_group x = ivf_flat_group<dummy_index>(
      ctx, tmp_uri, TILEDB_WRITE, TemporalPolicy{TimeTravel, 0}, "0.3", 10);

  CHECK_THROWS_WITH(
      ivf_flat_group<dummy_index>(
          ctx,
          tmp_uri,
          TILEDB_WRITE,
          TemporalPolicy{TimeTravel, 0},
          "different_version",
          10),
      "Version mismatch. Requested different_version but found 0.3");
}

TEST_CASE("clear history", "[ivf_flat_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "ivf_flat_group").string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  ivf_flat_group x =
      ivf_flat_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);

  x.append_ingestion_timestamp(1);
  x.append_base_size(2);
  x.append_num_partitions(3);

  x.append_ingestion_timestamp(11);
  x.append_base_size(22);
  x.append_num_partitions(33);

  x.append_ingestion_timestamp(111);
  x.append_base_size(222);
  x.append_num_partitions(333);

  CHECK(x.get_all_ingestion_timestamps().size() == 4);
  CHECK(x.get_all_base_sizes().size() == 4);
  CHECK(x.get_all_num_partitions().size() == 4);
  CHECK(x.get_all_ingestion_timestamps()[0] == 0);
  CHECK(x.get_all_ingestion_timestamps()[1] == 1);
  CHECK(x.get_all_ingestion_timestamps()[2] == 11);
  CHECK(x.get_all_ingestion_timestamps()[3] == 111);
  CHECK(x.get_all_base_sizes()[0] == 0);
  CHECK(x.get_all_base_sizes()[1] == 2);
  CHECK(x.get_all_base_sizes()[2] == 22);
  CHECK(x.get_all_base_sizes()[3] == 222);
  CHECK(x.get_all_num_partitions()[0] == 0);
  CHECK(x.get_all_num_partitions()[1] == 3);
  CHECK(x.get_all_num_partitions()[2] == 33);
  CHECK(x.get_all_num_partitions()[3] == 333);

  // Can clear the first three timestamps correctly.
  x.clear_history(100);
  CHECK(x.get_all_ingestion_timestamps().size() == 1);
  CHECK(x.get_all_base_sizes().size() == 1);
  CHECK(x.get_all_num_partitions().size() == 1);
  CHECK(x.get_all_ingestion_timestamps()[0] == 111);
  CHECK(x.get_all_base_sizes()[0] == 222);
  CHECK(x.get_all_num_partitions()[0] == 333);

  // If we clear after the last timestamp, we end up with zeroes.
  x.clear_history(112);
  CHECK(x.get_all_ingestion_timestamps().size() == 1);
  CHECK(x.get_all_base_sizes().size() == 1);
  CHECK(x.get_all_num_partitions().size() == 1);
  CHECK(x.get_all_ingestion_timestamps()[0] == 0);
  CHECK(x.get_all_ingestion_timestamps()[0] == 0);
  CHECK(x.get_all_base_sizes()[0] == 0);
  CHECK(x.get_all_num_partitions()[0] == 0);
}
