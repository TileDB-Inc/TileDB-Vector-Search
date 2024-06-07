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
#include "index/vamana_group.h"
#include "test/utils/array_defs.h"

struct dummy_index {
  using feature_type = float;
  using id_type = int;
  using adjacency_row_index_type = int;
  using adjacency_scores_type = float;

  using group_type = vamana_index_group<dummy_index>;
  using metadata_type = vamana_index_metadata;

  constexpr static tiledb_datatype_t feature_datatype = TILEDB_FLOAT32;
  constexpr static tiledb_datatype_t id_datatype = TILEDB_UINT64;
  constexpr static tiledb_datatype_t adjacency_row_index_datatype =
      TILEDB_UINT64;
  constexpr static tiledb_datatype_t adjacency_scores_datatype = TILEDB_FLOAT32;
  constexpr static tiledb_datatype_t adjacency_ids_datatype = TILEDB_UINT64;
};

TEST_CASE("read constructor for non-existent group", "[vamana_group]") {
  tiledb::Context ctx;

  CHECK_THROWS_WITH(
      vamana_index_group<dummy_index>(ctx, "I dont exist"),
      "Group uri I dont exist does not exist.");
}

TEST_CASE("write constructor - create", "[vamana_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "vamana_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x =
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);
}

TEST_CASE("write constructor - create and open", "[vamana_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "vamana_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x =
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  vamana_index_group y =
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);
}

TEST_CASE("write constructor - create and read", "[vamana_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "vamana_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  {
    vamana_index_group x =
        vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
    CHECK(x.get_dimensions() == 10);
    x.append_num_edges(0);
    x.append_base_size(0);
    x.append_ingestion_timestamp(0);

    // We throw b/c the vamana_index_group hasn't actually been written (the
    // write happens in the destructor).
    CHECK_THROWS_WITH(
        vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ),
        "No ingestion timestamps found.");
  }

  vamana_index_group y =
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
}

TEST_CASE("write constructor - invalid create and read", "[vamana_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "vamana_group_test_write_constructor")
                            .string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  {
    vamana_index_group x =
        vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
    CHECK(x.get_dimensions() == 10);
    // We throw b/c the vamana_index_group hasn't actually been written (the
    // write happens in the destructor).
    CHECK_THROWS_WITH(
        vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ),
        "No ingestion timestamps found.");
  }

  // If we don't append to the group, even if it's been written there will be no
  // timestamps.
  CHECK_THROWS_WITH(
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ),
      "No ingestion timestamps found.");
}

TEST_CASE("group metadata - bases, ingestions, partitions", "[vamana_group]") {
  std::string tmp_uri = (std::filesystem::temp_directory_path() /
                         "vamana_group_test_write_constructor")
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

  {
    vamana_index_group x =
        vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
    CHECK(x.get_dimensions() == 10);
    x.append_num_edges(0);
    x.append_base_size(0);
    x.append_ingestion_timestamp(0);
  }

  vamana_index_group x =
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  SECTION("Just set") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);
  }

  SECTION("Just append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.append_ingestion_timestamp(expected_ingestion);
    x.append_base_size(expected_base);
    x.append_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);
  }

  SECTION("Set then append") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);

    offset = 13;

    x.append_ingestion_timestamp(expected_ingestion + offset);
    x.append_base_size(expected_base + offset);
    x.append_num_edges(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimensions(expected_dimension + offset);

    CHECK(size(x.get_all_ingestion_timestamps()) == 2);
    CHECK(size(x.get_all_base_sizes()) == 2);
    CHECK(size(x.get_all_num_edges()) == 2);
  }

  SECTION("Set then set") {
    SECTION("After create") {
    }

    SECTION("After create and read") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and write") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    SECTION("After create and write and read") {
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
    }

    SECTION("After create and read and write") {
      x = vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_READ);
      x = vamana_index_group<dummy_index>(
          ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
      CHECK(x.get_dimensions() == 10);
    }

    x.set_ingestion_timestamp(expected_ingestion);
    x.set_base_size(expected_base);
    x.set_num_edges(expected_partitions);
    x.set_temp_size(expected_temp_size);
    x.set_dimensions(expected_dimension);

    offset = 13;

    x.set_ingestion_timestamp(expected_ingestion + offset);
    x.set_base_size(expected_base + offset);
    x.set_num_edges(expected_partitions + offset);
    x.set_temp_size(expected_temp_size + offset);
    x.set_dimensions(expected_dimension + offset);

    CHECK(size(x.get_all_ingestion_timestamps()) == 1);
    CHECK(size(x.get_all_base_sizes()) == 1);
    CHECK(size(x.get_all_num_edges()) == 1);
  }

  CHECK(x.get_previous_ingestion_timestamp() == expected_ingestion + offset);
  CHECK(x.get_previous_base_size() == expected_base + offset);
  CHECK(x.get_previous_num_edges() == expected_partitions + offset);
  CHECK(x.get_temp_size() == expected_temp_size + offset);
  CHECK(x.get_dimensions() == expected_dimension + offset);
}

TEST_CASE("storage version", "[vamana_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "vamana_group").string();

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

  vamana_index_group x =
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);
  CHECK(x.get_dimensions() == 10);

  SECTION("0.3") {
    x = vamana_index_group<dummy_index>(
        ctx, tmp_uri, TILEDB_WRITE, TemporalPolicy{TimeTravel, 0}, "0.3", 10);
    x.append_num_edges(0);
    x.append_base_size(0);
    x.append_ingestion_timestamp(0);
  }

  SECTION("current_storage_version") {
    x = vamana_index_group<dummy_index>(
        ctx,
        tmp_uri,
        TILEDB_WRITE,
        TemporalPolicy{TimeTravel, 0},
        current_storage_version);
    x.append_num_edges(0);
    x.append_base_size(0);
    x.append_ingestion_timestamp(0);
  }

  x.set_ingestion_timestamp(expected_ingestion + offset);
  x.set_base_size(expected_base + offset);
  x.set_num_edges(expected_partitions + offset);
  x.set_temp_size(expected_temp_size + offset);
  x.set_dimensions(expected_dimension + offset);

  CHECK(size(x.get_all_ingestion_timestamps()) == 1);
  CHECK(size(x.get_all_base_sizes()) == 1);
  CHECK(size(x.get_all_num_edges()) == 1);
  CHECK(x.get_previous_ingestion_timestamp() == expected_ingestion + offset);
  CHECK(x.get_previous_base_size() == expected_base + offset);
  CHECK(x.get_previous_num_edges() == expected_partitions + offset);
  CHECK(x.get_temp_size() == expected_temp_size + offset);
  CHECK(x.get_dimensions() == expected_dimension + offset);
}

TEST_CASE("invalid storage version", "[vamana_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "vamana_group").string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }
  CHECK_THROWS(vamana_index_group<dummy_index>(
      ctx,
      tmp_uri,
      TILEDB_WRITE,
      TemporalPolicy{TimeTravel, 0},
      "invalid",
      10));
}

TEST_CASE("mismatched storage version", "[vamana_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "vamana_group").string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x = vamana_index_group<dummy_index>(
      ctx, tmp_uri, TILEDB_WRITE, TemporalPolicy{TimeTravel, 0}, "0.3", 10);

  CHECK_THROWS_WITH(
      vamana_index_group<dummy_index>(
          ctx,
          tmp_uri,
          TILEDB_WRITE,
          TemporalPolicy{TimeTravel, 0},
          "different_version",
          10),
      "Version mismatch. Requested different_version but found 0.3");
}

TEST_CASE("clear history", "[vamana_group]") {
  std::string tmp_uri =
      (std::filesystem::temp_directory_path() / "vamana_group").string();

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_uri)) {
    vfs.remove_dir(tmp_uri);
  }

  vamana_index_group x =
      vamana_index_group<dummy_index>(ctx, tmp_uri, TILEDB_WRITE, {}, "", 10);

  x.append_ingestion_timestamp(1);
  x.append_base_size(2);
  x.append_num_edges(3);

  x.append_ingestion_timestamp(11);
  x.append_base_size(22);
  x.append_num_edges(33);

  x.append_ingestion_timestamp(111);
  x.append_base_size(222);
  x.append_num_edges(333);

  CHECK(x.get_all_ingestion_timestamps().size() == 3);
  CHECK(x.get_all_base_sizes().size() == 3);
  CHECK(x.get_all_num_edges().size() == 3);
  CHECK(x.get_all_ingestion_timestamps()[0] == 1);
  CHECK(x.get_all_ingestion_timestamps()[1] == 11);
  CHECK(x.get_all_ingestion_timestamps()[2] == 111);
  CHECK(x.get_all_base_sizes()[0] == 2);
  CHECK(x.get_all_base_sizes()[1] == 22);
  CHECK(x.get_all_base_sizes()[2] == 222);
  CHECK(x.get_all_num_edges()[0] == 3);
  CHECK(x.get_all_num_edges()[1] == 33);
  CHECK(x.get_all_num_edges()[2] == 333);

  // No-op if we clear before the first ingestion timestamp.
  x.clear_history(0);
  CHECK(x.get_all_ingestion_timestamps().size() == 3);
  CHECK(x.get_all_base_sizes().size() == 3);
  CHECK(x.get_all_num_edges().size() == 3);
  CHECK(x.get_all_ingestion_timestamps()[0] == 1);
  CHECK(x.get_all_ingestion_timestamps()[1] == 11);
  CHECK(x.get_all_ingestion_timestamps()[2] == 111);
  CHECK(x.get_all_base_sizes()[0] == 2);
  CHECK(x.get_all_base_sizes()[1] == 22);
  CHECK(x.get_all_base_sizes()[2] == 222);
  CHECK(x.get_all_num_edges()[0] == 3);
  CHECK(x.get_all_num_edges()[1] == 33);
  CHECK(x.get_all_num_edges()[2] == 333);

  // Can clear the first two timestamps correctly.
  x.clear_history(100);
  CHECK(x.get_all_ingestion_timestamps().size() == 1);
  CHECK(x.get_all_base_sizes().size() == 1);
  CHECK(x.get_all_num_edges().size() == 1);
  CHECK(x.get_all_ingestion_timestamps()[0] == 111);
  CHECK(x.get_all_base_sizes()[0] == 222);
  CHECK(x.get_all_num_edges()[0] == 333);

  // If we clear after the last timestamp, we end up with zeroes.
  x.clear_history(112);
  CHECK(x.get_all_ingestion_timestamps().size() == 1);
  CHECK(x.get_all_base_sizes().size() == 1);
  CHECK(x.get_all_num_edges().size() == 1);
  CHECK(x.get_all_ingestion_timestamps()[0] == 0);
  CHECK(x.get_all_base_sizes()[0] == 0);
  CHECK(x.get_all_num_edges()[0] == 0);
}
