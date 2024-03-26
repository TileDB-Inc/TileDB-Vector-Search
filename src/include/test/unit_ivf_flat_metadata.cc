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

std::vector<std::tuple<std::string, std::string>> expected_str{
    {"dataset_type", "vector_search"},
    {"storage_version", current_storage_version},
    {"dtype", "float32"},
    {"feature_type", "float32"},
    {"id_type", "uint32"},
    {"indices_type", "uint32"},
    {"index_type", "IVF_FLAT"},
    {"base_sizes", "[0,10000]"},
    {"partition_history", "[0,100]"},
};

std::vector<std::tuple<std::string, size_t>> expected_arithmetic{
    {"temp_size", 0},
    {"dimension", 128},
    {"feature_datatype", 2},
    {"id_datatype", 9},
    {"px_datatype", 9},
};

TEST_CASE("ivf_flat_metadata: test test", "[ivf_flat_metadata]") {
  REQUIRE(true);
}

TEST_CASE("ivf_flat_metadata: default constructor", "[ivf_flat_metadata]") {
  auto x = ivf_flat_index_metadata();
  ivf_flat_index_metadata y;
}

// TODO(paris): Modify the index and then also check for ingestion_timestamps
// and num_edges_history.
TEST_CASE(
    "ivf_flat_metadata: load metadata from index", "[ivf_flat_metadata]") {
  tiledb::Context ctx;
  tiledb::Config cfg;

  std::string uri =
      (std::filesystem::temp_directory_path() / "tmp_ivf_index").string();
  auto training_vectors =
      tdbColMajorPreLoadMatrix<float>(ctx, siftsmall_inputs_uri);
  auto idx = ivf_flat_index<float, uint32_t, uint32_t>(100, 1);
  idx.train(training_vectors, kmeans_init::kmeanspp);
  idx.add(training_vectors);
  idx.write_index(ctx, uri, true);

  auto read_group = tiledb::Group(ctx, uri, TILEDB_READ, cfg);
  auto x = ivf_flat_index_metadata();
  x.load_metadata(read_group);

  SECTION("Validate metadata.") {
    auto x = ivf_flat_index_metadata();
    x.load_metadata(read_group);
    x.dump();

    for (auto& [name, value] : expected_str) {
      tiledb_datatype_t v_type;
      uint32_t v_num;
      const void* v;
      CHECK(read_group.has_metadata(name, &v_type));
      if (!read_group.has_metadata(name, &v_type)) {
        continue;
      }

      read_group.get_metadata(name, &v_type, &v_num, &v);
      CHECK((v_type == TILEDB_STRING_ASCII || v_type == TILEDB_STRING_UTF8));
      std::string tmp = std::string(static_cast<const char*>(v), v_num);
      CHECK(!empty(value));
      CHECK(tmp == value);
    }
    for (auto& [name, value] : expected_arithmetic) {
      tiledb_datatype_t v_type;
      uint32_t v_num;
      const void* v;
      CHECK(read_group.has_metadata(name, &v_type));
      if (!read_group.has_metadata(name, &v_type)) {
        continue;
      }

      read_group.get_metadata(name, &v_type, &v_num, &v);

      if (name == "temp_size") {
        CHECK((v_type == TILEDB_INT64 || v_type == TILEDB_FLOAT64));
        if (v_type == TILEDB_INT64) {
          CHECK(value == *static_cast<const int64_t*>(v));
        } else if (v_type == TILEDB_FLOAT64) {
          CHECK(value == (int64_t) * static_cast<const double*>(v));
        }
      }
      CHECK(
          (v_type == TILEDB_UINT32 || v_type == TILEDB_INT64 ||
           v_type == TILEDB_UINT64 || v_type == TILEDB_FLOAT64 ||
           v_type == TILEDB_FLOAT32));

      std::cout << "name: " << name << std::endl;
      switch (v_type) {
        case TILEDB_FLOAT64:
          CHECK(value == *static_cast<const double*>(v));
          break;
        case TILEDB_FLOAT32:
          CHECK(value == *static_cast<const float*>(v));
          break;
        case TILEDB_INT64:
          CHECK(value == *static_cast<const int64_t*>(v));
          break;
        case TILEDB_UINT64:
          CHECK(value == *static_cast<const uint64_t*>(v));
          break;
        case TILEDB_UINT32:
          CHECK(value == *static_cast<const uint32_t*>(v));
          break;
        default:
          CHECK(false);
          break;
      }
    }

    SECTION("Compare with another load of the metadata.") {
      ivf_flat_index_metadata y;
      y.load_metadata(read_group);
      CHECK(x.compare_metadata(y));
    }
  }
}
