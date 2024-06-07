/**
 * @file   unit_ivf_pq_metadata.cc
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
#include <tiledb/tiledb>
#include <vector>
#include "api/feature_vector_array.h"
#include "detail/linalg/tdb_matrix.h"
#include "index/ivf_pq_index.h"
#include "index/ivf_pq_metadata.h"
#include "test/utils/array_defs.h"
#include "test/utils/test_utils.h"

TEST_CASE("default constructor", "[ivf_pq_metadata]") {
  auto x = ivf_pq_metadata();
  ivf_pq_metadata y;
}

TEST_CASE("default constructor compare", "[ivf_pq_metadata]") {
  auto x = ivf_pq_metadata();
  ivf_pq_metadata y;

  CHECK(x.compare_metadata(y));
  CHECK(y.compare_metadata(x));
}

TEST_CASE("load metadata from index", "[ivf_pq_metadata]") {
  tiledb::Context ctx;
  tiledb::Config cfg;

  std::string uri =
      (std::filesystem::temp_directory_path() / "tmp_ivf_pq_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(uri)) {
    vfs.remove_dir(uri);
  }
  auto idx = ivf_pq_index<siftsmall_feature_type, siftsmall_ids_type>(0, 16);

  std::vector<std::tuple<std::string, size_t>> expected_arithmetic{
      {"temp_size", 0},
      {"dimensions", 128},
      {"feature_datatype", 2},
      {"id_datatype", 10},
      {"px_datatype", 10},
      {"num_subspaces", 16},
      {"sub_dimensions", 8},
      {"bits_per_subspace", 8},
      {"num_clusters", 256},
  };

  {
    // Check the metadata after an initial write_index().
    auto training_vectors =
        ColMajorMatrixWithIds<siftsmall_feature_type, siftsmall_ids_type>(
            128, 0);
    idx.train(training_vectors, training_vectors.raveled_ids());
    idx.add(training_vectors, training_vectors.raveled_ids());
    idx.write_index(ctx, uri, TemporalPolicy(TimeTravel, 0));
    auto read_group = tiledb::Group(ctx, uri, TILEDB_READ, cfg);
    std::vector<std::tuple<std::string, std::string>> expected_str{
        {"dataset_type", "vector_search"},
        {"index_type", "IVF_PQ"},
        {"storage_version", current_storage_version},
        {"dtype", "float32"},
        {"feature_type", "float32"},
        {"id_type", "uint64"},
        {"indices_type", "uint64"},
        {"ingestion_timestamps", "[0]"},
        {"base_sizes", "[0]"},
        {"partition_history", "[0]"},
    };
    validate_metadata(read_group, expected_str, expected_arithmetic);

    auto x = ivf_pq_metadata();
    x.load_metadata(read_group);
    CHECK(x.ingestion_timestamps_.size() == 1);
    CHECK(x.ingestion_timestamps_[0] == 0);
    CHECK(x.base_sizes_.size() == 1);
    CHECK(x.base_sizes_[0] == 0);
    CHECK(x.partition_history_.size() == 1);
    CHECK(x.partition_history_[0] == 0);
  }

  {
    // Check that we can overwrite the last ingestion_timestamps, base_sizes,
    // and num_edges_history. We rely on this when creating an index from Python
    // during the initial ingest() so that we end up with the same metadata as
    // when creating with Python.
    auto training_vectors = tdbColMajorPreLoadMatrixWithIds<
        siftsmall_feature_type,
        siftsmall_ids_type>(ctx, siftsmall_inputs_uri, siftsmall_ids_uri, 222);

    idx.train(training_vectors, training_vectors.raveled_ids());
    idx.add(training_vectors, training_vectors.raveled_ids());
    idx.write_index(ctx, uri, TemporalPolicy(TimeTravel, 2), "");

    auto read_group = tiledb::Group(ctx, uri, TILEDB_READ, cfg);
    std::vector<std::tuple<std::string, std::string>> expected_str{
        {"dataset_type", "vector_search"},
        {"index_type", "IVF_PQ"},
        {"storage_version", current_storage_version},
        {"dtype", "float32"},
        {"feature_type", "float32"},
        {"id_type", "uint64"},
        {"indices_type", "uint64"},
        {"ingestion_timestamps", "[2]"},
        {"base_sizes", "[222]"},
        {"partition_history", "[14]"},
    };
    validate_metadata(read_group, expected_str, expected_arithmetic);

    auto x = ivf_pq_metadata();
    x.load_metadata(read_group);
    CHECK(x.ingestion_timestamps_.size() == 1);
    CHECK(x.ingestion_timestamps_[0] == 2);
    CHECK(x.base_sizes_.size() == 1);
    CHECK(x.base_sizes_[0] == 222);
    CHECK(x.partition_history_.size() == 1);
  }

  {
    // Check we appended to metadata after a second normal write_index().
    auto training_vectors = tdbColMajorPreLoadMatrixWithIds<
        siftsmall_feature_type,
        siftsmall_ids_type>(ctx, siftsmall_inputs_uri, siftsmall_ids_uri, 333);

    idx.train(training_vectors, training_vectors.raveled_ids());
    idx.add(training_vectors, training_vectors.raveled_ids());
    idx.write_index(ctx, uri, TemporalPolicy(TimeTravel, 3));

    auto read_group = tiledb::Group(ctx, uri, TILEDB_READ, cfg);
    std::vector<std::tuple<std::string, std::string>> expected_str{
        {"dataset_type", "vector_search"},
        {"index_type", "IVF_PQ"},
        {"storage_version", current_storage_version},
        {"dtype", "float32"},
        {"feature_type", "float32"},
        {"id_type", "uint64"},
        {"indices_type", "uint64"},
        {"ingestion_timestamps", "[2,3]"},
        {"base_sizes", "[222,333]"},
        {"partition_history", "[14,14]"},
    };
    validate_metadata(read_group, expected_str, expected_arithmetic);

    auto x = ivf_pq_metadata();
    x.load_metadata(read_group);
    CHECK(x.ingestion_timestamps_.size() == 2);
    CHECK(x.ingestion_timestamps_[0] == 2);
    CHECK(x.ingestion_timestamps_[1] == 3);
    CHECK(x.base_sizes_.size() == 2);
    CHECK(x.base_sizes_[0] == 222);
    CHECK(x.base_sizes_[1] == 333);
    CHECK(x.partition_history_.size() == 2);
  }

  {
    // Check we can clear history.
    auto read_group = tiledb::Group(ctx, uri, TILEDB_READ, cfg);
    auto x = ivf_pq_metadata();
    x.load_metadata(read_group);
    // Will clear less than or equal to 2, so we should just have ingestion at
    // timestamp 3.
    x.clear_history(2);
    CHECK(x.ingestion_timestamps_.size() == 1);
    CHECK(x.ingestion_timestamps_[0] == 3);
    CHECK(x.base_sizes_.size() == 1);
    CHECK(x.base_sizes_[0] == 333);
    CHECK(x.partition_history_.size() == 1);

    auto write_group = tiledb::Group(ctx, uri, TILEDB_WRITE, cfg);
    x.store_metadata(write_group);
  }

  {
    // And we can still load correctly after clearing history.
    auto read_group = tiledb::Group(ctx, uri, TILEDB_READ, cfg);
    std::vector<std::tuple<std::string, std::string>> expected_str{
        {"dataset_type", "vector_search"},
        {"index_type", "IVF_PQ"},
        {"storage_version", current_storage_version},
        {"dtype", "float32"},
        {"feature_type", "float32"},
        {"id_type", "uint64"},
        {"ingestion_timestamps", "[3]"},
        {"base_sizes", "[333]"},
        {"partition_history", "[14]"},
    };
    validate_metadata(read_group, expected_str, expected_arithmetic);
  }
}
