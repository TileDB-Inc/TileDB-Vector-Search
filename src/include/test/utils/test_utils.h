/**
 * * @file   test/test_utils.h
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
 */

#ifndef TILEDB_TEST_UTILS_H
#define TILEDB_TEST_UTILS_H

#include <catch2/catch_all.hpp>
#include <ranges>

#include <tiledb/tiledb>
#include "detail/linalg/tdb_io.h"

template <class id_type>
std::string write_ids_to_uri(
    const tiledb::Context& ctx, const tiledb::VFS vfs, size_t num_ids) {
  std::vector<id_type> ids(num_ids);
  std::iota(begin(ids), end(ids), 0);
  std::string ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_ids_uri").string();
  if (vfs.is_dir(ids_uri)) {
    vfs.remove_dir(ids_uri);
  }
  write_vector(ctx, ids, ids_uri);
  return ids_uri;
}

// Fill a matrix with sequentially increasing values. Will delete data from the
// URI if it exists.
template <class Matrix>
void fill_and_write_matrix(
    const tiledb::Context& ctx,
    Matrix& X,
    const std::string& uri,
    size_t rows,
    size_t cols,
    size_t offset,
    TemporalPolicy temporal_policy = {}) {
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(uri)) {
    vfs.remove_dir(uri);
  }
  std::iota(X.data(), X.data() + dimensions(X) * num_vectors(X), offset);
  write_matrix(ctx, X, uri, 0, true, temporal_policy);
}

/*
 * Fill a matrix with sequentially increasing values and write those values to
 * URIs. Will delete data from the URI if it exists.
 * @param ctx TileDB context.
 * @param X MatrixWithIds to fill with vectors and IDs.
 * @param uri The URI to write the vectors to in addition to filled X with them.
 * @param ids_uri The URI to write the IDs to in addition to filled X with them.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @param offset The value to start filling the matrix with, i.e. if it's 10 we
 * will fill with 10, 11, 12, etc.
 */
template <class MatrixWithIds>
void fill_and_write_matrix(
    const tiledb::Context& ctx,
    MatrixWithIds& X,
    const std::string& uri,
    const std::string& ids_uri,
    size_t rows,
    size_t cols,
    size_t offset,
    TemporalPolicy temporal_policy = {}) {
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(uri)) {
    vfs.remove_dir(uri);
  }
  if (vfs.is_dir(ids_uri)) {
    vfs.remove_dir(ids_uri);
  }
  std::iota(X.data(), X.data() + dimensions(X) * num_vectors(X), offset);
  std::iota(X.ids(), X.ids() + X.num_ids(), offset);

  // Write the vectors to their URI.
  write_matrix(ctx, X, uri, 0, true, temporal_policy);

  // Write the IDs to their URI.
  write_vector(ctx, X.raveled_ids(), ids_uri, 0, true, temporal_policy);
}

void validate_metadata(
    tiledb::Group& read_group,
    const std::vector<std::tuple<std::string, std::string>>& expected_str,
    const std::vector<std::tuple<std::string, size_t>>& expected_arithmetic) {
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
      case TILEDB_STRING_UTF8:
        CHECK(false);
        break;
      default:
        CHECK(false);
        break;
    }
  }
}

#endif  // TILEDB_TEST_UTILS_H
