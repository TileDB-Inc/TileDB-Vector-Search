/**
 * @file   unit_tdb_io.cc
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

#include <tiledb/tiledb>

#include <tiledb/group_experimental.h>
#include <catch2/catch_all.hpp>
#include "array_defs.h"
#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/tdb_matrix.h"
#include "query_common.h"

TEST_CASE("tdb_io: test test", "[tdb_io]") {
  REQUIRE(true);
}

// This may not work any longer as we are putting array sizes into the group
// metadata rather than assuming it is the same as the array dimensions.
#if 0
TEST_CASE("tdb_io: read vector", "[tdb_io]") {
  tiledb::Context ctx;

  // Nice hallucination
  // auto uri = "s3://tiledb-inc-demo-data/ivf-hnsw/ivfhnsw_1000_128_uint8/ids";
  auto ids = read_vector<uint64_t>(ctx, bigann1M_ids_uri);
  CHECK(ids.size() == num_bigann1M_vectors);
}
#endif

TEMPLATE_TEST_CASE("tdb_io: read / write vector", "[tdb_io]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_std_vector_uri = "/tmp/tmp_std_vector";
  std::string tmp_vector_uri = "/tmp/tmp_vector";
  int offset = 19;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_std_vector_uri)) {
    vfs.remove_dir(tmp_std_vector_uri);
  }
  if (vfs.is_dir(tmp_vector_uri)) {
    vfs.remove_dir(tmp_vector_uri);
  }

  size_t N = 100'000;
  std::vector<TestType> v(N);
  Vector<TestType> w(N);
  std::iota(begin(v), end(v), offset);
  std::iota(begin(w), end(w), offset);

  write_vector(ctx, v, tmp_std_vector_uri);
  write_vector(ctx, w, tmp_vector_uri);

  auto x = read_vector<TestType>(ctx, tmp_std_vector_uri);
  CHECK(x == v);
  auto y = read_vector<TestType>(ctx, tmp_vector_uri);
  CHECK(y == v);
}

TEST_CASE("tdb_io: read matrix", "[tdb_io]") {
  tiledb::Context ctx;

  auto X = tdbColMajorMatrix<uint8_t>(ctx, bigann1M_inputs_uri);
  CHECK(num_vectors(X) == num_bigann1M_vectors);
  CHECK(dimension(X) == bigann1M_dimension);
}

TEMPLATE_TEST_CASE("tdb_io: write matrix", "[tdb_io]", float, uint8_t) {
  tiledb::Context ctx;
  std::string tmp_matrix_uri = "/tmp/tmp_matrix";
  int offset = 13;

  size_t Mrows = 200;
  size_t Ncols = 500;

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  auto X = ColMajorMatrix<TestType>(Mrows, Ncols);
  std::iota(X.data(), X.data() + dimension(X) * num_vectors(X), offset);
  write_matrix(ctx, X, tmp_matrix_uri);

  auto Y = tdbColMajorMatrix<TestType>(ctx, tmp_matrix_uri);
  Y.load();

  CHECK(num_vectors(Y) == num_vectors(X));
  CHECK(dimension(Y) == dimension(X));
  CHECK(
      std::equal(X.data(), X.data() + dimension(X) * num_vectors(X), Y.data()));
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      CHECK(X(i, j) == Y(i, j));
    }
  }
}

TEST_CASE("tdb_io: create group", "[tdb_io]") {
  size_t N = 10'000;

  tiledb::Context ctx;
  tiledb::Config cfg;
  std::string tmp_group_uri = "/tmp/tmp_group";

  std::string ids_name = "ids";

  auto v = std::vector<uint32_t>(N);
  std::iota(begin(v), end(v), 17);
  auto w{v};

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_group_uri)) {
    vfs.remove_dir(tmp_group_uri);
  }

  tiledb::Group::create(ctx, tmp_group_uri);
  auto write_group = tiledb::Group(ctx, tmp_group_uri, TILEDB_WRITE, cfg);
  write_vector(ctx, v, tmp_group_uri + "/" + ids_name);
  write_group.add_member(ids_name, true, "ids");

  auto w_type = tiledb::impl::type_to_tiledb<
      std::remove_reference_t<decltype(w[0])>>::tiledb_type;
  CHECK(std::is_same_v<std::remove_reference_t<decltype(w[0])>, uint32_t>);
  CHECK(std::is_same_v<std::remove_reference_t<decltype(v[0])>, uint32_t>);
  CHECK(w_type == TILEDB_UINT32);

  write_group.put_metadata("w[0]", w_type, 1, &w[0]);
  write_group.put_metadata("w[1-16]", w_type, 16, &w[1]);
  write_group.close();

  auto read_group = tiledb::Group(ctx, tmp_group_uri, TILEDB_READ, cfg);
  auto num_members = read_group.member_count();
  CHECK(num_members == 1);
  auto a = read_group.member(0);
  CHECK(a.uri() == "file:///tmp/tmp_group/ids");

  auto b = read_group.member("ids");
  CHECK(a.uri() == "file:///tmp/tmp_group/ids");

  CHECK(read_group.has_metadata("w[0]", &w_type));
  CHECK(!read_group.has_metadata("w[1]", &w_type));
  CHECK(read_group.has_metadata("w[1-16]", &w_type));

  uint32_t* w0{nullptr};
  uint32_t* w2{nullptr};
  tiledb_datatype_t r_type{TILEDB_ANY};

  uint32_t one{1};
  read_group.get_metadata("w[0]", &w_type, &one, (const void**)&w0);
  CHECK(*w0 == w[0]);
  CHECK(one == 1);

  uint32_t sixteen{16};
  read_group.get_metadata("w[1-16]", &w_type, &sixteen, (const void**)&w2);
  CHECK(std::equal(begin(w) + 1, begin(w) + 17, w2));
  CHECK(sixteen == 16);

  read_group.close();
}
