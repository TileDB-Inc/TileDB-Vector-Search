/**
 * @file   unit_api_feature_vector_array.cc
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

#include <type_traits>
#include "api/feature_vector.h"
#include "api/feature_vector_array.h"
#include "catch2/catch_all.hpp"
#include "detail/ivf/qv.h"
#include "index/index_defs.h"
#include "tdb_defs.h"
#include "test/utils/array_defs.h"
#include "test/utils/query_common.h"
#include "test/utils/test_utils.h"
#include "utils/utils.h"

#include <tiledb/tiledb>

// ----------------------------------------------------------------------------
// FeatureVectorArray tests
// ----------------------------------------------------------------------------

TEST_CASE("feature vector array open", "[api]") {
  tiledb::Context ctx;

  auto a = FeatureVectorArray(ctx, sift_inputs_uri);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(a) == 128);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = FeatureVectorArray(ctx, bigann1M_inputs_uri);
  CHECK(b.feature_type() == TILEDB_UINT8);
  CHECK(dimensions(b) == 128);
  CHECK(num_vectors(b) == num_bigann1M_vectors);

  auto c = FeatureVectorArray(ctx, fmnist_inputs_uri);
  CHECK(c.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(c) == 784);
  CHECK(num_vectors(c) == num_fmnist_vectors);

  auto d = FeatureVectorArray(ctx, sift_inputs_uri);
  CHECK(d.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(d) == 128);
  CHECK(num_vectors(d) == num_sift_vectors);
}

template <query_vector_array M>
auto _ack(const M& m) {
}

auto ack() {
  _ack(MatrixView<float>{});
}

TEST_CASE("Matrix constructors and destructors", "[api]") {
  auto a = ColMajorMatrix<int>(3, 7);
  auto b = FeatureVectorArray(a);

  auto c = ColMajorMatrix<int>(3, 7);
  auto d = FeatureVectorArray(std::move(c));
}

TEMPLATE_TEST_CASE(
    "FeatureVectorArray feature_type",
    "[api]",
    int,
    int8_t,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  auto a = ColMajorMatrix<TestType>{3, 17};
  auto b = FeatureVectorArray(a);

  CHECK(b.feature_type() == t);
  CHECK(b.feature_size() == sizeof(TestType));

  auto c = FeatureVectorArray{ColMajorMatrix<TestType>{17, 3}};
  CHECK(c.feature_type() == t);
  CHECK(c.feature_size() == sizeof(TestType));

  auto f = ColMajorMatrix<TestType>{3, 17};
  auto d = FeatureVectorArray{std::move(f)};
  CHECK(d.feature_type() == t);
  CHECK(d.feature_size() == sizeof(TestType));

  auto e = FeatureVectorArray{std::move(ColMajorMatrix<TestType>{3, 9})};
  CHECK(e.feature_type() == t);
  CHECK(e.feature_size() == sizeof(TestType));

  auto g = std::move(e);
  CHECK(g.feature_type() == t);
  CHECK(g.feature_size() == sizeof(TestType));
}

TEST_CASE("tdbMatrix constructors and destructors", "[api]") {
  tiledb::Context ctx;
  auto c = ColMajorMatrix<int>(3, 7);

  const auto tmp = (std::filesystem::temp_directory_path() / "a").string();

  std::filesystem::remove_all(tmp);
  write_matrix(ctx, c, tmp);

  auto a = tdbColMajorMatrix<int>(ctx, tmp);
  a.load();
  auto b = FeatureVectorArray(a);

  auto d = tdbColMajorMatrix<int>(ctx, tmp);
  d.load();
  auto e = FeatureVectorArray(std::move(d));
}

#if 0  // This fails with 2.16.0
TEST_CASE("Arrays going out of scope", "[api]") {
  auto ctx = tiledb::Context{};
  auto foo = tiledb::Array(ctx, "/tmp/a", TILEDB_READ);
  auto bar = std::move(foo);
}
#endif

TEMPLATE_TEST_CASE(
    "tdb FeatureVectorArray feature_type",
    "[api]",
    int,
    int8_t,
    uint8_t,
    uint32_t,
    float,
    uint64_t) {
  auto t = tiledb::impl::type_to_tiledb<TestType>::tiledb_type;

  tiledb::Context ctx;
  const auto uri = (std::filesystem::temp_directory_path() / "a").string();

  auto cc = ColMajorMatrix<TestType>(3, 7);

  std::filesystem::remove_all(uri);
  write_matrix(ctx, cc, uri);

  {
    auto a = tdbColMajorMatrix<TestType>{ctx, uri};
    auto b = FeatureVectorArray(a);
    CHECK(b.feature_type() == t);
  }

  {
    auto c = FeatureVectorArray(tdbColMajorMatrix<TestType>{ctx, uri});
    CHECK(c.feature_type() == t);
  }

  {
    auto f = tdbColMajorMatrix<TestType>{ctx, uri};
    auto d = FeatureVectorArray{std::move(f)};
    CHECK(d.feature_type() == t);
  }

  {
    auto e =
        FeatureVectorArray{std::move(tdbColMajorMatrix<TestType>{ctx, uri})};
    CHECK(e.feature_type() == t);

    auto g = std::move(e);
    CHECK(g.feature_type() == t);
  }
}

TEST_CASE("query checks", "[api][index]") {
  tiledb::Context ctx;
  size_t k_nn = 10;
  size_t nthreads = 8;
  size_t num_queries = 50;

  SECTION("simple check") {
    auto z = FeatureVectorArray(ctx, sift_inputs_uri);
    auto nn = dimensions(z);
    auto nnn = num_vectors(z);
    CHECK(dimensions(z) == 128);
    CHECK(num_vectors(z) == num_sift_vectors);
  }

  SECTION("tdbMatrix") {
    auto ck = tdbColMajorMatrix<float>(ctx, sift_inputs_uri);
    ck.load();

    auto qk = tdbColMajorMatrix<float>(ctx, sift_query_uri, num_queries);
    load(qk);

    auto [ck_scores, ck_top_k] =
        detail::flat::qv_query_heap(ck, qk, k_nn, nthreads);

    auto gk =
        tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
    load(gk);

    auto ok = validate_top_k(ck_top_k, gk);
    CHECK(ok);
  }
}

// ----------------------------------------------------------------------------
// FeatureVectorArray with IDs tests
// ----------------------------------------------------------------------------

TEST_CASE("feature vector array with IDs open", "[api]") {
  tiledb::Context ctx;

  auto a = FeatureVectorArray(ctx, sift_inputs_uri, sift_ids_uri);
  CHECK(a.feature_type() == TILEDB_FLOAT32);
  CHECK(dimensions(a) == 128);
  CHECK(num_vectors(a) == num_sift_vectors);

  auto b = FeatureVectorArray(ctx, bigann1M_inputs_uri, bigann1M_ids_uri);
  CHECK(b.feature_type() == TILEDB_UINT8);
  CHECK(dimensions(b) == 128);
  CHECK(num_vectors(b) == num_bigann1M_vectors);
}

TEST_CASE("MatrixWithIds constructors and destructors", "[api]") {
  auto rows = 3;
  auto cols = 7;

  SECTION("copy constructor") {
    using DataType = int;
    using IdsType = float;

    auto a = ColMajorMatrixWithIds<DataType, IdsType>(rows, cols);
    std::iota(a.data(), a.data() + rows * cols, 0);
    std::iota(a.ids(), a.ids() + a.num_ids(), 0);
    auto b = FeatureVectorArray(a);

    CHECK(b.dimensions() == rows);
    CHECK(dimensions(b) == rows);
    CHECK(b.num_vectors() == cols);
    CHECK(num_vectors(b) == cols);
    CHECK(b.num_ids() == cols);
    CHECK(num_ids(b) == cols);

    auto feature_type = tiledb::impl::type_to_tiledb<DataType>::tiledb_type;
    CHECK(b.feature_type() == feature_type);
    CHECK(b.feature_type_string() == datatype_to_string(feature_type));
    CHECK(b.feature_size() == datatype_to_size(feature_type));

    auto ids_type = tiledb::impl::type_to_tiledb<IdsType>::tiledb_type;
    CHECK(b.ids_type() == ids_type);
    CHECK(b.ids_type_string() == datatype_to_string(ids_type));
    CHECK(b.ids_size() == datatype_to_size(ids_type));

    auto data = MatrixView<DataType, stdx::layout_left>{
        (DataType*)b.data(), extents(b)[0], extents(b)[1]};
    CHECK(data(0, 0) == 0);
    CHECK(data(5, 0) == 5);

    CHECK(b.num_ids() == cols);
    CHECK(b.ids() != nullptr);
    auto ids = std::span<IdsType>((IdsType*)b.ids(), b.num_vectors());
    CHECK(ids.size() == cols);
    CHECK(ids[0] == 0);
    CHECK(ids[5] == 5);
  }
  SECTION("move constructor") {
    using DataType = float;
    using IdsType = uint8_t;

    auto a = ColMajorMatrixWithIds<DataType, IdsType>(rows, cols);
    std::iota(a.data(), a.data() + rows * cols, 0);
    std::iota(a.ids(), a.ids() + a.num_ids(), 0);

    auto a_ptr = a.data();
    auto a_ptr_ids = a.ids();

    auto b = FeatureVectorArray(std::move(a));

    CHECK(a_ptr == b.data());
    CHECK(a_ptr_ids == b.ids());
    CHECK(a.data() == nullptr);
    CHECK(a.ids() == nullptr);

    CHECK(b.dimensions() == rows);
    CHECK(dimensions(b) == rows);
    CHECK(b.num_vectors() == cols);
    CHECK(num_vectors(b) == cols);
    CHECK(b.num_ids() == cols);

    auto feature_type = tiledb::impl::type_to_tiledb<DataType>::tiledb_type;
    CHECK(b.feature_type() == feature_type);
    CHECK(b.feature_type_string() == datatype_to_string(feature_type));
    CHECK(b.feature_size() == datatype_to_size(feature_type));

    auto ids_type = tiledb::impl::type_to_tiledb<IdsType>::tiledb_type;
    CHECK(b.ids_type() == ids_type);
    CHECK(b.ids_type_string() == datatype_to_string(ids_type));
    CHECK(b.ids_size() == datatype_to_size(ids_type));

    auto data = MatrixView<DataType, stdx::layout_left>{
        (DataType*)b.data(), extents(b)[0], extents(b)[1]};
    CHECK(data(0, 0) == 0);
    CHECK(data(5, 0) == 5);

    CHECK(b.ids() != nullptr);
    auto ids = std::span<IdsType>((IdsType*)b.ids(), b.num_vectors());
    CHECK(ids.size() == cols);
    CHECK(ids[0] == 0);
    CHECK(ids[5] == 5);
  }
}

TEMPLATE_PRODUCT_TEST_CASE(
    "FeatureVectorArray with IDs",
    "[api]",
    (ColMajorMatrixWithIds),
    ((int, uint32_t),
     (int8_t, uint32_t),
     (uint8_t, uint32_t),
     (uint32_t, uint32_t),
     (float, uint32_t),
     (uint64_t, uint32_t),
     (int, uint64_t),
     (int8_t, uint64_t),
     (uint8_t, uint64_t),
     (uint32_t, uint64_t),
     (float, uint64_t),
     (uint64_t, uint64_t))) {
  using DataType = typename TestType::value_type;
  using IdsType = typename TestType::ids_type;
  auto t = tiledb::impl::type_to_tiledb<DataType>::tiledb_type;
  auto t_ids = tiledb::impl::type_to_tiledb<IdsType>::tiledb_type;

  auto a = TestType{3, 17};
  auto b = FeatureVectorArray(a);
  CHECK(b.feature_type() == t);
  CHECK(b.feature_size() == sizeof(DataType));
  CHECK(b.ids_type() == t_ids);
  CHECK(b.ids_size() == sizeof(IdsType));

  auto c = FeatureVectorArray{TestType{17, 3}};
  CHECK(c.feature_type() == t);
  CHECK(c.feature_size() == sizeof(DataType));
  CHECK(c.ids_type() == t_ids);
  CHECK(c.ids_size() == sizeof(IdsType));

  auto f = TestType{3, 17};
  auto d = FeatureVectorArray{std::move(f)};
  CHECK(d.feature_type() == t);
  CHECK(d.feature_size() == sizeof(DataType));
  CHECK(d.ids_type() == t_ids);
  CHECK(d.ids_size() == sizeof(IdsType));

  auto e = FeatureVectorArray{std::move(TestType{3, 9})};
  CHECK(e.feature_type() == t);
  CHECK(e.feature_size() == sizeof(DataType));
  CHECK(e.ids_type() == t_ids);
  CHECK(e.ids_size() == sizeof(IdsType));

  auto g = std::move(e);
  CHECK(g.feature_type() == t);
  CHECK(g.feature_size() == sizeof(DataType));
  CHECK(g.ids_type() == t_ids);
  CHECK(g.ids_size() == sizeof(IdsType));
}

TEST_CASE("tdbMatrixWithIds constructors and destructors", "[api]") {
  tiledb::Context ctx;

  int offset = 13;
  size_t rows = 3;
  size_t cols = 7;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();

  auto c = ColMajorMatrixWithIds<int, float>(rows, cols);
  fill_and_write_matrix(
      ctx, c, tmp_matrix_uri, tmp_ids_uri, rows, cols, offset);

  auto a =
      tdbColMajorMatrixWithIds<int, float>(ctx, tmp_matrix_uri, tmp_ids_uri);
  a.load();
  auto b = FeatureVectorArray(a);

  auto d =
      tdbColMajorMatrixWithIds<int, float>(ctx, tmp_matrix_uri, tmp_ids_uri);
  d.load();
  auto e = FeatureVectorArray(std::move(d));
}

TEMPLATE_TEST_CASE(
    "tdb FeatureVectorArray with IDs feature_type",
    "[api]",
    uint32_t,
    uint64_t) {
  using DataType = float;
  using IdsType = TestType;
  auto t = tiledb::impl::type_to_tiledb<DataType>::tiledb_type;
  auto t_ids = tiledb::impl::type_to_tiledb<IdsType>::tiledb_type;

  tiledb::Context ctx;

  int offset = 13;
  size_t rows = 3;
  size_t cols = 7;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  std::string tmp_ids_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_ids_matrix").string();

  auto cc = ColMajorMatrixWithIds<DataType, IdsType>(rows, cols);
  fill_and_write_matrix(
      ctx, cc, tmp_matrix_uri, tmp_ids_uri, rows, cols, offset);

  {
    auto a = tdbColMajorMatrixWithIds<DataType, TestType>{
        ctx, tmp_matrix_uri, tmp_ids_uri};
    auto b = FeatureVectorArray(a);
    CHECK(b.feature_type() == t);
    CHECK(b.feature_size() == sizeof(DataType));
    CHECK(b.ids_type() == t_ids);
    CHECK(b.ids_size() == sizeof(TestType));
  }

  {
    auto c = FeatureVectorArray(tdbColMajorMatrixWithIds<DataType, IdsType>{
        ctx, tmp_matrix_uri, tmp_ids_uri});
    CHECK(c.feature_type() == t);
    CHECK(c.feature_size() == sizeof(DataType));
    CHECK(c.ids_type() == t_ids);
    CHECK(c.ids_size() == sizeof(TestType));
  }

  {
    auto f = tdbColMajorMatrixWithIds<DataType, IdsType>{
        ctx, tmp_matrix_uri, tmp_ids_uri};
    auto d = FeatureVectorArray{std::move(f)};
    CHECK(d.feature_type() == t);
    CHECK(d.feature_size() == sizeof(DataType));
    CHECK(d.ids_type() == t_ids);
    CHECK(d.ids_size() == sizeof(TestType));
  }

  {
    auto e = FeatureVectorArray{
        std::move(tdbColMajorMatrixWithIds<DataType, IdsType>{
            ctx, tmp_matrix_uri, tmp_ids_uri})};
    CHECK(e.feature_type() == t);
    CHECK(e.feature_size() == sizeof(DataType));
    CHECK(e.ids_type() == t_ids);
    CHECK(e.ids_size() == sizeof(TestType));

    auto g = std::move(e);
    CHECK(g.feature_type() == t);
    CHECK(g.feature_size() == sizeof(DataType));
    CHECK(g.ids_type() == t_ids);
    CHECK(g.ids_size() == sizeof(TestType));
  }
}

TEST_CASE("query checks with IDs", "[api][index]") {
  tiledb::Context ctx;
  size_t k_nn = 10;
  size_t nthreads = 8;
  size_t num_queries = 50;

  SECTION("simple check") {
    auto z = FeatureVectorArray(ctx, sift_inputs_uri, sift_ids_uri);
    auto nn = dimensions(z);
    auto nnn = num_vectors(z);
    CHECK(dimensions(z) == 128);
    CHECK(z.dimensions() == 128);
    CHECK(num_vectors(z) == num_sift_vectors);
    CHECK(z.num_vectors() == num_sift_vectors);
    CHECK(num_ids(z) == num_sift_vectors);
    CHECK(z.num_ids() == num_sift_vectors);
  }

  SECTION("tdbMatrixWithIds") {
    auto ck =
        tdbColMajorMatrixWithIds<float>(ctx, sift_inputs_uri, sift_ids_uri);
    ck.load();
    CHECK(num_ids(ck) == num_sift_vectors);
    CHECK(ck.num_ids() == num_sift_vectors);

    auto qk = tdbColMajorMatrixWithIds<float>(
        ctx, sift_query_uri, sift_ids_uri, num_queries);
    load(qk);
    CHECK(num_ids(qk) == num_queries);
    CHECK(qk.num_ids() == num_queries);

    auto [ck_scores, ck_top_k] =
        detail::flat::qv_query_heap(ck, qk, k_nn, nthreads);

    auto gk =
        tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
    load(gk);

    auto ok = validate_top_k(ck_top_k, gk);
    CHECK(ok);
  }
}

TEST_CASE("load empty matrix", "[api][index]") {
  tiledb::Context ctx;
  std::string tmp_matrix_uri =
      (std::filesystem::temp_directory_path() / "tmp_tdb_matrix").string();
  size_t dimension{10};
  int32_t domain{std::numeric_limits<int32_t>::max() - 1};
  int32_t tile_extent{100'000};

  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(tmp_matrix_uri)) {
    vfs.remove_dir(tmp_matrix_uri);
  }

  create_empty_for_matrix<float, stdx::layout_left>(
      ctx,
      tmp_matrix_uri,
      dimension,
      domain,
      dimension,
      tile_extent,
      TILEDB_FILTER_NONE);

  auto X = FeatureVectorArray(ctx, tmp_matrix_uri);
}

TEST_CASE("temporal_policy", "[api]") {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  std::string feature_vectors_uri =
      (std::filesystem::temp_directory_path() / "temp_feature_vectors_uri")
          .string();
  std::string ids_uri =
      (std::filesystem::temp_directory_path() / "temp_ids_uri").string();
  if (vfs.is_dir(feature_vectors_uri)) {
    vfs.remove_dir(feature_vectors_uri);
  }
  if (vfs.is_dir(ids_uri)) {
    vfs.remove_dir(ids_uri);
  }

  auto dimension = 3;

  using FeatureType = float;
  using IdsType = uint32_t;

  int32_t default_domain{std::numeric_limits<int32_t>::max() - 1};
  int32_t default_tile_extent{100'000};
  int32_t tile_size_bytes{64 * 1024 * 1024};
  tiledb_filter_type_t default_compression{string_to_filter(
      storage_formats[current_storage_version]["default_attr_filters"])};
  int32_t tile_size{
      (int32_t)(tile_size_bytes / sizeof(FeatureType) / dimension)};

  // First we create the empty vectors matrix and the ids vector.
  {
    create_empty_for_matrix<FeatureType, stdx::layout_left>(
        ctx,
        feature_vectors_uri,
        dimension,
        default_domain,
        dimension,
        default_tile_extent,
        default_compression);

    create_empty_for_vector<IdsType>(
        ctx, ids_uri, default_domain, tile_size, default_compression);
  }

  // Write to them at timestamp 99.
  {
    auto temporal_policy = TemporalPolicy(TimeTravel, 99);
    auto matrix_with_ids = ColMajorMatrixWithIds<FeatureType, IdsType>{
        {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}, {1, 2, 3, 4}};

    write_matrix(
        ctx, matrix_with_ids, feature_vectors_uri, 0, false, temporal_policy);

    write_vector(
        ctx, matrix_with_ids.raveled_ids(), ids_uri, 0, false, temporal_policy);
  }

  // Read the data and validate our initial write worked.
  {
    auto feature_vector_array =
        FeatureVectorArray(ctx, feature_vectors_uri, ids_uri);
    auto data = MatrixView<FeatureType, stdx::layout_left>{
        (FeatureType*)feature_vector_array.data(),
        extents(feature_vector_array)[0],
        extents(feature_vector_array)[1]};
    auto ids = std::span<IdsType>(
        (IdsType*)feature_vector_array.ids(),
        feature_vector_array.num_vectors());
    auto expected_ids = {1, 2, 3, 4};
    CHECK(std::equal(ids.begin(), ids.end(), expected_ids.begin()));
    CHECK(ids.size() == 4);
    for (size_t i = 0; i < ids.size(); ++i) {
      for (size_t j = 0; j < dimension; ++j) {
        CHECK(data(j, i) == i + 1);
      }
    }
  }

  // Write to them at timestamp 100.
  {
    auto temporal_policy = TemporalPolicy(TimeTravel, 100);
    auto matrix_with_ids = ColMajorMatrixWithIds<FeatureType, IdsType>{
        {{11, 11, 11}, {22, 22, 22}, {33, 33, 33}, {44, 44, 44}},
        {11, 22, 33, 44}};

    write_matrix(
        ctx, matrix_with_ids, feature_vectors_uri, 0, false, temporal_policy);

    write_vector(
        ctx, matrix_with_ids.raveled_ids(), ids_uri, 0, false, temporal_policy);
  }

  // Read the data and validate we read at temporal_policy 100 by default.
  {
    auto feature_vector_array =
        FeatureVectorArray(ctx, feature_vectors_uri, ids_uri);
    auto data = MatrixView<FeatureType, stdx::layout_left>{
        (FeatureType*)feature_vector_array.data(),
        extents(feature_vector_array)[0],
        extents(feature_vector_array)[1]};
    auto ids = std::span<IdsType>(
        (IdsType*)feature_vector_array.ids(),
        feature_vector_array.num_vectors());
    auto expected_ids = {11, 22, 33, 44};
    CHECK(std::equal(ids.begin(), ids.end(), expected_ids.begin()));
    CHECK(ids.size() == 4);
    for (size_t i = 0; i < ids.size(); ++i) {
      for (size_t j = 0; j < dimension; ++j) {
        CHECK(data(j, i) == (i + 1) * 11);
      }
    }
  }

  // Read the data at timestamp 99.
  {
    auto feature_vector_array = FeatureVectorArray(
        ctx, feature_vectors_uri, ids_uri, 0, TemporalPolicy(TimeTravel, 99));
    auto data = MatrixView<FeatureType, stdx::layout_left>{
        (FeatureType*)feature_vector_array.data(),
        extents(feature_vector_array)[0],
        extents(feature_vector_array)[1]};
    auto ids = std::span<IdsType>(
        (IdsType*)feature_vector_array.ids(),
        feature_vector_array.num_vectors());
    auto expected_ids = {1, 2, 3, 4};
    CHECK(std::equal(ids.begin(), ids.end(), expected_ids.begin()));
    CHECK(ids.size() == 4);
    for (size_t i = 0; i < ids.size(); ++i) {
      for (size_t j = 0; j < dimension; ++j) {
        CHECK(data(j, i) == i + 1);
      }
    }
  }

  // Read the data at timestamp 50.
  {
    auto feature_vector_array = FeatureVectorArray(
        ctx, feature_vectors_uri, ids_uri, 0, TemporalPolicy(TimeTravel, 50));
    CHECK(extents(feature_vector_array)[0] == 0);
    CHECK(extents(feature_vector_array)[1] == 0);
    CHECK(feature_vector_array.num_vectors() == 0);
    CHECK(feature_vector_array.num_ids() == 0);
    CHECK(feature_vector_array.dimensions() == 0);
    auto data = MatrixView<FeatureType, stdx::layout_left>{
        (FeatureType*)feature_vector_array.data(),
        extents(feature_vector_array)[0],
        extents(feature_vector_array)[1]};
    auto ids = std::span<IdsType>(
        (IdsType*)feature_vector_array.ids(),
        feature_vector_array.num_vectors());
    CHECK(ids.size() == 0);
    CHECK(data.size() == 0);
  }
}
