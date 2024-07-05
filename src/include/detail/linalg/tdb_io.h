/**
 * @file   tdb_io.h
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

#ifndef TILEDB_TDB_IO_H
#define TILEDB_TDB_IO_H

#include <filesystem>
#include <numeric>
#include <vector>

#include <tiledb/tiledb>
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_helpers.h"
#include "utils/logging.h"
#include "utils/print_types.h"
#include "utils/timer.h"

namespace {

template <class T>
std::vector<T> read_vector_helper(
    const tiledb::Context& ctx,
    const std::string& uri,
    size_t start_pos,
    size_t end_pos,
    TemporalPolicy temporal_policy,
    bool read_full_vector) {
  scoped_timer _{tdb_func__ + " " + std::string{uri}};

  auto array_ = tiledb_helpers::open_array(
      tdb_func__, ctx, uri, TILEDB_READ, temporal_policy);
  auto schema_ = array_->schema();

  using domain_type = int32_t;
  const size_t idx = 0;

  auto domain_{schema_.domain()};

  auto dim_num_{domain_.ndim()};
  auto array_rows_{domain_.dimension(0)};

  if (read_full_vector) {
    if (start_pos == 0) {
      start_pos = array_rows_.template domain<domain_type>().first;
    }
    if (end_pos == 0) {
      end_pos = array_rows_.template domain<domain_type>().second + 1;
    }
  }

  auto vec_rows_{end_pos - start_pos};

  if (vec_rows_ == 0) {
    return {};
  }

  auto attr_num{schema_.attribute_num()};
  auto attr = schema_.attribute(idx);

  std::string attr_name = attr.name();
  tiledb_datatype_t attr_type = attr.type();

  // Create a subarray that reads the array up to the specified subset.
  std::vector<int32_t> subarray_vals = {
      (int32_t)start_pos, std::max(0, (int32_t)end_pos - 1)};
  tiledb::Subarray subarray(ctx, *array_);
  subarray.set_subarray(subarray_vals);

  // @todo: use something non-initializing
  std::vector<T> data_(vec_rows_);

  tiledb::Query query(ctx, *array_);
  query.set_subarray(subarray).set_data_buffer(
      attr_name, data_.data(), vec_rows_);
  tiledb_helpers::submit_query(tdb_func__, uri, query);
  _memory_data.insert_entry(tdb_func__, vec_rows_ * sizeof(T));

  array_->close();
  assert(tiledb::Query::Status::COMPLETE == query.query_status());

  return data_;
}
}  // namespace

/******************************************************************************
 * Matrix creation and writing.  Because we support out-of-core operations with
 * matrices, reading a matrix is more subtle and is contained in the tdb_matrix
 * and tdb_partitioned_matrix classes.
 ******************************************************************************/

/**
 * Create an empty TileDB array to eventually contain a matrix (a
 * feature_vector_array).
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
void create_empty_for_matrix(
    const tiledb::Context& ctx,
    const std::string& uri,
    size_t rows,
    size_t cols,
    size_t row_extent,
    size_t col_extent,
    tiledb_filter_type_t filter) {
  tiledb::FilterList filter_list(ctx);
  filter_list.add_filter({ctx, filter});

  tiledb::Domain domain(ctx);
  domain
      .add_dimensions(tiledb::Dimension::create<int>(
          ctx, "rows", {{0, std::max(0, (int)rows - 1)}}, row_extent))
      .add_dimensions(tiledb::Dimension::create<int>(
          ctx, "cols", {{0, std::max(0, (int)cols - 1)}}, col_extent));

  tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
  auto order = std::is_same_v<LayoutPolicy, stdx::layout_right> ?
                   TILEDB_ROW_MAJOR :
                   TILEDB_COL_MAJOR;
  schema.set_domain(domain).set_order({{order, order}});
  schema.add_attribute(
      tiledb::Attribute::create<T>(ctx, "values", filter_list));
  schema.set_coords_filter_list(filter_list);

  tiledb::Array::create(uri, schema);
}

/**
 * @brief Create an empty TileDB array into which to write a Matrix.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
void create_matrix(
    const tiledb::Context& ctx,
    const Matrix<T, LayoutPolicy, I>& A,
    const std::string& uri,
    tiledb_filter_type_t filter) {
  // @todo: make this a parameter
  size_t num_parts = 10;

  size_t row_extent = std::max<size_t>(
      (A.num_rows() + num_parts - 1) / num_parts, A.num_rows() >= 2 ? 2 : 1);
  size_t col_extent = std::max<size_t>(
      (A.num_cols() + num_parts - 1) / num_parts, A.num_cols() >= 2 ? 2 : 1);

  create_empty_for_matrix<T, LayoutPolicy, I>(
      ctx, uri, A.num_rows(), A.num_cols(), row_extent, col_extent, filter);
}

/**
 * @brief Write the contents of a Matrix to a TileDB array.
 *
 * @tparam T Type of the matrix elements
 * @tparam LayoutPolicy Layout policy of the matrix (left or right)
 * @tparam I Index type of the matrix
 * @param ctx Context for the write
 * @param A The matrix to write
 * @param uri The URI of the TileDB array to write to
 * @param start_pos Offset row/colum to start writing at
 * @param create Flag to create the array if it does not exist
 * @param temporal_policy Temporal policy for the write
 *
 * @note If we create the matrix here, it will not have any compression
 * @todo Add compressor argument
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
void write_matrix(
    const tiledb::Context& ctx,
    const Matrix<T, LayoutPolicy, I>& A,
    const std::string& uri,
    size_t start_pos = 0,
    bool create = true,
    TemporalPolicy temporal_policy = {}) {
  scoped_timer _{tdb_func__ + " " + std::string{uri}};

  if (create) {
    create_matrix<T, LayoutPolicy, I>(ctx, A, uri, TILEDB_FILTER_NONE);
  }

  if (A.num_rows() == 0 || A.num_cols() == 0) {
    return;
  }

  std::vector<int32_t> subarray_vals{
      0,
      std::max(0, (int)A.num_rows() - 1),
      std::max(0, (int)start_pos),
      std::max(0, (int)start_pos + (int)A.num_cols() - 1)};
  // Open array for writing
  auto array = tiledb_helpers::open_array(
      tdb_func__, ctx, uri, TILEDB_WRITE, temporal_policy);

  tiledb::Subarray subarray(ctx, *array);
  subarray.set_subarray(subarray_vals);

  tiledb::Query query(ctx, *array);
  auto order = std::is_same_v<LayoutPolicy, stdx::layout_right> ?
                   TILEDB_ROW_MAJOR :
                   TILEDB_COL_MAJOR;
  query.set_layout(order)
      .set_data_buffer(
          "values", &A(0, 0), (uint64_t)A.num_rows() * (uint64_t)A.num_cols())
      .set_subarray(subarray);
  tiledb_helpers::submit_query(tdb_func__, uri, query);

  assert(tiledb::Query::Status::COMPLETE == query.query_status());

  array->close();
}

/******************************************************************************
 * Vector I/O
 ******************************************************************************/

/**
 * Create an empty TileDB array to eventually contain a vector
 * @tparam feature_type
 * @param ctx
 * @param uri
 * @param rows
 * @param row_extent
 */
template <class feature_type>
void create_empty_for_vector(
    const tiledb::Context& ctx,
    const std::string& uri,
    size_t rows,
    int32_t row_extent,
    tiledb_filter_type_t filter) {
  tiledb::FilterList filter_list(ctx);
  filter_list.add_filter({ctx, filter});

  tiledb::Domain domain(ctx);
  domain.add_dimensions(tiledb::Dimension::create<int32_t>(
      ctx, "rows", {{0, std::max(0, (int)rows - 1)}}, row_extent));

  tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
  schema.set_domain(domain).set_order({{TILEDB_COL_MAJOR, TILEDB_COL_MAJOR}});
  schema.add_attribute(
      tiledb::Attribute::create<feature_type>(ctx, "values", filter_list));
  schema.set_coords_filter_list(filter_list);

  tiledb::Array::create(uri, schema);
}

/**
 * Create an empty TileDB array to into which to write a vector
 * @tparam feature_type
 * @param ctx
 * @param uri
 * @todo change order of arguments
 */
template <std::ranges::contiguous_range V>
void create_vector(
    const tiledb::Context& ctx,
    const V& v,
    const std::string& uri,
    tiledb_filter_type_t filter) {
  using value_type = std::ranges::range_value_t<V>;

  size_t num_parts = 10;
  size_t tile_extent = (size(v) + num_parts - 1) / num_parts;

  create_empty_for_vector<value_type>(ctx, uri, size(v), tile_extent, filter);
}

/**
 * Write the contents of a std::vector to a TileDB array.
 * @tparam V The type of the vector
 * @param ctx The TileDB context for writing the vector
 * @param v The vector to write
 * @param uri The URI at which to write it
 * @param start_pos Offset to start writing from
 * @param create Flag to create the array if it does not exist
 *
 * @todo change the naming of this function to something more appropriate
 * @todo change the order of arguments
 */
template <std::ranges::contiguous_range V>
void write_vector(
    const tiledb::Context& ctx,
    const V& v,
    const std::string& uri,
    size_t start_pos = 0,
    bool create = true,
    TemporalPolicy temporal_policy = {}) {
  scoped_timer _{tdb_func__ + " " + std::string{uri}};

  using value_type = std::remove_const_t<std::ranges::range_value_t<V>>;

  if (create) {
    create_vector(ctx, v, uri, TILEDB_FILTER_NONE);
  }

  if (size(v) == 0) {
    return;
  }

  // Set the subarray to write into
  std::vector<int32_t> subarray_vals{
      (int)start_pos, (int)start_pos + (int)size(v) - 1};

  // Open array for writing
  auto array = tiledb_helpers::open_array(
      tdb_func__, ctx, uri, TILEDB_WRITE, temporal_policy);

  tiledb::Subarray subarray(ctx, *array);
  subarray.set_subarray(subarray_vals);

  // print_types(v, v.data(), v.size());

  tiledb::Query query(ctx, *array);
  query.set_layout(TILEDB_COL_MAJOR)
      .set_data_buffer("values", const_cast<value_type*>(v.data()), size(v))
      .set_subarray(subarray);

  query.submit();
  assert(tiledb::Query::Status::COMPLETE == query.query_status());
  tiledb_helpers::submit_query(tdb_func__, uri, query);

  array->close();
}

/**
 * Read the contents of a TileDB array into a std::vector.
 *
 * @todo Return a Vector instead of a std::vector
 */
template <class T>
std::vector<T> read_vector(
    const tiledb::Context& ctx,
    const std::string& uri,
    size_t start_pos,
    size_t end_pos,
    TemporalPolicy temporal_policy = {}) {
  return read_vector_helper<T>(
      ctx, uri, start_pos, end_pos, temporal_policy, false);
}

/**
 * @brief Overload to read the entire vector.
 */
template <class T>
std::vector<T> read_vector(
    const tiledb::Context& ctx,
    const std::string& uri,
    TemporalPolicy temporal_policy = {}) {
  return read_vector_helper<T>(ctx, uri, 0, 0, temporal_policy, true);
}

template <class T>
auto sizes_to_indices(const std::vector<T>& sizes) {
  std::vector<T> indices(size(sizes) + 1);
  std::inclusive_scan(begin(sizes), end(sizes), begin(indices) + 1);

  return indices;
}

/**
 * Read binary file format of "sift" datafiles (cf.
 * http://corpus-texmex.irisa.fr) This reads the entire file into memory, from a
 * locally stored file.
 * @tparam T Type of data element stored
 * @param bin_file Path to data file
 * @param subset Number of vectors to read (0 = all)
 * @return a Matrix of the data
 */
template <class T>
auto read_bin_local(
    const tiledb::Context& ctx,
    const std::string& bin_file,
    size_t subset = 0) {
  auto vfs = tiledb::VFS{ctx};

  uint32_t dimension = 0u;

  const auto file_size = vfs.file_size(bin_file);
  auto filebuf = tiledb::VFS::filebuf{vfs};
  filebuf.open(bin_file, std::ios::in | std::ios::binary);

  auto file = std::ifstream{bin_file, std::ios::binary};
  if (!file.good()) {
    throw std::runtime_error("failed to open file: " + bin_file);
  }

  if (!file.read(reinterpret_cast<char*>(&dimension), sizeof(dimension))) {
    throw std::runtime_error("failed to read dimension for the first vector");
  }
  file.seekg(0);

  const auto max_vectors = file_size / (4u + dimension * sizeof(T));
  if (subset > max_vectors) {
    throw std::runtime_error(
        "specified subset is too large " + std::to_string(subset) + " > " +
        std::to_string(max_vectors));
  }

  const auto num_vectors = subset == 0 ? max_vectors : subset;

  auto result = ColMajorMatrix<T>{dimension, num_vectors};
  auto* result_ptr = result.data();

  for (size_t i = 0; i < num_vectors; ++i) {
    uint32_t d = 0u;

    if (!file.read(reinterpret_cast<char*>(&d), sizeof(d))) {
      throw std::runtime_error(
          "failed to read dimension for vector at pos: " + std::to_string(i));
    }

    if (d != dimension) {
      throw std::runtime_error(
          "dimension mismatch: " + std::to_string(d) +
          " != " + std::to_string(dimension));
    }
    if (!file.read(reinterpret_cast<char*>(result_ptr), d * sizeof(T))) {
      throw std::runtime_error("file read error");
    }
    result_ptr += dimension;
  }

  return result;
}

#endif  // TILEDB_TDB_IO_H
