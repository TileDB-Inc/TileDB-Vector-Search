/**
 * @file   partitioned.h
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
 * Defines class for partitioned matrix, used for indexing knn search.
 *
 * In the naive case, we can load the entire partitioned matrix into memory and pull out the partitions we need.
 *
 */

#include <cassert>
#include <memory>
#include <span>
#include "linalg.h"
#include "flat_query.h"

#include <mdspan/mdspan.hpp>
#include <tiledb/tiledb>

namespace stdx {
using namespace Kokkos;
using namespace Kokkos::Experimental;
}  // namespace stdx

template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class tdbPartitionedMatrix {
  std::unique_ptr<T[]> storage_;
  std::vector<std::span<T>> partitions_1d_;
  std::vector<stdx::mdspan<T, matrix_extents<I>, LayoutPolicy>> partitions_2d_;
  std::vector<uint32_t> original_ids_;

 public:
  template <class C, class Q, class Idx, class Ids>
  tdbPartitionedMatrix(
      const std::string& uri,
      const C& centroids,
      const Q& queries,
      const Idx& indices,
      const Ids& vector_ids,
      size_t nprobe,
      size_t nthreads) {
    assert(centroids.num_rows() == queries.num_rows());

    size_t dimension = centroids.num_cols();

    // @todo optimize for best choice based on size of centroids and queries
    auto proto_parts = vq_query_heap(centroids, queries, nprobe, nthreads);

    size_t total_size = 0;
    std::vector<size_t> parts(nprobe);
    for (size_t i = 0; i < nprobe; ++i) {
      parts[i] = proto_parts(i, 0);
      total_size += indices[parts[i] + 1] - indices[parts[i]];
    }

    original_ids_.reserve(total_size);

    // Correctness first -- read each partition into its own chunk of memory
    // @todo Load new partitions into memory as needed
    // @todo Read multiple partitions at one time

#ifndef __APPLE__
    auto data_ = std::make_unique_for_overwrite<T[]>(dimension * total_size);
#else
    // auto data_ = std::make_unique<T[]>(new T[mat_rows_ * mat_cols_]);
    storage_ = std::unique_ptr<T[]>(new T[dimension * total_size]);
#endif

    partitions_1d_.reserve(nprobe);
    partitions_2d_.reserve(nprobe);

    for (size_t i = 0; i < nprobe; ++i) {
      size_t start = indices[parts[i]];
      size_t stop = indices[parts[i] + 1];

      auto part_size = stop - start;
      partitions_1d_.emplace_back(
          storage_.get() + i * dimension * part_size, dimension * part_size);
      partitions_2d_.emplace_back(storage_.get() + i * dimension * part_size, dimension, part_size);

      while (start != stop) {
        original_ids_.push_back(vector_ids[start++]);
      }
    }

    auto matrix_order_{order_v<LayoutPolicy>};

    // Open array for reading
    auto init_ =
        std::map<std::string, std::string>{{"vfs.s3.region", "us-east-1"}};
    auto config_ = tiledb::Config{init_};
    auto ctx_ = tiledb::Context{config_};

    auto array_ = tiledb::Array{ctx_, uri, TILEDB_READ};
    auto schema_ = array_.schema();

    const size_t attr_idx = 0;
    auto attr = schema_.attribute(attr_idx);

    std::string attr_name = attr.name();
    tiledb_datatype_t attr_type = attr.type();

    // For each partition, create a tiledb query
    auto cell_order = schema_.cell_order();
    auto tile_order = schema_.tile_order();

    assert(cell_order == tile_order);

    auto domain_{schema_.domain()};

    auto array_rows_{domain_.dimension(0)};
    auto array_cols_{domain_.dimension(1)};

    using row_domain_type = int32_t;
    using col_domain_type = int32_t;

    auto num_array_rows_ =
        (array_rows_.template domain<row_domain_type>().second -
         array_rows_.template domain<row_domain_type>().first + 1);
    auto num_array_cols_ =
        (array_cols_.template domain<col_domain_type>().second -
         array_cols_.template domain<col_domain_type>().first + 1);

    for (size_t partno = 0; partno < nprobe; ++partno) {
      size_t row_begin = 0;
      size_t row_end = dimension;
      size_t col_begin = indices[parts[partno]];
      size_t col_end = indices[parts[partno] + 1];

      if ((matrix_order_ == TILEDB_ROW_MAJOR &&
           cell_order == TILEDB_COL_MAJOR) ||
          (matrix_order_ == TILEDB_COL_MAJOR &&
           cell_order == TILEDB_ROW_MAJOR)) {
        std::swap(row_begin, col_begin);
        std::swap(row_end, col_end);
      }

      // sanity check dimension == row size or column size

      // my_mat.extent(0)

      auto addr = partitions_1d_[partno].data();
      auto num_elts = partitions_1d_[partno].size();

      assert(partitions_1d_[partno].data() == partitions_2d_[partno].data_handle());
      assert(dimension == partitions_2d_[partno].extent(0));
      assert(num_elts == partitions_2d_[partno].extent(1) * dimension);
      assert(
          indices[parts[partno] + 1] - indices[parts[partno]] ==
          partitions_2d_[partno].extent(1));

      std::vector<int32_t> subarray_vals = {
          (int32_t)row_begin,
          (int32_t)row_end - 1,
          (int32_t)col_begin,
          (int32_t)col_end - 1};
      tiledb::Subarray subarray(ctx_, array_);
      subarray.set_subarray(subarray_vals);

      auto layout_order = cell_order;

      tiledb::Query query(ctx_, array_);

      query.set_subarray(subarray)
          .set_layout(layout_order)
          .set_data_buffer(attr_name, addr, num_elts);
      query.submit();
      assert(tiledb::Query::Status::COMPLETE == query.query_status());
    }

    array_.close();
  }
};
