/**
 * @file   feature_vector.h
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 */

#include <span>
#include <tiledb/tiledb>
#include "mdspan/mdspan.hpp"

#include "linalg.h"

namespace stdx {
using namespace Kokkos;
using namespace Kokkos::Experimental;
}  // namespace stdx

template <class T>
using FeatureVector = std::span<T>;

template <class T>
using FeatureVectorRange = ColMajorMatrix<T>;

// @todo operator[] should be part of FeatureVectorRange rather than Matrix
template <class T>
using tdbFeatureVectorRange = tdbColMajorMatrix<T>

template <typename T>
class FeatureVectorRangeReader {
 public:
  FeatureVector<T> read(
      tiledb::Context& ctx,
      const std::string& uri,
      const std::string& attr_name,
      const std::vector<uint64_t>& start,
      const std::vector<uint64_t>& end) {
    tiledb::Array array(ctx, uri, TILEDB_READ);
    tiledb::Query query(ctx, array);
    tiledb::Subarray subarray(ctx, array);
    subarray.set_subarray(start, end);
    query.set_subarray(subarray);
    query.set_layout(TILEDB_ROW_MAJOR);
    query.set_data_buffer(attr_name, buffer);
    query.submit();
    return FeatureVector<T>(buffer, buffer_size);
  }
};

template <class Reader, class T = typename Reader::value_type>
class tdbFeatureVectorRange : FeatureVectorRange<T> {
  tiledb::Array array_;

 public:
  tdbFeatureVectorRange(const std::string& uri)
      : array_(Reader::open(uri)) {
  }
  ~tdbFeatureVectorRange() {
    array_.close();
  }

  FeatureVector<T> read(
      const std::vector<uint64_t>& start, const std::vector<uint64_t>& end) {
    return Reader::read(array_, start, end);
  }
};
