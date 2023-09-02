/**
 * @file   index.h
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
 * Concrete index classes.
 *
 */

#ifndef TILEDB_INDEX_H
#define TILEDB_INDEX_H

#include <tiledb/tiledb>
#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/flat/qv.h"
#include "detail/flat/vq.h"

/**
 *
 * @tparam DB
 * @tparam Q
 * @tparam Index
 *
 * @todo Should these be contiguous?  Doesn't seem necessary
 */

// tiledb::tiledb_to_type

template <tiledb_datatype_t T>
struct flatter_index {
  using attribute_type = tiledb::impl::tiledb_to_type<T>;
};


// In the arrays we have created, V is float or unint8_t, Index uint64
// We can standardize on this schema -- though we may want to consider int32
// Other external arrays (fmnist, sift) have int32
template <class attribute_type>
class flat_index {
 private:
  std::unique_ptr<tdbColMajorMatrix<attribute_type>> feature_vectors_;

 public:
  flat_index() = delete;
  flat_index(const flat_index& index) = delete;
  flat_index& operator=(const flat_index& index) = delete;

  flat_index(const flat_index&& index) {}
  flat_index& operator=(const flat_index&& index) = default;

  ~flat_index() = default;

  /**
   * @brief
   * @param db
   * @param q
   * @return
   */
  template <class Opt>
  flat_index(const tiledb::Context& ctx, const std::string& uri, const Opt& opt) : feature_vectors_{ctx, uri}{
  }

  template <class M>
  auto query(M&& query, size_t k_nn, size_t nthreads = std::thread::hardware_concurrency()) const {
    return detail::flat::qv_query_heap_tiled(*feature_vectors_, query, k_nn, nthreads);
  }

  constexpr auto load() {
    feature_vectors_->load();
  }

  auto store() {
  }

  void remove() {
  }

  void update(const std::string& uri) {
  }
};

#endif  // TILEDB_INDEX_H