/**
 * @file   gemm.h
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
 *
 */

#ifndef TILEDB_IVF_GEMM_H
#define TILEDB_IVF_GEMM_H

#include "linalg.h"

namespace detail::ivf {

/**
 * @brief Query a set of query vectors against a vector database.
 *
 * This will need to be restructured to support blocking.
 */
auto ivf_blocked_gemm_contiguous(
    auto&& shuffled_db,
    auto&& centroids,
    auto&& q,
    auto&& indices,
    auto&& shuffled_ids,
    size_t nprobe,
    size_t k_nn,
    bool nth,
    size_t nthreads) {
  // get closest centroid for each query vector
  // Does this even need to be blocked...?
  // The whole point of ivf is to avoid loading everything
  // The shuffled_db is the big array to avoid loading
  auto top_k = blocked_gemm_query(centroids, q, nprobe, nth, nthreads);

  // Copy top k from Matrix to vector
  std::vector<size_t> top_top_k(nprobe, 0);
  for (size_t i = 0; i < nprobe; ++i) {
    top_top_k[i] = top_k(i, 0);
  }

  // gather all the probed partitions into a single matrix
  size_t total_size = 0;
  for (size_t i = 0; i < size(top_top_k); ++i) {
    total_size += indices[top_top_k[i] + 1] - indices[top_top_k[i]];
  }

  // Storage for the probed partitions and their ids
  auto all_results = ColMajorMatrix<float>{centroids.num_rows(), total_size};
  auto all_ids = std::vector<uint64_t>(total_size);

  // Tracks next location to copy into
  size_t ctr = 0;

  // @todo parallelize this loop
  // @todo don't make contiguous copy -- just search each cluster separately
  // Copy the probed partitions into contiguous storage
  // For each probed partition
  for (size_t j = 0; j < nprobe; ++j) {
    // Get begin and end indices of the partition
    size_t start = indices[top_top_k[j]];
    size_t end = indices[top_top_k[j] + 1];

    // Copy the partition into the storage
    // For each vector in the partition
    for (size_t i = start; i < end; ++i) {
      // Copy the vector into all_results and ids into all_ids
      // @todo Abstract all of this explicit loop based assignment
      size_t l_end = shuffled_db.num_rows();
      for (size_t l = 0; l < l_end; ++l) {
        all_results(l, ctr) = shuffled_db(l, i);
        all_ids[ctr] = shuffled_ids[i];
      }
      ++ctr;
    }
  }

  // Now, with the single matrix of probed partitions, find the closest vectors
  auto kmeans_ids = blocked_gemm_query(all_results, q, k_nn, nth, nthreads);

  // Original ids are: all_ids[kmeans_ids(i, 0)]
  // Maybe that is what should be returned?

  return std::make_tuple(std::move(kmeans_ids), all_ids);
}

}  // namespace detail::ivf

#endif  // TILEDB_IVF_GEMM_H
