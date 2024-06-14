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
 *
 */

#ifndef TILEDB_IVF_IVF_INDEX_H
#define TILEDB_IVF_IVF_INDEX_H

#include <filesystem>
#include <string>
#include <thread>
#include <tiledb/tiledb>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <docopt.h>

#include "detail/flat/qv.h"
#include "detail/linalg/tdb_matrix.h"

#include "utils/utils.h"

namespace detail::ivf {

/**
 * Partitions a set of vectors, given a set of centroids.
 * @return
 */
template <typename T, class ids_type, class centroids_type>
int ivf_index(
    tiledb::Context& ctx,
    const ColMajorMatrix<T>& db,                // IN
    const std::vector<ids_type>& external_ids,  // IN
    const std::vector<ids_type>& deleted_ids,   // IN
    const std::string& centroids_uri,           // IN (from array centroids_uri)
    const std::string& parts_uri,               // OUT (to array at parts_uri)
    const std::string& index_uri,               // OUT (to array at index_uri)
    const std::string& id_uri,                  // OUT (to array at id_uri)
    size_t start_pos,
    size_t end_pos,
    size_t nthreads,
    uint64_t timestamp) {
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }
  // NOTE(paris): Is it correct to do `TimestampStartEnd, timestamp, timestamp`?
  auto centroid_read_temporal_policy =
      (timestamp == 0) ?
          TemporalPolicy() :
          TemporalPolicy(TimestampStartEnd, timestamp, timestamp);
  tiledb::Array array(
      ctx,
      centroids_uri,
      TILEDB_READ,
      centroid_read_temporal_policy.to_tiledb_temporal_policy());
  auto non_empty = array.non_empty_domain<int32_t>();
  auto partitions = non_empty[1].second.second + 1;

  // Read all rows from column 0 -> `partitions`. Set no upper_bound.
  auto centroids = tdbColMajorMatrix<centroids_type>(
      ctx,
      centroids_uri,
      0,
      std::nullopt,
      0,
      partitions,
      0,
      centroid_read_temporal_policy);
  centroids.load();
  auto parts = detail::flat::qv_partition(centroids, db, nthreads);
  // debug_matrix(parts, "parts");
  {
    scoped_timer _{"shuffling data"};
    std::unordered_set<ids_type> deleted_ids_set(
        deleted_ids.begin(), deleted_ids.end());
    std::vector<size_t> degrees(centroids.num_cols());
    std::vector<ids_type> indices(centroids.num_cols() + 1);
    if (deleted_ids.empty()) {
      for (size_t i = 0; i < db.num_cols(); ++i) {
        auto j = parts[i];
        ++degrees[j];
      }
    } else {
      for (size_t i = 0; i < db.num_cols(); ++i) {
        if (auto iter = deleted_ids_set.find(external_ids[i]);
            iter == deleted_ids_set.end()) {
          auto j = parts[i];
          ++degrees[j];
        }
      }
    }
    indices[0] = 0;
    std::inclusive_scan(begin(degrees), end(degrees), begin(indices) + 1);

    std::vector<size_t> check(indices.size());
    std::copy(begin(indices), end(indices), begin(check));

    // @todo Add systematic mechanism for debugging algorithms / dat structures
    // debug_matrix(degrees, "degrees");
    // debug_matrix(indices, "indices");

    // Some variables for debugging
    // @todo remove these once we are confident in the code
    auto mis = std::max_element(begin(indices), end(indices));
    auto a = std::distance(begin(indices), mis);
    auto b = std::distance(mis, end(indices));
    auto misx = *mis;

    // Array for storing the shuffled data
    auto shuffled_db = ColMajorMatrix<T>{db.num_rows(), db.num_cols()};
    std::vector shuffled_ids = std::vector<ids_type>(db.num_cols());

    // @todo Add systematic mechanism for debugging algorithms / dat structures
    // debug_matrix(shuffled_db, "shuffled_db");
    // debug_matrix(shuffled_ids, "shuffled_ids");

    // @todo parallelize
    // Unfortunately this variant of the algorithm is not parallelizable.
    // The other approach involves doing parallel sort on the indices,
    // which will group them nicely -- but a distributed parallel sort may
    // be difficult to implement.  Even this algorithm is not trivial to
    // parallelize, because of the random access to the indices array.
    if (deleted_ids.empty()) {
      for (size_t i = 0; i < db.num_cols(); ++i) {
        size_t bin = parts[i];
        size_t ibin = indices[bin];

        shuffled_ids[ibin] = external_ids[i];

        if (ibin >= shuffled_db.num_cols()) {
          throw std::runtime_error(
              "[ivf_index] ibin >= shuffled_db.num_cols()");
        }
        for (size_t j = 0; j < db.num_rows(); ++j) {
          shuffled_db(j, ibin) = db(j, i);
        }
        ++indices[bin];
      }
    } else {
      for (size_t i = 0; i < db.num_cols(); ++i) {
        if (auto iter = deleted_ids_set.find(external_ids[i]);
            iter == deleted_ids_set.end()) {
          size_t bin = parts[i];
          size_t ibin = indices[bin];

          shuffled_ids[ibin] = external_ids[i];

          if (ibin >= shuffled_db.num_cols()) {
            throw std::runtime_error(
                "[ivf_index] ibin >= shuffled_db.num_cols()");
          }
          for (size_t j = 0; j < db.num_rows(); ++j) {
            shuffled_db(j, ibin) = db(j, i);
          }
          ++indices[bin];
        }
      }
    }

    std::shift_right(begin(indices), end(indices), 1);
    indices[0] = 0;

    // A check for debugging
    auto x = std::equal(begin(indices), end(indices), begin(check));

    for (size_t i = 0; i < size(indices); ++i) {
      indices[i] = indices[i] + start_pos;
    }

    // Write out the arrays
    TemporalPolicy temporal_policy = (timestamp == 0) ?
                                         TemporalPolicy() :
                                         TemporalPolicy(TimeTravel, timestamp);
    if (parts_uri != "") {
      write_matrix<T, stdx::layout_left, size_t>(
          ctx, shuffled_db, parts_uri, start_pos, false, temporal_policy);
    }
    if (index_uri != "") {
      write_vector(ctx, indices, index_uri, 0, false, temporal_policy);
    }
    if (id_uri != "") {
      write_vector(
          ctx, shuffled_ids, id_uri, start_pos, false, temporal_policy);
    }
  }
  return 0;
}

/**
 * Open db and set up external ids to either be a contiguous set of integers
 * (i.e., the index of the vector in the db), or read from an external array.
 * Call the main ivf_index function above.
 */
template <typename T, class ids_type, class centroids_type>
int ivf_index(
    tiledb::Context& ctx,
    const std::string& db_uri,
    const std::string& external_ids_uri,
    const std::vector<ids_type>& deleted_ids,
    const std::string& centroids_uri,
    const std::string& parts_uri,
    const std::string& index_uri,
    const std::string& id_uri,
    size_t start_pos = 0,
    size_t end_pos = 0,
    size_t nthreads = 0,
    uint64_t timestamp = 0) {
  TemporalPolicy temporal_policy = (timestamp == 0) ?
                                       TemporalPolicy() :
                                       TemporalPolicy(TimeTravel, timestamp);
  // Read all rows from column `start_pos` -> `end_pos`. Set no upper_bound.
  auto db = tdbColMajorMatrix<T>(
      ctx, db_uri, 0, std::nullopt, start_pos, end_pos, 0, temporal_policy);
  db.load();
  std::vector<ids_type> external_ids;
  if (external_ids_uri.empty()) {
    external_ids = std::vector<ids_type>(db.num_cols());
    std::iota(begin(external_ids), end(external_ids), start_pos);
  } else {
    external_ids = read_vector<ids_type>(
        ctx, external_ids_uri, start_pos, end_pos, temporal_policy);
  }
  return ivf_index<T, ids_type, centroids_type>(
      ctx,
      db,
      external_ids,
      deleted_ids,
      centroids_uri,
      parts_uri,
      index_uri,
      id_uri,
      start_pos,
      end_pos,
      nthreads,
      timestamp);
}

/**
 * Open db and call main ivf_index function above.
 */
template <typename T, class ids_type, class centroids_type>
int ivf_index(
    tiledb::Context& ctx,
    const std::string& db_uri,
    const std::vector<ids_type>& external_ids,
    const std::vector<ids_type>& deleted_ids,
    const std::string& centroids_uri,
    const std::string& parts_uri,
    const std::string& index_uri,
    const std::string& id_uri,
    size_t start_pos = 0,
    size_t end_pos = 0,
    size_t nthreads = 0,
    uint64_t timestamp = 0) {
  // Read all rows from column `start_pos` -> `end_pos`. Set no upper_bound.
  auto db = tdbColMajorMatrix<T>(
      ctx, db_uri, 0, std::nullopt, start_pos, end_pos, 0, timestamp);
  db.load();
  return ivf_index<T, ids_type, centroids_type>(
      ctx,
      db,
      external_ids,
      deleted_ids,
      centroids_uri,
      parts_uri,
      index_uri,
      id_uri,
      start_pos,
      end_pos,
      nthreads,
      timestamp);
}

/*
 * Set up external ids to be either the indices of the vectors in the db,
 * or read from an external array.  Call the main ivf_index function above.
 */
template <typename T, class ids_type, class centroids_type>
int ivf_index(
    tiledb::Context& ctx,
    const ColMajorMatrix<T>& db,
    const std::string& external_ids_uri,
    const std::vector<ids_type>& deleted_ids,
    const std::string& centroids_uri,
    const std::string& parts_uri,
    const std::string& index_uri,
    const std::string& id_uri,
    size_t start_pos = 0,
    size_t end_pos = 0,
    size_t nthreads = 0,
    uint64_t timestamp = 0) {
  std::vector<ids_type> external_ids;
  if (external_ids_uri.empty()) {
    external_ids = std::vector<ids_type>(db.num_cols());
    std::iota(begin(external_ids), end(external_ids), start_pos);
  } else {
    external_ids = read_vector<ids_type>(
        ctx, external_ids_uri, start_pos, end_pos, timestamp);
  }
  return ivf_index<T, ids_type, centroids_type>(
      ctx,
      db,
      external_ids,
      deleted_ids,
      centroids_uri,
      parts_uri,
      index_uri,
      id_uri,
      start_pos,
      end_pos,
      nthreads,
      timestamp);
}

}  // namespace detail::ivf

#endif  // TILEDB_IVF_IVF_INDEX_H
