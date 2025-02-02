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

namespace {
// Compute indices.
template <
    typename FeatureType,
    class Vector,
    class IdsType,
    class IndicesType,
    class CentroidsType>
std::vector<IndicesType> compute_indices(
    const Vector& external_ids,                          // IN
    const std::unordered_set<IdsType>& deleted_ids_set,  // IN
    const ColMajorMatrix<CentroidsType>& centroids,      // IN
    const std::vector<size_t>& parts) {
  // The number of vectors assigned to each centroid.
  std::vector<size_t> degrees(centroids.num_cols());
  if (deleted_ids_set.empty()) {
    for (size_t i = 0; i < external_ids.size(); ++i) {
      auto j = parts[i];
      ++degrees[j];
    }
  } else {
    for (size_t i = 0; i < external_ids.size(); ++i) {
      if (deleted_ids_set.find(external_ids[i]) == deleted_ids_set.end()) {
        auto j = parts[i];
        ++degrees[j];
      }
    }
  }

  // The starting index of each partition in the shuffled data.
  std::vector<IndicesType> indices(centroids.num_cols() + 1);
  indices[0] = 0;
  std::inclusive_scan(begin(degrees), end(degrees), begin(indices) + 1);

  return indices;
}

}  // namespace

/**
 * Partitions a set of vectors, given a set of centroids.
 * @return
 */
template <
    typename MatrixType,
    typename FeatureType,
    typename PQFeatureType,
    class IdsType,
    class IndicesType,
    class CentroidsType>
int ivf_pq_index(
    tiledb::Context& ctx,
    const MatrixType& input_vectors,                        // IN
    const ColMajorMatrix<PQFeatureType>& input_pq_vectors,  // IN
    const std::span<IdsType>& external_ids,                 // IN
    const std::span<IdsType>& deleted_ids,                  // IN
    const ColMajorMatrix<CentroidsType>& centroids,         // IN
    const std::string& parts_uri,     // OUT (to array at parts_uri)
    const std::string& index_uri,     // OUT (to array at index_uri)
    const std::string& id_uri,        // OUT (to array at id_uri)
    const std::string& pq_parts_uri,  // OUT (to array at parts_uri)
    size_t start_pos,
    size_t end_pos,
    size_t nthreads,
    TemporalPolicy temporal_policy,
    size_t partition_start = 0) {
  if (input_vectors.num_cols() != input_pq_vectors.num_cols()) {
    throw std::runtime_error(
        "[index@ivf_pq_index] input_vectors.num_cols() != "
        "input_pq_vectors.num_cols()");
  }
  if (input_vectors.num_cols() != external_ids.size()) {
    throw std::runtime_error(
        "[index@ivf_pq_index] input_vectors.num_cols() != external_ids.size()");
  }

  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }

  // Find the centroid that is closest to each input vector.
  auto parts = detail::flat::qv_partition(centroids, input_vectors, nthreads);
  {
    scoped_timer _{"index@ivf_pq_index@shuffling_data"};
    std::unordered_set<IdsType> deleted_ids_set(
        deleted_ids.begin(), deleted_ids.end());
    auto indices = compute_indices<
        FeatureType,
        std::span<IdsType>,
        IdsType,
        IndicesType,
        CentroidsType>(external_ids, deleted_ids_set, centroids, parts);

    // Array for storing the shuffled data
    auto shuffled_input_vectors = ColMajorMatrix<FeatureType>{
        input_vectors.num_rows(), input_vectors.num_cols()};
    auto shuffled_pq_input_vectors = ColMajorMatrix<PQFeatureType>{
        input_pq_vectors.num_rows(), input_pq_vectors.num_cols()};
    std::vector shuffled_ids = std::vector<IdsType>(input_vectors.num_cols());

    // @todo parallelize
    // Unfortunately this variant of the algorithm is not parallelizable.
    // The other approach involves doing parallel sort on the indices,
    // which will group them nicely -- but a distributed parallel sort may
    // be difficult to implement.  Even this algorithm is not trivial to
    // parallelize, because of the random access to the indices array.
    if (deleted_ids.empty()) {
      for (size_t i = 0; i < input_vectors.num_cols(); ++i) {
        // First get the centroid that this vector is in.
        size_t bin = parts[i];
        // Then find where in the shuffled data this vector will go.
        size_t ibin = indices[bin];
        if (ibin >= shuffled_input_vectors.num_cols()) {
          throw std::runtime_error(
              "[index@ivf_index] ibin >= shuffled_input_vectors.num_cols()");
        }

        // Copy over the id and the vector.
        shuffled_ids[ibin] = external_ids[i];
        for (size_t j = 0; j < input_vectors.num_rows(); ++j) {
          shuffled_input_vectors(j, ibin) = input_vectors(j, i);
        }
        for (size_t j = 0; j < input_pq_vectors.num_rows(); ++j) {
          shuffled_pq_input_vectors(j, ibin) = input_pq_vectors(j, i);
        }

        // Increment indices so that the next vector in this bin goes to the
        // next spot.
        ++indices[bin];
      }
    } else {
      for (size_t i = 0; i < input_vectors.num_cols(); ++i) {
        if (deleted_ids_set.find(external_ids[i]) == deleted_ids_set.end()) {
          size_t bin = parts[i];
          size_t ibin = indices[bin];
          if (ibin >= shuffled_input_vectors.num_cols()) {
            throw std::runtime_error(
                "[index@ivf_index] ibin >= shuffled_input_vectors.num_cols()");
          }

          shuffled_ids[ibin] = external_ids[i];
          for (size_t j = 0; j < input_vectors.num_rows(); ++j) {
            shuffled_input_vectors(j, ibin) = input_vectors(j, i);
          }
          for (size_t j = 0; j < input_pq_vectors.num_rows(); ++j) {
            shuffled_pq_input_vectors(j, ibin) = input_pq_vectors(j, i);
          }
          ++indices[bin];
        }
      }
    }

    std::shift_right(begin(indices), end(indices), 1);
    indices[0] = 0;

    for (size_t i = 0; i < size(indices); ++i) {
      indices[i] = indices[i] + start_pos;
    }

    // Write out the arrays
    if (!parts_uri.empty()) {
      write_matrix<FeatureType, stdx::layout_left, size_t>(
          ctx,
          shuffled_input_vectors,
          parts_uri,
          start_pos,
          false,
          temporal_policy);
    }
    if (!pq_parts_uri.empty()) {
      write_matrix<PQFeatureType, stdx::layout_left, size_t>(
          ctx,
          shuffled_pq_input_vectors,
          pq_parts_uri,
          start_pos,
          false,
          temporal_policy);
    }
    if (!index_uri.empty()) {
      write_vector(
          ctx, indices, index_uri, partition_start, false, temporal_policy);
    }
    if (!id_uri.empty()) {
      write_vector(
          ctx, shuffled_ids, id_uri, start_pos, false, temporal_policy);
    }
  }
  return 0;
}

/**
 * Partitions a set of vectors, given a set of centroids.
 * @return
 */
template <
    typename FeatureType,
    class IdsType,
    class IndicesType,
    class CentroidsType>
int ivf_index(
    tiledb::Context& ctx,
    const ColMajorMatrix<FeatureType>& input_vectors,  // IN
    const std::vector<IdsType>& external_ids,          // IN
    const std::vector<IdsType>& deleted_ids,           // IN
    const ColMajorMatrix<CentroidsType>& centroids,    // IN
    const std::string& parts_uri,  // OUT (to array at parts_uri)
    const std::string& index_uri,  // OUT (to array at index_uri)
    const std::string& id_uri,     // OUT (to array at id_uri)
    size_t start_pos,
    size_t end_pos,
    size_t nthreads,
    TemporalPolicy temporal_policy,
    size_t partition_start = 0) {
  if (input_vectors.num_cols() != external_ids.size()) {
    throw std::runtime_error(
        "[index@ivf_index] input_vectors.num_cols() != external_ids.size()");
  }
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }

  // Find the centroid that is closest to each input vector.
  auto parts = detail::flat::qv_partition(centroids, input_vectors, nthreads);
  {
    scoped_timer _{"index@ivf_index@shuffling_data"};
    std::unordered_set<IdsType> deleted_ids_set(
        deleted_ids.begin(), deleted_ids.end());

    auto indices = compute_indices<
        FeatureType,
        std::vector<IdsType>,
        IdsType,
        IndicesType,
        CentroidsType>(external_ids, deleted_ids_set, centroids, parts);

    // Array for storing the shuffled data
    auto shuffled_input_vectors = ColMajorMatrix<FeatureType>{
        input_vectors.num_rows(), input_vectors.num_cols()};
    std::vector shuffled_ids = std::vector<IdsType>(input_vectors.num_cols());

    // @todo parallelize
    // Unfortunately this variant of the algorithm is not parallelizable.
    // The other approach involves doing parallel sort on the indices,
    // which will group them nicely -- but a distributed parallel sort may
    // be difficult to implement.  Even this algorithm is not trivial to
    // parallelize, because of the random access to the indices array.
    if (deleted_ids.empty()) {
      for (size_t i = 0; i < input_vectors.num_cols(); ++i) {
        // First get the centroid that this vector is in.
        size_t bin = parts[i];
        // Then find where in the shuffled data this vector will go.
        size_t ibin = indices[bin];
        if (ibin >= shuffled_input_vectors.num_cols()) {
          throw std::runtime_error(
              "[index@ivf_index] ibin >= shuffled_input_vectors.num_cols()");
        }

        // Copy over the id and the vector.
        shuffled_ids[ibin] = external_ids[i];
        for (size_t j = 0; j < input_vectors.num_rows(); ++j) {
          shuffled_input_vectors(j, ibin) = input_vectors(j, i);
        }

        // Increment indices so that the next vector in this bin goes to the
        // next spot.
        ++indices[bin];
      }
    } else {
      for (size_t i = 0; i < input_vectors.num_cols(); ++i) {
        if (deleted_ids_set.find(external_ids[i]) == deleted_ids_set.end()) {
          size_t bin = parts[i];
          size_t ibin = indices[bin];
          if (ibin >= shuffled_input_vectors.num_cols()) {
            throw std::runtime_error(
                "[index@ivf_index] ibin >= shuffled_input_vectors.num_cols()");
          }

          shuffled_ids[ibin] = external_ids[i];
          for (size_t j = 0; j < input_vectors.num_rows(); ++j) {
            shuffled_input_vectors(j, ibin) = input_vectors(j, i);
          }
          ++indices[bin];
        }
      }
    }

    std::shift_right(begin(indices), end(indices), 1);
    indices[0] = 0;

    for (size_t i = 0; i < size(indices); ++i) {
      indices[i] = indices[i] + start_pos;
    }

    // Write out the arrays
    if (!parts_uri.empty()) {
      write_matrix<FeatureType, stdx::layout_left, size_t>(
          ctx,
          shuffled_input_vectors,
          parts_uri,
          start_pos,
          false,
          temporal_policy);
    }
    if (!index_uri.empty()) {
      write_vector(
          ctx, indices, index_uri, partition_start, false, temporal_policy);
    }
    if (!id_uri.empty()) {
      write_vector(
          ctx, shuffled_ids, id_uri, start_pos, false, temporal_policy);
    }
  }
  return 0;
}

/**
 * Partitions a set of vectors, given a set of centroids.
 * @return
 */
template <typename FeatureType, class IdsType, class CentroidsType>
int ivf_index(
    tiledb::Context& ctx,
    const ColMajorMatrix<FeatureType>& input_vectors,  // IN
    const std::vector<IdsType>& external_ids,          // IN
    const std::vector<IdsType>& deleted_ids,           // IN
    const std::string& centroids_uri,  // IN (from array centroids_uri)
    const std::string& parts_uri,      // OUT (to array at parts_uri)
    const std::string& index_uri,      // OUT (to array at index_uri)
    const std::string& id_uri,         // OUT (to array at id_uri)
    size_t start_pos,
    size_t end_pos,
    size_t nthreads,
    uint64_t timestamp,
    size_t partition_start = 0) {
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
  auto centroids = tdbColMajorMatrix<CentroidsType>(
      ctx,
      centroids_uri,
      0,
      std::nullopt,
      0,
      partitions,
      0,
      centroid_read_temporal_policy);
  centroids.load();

  TemporalPolicy temporal_policy = (timestamp == 0) ?
                                       TemporalPolicy() :
                                       TemporalPolicy(TimeTravel, timestamp);

  return ivf_index<FeatureType, IdsType, IdsType, CentroidsType>(
      ctx,
      input_vectors,
      external_ids,
      deleted_ids,
      centroids,
      parts_uri,
      index_uri,
      id_uri,
      start_pos,
      end_pos,
      nthreads,
      temporal_policy,
      partition_start);
}

/**
 * Open db and set up external ids to either be a contiguous set of integers
 * (i.e., the index of the vector in the db), or read from an external array.
 * Call the main ivf_index function above.
 */
template <typename FeatureType, class IdsType, class CentroidsType>
int ivf_index(
    tiledb::Context& ctx,
    const std::string& input_vectors_uri,
    const std::string& external_ids_uri,
    const std::vector<IdsType>& deleted_ids,
    const std::string& centroids_uri,
    const std::string& parts_uri,
    const std::string& index_uri,
    const std::string& id_uri,
    size_t start_pos = 0,
    size_t end_pos = 0,
    size_t nthreads = 0,
    uint64_t timestamp = 0,
    size_t partition_start = 0) {
  TemporalPolicy temporal_policy = (timestamp == 0) ?
                                       TemporalPolicy() :
                                       TemporalPolicy(TimeTravel, timestamp);
  // Read all rows from column `start_pos` -> `end_pos`. Set no upper_bound.
  auto input_vectors = tdbColMajorMatrix<FeatureType>(
      ctx,
      input_vectors_uri,
      0,
      std::nullopt,
      start_pos,
      end_pos,
      0,
      temporal_policy);
  input_vectors.load();
  std::vector<IdsType> external_ids;
  if (external_ids_uri.empty()) {
    external_ids = std::vector<IdsType>(input_vectors.num_cols());
    std::iota(begin(external_ids), end(external_ids), start_pos);
  } else {
    external_ids = read_vector<IdsType>(
        ctx, external_ids_uri, start_pos, end_pos, temporal_policy);
  }
  return ivf_index<FeatureType, IdsType, CentroidsType>(
      ctx,
      input_vectors,
      external_ids,
      deleted_ids,
      centroids_uri,
      parts_uri,
      index_uri,
      id_uri,
      start_pos,
      end_pos,
      nthreads,
      timestamp,
      partition_start);
}

/**
 * Open db and call main ivf_index function above.
 */
template <typename FeatureType, class IdsType, class CentroidsType>
int ivf_index(
    tiledb::Context& ctx,
    const std::string& input_vectors_uri,
    const std::vector<IdsType>& external_ids,
    const std::vector<IdsType>& deleted_ids,
    const std::string& centroids_uri,
    const std::string& parts_uri,
    const std::string& index_uri,
    const std::string& id_uri,
    size_t start_pos = 0,
    size_t end_pos = 0,
    size_t nthreads = 0,
    uint64_t timestamp = 0,
    size_t partition_start = 0) {
  // Read all rows from column `start_pos` -> `end_pos`. Set no upper_bound.
  auto input_vectors = tdbColMajorMatrix<FeatureType>(
      ctx,
      input_vectors_uri,
      0,
      std::nullopt,
      start_pos,
      end_pos,
      0,
      timestamp);
  input_vectors.load();
  return ivf_index<FeatureType, IdsType, CentroidsType>(
      ctx,
      input_vectors,
      external_ids,
      deleted_ids,
      centroids_uri,
      parts_uri,
      index_uri,
      id_uri,
      start_pos,
      end_pos,
      nthreads,
      timestamp,
      partition_start);
}

/*
 * Set up external ids to be either the indices of the vectors in the
 * input_vectors, or read from an external array.  Call the main ivf_index
 * function above.
 */
template <typename FeatureType, class IdsType, class CentroidsType>
int ivf_index(
    tiledb::Context& ctx,
    const ColMajorMatrix<FeatureType>& input_vectors,
    const std::string& external_ids_uri,
    const std::vector<IdsType>& deleted_ids,
    const std::string& centroids_uri,
    const std::string& parts_uri,
    const std::string& index_uri,
    const std::string& id_uri,
    size_t start_pos = 0,
    size_t end_pos = 0,
    size_t nthreads = 0,
    uint64_t timestamp = 0,
    size_t partition_start = 0) {
  std::vector<IdsType> external_ids;
  if (external_ids_uri.empty()) {
    external_ids = std::vector<IdsType>(input_vectors.num_cols());
    std::iota(begin(external_ids), end(external_ids), start_pos);
  } else {
    external_ids = read_vector<IdsType>(
        ctx, external_ids_uri, start_pos, end_pos, timestamp);
  }
  return ivf_index<FeatureType, IdsType, CentroidsType>(
      ctx,
      input_vectors,
      external_ids,
      deleted_ids,
      centroids_uri,
      parts_uri,
      index_uri,
      id_uri,
      start_pos,
      end_pos,
      nthreads,
      timestamp,
      partition_start);
}

}  // namespace detail::ivf

#endif  // TILEDB_IVF_IVF_INDEX_H
