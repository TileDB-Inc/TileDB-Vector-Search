/**
 * @file   index_defs.h
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
 * Some info shared across index implementations.
 *
 */

#ifndef TILEDB_INDEX_DEFS_H
#define TILEDB_INDEX_DEFS_H

#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <tiledb/tiledb>

/******************************************************************************
 * Static info for index kinds
 ******************************************************************************/

enum class IndexKind { FlatL2, IVFFlat, Vamana, IVFPQ };

[[maybe_unused]] static std::vector<std::string> index_kind_strings{
    "FLAT", "IVF_FLAT", "VAMANA", "IVF_PQ"};

[[maybe_unused]] static inline auto str(IndexKind kind) {
  return index_kind_strings[static_cast<int>(kind)];
}

enum class QueryType { FiniteRAM, InfiniteRAM };

/******************************************************************************
 * Static info for arrays associated with an index group
 ******************************************************************************/

// @todo C++20 should have constexpr std::string
[[maybe_unused]] static std::string current_storage_version{"0.3"};

// @todo Use enum for key rather than string?
using StorageFormat =
    std::map<std::string, std::unordered_map<std::string, std::string>>;
[[maybe_unused]] static StorageFormat storage_formats = {
    {"0.1",
     {
         {"centroids_array_name", "centroids.tdb"},
         {"index_array_name", "index.tdb"},
         {"ids_array_name", "ids.tdb"},
         {"parts_array_name", "parts.tdb"},
         {"input_vectors_array_name", "input_vectors"},
         {"external_ids_array_name", "external_ids"},
         {"partial_write_array_dir", "write_temp"},
         {"default_attr_filters", ""},
         {"updates_array_name", "updates"},
         {"support_timetravel", "false"},
     }},
    {"0.2",
     {
         {"centroids_array_name", "partition_centroids"},
         {"index_array_name", "partition_indexes"},
         {"ids_array_name", "shuffled_vector_ids"},
         {"parts_array_name", "shuffled_vectors"},
         {"input_vectors_array_name", "input_vectors"},
         {"external_ids_array_name", "external_ids"},
         {"partial_write_array_dir", "temp_data"},
         {"default_attr_filters", "zstd"},
         {"updates_array_name", "updates"},
         {"support_timetravel", "false"},
     }},
    {"0.3",
     {
         {"centroids_array_name", "partition_centroids"},
         {"index_array_name", "partition_indexes"},
         {"ids_array_name", "shuffled_vector_ids"},
         {"parts_array_name", "shuffled_vectors"},
         {"input_vectors_array_name", "input_vectors"},
         {"external_ids_array_name", "external_ids"},
         {"partial_write_array_dir", "temp_data"},
         {"default_attr_filters", "zstd"},
         {"updates_array_name", "updates"},
         {"support_timetravel", "true"},
     }},
};

/******************************************************************************
 * Type translation functions
 ******************************************************************************/

template <tiledb_datatype_t T>
using tiledb_to_type_t = typename tiledb::impl::tiledb_to_type<T>::type;

template <class T>
constexpr tiledb_datatype_t type_to_tiledb_t =
    tiledb::impl::type_to_tiledb<T>::tiledb_type;

#endif  // TILEDB_INDEX_DEFS_H
