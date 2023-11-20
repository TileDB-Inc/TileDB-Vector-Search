/**
 * @file   index/index_defs.h
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

#include <map>
#include <string>
#include <tiledb/tiledb>
#include <tuple>

/******************************************************************************
 * Static info for index kinds
 ******************************************************************************/

enum class IndexKind {
  FlatL2,
  IVFFlat,
  FlatPQ,
  IVFPQ,
  Vamana,
  VamanaPQ,
  NNDescent,
  Last
};

static std::vector<const std::string> index_kind_strings{
    "FlatL2",
    "IVFFlat",
    "FlatPQ",
    "IVFPQ",
    "Vamana",
    "VamanaPQ",
    "NNDescent",
    "Last"};

constexpr static inline auto str(IndexKind kind) {
  return index_kind_strings[static_cast<int>(kind)];
}

/******************************************************************************
 * Static info for arrays associated with an index group
 ******************************************************************************/

// @todo C++20 should have constexpr std::string
static std::string current_storage_version{"0.3"};

using StorageFormat = std::map<std::string, std::map<std::string, std::string>>;
StorageFormat storage_formats = {
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

static const std::map<std::string, std::vector<std::string>>
    ivf_flat_index_metadata_fields{
        {"0.3",
         {
             // From Python ivf_flat_index.py
             "dataset_type",          // "vector_search"
             "dtype",                 // "float32", etc (Python dtype names)
             "storage_version",       // "0.3"
             "index_type",            // "FLAT", "IVF_FLAT"
             "base_sizes",            // (json) list
             "ingestion_timestamps",  // (json) list

             // From C++ ivf_flat_index -- these are constant for a given index
             "feature_datatype",      // tiledb_datatype_t
             "id_datatype",           // tiledb_datatype_t
             "px_datatype",           // tiledb_datatype_t
             "dimension",             // uint64_t

             // From C++ ivf_flat_index -- this needs to be timestamped
             "num_partitions",        // uint64_t @todo change to (json) list?

             // These all have to do with individual instantiations of the
             // index and might not really belong in the metadata -- should
             // probably keep with a timestamp if kept?
             "max_iter",              // uint64_t
             "tol",                   // float
             "reassign_ratio",        // float
             "num_threads",           // float
             // "seed", &seed_, TILEDB_UINT64},
             // "timestamp", &timestamp_, TILEDB_UINT64},
         }}};

// convert tiledb dataypes into python name strings with datatype_to_string
/******************************************************************************
 * Type translation functions
 ******************************************************************************/

template <tiledb_datatype_t T>
using tiledb_to_type_t = typename tiledb::impl::tiledb_to_type<T>::type;

template <class T>
constexpr tiledb_datatype_t type_to_tiledb_t =
    tiledb::impl::type_to_tiledb<T>::tiledb_type;

#endif  // TILEDB_INDEX_DEFS_H