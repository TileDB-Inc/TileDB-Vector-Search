/**
 * @file   ivf_flat_metadata.h
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
 */

#ifndef TILEDB_IVF_FLAT_METADATA_H
#define TILEDB_IVF_FLAT_METADATA_H

#include <tiledb/group_experimental.h>
#include <tiledb/tiledb>

#include "index/index_defs.h"
#include "index/ivf_flat_index.h"

class ivf_flat_index_metadata {

 private:

  using base_size_type = uint32_t;
  using ingestion_timestamp_type = uint32_t;
  using partition_history_type = uint32_t;

  std::vector<base_size_type> base_sizes_;
  std::string dataset_type_{"vector_search"};
  std::string dtype_{""};
  std::string index_type_{"IVF_FLAT"};
  std::string ingestion_timestamps_{""};
  std::string partition_history_{""};
  std::string storage_version_{current_storage_version};
  double temp_size_{0};   // @todo ???

   /*
    * Group Metadata:
    *
    "base_sizes",            // (json) list
    "dataset_type",          // "vector_search"
    "dtype",                 // "float32", etc (Python dtype names)
    "index_type",            // "FLAT", "IVF_FLAT"
    "ingestion_timestamps",  // (json) list
    "partition_history",     // (json) list
    "storage_version",       // "0.3"
    "temp_size",
    */

  public:
   ivf_flat_index_metadata() = default;




};



#endif // TILEDB_IVF_FLAT_METADATA_H