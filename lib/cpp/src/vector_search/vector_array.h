/**
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023-2023 TileDB, Inc.
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
 */

#ifndef TILEDB_VECTOR_SEARCH_VECTOR_ARRAY_H
#define TILEDB_VECTOR_SEARCH_VECTOR_ARRAY_H

#include <iostream>
#include <tiledb/tiledb>

namespace tiledb::vector_search {

using namespace tiledb;

/* **************************** */
/*         VECTOR ARRAY         */
/* **************************** */

/**
 */
class VectorArray {
 
 public:
  static VectorArray* open(const Context& ctx,
      const std::string& array_uri,
      tiledb_query_type_t query_type);
  
  VectorArray(const Context& ctx,
      const std::string& array_uri,
      tiledb_query_type_t query_type,
      const std::string& array_type): 
      ctx_(ctx)
      , array_uri_(array_uri)
      , array_type_(array_type)
      , query_type_(query_type){}

  virtual std::vector<std::vector<uint8_t>> read_vector_partition(int partition_id) = 0;
  virtual std::vector<std::vector<float>> get_centroids() = 0;

 protected:
  /* ********************************* */
  /*         PRIVATE ATTRIBUTES        */
  /* ********************************* */

  /** The TileDB context. */
  std::reference_wrapper<const Context> ctx_;
  const std::string array_uri_;
  const std::string array_type_;
  const tiledb_query_type_t query_type_;
  int num_vector_dim_;
  int num_partitions_;

 private:
  static std::string readArrayTypeFromMetadata(Array* index_array);
  static int readArrayTypeVersionFromMetadata(Array* index_array);
};

}  // namespace tiledb::vector_search

#endif  // TILEDB_VECTOR_SEARCH_VECTOR_ARRAY_H