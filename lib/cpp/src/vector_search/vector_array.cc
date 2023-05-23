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

#include "vector_array.h"
#include "dense_vector_array.h"
#include "sparse_vector_array.h"

namespace tiledb::vector_search {

const std::string INDEX_ARRAY_NAME = "/index";

using namespace tiledb;

VectorArray* VectorArray::open(
      const Context& ctx,
      const std::string& array_uri,
      tiledb_query_type_t query_type) {
    Array index_array(ctx, array_uri + INDEX_ARRAY_NAME, TILEDB_READ);
    std::string array_type = readArrayTypeFromMetadata(&index_array);
    int array_type_version = readArrayTypeVersionFromMetadata(&index_array);
    index_array.close();
    if (array_type == "DENSE") {
        return new DenseVectorArray(ctx, array_uri, query_type, array_type, array_type_version);
    } else if (array_type == "SPARSE" ) {
        return new SparseVectorArray(ctx, array_uri, query_type, array_type, array_type_version);
    } 
    throw TileDBError(
        "[TileDB::VectorArray] Error: Unsopported array type " + array_type);
}

std::string VectorArray::readArrayTypeFromMetadata(Array* index_array){
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;
    index_array->get_metadata("type", &v_type, &v_num, &v);
    if (v_type != TILEDB_STRING_ASCII){
      throw TileDBError(
          "[TileDB::VectorArray] Error: Failed to read vector array type from array metadata");
    }
    return (const char*)v;
}

int VectorArray::readArrayTypeVersionFromMetadata(Array* index_array){
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;
    index_array->get_metadata("version", &v_type, &v_num, &v);
    if (v_type != TILEDB_INT32){
      throw TileDBError(
          "[TileDB::VectorArray] Error: Failed to read vector array version from array metadata");
    }
    return *(const int*)v;
}

}   // namespace tiledb::vector_search
