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
 * Some useful definitions for working with TileDB indexes.
 *
 */
#ifndef TDB_INDEX_DEFS_H
#define TDB_INDEX_DEFS_H

#include <tiledb/type.h>
#include <string>

enum class IndexKind { FlatL2, IVFFlat, FlatPQ, IVFPQ, NNDescent, Vamana };

template <tiledb_datatype_t T>
using tiledb_to_type_t = typename tiledb::impl::tiledb_to_type<T>::type;

template <class T>
constexpr tiledb_datatype_t type_to_tiledb_t =
    tiledb::impl::type_to_tiledb<T>::tiledb_type;

tiledb_datatype_t string_to_datatype(const std::string& str) {
  if (str == "float32") {
    return TILEDB_FLOAT32;
  }
  if (str == "uint8") {
    return TILEDB_UINT8;
  }
  if (str == "int32") {
    return TILEDB_INT32;
  }
  if (str == "uint32") {
    return TILEDB_UINT32;
  }
  if (str == "int64") {
    return TILEDB_INT64;
  }
  if (str == "uint64") {
    return TILEDB_UINT64;
  }
  throw std::runtime_error("Unsupported datatype");
}

std::string datatype_to_string(tiledb_datatype_t datatype) {
  switch (datatype) {
    case TILEDB_FLOAT32:
      return "float32";
    case TILEDB_UINT8:
      return "uint8";
    case TILEDB_INT32:
      return "int32";
    case TILEDB_UINT32:
      return "uint32";
    case TILEDB_INT64:
      return "int64";
    case TILEDB_UINT64:
      return "uint64";
    default:
      throw std::runtime_error("Unsupported datatype");
  }
}

#endif  // TDB_INDEX_DEFS_H
