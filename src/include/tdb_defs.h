/**
 * @file   tdb_defs.h
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

#ifndef TILEDB_TDB_DEFS_H
#define TILEDB_TDB_DEFS_H

#include <string>
#include <tiledb/tiledb>

template <class... T>
constexpr bool always_false = false;

auto get_array_datatype(const tiledb::Array& array) {
  auto schema = array.schema();
  auto num_attributes = schema.attribute_num();
  if (num_attributes == 1) {
    return schema.attribute(0).type();
  }
  if (schema.has_attribute("values")) {
    return schema.attribute("values").type();
  }
  if (schema.has_attribute("a")) {
    return schema.attribute("a").type();
  }
  throw std::runtime_error("Could not determine datatype of array attributes");
}

// @todo Implement this with a map
tiledb_datatype_t string_to_datatype(const std::string& str) {
  if (str == "float32") {
    return TILEDB_FLOAT32;
  }
  if (str == "float64") {
    return TILEDB_FLOAT64;
  }
  if (str == "int8") {
    return TILEDB_INT8;
  }
  if (str == "uint8") {
    return TILEDB_UINT8;
  }
  if (str == "int16") {
    return TILEDB_INT8;
  }
  if (str == "uint16") {
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
    case TILEDB_FLOAT64:
      return "float64";
    case TILEDB_INT8:
      return "int8";
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

size_t datatype_to_size(tiledb_datatype_t datatype) {
        switch (datatype) {
        case TILEDB_FLOAT32:
        return sizeof(float);
        case TILEDB_FLOAT64:
        return sizeof(double);
        case TILEDB_INT8:
        return sizeof(int8_t);
        case TILEDB_UINT8:
        return sizeof(uint8_t);
        case TILEDB_INT32:
        return sizeof(int32_t);
        case TILEDB_UINT32:
        return sizeof(uint32_t);
        case TILEDB_INT64:
        return sizeof(int64_t);
        case TILEDB_UINT64:
        return sizeof(uint64_t);
        default:
        throw std::runtime_error("Unsupported datatype");
        }
}

#endif  // TILEDB_TDB_DEFS_H
