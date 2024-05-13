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

template <class T>
constexpr auto type_to_tiledb_v = tiledb::impl::type_to_tiledb<T>::tiledb_type;

[[maybe_unused]] static auto get_array_datatype(const tiledb::Array& array) {
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
[[maybe_unused]] static tiledb_datatype_t string_to_datatype(
    const std::string& str) {
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
  throw std::runtime_error("Unsupported datatype: " + str);
}

// cf type_to_str(tiledb_datatype_t type) in tiledb/type.h
// type_to_tiledb<T>::tiledb_type
// tiledb_to_type<tiledb_datatype_t>::type
[[maybe_unused]] inline static std::string datatype_to_string(
    tiledb_datatype_t datatype) {
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
    case TILEDB_ANY:
      return "any";
    default:
      throw std::runtime_error("Unsupported datatype");
  }
}

[[maybe_unused]] constexpr inline static size_t datatype_to_size(
    tiledb_datatype_t datatype) {
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

[[maybe_unused]] inline static auto string_to_filter(const std::string& str) {
  if (str == "gzip") {
    return TILEDB_FILTER_GZIP;
  }
  if (str == "zstd") {
    return TILEDB_FILTER_ZSTD;
  }
  if (str == "lz4") {
    return TILEDB_FILTER_LZ4;
  }
  if (str == "rle") {
    return TILEDB_FILTER_RLE;
  }
  if (str == "bzip2") {
    return TILEDB_FILTER_BZIP2;
  }
  if (str == "double-delta") {
    return TILEDB_FILTER_DOUBLE_DELTA;
  }
  throw std::runtime_error("Unsupported filter name " + str);
}

[[maybe_unused]] static inline std::string filter_to_string(
    tiledb_filter_type_t filter) {
  switch (filter) {
    case TILEDB_FILTER_NONE:
      return "none";
    case TILEDB_FILTER_GZIP:
      return "gzip";
    case TILEDB_FILTER_ZSTD:
      return "zstd";
    case TILEDB_FILTER_LZ4:
      return "lz4";
    case TILEDB_FILTER_RLE:
      return "rle";
    case TILEDB_FILTER_BZIP2:
      return "bzip2";
    case TILEDB_FILTER_DOUBLE_DELTA:
      return "double-delta";
    default:
      throw std::runtime_error("Unsupported filter");
  }
}

template <class T>
struct type_to_string {
  static constexpr auto value = "unknown";
};

template <>
struct type_to_string<float> {
  static constexpr auto value = "float32";
};

template <>
struct type_to_string<double> {
  static constexpr auto value = "float64";
};

template <>
struct type_to_string<int8_t> {
  static constexpr auto value = "int8";
};

template <>
struct type_to_string<uint8_t> {
  static constexpr auto value = "uint8";
};

template <>
struct type_to_string<int32_t> {
  static constexpr auto value = "int32";
};

template <>
struct type_to_string<uint32_t> {
  static constexpr auto value = "uint32";
};

template <>
struct type_to_string<int64_t> {
  static constexpr auto value = "int64";
};

template <>
struct type_to_string<uint64_t> {
  static constexpr auto value = "uint64";
};

template <class T>
inline constexpr auto type_to_string_v = type_to_string<T>::value;

#endif  // TILEDB_TDB_DEFS_H
