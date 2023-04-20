//
// Created by Andrew Lumsdaine on 4/19/23.
//

#ifndef TDB_SIFT_ARRAY_H
#define TDB_SIFT_ARRAY_H

#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <span>
#include <vector>
#include <unistd.h>

#include <tiledb/tiledb>

template <class T>
class sift_array : public std::vector<std::span<T>> {

  using Base = std::vector<std::span<T>>;
  tiledb::Context ctx_;
  std::unique_ptr<T> data_;

public:
  sift_array(const std::string& array_name, size_t dimension) {

    tiledb::Array array(ctx_, array_name, TILEDB_READ);
    auto schema = array.schema();

    // Or, get schema directly from storage
    // tiledb::ArraySchema schema(ctx_, "<array-uri>");

    // Get the array domain -- "rows" and "cols" are int32
    auto domain = schema.domain();

    tiledb_datatype_t type = domain.type();
    assert(type == TILEDB_INT32);

    // Get number of dimensions
    uint32_t dim_num = domain.ndim();
    assert(dim_num == 2);

    // Get dimension from name
    tiledb::Dimension rows = domain.dimension("rows");
    tiledb::Dimension cols = domain.dimension("cols");

    auto num_rows = rows.domain<uint32_t>().second;
    auto num_cols = cols.domain<uint32_t>().second;

#ifndef __APPLE__
    data_ = std::make_unique_for_overwrite<float[]>(num_rows * num_cols);
#else
    data_ = std::make_unique<float[]>(num_rows * num_cols);
#endif

    this->resize(num_cols);

    for (size_t j = 0; j < num_cols; ++j) {
      Base::operator[](j) = std::span<float>(data_.get() + j * num_rows, num_rows);
    }

    std::vector<unsigned> subarray = {0, num_rows-1, 0, num_cols-1};

    // Allocate query and set subarray.
    tiledb::Query query(ctx_, array);
    query.set_layout(TILEDB_COL_MAJOR)
            .set_data_buffer("a", data_.get());
    // Read from the array.
    query.submit();
    array.close();
    assert(tiledb::Query::Status::COMPLETE == query.query_status());
  }
};



#endif//TDB_SIFT_ARRAY_H
