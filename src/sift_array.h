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

  std::map<std::string, std::string> init_ {{ "vfs.s3.region", "us-west-2" }};
  tiledb::Config config_ {init_};
  tiledb::Context ctx_ {config_};

  tiledb::Array array_;
  tiledb::ArraySchema schema_;
  tiledb::Domain domain_;
  tiledb::Dimension rows_;
  tiledb::Dimension cols_;
  uint32_t dim_num_;
  int32_t num_rows_;
  int32_t num_cols_;

  std::unique_ptr<T[]> data_;

  public:
  sift_array(const std::string& array_name, size_t subset = 0)
          : array_(ctx_, array_name, TILEDB_READ)
          , schema_(array_.schema())
          , domain_(schema_.domain())
          , rows_(domain_.dimension("rows"))
          , cols_(domain_.dimension("cols"))
          , dim_num_(domain_.ndim())
          , num_rows_(rows_.domain<int32_t>().second - rows_.domain<int32_t>().first + 1)
          , num_cols_(subset == 0 ? (cols_.domain<int32_t>().second - cols_.domain<int32_t>().first + 1) : subset)

#ifndef __APPLE__
	  , data_ { std::make_unique_for_overwrite<T[]>(num_rows_ * num_cols_) }
#else
  //, data_ {std::make_unique<T[]>(new T[num_rows_ * num_cols_])}
  , data_ {new T[num_rows_ * num_cols_]}
#endif
  {
    ctx_.set_tag("vfs.s3.region", "us-west-2");

    this->resize(num_cols_);

    for (decltype(num_cols_) j = 0; j < num_cols_; ++j) {
      Base::operator[](j) = std::span<T>(data_.get() + j * num_rows_, num_rows_);
    }

    std::vector<int> subarray_vals = {0, num_rows_-1, 0, num_cols_-1};
    tiledb::Subarray subarray(ctx_, array_);
    subarray.set_subarray(subarray_vals);

    // Allocate query and set subarray.
    tiledb::Query query(ctx_, array_);
    query.set_subarray(subarray)
            .set_layout(TILEDB_COL_MAJOR)
            .set_data_buffer("a", data_.get(), num_rows_ * num_cols_);
    // Read from the array.
    query.submit();
    array_.close();
    assert(tiledb::Query::Status::COMPLETE == query.query_status());
  }
};



#endif//TDB_SIFT_ARRAY_H
