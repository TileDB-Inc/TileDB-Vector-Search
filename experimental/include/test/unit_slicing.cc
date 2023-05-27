

#include <catch2/catch_all.hpp>
#include "../linalg.h"

std::string global_region = "us-east-1";

TEST_CASE("slice", "[linalg]") {

  std::string uri = "s3://tiledb-andrew/sift/sift_base";

  std::map<std::string, std::string> init_{
      {"vfs.s3.region", global_region.c_str()}};
  tiledb::Config config_{init_};;
  tiledb::Context ctx_{config_};;

  std::vector<int> data_(288);
  std::vector<int> data2_(288);
  std::vector<float> value_(288);

  tiledb::Array array_{ctx_, uri, TILEDB_READ};
  tiledb::ArraySchema schema_{array_.schema()};
  tiledb::Query query(ctx_, array_);

  tiledb::Subarray subarray(ctx_, array_);
  subarray.add_range(0, 0, 5)
      .add_range(1, 88, 100)
  .add_range(0, 10, 13);

//      .add_range(1, col_0_start, col_0_end);
  query.set_subarray(subarray);

  query.set_subarray(subarray)
      .set_layout(TILEDB_COL_MAJOR)
      .set_data_buffer("cols", data2_.data(), 288)
      .set_data_buffer("rows", data_.data(), 288)
      .set_data_buffer("a", value_.data(), 288);

    query.submit();

  for (int i = 0; i < 135; i++) {
    std::cout << data_[i] << ", " << data2_[i] << ": " << value_[i] <<std::endl;
  }

}

