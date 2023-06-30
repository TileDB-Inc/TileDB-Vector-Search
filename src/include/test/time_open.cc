
#include <string>
#include <tiledb/tiledb>
#include "detail/linalg/tdb_helpers.h"
#include "utils/timer.h"

void open_array(const std::string& uri) {
  scoped_timer _{"open_array " + uri};

  tiledb::Context ctx;
  tiledb::Array array =
      tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);

  scoped_timer _2{"get_schema portion"};
  tiledb::ArraySchema schema = array.schema();
}

int main() {
  for (
      const std::string& s :
      {std::string(
           "s3://tiledb-nikos/vector-search/datasets/arrays/sift-1b-col-major"),
       std::string("s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/"
                   "centroids.tdb"),
       std::string("s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/"
                   "parts.tdb"),
       std::string("s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/"
                   "ids.tdb"),
       std::string("/home/lums/TileDB-Vector-Search/external/data/gp3/1B/"
                   "sift-1b-col-major"),
       std::string("/home/lums/TileDB-Vector-Search/external/data/gp3/1B/"
                   "centroids.tdb"),
       std::string(
           "/home/lums/TileDB-Vector-Search/external/data/gp3/1B/parts.tdb")}) {
    open_array(s);
  }
}
