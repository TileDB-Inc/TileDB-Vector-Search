
#include <string>
#include <tiledb/tiledb>
#include "utils/timer.h"

void open_array(const std::string& uri) {
  scoped_timer _{"open_array " + uri};

  tiledb::Context ctx;
  tiledb::Array array(ctx, uri, TILEDB_READ);

  scoped_timer _2{"get_schema portion"};
  tiledb::ArraySchema schema = array.schema();
}


int main() {
  for (const std::string& s : {
      "s3://tiledb-nikos/vector-search/datasets/arrays/sift-1b-col-major",
      "s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb",
      "s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb",
      "/home/lums/feature-vector-prototype/experimental/external/data/gp3/1B/sift-1b-col-major",
      "/home/lums/feature-vector-prototype/experimental/external/data/gp3/1B/centroids.tdb",
      "/home/lums/feature-vector-prototype/experimental/external/data/gp3/1B/parts.tdb"
       }) {
    open_array(s);
  }
}
