
#include "detail/linalg/tdb_io.h"

#include <highfive/H5File.hpp>
#include "detail/linalg/matrix.h"

using namespace HighFive;
bool global_debug = false;

template <class T>
auto to_matrix(const std::vector<std::vector<T>>& v) {
  size_t cols = v.size();
  size_t rows = v[0].size();
  auto m = ColMajorMatrix<T>(rows, cols);
  for (size_t i = 0; i < cols; ++i) {
    for (size_t j = 0; j < rows; ++j) {
      m(j, i) = v[i][j];
    }
  }
  return m;
}

int main() {
  std::string filename =
      "/Users/lums/TileDB/TileDB-Vector-Search/external/pynndescent/"
      "pynndescent/fashion-mnist-784-euclidean.hdf5";

  // fname =
  // '/Users/lums/TileDB/TileDB-Vector-Search/external/pynndescent/pynndescent/fashion-mnist-784-euclidean.hdf5'
  //  ['distances', 'neighbors', 'test', 'train'] float int32 float float

  {
    File file(filename, File::ReadOnly);
    tiledb::Context ctx;
    auto dist_set = file.getDataSet("distances");
    auto dist_data = dist_set.read<std::vector<std::vector<float>>>();
    auto dist_mat = to_matrix(dist_data);
    write_matrix(ctx, dist_mat, "fmnist_distances.tdb");

    auto nbr_set = file.getDataSet("neighbors");
    auto nbr_data = nbr_set.read<std::vector<std::vector<int>>>();
    auto nbr_mat = to_matrix(nbr_data);
    write_matrix(ctx, nbr_mat, "fmnist_neighbors.tdb");

    auto test_set = file.getDataSet("test");
    auto test_data = test_set.read<std::vector<std::vector<float>>>();
    auto test_mat = to_matrix(test_data);
    write_matrix(ctx, test_mat, "fmnist_test.tdb");

    auto train_set = file.getDataSet("train");
    auto train_data = train_set.read<std::vector<std::vector<float>>>();
    auto train_mat = to_matrix(train_data);
    write_matrix(ctx, train_mat, "fmnist_train.tdb");

    // Because `pre_allocated` has the correct size, this will
    // not cause `pre_allocated` to be reallocated:
    //    auto pre_allocated = std::vector<int>(50);
    //    dataset.read(pre_allocated);
  }

  return 0;
}