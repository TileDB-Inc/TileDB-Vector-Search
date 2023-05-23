#include <cstdlib>
#include <iostream>
#include <tiledb/tiledb>
#include "vector_search/vector_array.h"

using namespace tiledb;
using namespace tiledb::vector_search;

// Name of array.
std::string array_name("s3://tiledb-nikos/vector-search/sift-10m-1000p");

int main(int argc, char* argv[]) {
  // Create a TileDB context.
  Config config;
  config["vfs.s3.aws_access_key_id"] = std::string(std::getenv("AWS_ACCESS_KEY_ID"));
  config["vfs.s3.aws_secret_access_key"] = std::string(std::getenv("AWS_SECRET_ACCESS_KEY"));
  Context ctx(config);

  std::string sparse_array_name = array_name+"/sparse";
  std::cout << "Sparse Array: " << sparse_array_name << std::endl;
  auto a_sparse = VectorArray::open(ctx, sparse_array_name, TILEDB_READ);
  auto sparse_centroids = a_sparse->get_centroids();
  for (int i = 0; i < 128; i++) {
    std::cout << sparse_centroids[0][i] << " ";
  }
  std::cout << std::endl;
  auto sparse_partition = a_sparse->read_vector_partition(42);
  std::cout << "Partition size:" << sparse_partition.size() << std::endl;
  for (int j = 0; j < 5; j++) {
    std::cout << "Vector:" << j << std::endl;
    for (int i = 0; i < 128; i++) {
      std::cout << int(sparse_partition[j][i]) << " ";
    }
    std::cout << std::endl;
  }

  std::string dense_array_name = array_name+"/dense";
  std::cout << "Dense Array: " << dense_array_name << std::endl;
  auto a_dense = VectorArray::open(ctx, dense_array_name, TILEDB_READ);
  auto dense_centroids = a_dense->get_centroids();
  for (int i = 0; i < 128; i++) {
    std::cout << dense_centroids[0][i] << " ";
  }
  std::cout << std::endl;
  auto dense_partition = a_dense->read_vector_partition(42);
  std::cout << "Partition size:" << dense_partition.size() << std::endl;
  for (int j = 0; j < 5; j++) {
    std::cout << "Vector:" << j << std::endl;
    for (int i = 0; i < 128; i++) {
      std::cout << int(dense_partition[j][i]) << " ";
    }
    std::cout << std::endl;
  }
}