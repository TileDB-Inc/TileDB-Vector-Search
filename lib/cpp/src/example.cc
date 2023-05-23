#include <chrono>
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

  std::chrono::steady_clock::time_point sparse_centroids_begin = std::chrono::steady_clock::now();
  auto a_sparse = VectorArray::open(ctx, sparse_array_name, TILEDB_READ);
  auto sparse_centroids = a_sparse->get_centroids();
  std::chrono::steady_clock::time_point sparse_centroids_end = std::chrono::steady_clock::now();
  std::cout << "Centroids fetch: " << std::chrono::duration_cast<std::chrono::milliseconds>(sparse_centroids_end - sparse_centroids_begin).count() << "[ms]" << std::endl;
  
  std::chrono::steady_clock::time_point sparse_partition_begin = std::chrono::steady_clock::now();
  auto sparse_partition = a_sparse->read_vector_partition(42);
  std::chrono::steady_clock::time_point sparse_partition_end = std::chrono::steady_clock::now();
  std::cout << "Partition fetch: " << std::chrono::duration_cast<std::chrono::milliseconds>(sparse_partition_end - sparse_partition_begin).count() << "[ms]" << std::endl;
  
  std::chrono::steady_clock::time_point sparse_ann_begin = std::chrono::steady_clock::now();
  auto ann_results_sparse = a_sparse->ann_query(sparse_partition[0],10,2,4);  
  std::chrono::steady_clock::time_point sparse_ann_end = std::chrono::steady_clock::now();
  std::cout << "Sparse ANN: " << std::chrono::duration_cast<std::chrono::milliseconds>(sparse_ann_end - sparse_ann_begin).count() << "[ms]" << std::endl;


  std::string dense_array_name = array_name+"/dense";
  std::cout << "Dense Array: " << dense_array_name << std::endl;

  std::chrono::steady_clock::time_point dense_centroids_begin = std::chrono::steady_clock::now();
  auto a_dense = VectorArray::open(ctx, dense_array_name, TILEDB_READ);
  auto dense_centroids = a_dense->get_centroids();
  std::chrono::steady_clock::time_point dense_centroids_end = std::chrono::steady_clock::now();
  std::cout << "Centroids fetch: " << std::chrono::duration_cast<std::chrono::milliseconds>(dense_centroids_end - dense_centroids_begin).count() << "[ms]" << std::endl;
  
  std::chrono::steady_clock::time_point dense_partition_begin = std::chrono::steady_clock::now();
  auto dense_partition = a_dense->read_vector_partition(42);
  std::chrono::steady_clock::time_point dense_partition_end = std::chrono::steady_clock::now();
  std::cout << "Partition fetch: " << std::chrono::duration_cast<std::chrono::milliseconds>(dense_partition_end - dense_partition_begin).count() << "[ms]" << std::endl;
  
  std::chrono::steady_clock::time_point dense_ann_begin = std::chrono::steady_clock::now();
  auto ann_results_dense = a_dense->ann_query(dense_partition[0],10,2,4);
  std::chrono::steady_clock::time_point dense_ann_end = std::chrono::steady_clock::now();
  std::cout << "Dense ANN: " << std::chrono::duration_cast<std::chrono::milliseconds>(dense_ann_end - dense_ann_begin).count() << "[ms]" << std::endl;

  std::cout << "Sparse ANN results:" << std::endl;
  for (const auto& ann_result : ann_results_sparse){
    std::cout << "similarity_score: " << ann_result.similarity_score << std::endl;
    std::cout << "Vector: " << std::endl;
    for (int i = 0; i < 128; i++) {
      std::cout << int(ann_result.vector[i]) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Dense ANN results:" << std::endl;
  for (const auto& ann_result : ann_results_dense){
    std::cout << "similarity_score: " << ann_result.similarity_score << std::endl;
    std::cout << "Vector: " << std::endl;
    for (int i = 0; i < 128; i++) {
      std::cout << int(ann_result.vector[i]) << " ";
    }
    std::cout << std::endl;
  }
}