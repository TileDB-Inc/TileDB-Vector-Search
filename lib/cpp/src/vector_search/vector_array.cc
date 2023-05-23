/**
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023-2023 TileDB, Inc.
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
 */

#include <future>

#include "vector_array.h"
#include "dense_vector_array.h"
#include "sparse_vector_array.h"
#include "../utils/vector_distance.h"

namespace tiledb::vector_search {

const std::string INDEX_ARRAY_NAME = "/index";

using namespace tiledb;

VectorArray* VectorArray::open(
      const Context& ctx,
      const std::string& array_uri,
      tiledb_query_type_t query_type) {
    Array index_array(ctx, array_uri + INDEX_ARRAY_NAME, TILEDB_READ);
    std::string array_type = readArrayTypeFromMetadata(&index_array);
    int array_type_version = readArrayTypeVersionFromMetadata(&index_array);
    index_array.close();
    if (array_type == "DENSE") {
        return new DenseVectorArray(ctx, array_uri, query_type, array_type, array_type_version);
    } else if (array_type == "SPARSE" ) {
        return new SparseVectorArray(ctx, array_uri, query_type, array_type, array_type_version);
    } 
    throw TileDBError(
        "[TileDB::VectorArray] Error: Unsopported array type " + array_type);
}

std::string VectorArray::readArrayTypeFromMetadata(Array* index_array){
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;
    index_array->get_metadata("type", &v_type, &v_num, &v);
    if (v_type != TILEDB_STRING_ASCII){
      throw TileDBError(
          "[TileDB::VectorArray] Error: Failed to read vector array type from array metadata");
    }
    return (const char*)v;
}

int VectorArray::readArrayTypeVersionFromMetadata(Array* index_array){
    tiledb_datatype_t v_type;
    uint32_t v_num;
    const void* v;
    index_array->get_metadata("version", &v_type, &v_num, &v);
    if (v_type != TILEDB_INT32){
      throw TileDBError(
          "[TileDB::VectorArray] Error: Failed to read vector array version from array metadata");
    }
    return *(const int*)v;
}

std::vector<SimilarityResult> VectorArray::ann_query(std::vector<uint8_t> query_vector, int k, int nprobe, int nthreads){
  using partition_element = std::pair<double, int>;
  Topk<partition_element>top_partitions(nprobe);
  auto centroids = get_centroids();
  int part_id=0;
  for (const auto& centroid : centroids) {
    auto score = L22(query_vector, centroid);
    top_partitions.insert(partition_element{score, part_id});
    ++part_id;
  }

  std::vector<SimilarityResult> result(k);
  using element = std::pair<double, std::vector<uint8_t>>;
  std::vector<std::future<void>> futs;
  futs.reserve(nthreads);

  std::vector<Topk<element>> top_k;
  for (int part_id = 0; part_id < nprobe; part_id++) {
    top_k.push_back(Topk<element>(k));
  }

  int i=0;
  while(!top_partitions.q_.empty()){
    partition_element e = top_partitions.q_.top();
    futs.emplace_back(std::async(
          std::launch::async, [this, k, i, part_id, e, &query_vector, &top_k]() {
            auto partition = read_vector_partition(e.second);
            for (const auto& vector : partition) {
                auto score = L21(query_vector, vector);
                top_k[i].insert(element{score, vector});
            }
          }));
    top_partitions.q_.pop();
    ++i;
  }

  for (size_t n = 0; n < size(futs); ++n) {
    futs[n].wait();
  }
  Topk<element>top_k_res(k);
  for (int i = 0; i < nprobe; i++) {
    while(!top_k[i].q_.empty()){
      top_k_res.insert(top_k[i].q_.top());
      top_k[i].q_.pop();
    }
  }

  i=k-1;
  while(!top_k_res.q_.empty()){
    const element& t = top_k_res.q_.top();
    result[i] = SimilarityResult{t.first, 0, t.second};
    top_k_res.q_.pop();
    --i;
  }
  return result;
}

std::vector<std::vector<SimilarityResult>> VectorArray::ann_query_batch(std::vector<std::vector<uint8_t>> query_vectors, int k, int nprobe, int nthreads){
      throw TileDBError(
          "[TileDB::VectorArray] Error:ann_query_batch not implemented");
}

}   // namespace tiledb::vector_search
