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

#include "vector_array.h"
#include "sparse_vector_array.h"

namespace tiledb::vector_search {

using namespace tiledb;

const std::string INDEX_ARRAY_NAME = "/index";
const std::string READ_ARRAY_NAME = "/read_array";

SparseVectorArray::SparseVectorArray(
      const Context& ctx,
      const std::string& array_uri,
      tiledb_query_type_t query_type,
      const std::string array_type,
      int array_type_version)
      : VectorArray(ctx, array_uri, query_type, array_type)
      , index_array_(Array(ctx, array_uri + INDEX_ARRAY_NAME, TILEDB_READ))
      , read_array_(Array(ctx, array_uri + READ_ARRAY_NAME, TILEDB_READ))
      , array_type_version_(array_type_version) {
        num_partitions_ = index_array_.schema().domain().dimension(0).domain<int>().second+1;
        num_vector_dim_ = index_array_.schema().domain().dimension(1).domain<int>().second+1;
        read_index_data();
}

std::vector<std::vector<float>> SparseVectorArray::get_centroids(){
    return centroids_;
}

std::vector<std::vector<uint8_t>> SparseVectorArray::read_vector_partition(int partition_id){
    Subarray subarray(ctx_, read_array_);
    subarray.add_range(0, partition_id, partition_id);
    std::vector<uint8_t> data(100*num_partitions_*num_vector_dim_);
    Query query(ctx_, read_array_, TILEDB_READ);
    query.set_subarray(subarray)
        .set_layout(TILEDB_ROW_MAJOR)
        .set_data_buffer("vector", data);
    query.submit();
    Query::Status status = query.query_status();
    if(status != Query::Status::COMPLETE){
        throw TileDBError(
            "[TileDB::SparseVectorArray] Error: Partition query status not complete.");
    }
    auto result_num = int(int(query.result_buffer_elements()["vector"].second)/num_vector_dim_);
    std::vector<std::vector<uint8_t>> res;
    res.resize(result_num);
    for (int i = 0; i < result_num; i++) {
        res[i].resize(num_vector_dim_);
    }
    for (int i = 0; i < result_num*num_vector_dim_; i++)
    {
        int row = i / num_vector_dim_;
        int col = i %num_vector_dim_;
        res[row][col] = data[i];
    }
    return res;
}

void SparseVectorArray::read_index_data(){
    Subarray subarray(ctx_, index_array_);
    subarray.add_range(0, 0, num_partitions_-1).add_range(1, 0, num_vector_dim_-1);

    std::vector<float> data(num_vector_dim_ *num_partitions_);
    Query query(ctx_, index_array_, TILEDB_READ);
    query.set_subarray(subarray)
        .set_layout(TILEDB_ROW_MAJOR)
        .set_data_buffer("value", data);
    query.submit();
    std::vector<std::vector<float>> res;
    centroids_.resize(num_partitions_);
    for (int i = 0; i < num_partitions_; i++)
    {
        centroids_[i].resize(num_vector_dim_);
    }
    for (int i = 0; i < data.size(); i++)
    {
        int row = i / num_vector_dim_;
        int col = i %num_vector_dim_;
        centroids_[row][col] = data[i];
    }
}

}   // namespace tiledb::vector_search
