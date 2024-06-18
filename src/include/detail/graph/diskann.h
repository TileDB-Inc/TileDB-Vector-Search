/**
 * @file   diskann.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023 TileDB, Inc.
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
 *
 * @section DESCRIPTION
 *
 *
 */

#ifndef TDB_DISKANN_H
#define TDB_DISKANN_H

#include <filesystem>
#include <fstream>
#include <string>

#include "detail/graph/adj_list.h"
#include "scoring.h"

auto read_diskann_data(const std::string& path) {
  uint32_t npoints{0};
  uint32_t ndim{0};

  std::ifstream binary_file(path, std::ios::binary);
  binary_file.exceptions(std::ifstream::failbit);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + path);
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);

  auto x = ColMajorMatrix<float>(ndim, npoints);

  binary_file.read((char*)x.data(), npoints * ndim * sizeof(float));
  if ((size_t)binary_file.gcount() != (size_t)npoints * ndim * sizeof(float)) {
    throw std::runtime_error("Could not read all data from " + path);
  }

  binary_file.close();

  return x;
}

/**
 * @brief Read a diskann index from disk
 *
 * @note: Currently this will read the test index that has a constant
 * number of neighbors for each vertex.
 * @todo: Implement loop to compare total bytes read to file size,
 * cf DiskANN/src/in_mem_graph_store.cpp:135
 */
auto read_diskann_mem_index(const std::string& index) {
  std::ifstream binary_file(index, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + index);
  }

  uint64_t index_file_size;
  uint32_t max_degree;
  uint32_t medioid;
  uint64_t vamana_frozen_num;

  binary_file.read((char*)&index_file_size, 8);
  binary_file.read((char*)&max_degree, 4);
  binary_file.read((char*)&medioid, 4);
  binary_file.read((char*)&vamana_frozen_num, 8);

  size_t num_nodes = (index_file_size - 24) / (max_degree * 4 + 4);
  auto g = detail::graph::index_adj_list<uint32_t>(num_nodes);

  for (size_t node = 0; node < num_nodes; ++node) {
    uint32_t num_neighbors;
    binary_file.read((char*)&num_neighbors, 4);
    for (size_t i = 0; i < num_neighbors; ++i) {
      uint32_t id;
      binary_file.read((char*)&id, 4);
      g.add_edge(node, id);
    }
    binary_file.seekg(max_degree - num_neighbors, std::ios_base::cur);
  }
  binary_file.close();

  return g;
}

/**
 * Simple reader for reading diskann test index with scores
 * @param index File name for index
 * @param data File name for scores
 * @param num_nodes Number of nodes in the graph
 * @return loaded graph
 * @note A reader for read_diskann_disk_index would be nice but is much more
 * complicated and probably not necessary to implement.
 */
auto read_diskann_mem_index_with_scores(
    const std::string& index, const std::string& data, size_t num_nodes = 0) {
  auto x = read_diskann_data(data);

  // @todo get rid of copy pasta
  std::ifstream binary_file(index, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + index);
  }

  uint64_t index_file_size;
  uint32_t max_degree;
  uint32_t medioid;
  uint64_t vamana_frozen_num;

  binary_file.read((char*)&index_file_size, 8);
  binary_file.read((char*)&max_degree, 4);
  binary_file.read((char*)&medioid, 4);
  binary_file.read((char*)&vamana_frozen_num, 8);

  if (num_nodes == 0) {
    num_nodes = (index_file_size - 24) / (max_degree * 4 + 4);
  }
  auto g = detail::graph::adj_list<float, uint32_t>(num_nodes);

  size_t node = 0;
  while (!binary_file.eof()) {
    size_t where = binary_file.tellg();
    if (where == index_file_size) {
      break;
    }
    uint32_t num_neighbors;
    binary_file.read((char*)&num_neighbors, 4);
    for (size_t i = 0; i < num_neighbors; ++i) {
      uint32_t id;
      binary_file.read((char*)&id, 4);
      if (id >= num_nodes) {
        throw std::runtime_error(
            "[read_diskann_mem_index_with_scores] id >= num_nodes");
      }
      g.add_edge(node, id, l2_distance(x[node], x[id]));
    }
    ++node;
  }
  binary_file.close();
  if (node != num_nodes) {
    throw std::runtime_error(
        "[read_diskann_mem_index_with_scores] node != num_nodes");
  }

  return g;
}

#endif  // TDB_DISKANN_H
