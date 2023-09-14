/**
* @file   gen_graphs.h
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
* Program to test performance of different min-heap implementations.
*
*/

#ifndef TDB_GEN_GRAPHS_H
#define TDB_GEN_GRAPHS_H

#include <cstddef>
#include <random>
#include <string>
#include <fstream>
#include <tuple>
#include <vector>
#include "detail/linalg/matrix.h"

auto random_geometric_2D(size_t N) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> coord(-1.0, 1.0);

  auto X = ColMajorMatrix<float>(2, N);

  for (size_t i = 0; i < N; ++i) {
    X(0, i) = coord(gen);
    X(1, i) = coord(gen);
  }

  return X;
}

template <class T>
void dump_coordinates(const std::string& filename, const ColMajorMatrix<T>& X) {
  std::ofstream out(filename);
  for (size_t i = 0; i < X.num_cols(); ++i) {
    out << X(0, i) << " " << X(1, i) << "\n";
  }
  out.close();
}

void dump_edgelist(
    const std::string& filename,
    const std::vector<std::tuple<size_t, size_t>>& edges) {
  std::ofstream out(filename);
  out << "# source target\n";
  for (auto&& [i, j] : edges) {
    out << i << " " << j << "\n";
  }
  out.close();
}

auto gen_uni_grid(size_t M, size_t N) {
  auto dim = 2;
  auto nvectors = M * N;
  auto nedges = (M - 1) * N + M * (N-1);

  std::vector<std::tuple<size_t, size_t>> edges;
  edges.reserve(nedges);

  auto vec_array = ColMajorMatrix<size_t>(dim, nvectors);

  size_t k = 0;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      vec_array(0, i * N + j) = i;
      vec_array(1, i * N + j) = j;
      ++k;
    }
  }


  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (j < N-1) {
        edges.emplace_back(i * N + j, i * N + j + 1);
      }
      if (i < M-1) {
        edges.emplace_back(i * N + j, (i + 1) * N + j);
      }
    }
  }


  return std::make_tuple(std::move(vec_array), edges);
}

auto gen_bi_grid(size_t M, size_t N) {
  auto dim = 2;
  auto nvectors = M * N;
  auto nedges = (M - 1) * N + M * (N-1);

  std::vector<std::tuple<size_t, size_t>> edges;
  edges.reserve(nedges);

  auto vec_array = ColMajorMatrix<size_t>(dim, nvectors);

  size_t k = 0;
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      vec_array(0, i * N + j) = i;
      vec_array(1, i * N + j) = j;
      ++k;
    }
  }


  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (j < N-1) {
        auto src = i * N + j;
        auto dst = i * N + j + 1;
        edges.emplace_back(src, dst);
        edges.emplace_back(dst, src);
      }
      if (i < M-1) {
        auto src = i * N + j;
        auto dst = (i + 1) * N + j;
        edges.emplace_back(src, dst);
        edges.emplace_back(dst, src);
      }
    }
  }


  return std::make_tuple(std::move(vec_array), edges);
}


#endif //TDB_GEN_GRAPHS_H