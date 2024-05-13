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
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "detail/linalg/matrix.h"
#include "detail/linalg/matrix_with_ids.h"

auto random_geometric_2D(size_t N) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> coord(-1.0, 1.0);

  auto X = ColMajorMatrixWithIds<float>(2, N);
  std::iota(X.ids(), X.ids() + X.num_ids(), 0);
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
  auto nedges = (M - 1) * N + M * (N - 1);

  std::vector<std::tuple<size_t, size_t>> edges;
  edges.reserve(nedges);

  auto vec_array = ColMajorMatrixWithIds<float>(dim, nvectors);
  std::iota(vec_array.ids(), vec_array.ids() + vec_array.num_ids(), 0);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      vec_array(0, i * N + j) = i;
      vec_array(1, i * N + j) = j;
    }
  }

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (j < N - 1) {
        edges.emplace_back(i * N + j, i * N + j + 1);
      }
      if (i < M - 1) {
        edges.emplace_back(i * N + j, (i + 1) * N + j);
      }
    }
  }

  return std::make_tuple(std::move(vec_array), edges);
}

auto gen_bi_grid(size_t M, size_t N) {
  auto dim = 2;
  auto nvectors = M * N;
  auto nedges = (M - 1) * N + M * (N - 1);

  std::vector<std::tuple<size_t, size_t>> edges;
  edges.reserve(nedges);

  auto vec_array = ColMajorMatrixWithIds<float>(dim, nvectors);
  std::iota(vec_array.ids(), vec_array.ids() + vec_array.num_ids(), 0);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      vec_array(0, i * N + j) = i;
      vec_array(1, i * N + j) = j;
    }
  }

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (j < N - 1) {
        auto src = i * N + j;
        auto dst = i * N + j + 1;
        edges.emplace_back(src, dst);
        edges.emplace_back(dst, src);
      }
      if (i < M - 1) {
        auto src = i * N + j;
        auto dst = (i + 1) * N + j;
        edges.emplace_back(src, dst);
        edges.emplace_back(dst, src);
      }
    }
  }

  return std::make_tuple(std::move(vec_array), edges);
}

auto gen_star_grid(size_t M, size_t N) {
  auto dim = 2;
  auto nvectors = M * N;
  auto nedges = (M - 1) * N + M * (N - 1);

  std::vector<std::tuple<size_t, size_t>> edges;
  edges.reserve(nedges);

  auto vec_array = ColMajorMatrix<size_t>(dim, nvectors);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      vec_array(0, i * N + j) = i;
      vec_array(1, i * N + j) = j;
    }
  }

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (j < N - 1) {
        auto src = i * N + j;
        auto dst = i * N + j + 1;
        edges.emplace_back(src, dst);
        if (i < M - 1) {
          auto src = i * N + j;
          auto dst = (i + 1) * N + j + 1;
          edges.emplace_back(src, dst);
        }
      }
      if (i < M - 1) {
        auto src = i * N + j;
        auto dst = (i + 1) * N + j;
        edges.emplace_back(src, dst);
        if (j < N - 1) {
          auto src = i * N + j;
          auto dst = (i + 1) * N + j + 1;
          edges.emplace_back(src, dst);
        }
        if (j > 0) {
          auto src = i * N + j;
          auto dst = (i + 1) * N + j - 1;
          edges.emplace_back(src, dst);
        }
      }
    }
  }

  return std::make_tuple(std::move(vec_array), edges);
}

template <class F = float, class T = uint8_t>
auto normalize_matrix(
    const ColMajorMatrixWithIds<F>& from,
    size_t min_val = 0,
    size_t max_val = 127) {
  auto&& [min_loc, max_loc] = std::minmax_element(
      from.data(), from.data() + from.num_rows() * from.num_cols());
  auto min = *min_loc;
  auto max = *max_loc;

  auto to = ColMajorMatrixWithIds<T>(from.num_rows(), from.num_cols());
  std::copy(from.ids(), from.ids() + from.num_ids(), to.ids());
  for (size_t i = 0; i < from.num_rows(); ++i) {
    for (size_t j = 0; j < from.num_cols(); ++j) {
      auto foo = from(i, j) - min;
      auto bar = max - min;
      auto baz = foo / bar;
      to(i, j) =
          (T)(((max_val - min_val) * ((from(i, j) - min) / (max - min))) +
              min_val);
    }
  }
  return to;
}

template <class T = float>
auto build_hypercube(size_t k_near, size_t k_far, size_t seed = 0) {
  const bool debug = false;

  size_t N = 8 * (k_near + k_far + 1);

  std::random_device rd;
  std::mt19937 gen(seed == 0 ? rd() : seed);
  std::uniform_real_distribution<float> dist_near(-0.1, 0.1);
  std::uniform_real_distribution<float> dist_far(0.2, 0.3);
  std::uniform_int_distribution<int> heads(0, 1);

  ColMajorMatrixWithIds<float> nn_hypercube(3, N + 1);
  std::iota(nn_hypercube.ids(), nn_hypercube.ids() + nn_hypercube.num_ids(), 0);
  size_t n{0};
  nn_hypercube(0, n) = 0;
  nn_hypercube(1, n) = 0;
  nn_hypercube(2, n) = 0;
  ++n;

  for (auto i : {-1, 1}) {
    for (auto j : {-1, 1}) {
      for (auto k : {-1, 1}) {
        nn_hypercube(0, n) = i;
        nn_hypercube(1, n) = j;
        nn_hypercube(2, n) = k;
        ++n;
      }
    }
  }

  for (size_t m = 0; m < k_near; ++m) {
    for (auto i : {-1, 1}) {
      for (auto j : {-1, 1}) {
        for (auto k : {-1, 1}) {
          nn_hypercube(0, n) = i + dist_near(gen);
          nn_hypercube(1, n) = j + dist_near(gen);
          nn_hypercube(2, n) = k + dist_near(gen);
          ++n;
        }
      }
    }
  }

  for (size_t m = 0; m < k_far; ++m) {
    for (auto i : {-1, 1}) {
      for (auto j : {-1, 1}) {
        for (auto k : {-1, 1}) {
          nn_hypercube(0, n) = i + (heads(gen) ? 1 : -1) * dist_far(gen);
          nn_hypercube(1, n) = j + (heads(gen) ? 1 : -1) * dist_far(gen);
          nn_hypercube(2, n) = k + (heads(gen) ? 1 : -1) * dist_far(gen);
          ++n;
        }
      }
    }
  }

  if (debug) {
    std::cout << "Hypercube stats:" << std::endl;
    std::cout << "  num_rows: " << nn_hypercube.num_rows() << " ";
    std::cout << "  num_cols: " << nn_hypercube.num_cols() << std::endl;

    std::cout << "Hypercube (transpose):" << std::endl;
    for (size_t j = 0; j < nn_hypercube.num_cols(); ++j) {
      for (size_t i = 0; i < nn_hypercube.num_rows(); ++i) {
        std::cout << nn_hypercube(i, j) << ", ";
      }
      std::cout << std::endl;
    }
  }
  return normalize_matrix<float, T>(nn_hypercube);
}

#endif  // TDB_GEN_GRAPHS_H
