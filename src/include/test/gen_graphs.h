

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

void dump_coordinates(const std::string& filename, const ColMajorMatrix<float>& X) {
  std::ofstream out(filename);
  for (size_t i = 0; i < X.num_cols(); ++i) {
    out << X(0, i) << " " << X(1, i) << "\n";
  }
  out.close();
}

void dump_edgelist(
    const std::string& filename,
    const std::vector<std::pair<size_t, size_t>>& edges) {
  std::ofstream out(filename);
  for (auto&& [i, j] : edges) {
    out << i << " " << j << "\n";
  }
  out.close();
}

auto gen_grid(size_t M, size_t N) {
  auto dim = 2;
  auto nvectors = M * N;
  auto nedges = (M - 1) * N + M * (N-1);

  std::vector<std::tuple<size_t, size_t>> edges(nedges);

  auto vec_array = ColMajorMatrix<size_t>(dim, nvectors);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      vec_array(0, i * N + j) = i;
      vec_array(1, i * N + j) = j;
    }
  }

  for (size_t i = 0; i < M-1; ++i) {
    for (size_t j = 0; j < N - 1; ++j) {
      edges[i * (N-1) + j] = {i * N + j, i * N + j + 1};
      edges[(M-1) * (N-1) + i * N + j] = {i * N + j, (i+1) * N + j};
    }
  }
  return std::make_tuple(std::move(vec_array), edges);
}